"""
conversation_memory.py -- Multi-Turn Conversation Memory for RAG

Manage conversation history so chatbot cna handle follow up questions

HOW IT WORKS (THE KEY INSIGHT)
The Anthropic Messages API already supports multi-turn conversations.
need to send ALL previous messages with each new request:

    client.messages.create(
        messages=[
            {"role": "user",      "content": "Tell me about DIM_STORE"},
            {"role": "assistant", "content": "DIM_STORE is a dimension..."},
            {"role": "user",      "content": "What about its foreign keys?"},
        ]
    )
Claude reads the entire message list and understands "its" = DIM_STORE.
The ENGINEERING CHALLENGE is not the API call -- it is managing the
message list so it does not:
  1. Exceed the model's context window (Claude Sonnet: ~200K tokens)
  2. Cost too much (every token in messages costs money per call)
  3. Include irrelevant old context (noise hurts answer quality)

OUR SOLUTION: SLIDING WINDOW + SUMMARY
---------------------------------------
We keep:
  - The last N message pairs (the "window") in full detail
  - A compressed summary of everything before the window

This mirrors human memory:
  - Recent: "5 minutes ago we talked about DIM_STORE foreign keys"
  - Older:  "Earlier we discussed several dimension tables"

dbt ANALOGY:
  The sliding window = incremental model (processes recent data in detail)
  The summary   = snapshot (compressed view of historical state)
  Dropping old messages = data retention policy (delete after N days)
"""
from collections import deque
import json
import time

#Token estimation
# We need to estimate how many tokens a message will consume WITHOUT
# making an API call. This lets us decide when to trim the window.
#
# PYTHON REFRESHER: Integer division with //
# -------------------------------------------
# 15 // 4  -> 3   (drops the remainder)
# 15 / 4   -> 3.75 (keeps the decimal)
#
# We use // because token counts must be integers.
# We add 1 as a safety margin: better to slightly overestimate
# than to exceed the context window.

def estimate_tokens(text:str)-> int:
    if not text:
        return 0
    # +1 is a safety margin so we never return 0 for non-empty text
    return len(text) // 4 +1
#THE CONVERSATION MEMORY CLASS
class ConversationMemory:
    """
    Manages multi-turn conversation history with a sliding window.

    DESIGN DECISIONS
    ----------------
    1. Window size is measured in TURNS, not messages.
       One turn = one user message + one assistant response = 2 messages.
       max_turns=5 means we keep the last 10 messages (5 pairs).

    2. When messages fall off the window, we generate a summary.
       The summary is prepended to the system prompt, not added as
       a user message. This prevents Claude from treating the summary
       as something the user said.

    3. Token budget is a soft limit. We estimate tokens and warn,
       but do not truncate mid-message. If a single response is
       huge, we let it through and trim on the NEXT turn.

    PARAMETERS
    ----------
    max_turns : int
        How many recent Q&A pairs to keep in full detail.
        Default 5 means the last 5 exchanges are preserved verbatim.
        Older exchanges are summarized.

        TUNING GUIDANCE:
        - 3 turns: Minimal context, lowest cost, fastest. Good for
          simple lookup chatbots where follow-ups are rare.
        - 5 turns: Good default. Handles "tell me about X" then
          "what about Y" then "compare X and Y" patterns.
        - 10 turns: Rich context, higher cost. Good for complex
          multi-step analysis conversations.

    max_tokens : int
        Soft limit on total tokens in the message history.
        Default 4000 is ~16K characters, which is well within
        Claude Sonnet's 200K context window while leaving room
        for the system prompt (~1K tokens) and retrieved chunks
        (~2K tokens per chunk * TOP_K chunks).

        COST MATH:
        Claude Sonnet input pricing: $3 per million tokens
        4000 tokens * $3/1M = $0.012 per API call for memory alone
        At 100 queries/day = $1.20/day for memory overhead

    PYTHON REFRESHER: Default parameter values
    -------------------------------------------
    def __init__(self, max_turns=5, max_tokens=4000):

    This means you can create instances in three ways:
      ConversationMemory()                    # max_turns=5, max_tokens=4000
      ConversationMemory(max_turns=10)        # max_turns=10, max_tokens=4000
      ConversationMemory(3, 2000)             # max_turns=3, max_tokens=2000

    Default values are evaluated ONCE when the function is defined,
    not each time it is called. This matters for mutable defaults:

    GOTCHA: Never use a mutable default like def f(x=[]):
      def bad(items=[]):    # This list is shared across ALL calls!
          items.append(1)
          return items
      bad()  -> [1]
      bad()  -> [1, 1]     # Same list object, not a new one!

    For our class, all defaults are immutable (int), so this is safe.
    """
    def __init__(self,max_turns:int = 5, max_tokens:int =40000):
        """
        This sliding window: a deque of message dicts

        each message is {"role":"user", "content":"......"}
        
        maxalen = max_turns *2
        as for each turns there are 2 messages 1 user 1 assistant

        when the deque is fully appended, the OLDEST item is removed
        """
        self.max_turns = max_turns
        self.max_tokens = max_tokens
        self.messages = deque(maxlen=max_turns * 2)

        #Summary of older context (everything outside of the windows)
        #This get updated whenevery message are drop from deque
        #the summary then inject into system prompt not messages list
        
        self.summary = None
        #Metadata for debugging
        self.total_turns = 0 #lifetime turn counter
        self.evicted_turns = 0 #turns were summarized
        self.estimated_tokens = 0
        self.created_ad = time.time() #When session started

    def add_turn(self,user_message:str,assistant_message:str)-> dict:
        """
        Record one completed Q&A exchange

        this is called after the assistant has generated a response
        
        UX Experience 
        - User type in question
        - retrieve() finds relevant chunks
        - ask_claude() generate an answer
        - memory.add_turn(question,answer)
        - Display answer in UI
        RETURNS
      
        dict with keys:
            "turn_number": int -- which turn this is (1-indexed)
            "was_evicted": bool -- whether an older turn was pushed out
            "estimated_tokens": int -- current total token estimate
            "window_size": int -- how many messages are in the window
        """
        #Check if this apopend will evict old message
        was_full = len(self.messages) >= self.messages.maxlen
        evicted_message = []

        if was_full:
            evicted_message = [self.messages[0],self.messages[1]]
            self.evicted_turns += 1
        #Append new turn, if deque is full then the oldest 2 are auto evicted
        self.messages.append({"role":"user","content":user_message})
        self.messages.append({"role":"assistant","content":assistant_message})
        self.total_turns+=1

        #Update summary if message is evicted
        if evicted_message:
            self._update_summary(evicted_message)
        #Recalucate esitmated token usage
        self.estimated_tokens = self._estimate_total_tokens()
        return {
            "turn_number":self.total_turns,
            "was_evicted":was_full,
            "estimated_tokens":self.estimated_tokens,
            "window_size":len(self.messages),
        }
    
    def _update_summary(self,evicted_messages:list[dict]) -> None:
        """
        private method only to be use by add_turn()
        """

        evicted_preview_parts = []
        for msg in evicted_messages:
            role = msg["role"]
            preview = msg["content"][:100].replace("\n"," ")
            evicted_preview_parts.append(f"{role}:{preview}")
        evicted_preview = " | ".join(evicted_preview_parts)

        #Build and/pr append running summary 
        if self.summary is None:
            self.summary = (
                f"Earlier in this conversation, the user dicussed:"
                f" {evicted_preview}"
            )
        #Append to existing summary keeping concise
        else:
            new_part = f" Then discussed: {evicted_preview}"
            self.summary = self.summary + new_part\
            
            #We cap summary at  500 caharacter to prevent unbounded groth

            if len(self.summary) > 500:
                self.summary = self.summary[:497]+'...'


    def _estimate_total_tokens(self) -> int:
        """
        Estimate the total token count of all messages in the window plus summary
        """
        total = 0
        for msg in self.messages:
            total += estimate_tokens(msg["content"])
            #Each message has overhed role name,json formatting, hence we expect 4 extrac per message
            total += 4
        if self.summary:
            total += estimate_tokens(self.summary)
        return total
    
    def get_messages_for_api(self) -> list[dict]:
        """
        Get the message history fromatted for Claude Mesasge API

        This is the method app.py calls right before making an API request

        This filters out any extra fields that app.py might
        have stored in session_state messages (like "citations" or
        "debug_info"). The Anthropic API only reads "role" and "content".

        RETURN
        list[dict]
        Messages formatted for the API. Each dict has exactly
            two keys: "role" (str) and "content" (str).

        EXAMPLE
        -------
        >>> memory = ConversationMemory(max_turns=3)
        >>> memory.add_turn("What is DIM_STORE?", "DIM_STORE is a dimension...")
        >>> memory.add_turn("What are its keys?", "It has SK_STORE_KEY...")
        >>> memory.get_messages_for_api()
        [
            {"role": "user", "content": "What is DIM_STORE?"},
            {"role": "assistant", "content": "DIM_STORE is a dimension..."},
            {"role": "user", "content": "What are its keys?"},
            {"role": "assistant", "content": "It has SK_STORE_KEY..."},
        ]
        """
        return [
            {"role": msg["role"], "content": msg["content"]}
            for msg in self.messages
        ]
    
    def get_system_prompt_addition(self) -> str:
        """
        Get the summary text to prepend to the system prompt.

        If there is no summary (conversation is short enough to fit
        entirely in the window), returns an empty string.

        This is appended to IMPROVED_SYSTEM_PROMPT in rag.py:

            system_prompt = IMPROVED_SYSTEM_PROMPT
            summary_addition = memory.get_system_prompt_addition()
            if summary_addition:
                system_prompt += "\\n\\n" + summary_addition

        WHY SYSTEM PROMPT AND NOT A USER MESSAGE?
        -------------------------------------------
        The summary is context about the conversation, not something
        the user said. Putting it in the system prompt tells Claude:
        "This is background information you should know" rather than
        "The user said this."

        If we put it as a user message, Claude might try to respond
        to it directly: "Yes, I remember discussing DIM_STORE..."
        which wastes tokens and is weird.
        """
        if not self.summary:
            return ""
        return (
            f"CONVERSATION CONTEXT:\n"
            f"---\n"
            f"{self.summary}\n"
            f"---\n"
            f"Use the above context to understand references to earlier "
            f"topics in this conversation (e.g., pronouns like 'it', "
            f"'that table', 'the same columns')."
        )
    
    def clear(self) -> None:
        """
        Reset all memory state
        Called when click "New conversation"
        """
        self.messages.clear()
        self.summary = None
        self.total_turns = 0
        self.evicted_turns = 0
        self.estimated_tokens = 0

    def get_stats(self) -> dict:
        """
        Return memory statistics for the debug panel in app.py.

        These stats help  understand memory behavior during development.
        In production,  needs to log these to our observability system.
        """
        return {
            "total_turns": self.total_turns,
            "window_turns": len(self.messages) // 2,
            "max_turns": self.max_turns,
            "evicted_turns": self.evicted_turns,
            "has_summary": self.summary is not None,
            "estimated_tokens": self.estimated_tokens,
            "max_tokens": self.max_tokens,
            "token_utilization": (
                f"{self.estimated_tokens / self.max_tokens * 100:.0f}%"
                if self.max_tokens > 0 else "N/A"
            ),
        }
    
    def is_follow_up(self, query: str) -> bool:
        """
        Heuristic to detect if a query is a follow-up to the previous turn.

        This is used by query_router.py to decide whether to include
        conversation memory in the API call.

        HOW IT WORKS
        ------------
        We check for signals that suggest the query references something
        from the previous turn:
          - Pronouns: "it", "its", "that", "those", "them"
          - Short queries: "and the keys?" (< 6 words, no table name)
          - Explicit references: "the same table", "that one"

        This is a HEURISTIC, not a classifier. It will have false
        positives and false negatives. But it is fast (no API call)
        and good enough for routing decisions.

        dbt ANALOGY: This is like a simple WHERE clause filter.
        It does not perfectly classify every row, but it filters
        out the obvious non-matches cheaply before expensive processing.

        RETURNS
        -------
        bool
            True if the query appears to be a follow-up.
        """
        #No distry means it cannot be a follow up
        if self.total_turns == 0:
            return False
        
        query_lower = query.lower().strip()
        words = query_lower.split()

        #Signal 1: Pronouns that reference previous context.
        #Below words are likely in a follow up query
        pronoun_signals = [
            "it", "its", "that", "those", "them", "this",
            "the same", "that table", "that one", "which ones",
            "what about", "how about", "and the", "also",
            "same table", "same columns",
        ]

        if any(signal in query_lower for signal in pronoun_signals):
            return True
        
        # Signal 2: Very short query without a table name.
        # "and the keys?" is likely a follow-up.
        # "What is FACT_SALES_ORDER?" is likely a new topic.
        has_table_name = any(
            prefix in query.upper()
            for prefix in ["FACT_", "DIM_", "BRIDGE_"]
        )

        if len(words) <= 5 and not has_table_name:
            return True
        
        return False

#SECTION 3: ask_claude_with_memory() -- THE UPGRADED GENERATION CALL
# This function wraps  existing ask_claude() with conversation
# memory support. It is the function that app.py will call instead
# of ask_claude() when memory is enabled.
#
# CRITICAL DESIGN DECISION: This is a SEPARATE function, not a
# modification to ask_claude() in rag.py. This means:
#   - rag.py remains untouched (eval.py still works)
#   - Memory is opt-in (app.py chooses which function to call)
#   - can A/B test memory vs no-memory easily

def ask_claude_with_memory(
        query:str,
        context_chunks:list[dict],
        memory:ConversationMemory,
        system_prompt:str = None
):
    """
    Generate an answer using Claude with conversation memory.

    This is functionally identical to ask_claude() in rag.py, except:
    1. It includes previous conversation messages in the API call
    2. It includes a summary of older context in the system prompt
    3. It adds the new Q&A pair to memory after getting a response

    PARAMETERS
    ----------
    query : str
        The current user question.
    context_chunks : list[dict]
        Retrieved chunks from ChromaDB (same as ask_claude).
    memory : ConversationMemory
        The conversation memory instance for this session.
    system_prompt : str, optional
        Override the system prompt. If None, uses IMPROVED_SYSTEM_PROMPT
        from rag.py.

    RETURNS
    -------
    str
        Claude's response text.
    """
    import anthropic
    from rag import IMPROVED_SYSTEM_PROMPT as DEFAULT_PROMPT

    #Build system prompt with convesation memory
    effective_prompt = system_prompt or DEFAULT_PROMPT
    summary_addition = memory.get_system_prompt_addition()
    if summary_addition:
        effective_prompt = effective_prompt + "\n\n" + summary_addition

    #Build context string form retrieved chunks
    #same as ask_claude()
    context_parts = []
    for chunk in context_chunks:
        label_parts = []
        if chunk.get("table_name"):
            label_parts.append(chunk['table_name'])
        if chunk.get("doc_type") and chunk["doc_type"] != "text":
            label_parts.append(chunk["doc_type"])
        label = " - ".join(label_parts) if label_parts else chunk.get("source", "unknown")
        context_parts.append(f"[Source: {label}]\n{chunk['text']}")

    context = "\n\n---\n\n".join(context_parts)

    #Build the messages list with conversation history
    prior_messages =  memory.get_messages_for_api()
    current_message = {
        "role": "user",
        "content": f"""Context from documents:

{context}

---

Question: {query}""",
    }

    all_messages = prior_messages + [current_message]

    #Call Anthropic API
    client = anthropic.Anthropic()
    response = client.messages.create(
        model="claude-sonnet-4-5-20250929",
        max_tokens=1024,
        system=effective_prompt,
        messages=all_messages,
    )

    answer = response.content[0].text
      

    #Record this turn in memory
    memory.add_turn(query, answer)

    return answer



#TESTING SCRIPT
if __name__ == "__main__":
    print("=" * 60)
    print("CONVERSATION MEMORY -- STANDALONE TEST")
    print("=" * 60)

    # Create a memory with a small window for testing
    memory = ConversationMemory(max_turns=3)

    print(f"\nCreated memory with max_turns={memory.max_turns}")
    print(f"Window capacity: {memory.messages.maxlen} messages")

    # Simulate 5 turns (more than the window of 3)
    test_turns = [
        ("What is DIM_STORE?", "DIM_STORE is a dimension table representing store locations."),
        ("What about its keys?", "DIM_STORE has SK_STORE_KEY as surrogate and BK_STORE_KEY as business key."),
        ("Which facts reference it?", "FACT_SALES_ORDER and FACT_STORE_INVENTORY both have FK_STORE_KEY."),
        ("Tell me about FACT_SALES_ORDER", "FACT_SALES_ORDER records individual sales transactions."),
        ("What is the grain?", "The grain is one row per sales order line item."),
    ]

    for i, (user_msg, assistant_msg) in enumerate(test_turns, 1):
        print(f"\n--- Turn {i} ---")
        print(f"  User: {user_msg}")
        print(f"  Assistant: {assistant_msg[:60]}...")

        result = memory.add_turn(user_msg, assistant_msg)
        print(f"  Result: {result}")

    # Show final state
    print("\n" + "=" * 60)
    print("FINAL STATE")
    print("=" * 60)

    stats = memory.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")

    print(f"\nSummary: {memory.summary}")

    print(f"\nMessages in window:")
    for msg in memory.get_messages_for_api():
        preview = msg["content"][:60]
        print(f"  [{msg['role']}] {preview}...")

    # Test follow-up detection
    print(f"\n--- Follow-up Detection ---")
    test_queries = [
        "What about its columns?",        # Should be True (pronoun "its")
        "Tell me about DIM_DATE",          # Should be False (new topic)
        "and the keys?",                   # Should be True (short, no table)
        "Compare FACT_SALES and DIM_STORE", # Should be False (new topic)
        "those foreign keys",              # Should be True (pronoun "those")
    ]

    for q in test_queries:
        result = memory.is_follow_up(q)
        print(f"  '{q}' -> is_follow_up={result}")

    print("\nAll tests passed. conversation_memory.py is ready.")


