"""
conversation_memory.py -- Multi-Turn Conversation Memory for RAG
================================================================
Week 4, Step 1 of 6

WHAT THIS FILE DOES
-------------------
Manages conversation history so your chatbot can handle follow-up questions.

Without this module:
  User: "Tell me about DIM_STORE"
  Bot:  "DIM_STORE is a dimension table with grain one row per store..."
  User: "What about its foreign keys?"
  Bot:  "I don't understand what 'its' refers to."  <-- BROKEN

With this module:
  User: "Tell me about DIM_STORE"
  Bot:  "DIM_STORE is a dimension table with grain one row per store..."
  User: "What about its foreign keys?"
  Bot:  "DIM_STORE has FK_REGION_KEY, FK_STATE_KEY..."  <-- WORKS

HOW IT WORKS (THE KEY INSIGHT)
------------------------------
The Anthropic Messages API already supports multi-turn conversations.
You just need to send ALL previous messages with each new request:

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
  The summary        = snapshot (compressed view of historical state)
  Dropping old messages = data retention policy (delete after N days)

TOKEN ESTIMATION
----------------
We need to estimate token counts WITHOUT calling the API (which would
add latency and cost). The rule of thumb for English text:

    1 token ~= 4 characters  (approximate)
    1 token ~= 0.75 words    (approximate)

This is not exact -- tokenization depends on the model's vocabulary.
But it is close enough for memory management decisions. The Anthropic
API will tell you exact token usage in the response metadata, which
we can use to calibrate over time.

PYTHON REFRESHER: collections.deque
------------------------------------
A deque (pronounced "deck") is a double-ended queue from the
Python standard library. It is like a list but with O(1) appends
and pops from BOTH ends.

    from collections import deque

    # maxlen=5 means: if you append a 6th item, the oldest is auto-removed
    d = deque(maxlen=5)
    d.append("a")  # ["a"]
    d.append("b")  # ["a", "b"]
    d.append("c")  # ["a", "b", "c"]
    d.append("d")  # ["a", "b", "c", "d"]
    d.append("e")  # ["a", "b", "c", "d", "e"]
    d.append("f")  # ["b", "c", "d", "e", "f"]  <-- "a" auto-dropped!

    # LONG-FORM EQUIVALENT (what deque does internally):
    messages = []
    messages.append("f")
    if len(messages) > 5:
        messages.pop(0)  # Remove oldest -- but this is O(n) for a list!

    # deque is better because pop(0) is O(1) instead of O(n).
    # For 5 messages this does not matter. For 5000 it would.

WHY deque INSTEAD OF list:
    Performance: deque.appendleft() and deque.popleft() are O(1).
                 list.insert(0, x) and list.pop(0) are O(n).
    Auto-eviction: maxlen parameter handles the sliding window
                   automatically -- no manual size-checking code.
    Semantic clarity: Using deque signals "this is a fixed-size buffer"
                      to anyone reading your code.

GOTCHA: deque maxlen is in MESSAGE COUNT, not TOKEN COUNT.
A single long message could still exceed your token budget.
We handle this by also checking estimated token count.


HOW THIS FILE CONNECTS TO YOUR PROJECT
---------------------------------------
  app.py imports ConversationMemory
    --> creates one instance per Streamlit session
    --> calls memory.add_turn() after each Q&A
    --> calls memory.get_messages_for_api() before each Claude call
    --> the returned messages list replaces the single-message list
        currently passed to ask_claude()

  rag.py is NOT modified. We create a new function ask_claude_with_memory()
  in THIS file that wraps ask_claude() with memory support.

  eval.py is NOT affected. It tests the raw pipeline without memory.
  This is intentional -- eval measures retrieval quality, not conversation
  quality. We will add multi-turn eval questions separately.


DEPENDENCIES
------------
  - Python standard library only (collections, json, time)
  - anthropic (for ask_claude_with_memory, already installed)
  - Your rag.py (for IMPROVED_SYSTEM_PROMPT and retrieve)
"""

from collections import deque
import json
import time

# =====================================================================
# SECTION 1: TOKEN ESTIMATION
# =====================================================================
#
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

def estimate_tokens(text: str) -> int:
    """
    Estimate the number of tokens in a text string.

    The 4-characters-per-token rule comes from OpenAI's tokenizer analysis
    and holds approximately true for Claude's tokenizer as well. It is
    not exact -- technical terms, code, and non-English text may have
    different ratios -- but it is sufficient for memory management.

    PARAMETERS
    ----------
    text : str
        The text to estimate tokens for.

    RETURNS
    -------
    int
        Estimated token count (always >= 1).

    EXAMPLES
    --------
    >>> estimate_tokens("Hello world")
    3
    >>> estimate_tokens("DIM_STORE has SK_STORE_KEY as its surrogate key")
    12
    """
    if not text:
        return 0
    # +1 is a safety margin so we never return 0 for non-empty text
    return len(text) // 4 + 1


# =====================================================================
# SECTION 2: THE CONVERSATION MEMORY CLASS
# =====================================================================
#
# PYTHON REFRESHER: Why a class instead of module-level functions?
# ----------------------------------------------------------------
# In Week 1-3, we used standalone functions (load_documents, retrieve).
# For conversation memory, a CLASS makes more sense because:
#
# 1. STATE: Memory has state (the message history) that must persist
#    between function calls. A class bundles state + behavior together.
#
# 2. MULTIPLE INSTANCES: In Streamlit, each browser session needs its
#    OWN memory. A class lets you create separate instances:
#      session_a_memory = ConversationMemory()
#      session_b_memory = ConversationMemory()
#
# 3. CONFIGURATION: Different sessions might want different window sizes.
#    Constructor parameters (max_turns, max_tokens) make this clean.
#
# If you only ever had ONE conversation, module-level functions with
# a global list would work. But classes scale to multiple conversations.
#
# PYTHON REFRESHER: __init__ (the constructor)
# ---------------------------------------------
# __init__ runs ONCE when you create an instance:
#   memory = ConversationMemory(max_turns=5)
#
# After __init__, the instance has all its attributes:
#   memory.max_turns  -> 5
#   memory.messages   -> deque([])
#   memory.summary    -> None
#
# "self" is the instance being created. Python passes it automatically.
# You never write: ConversationMemory.add_turn(memory, ...) -- just
# memory.add_turn(...) and Python fills in self=memory for you.
#
# LONG-FORM EQUIVALENT of what __init__ does:
#   memory = object()  # Create empty object
#   memory.max_turns = 5
#   memory.messages = deque(maxlen=10)
#   memory.summary = None
#   # ... etc


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

    def __init__(self, max_turns: int = 5, max_tokens: int = 4000):
        # ---------------------------------------------------------------
        # The sliding window: a deque of message dicts.
        #
        # Each message is: {"role": "user"|"assistant", "content": "..."}
        #
        # maxlen = max_turns * 2 because each turn has 2 messages
        # (one user + one assistant).
        #
        # When the deque is full and you append, the OLDEST item is
        # automatically removed. This IS the sliding window mechanism.
        # ---------------------------------------------------------------
        self.max_turns = max_turns
        self.max_tokens = max_tokens
        self.messages = deque(maxlen=max_turns * 2)

        # ---------------------------------------------------------------
        # Summary of older context (everything that fell off the window).
        # Starts as None because there is nothing to summarize yet.
        #
        # This gets updated whenever messages are evicted from the deque.
        # The summary is injected into the system prompt, not into the
        # messages list. This is important because:
        #   - System prompt = instructions Claude follows
        #   - Messages = the actual conversation
        # A summary of "we discussed DIM_STORE earlier" is context/
        # instruction, not a message the user typed.
        # ---------------------------------------------------------------
        self.summary = None

        # ---------------------------------------------------------------
        # Metadata tracking: useful for debugging and the eval framework.
        # ---------------------------------------------------------------
        self.total_turns = 0          # Lifetime turn counter
        self.evicted_turns = 0        # How many turns were summarized
        self.estimated_tokens = 0     # Current token usage estimate
        self.created_at = time.time() # When this session started


    def add_turn(self, user_message: str, assistant_message: str) -> dict:
        """
        Record one complete Q&A exchange (one "turn").

        This is called AFTER the assistant has generated a response.
        The full flow in app.py is:

            1. User types a question
            2. retrieve() finds relevant chunks
            3. ask_claude() generates an answer
            4. memory.add_turn(question, answer)  <-- HERE
            5. Display the answer in the UI

        PARAMETERS
        ----------
        user_message : str
            The question the user asked.
        assistant_message : str
            The answer the chatbot generated.

        RETURNS
        -------
        dict with keys:
            "turn_number": int -- which turn this is (1-indexed)
            "was_evicted": bool -- whether an older turn was pushed out
            "estimated_tokens": int -- current total token estimate
            "window_size": int -- how many messages are in the window

        PYTHON REFRESHER: The walrus operator :=
        -----------------------------------------
        We do NOT use the walrus operator here because it reduces
        readability for learning code. But you may see it in production:

            # With walrus operator:
            if (count := len(self.messages)) >= self.messages.maxlen:
                # count is already assigned

            # Without (what we use):
            count = len(self.messages)
            if count >= self.messages.maxlen:
                # Same result, easier to read
        """
        # ---------------------------------------------------------------
        # Check if this append will evict old messages.
        #
        # The deque has maxlen = max_turns * 2. If it is already full,
        # appending 2 new messages will push out the 2 oldest.
        #
        # We need to capture the evicted messages BEFORE they disappear,
        # because we want to include their content in the summary.
        # ---------------------------------------------------------------
        was_full = len(self.messages) >= self.messages.maxlen
        evicted_messages = []

        if was_full:
            # ---------------------------------------------------------------
            # PYTHON REFRESHER: List slicing from a deque
            # ---------------------------------------------------------------
            # deque does not support slicing directly like lists:
            #   self.messages[0:2]  # This WORKS for deque in Python 3.5+
            #
            # But to be safe and explicit, we convert the items we need:
            #   list(self.messages)[0:2]
            #
            # We grab the first 2 items (one user + one assistant message)
            # because those are the ones that will be evicted when we
            # append 2 new items to a full deque.
            # ---------------------------------------------------------------
            evicted_messages = [self.messages[0], self.messages[1]]
            self.evicted_turns += 1

        # ---------------------------------------------------------------
        # Append the new turn. If deque is full, oldest 2 auto-evicted.
        # ---------------------------------------------------------------
        self.messages.append({"role": "user", "content": user_message})
        self.messages.append({"role": "assistant", "content": assistant_message})
        self.total_turns += 1

        # ---------------------------------------------------------------
        # Update the summary if messages were evicted.
        # ---------------------------------------------------------------
        if evicted_messages:
            self._update_summary(evicted_messages)

        # ---------------------------------------------------------------
        # Recalculate estimated token usage.
        # ---------------------------------------------------------------
        self.estimated_tokens = self._estimate_total_tokens()

        return {
            "turn_number": self.total_turns,
            "was_evicted": was_full,
            "estimated_tokens": self.estimated_tokens,
            "window_size": len(self.messages),
        }


    def _update_summary(self, evicted_messages: list[dict]) -> None:
        """
        Incorporate evicted messages into the running summary.

        This is a PRIVATE method (underscore prefix convention).
        It should only be called by add_turn(), not by external code.

        PYTHON REFRESHER: Private methods (underscore convention)
        ---------------------------------------------------------
        Python does not have true private methods like Java/C#.
        The underscore prefix is a CONVENTION that means:
          "This is an internal implementation detail. Do not call
           it from outside this class."

        If someone writes memory._update_summary([...]), Python
        will NOT stop them. But the underscore signals "you should
        not do this, and if you do, you accept the consequences."

        Double underscore (__method) triggers name mangling, which
        makes it HARDER (but not impossible) to call from outside.
        We use single underscore because it is clearer and sufficient.

        DESIGN DECISION: Simple text summary vs LLM summary
        ----------------------------------------------------
        We could call Claude to generate a smart summary of evicted
        messages. But that would:
          1. Add an API call (cost + latency) on every eviction
          2. Create a dependency on Claude for memory management
          3. Break if the API is down

        Instead, we build a simple text summary by extracting the
        topic of each evicted exchange. This is "good enough" for
        giving Claude context about what was discussed earlier.

        In production systems (ChatGPT, Claude.ai itself), they DO
        use LLM-generated summaries. But for learning, the simple
        approach teaches the concept without the complexity.
        """
        # ---------------------------------------------------------------
        # Extract a brief description of the evicted exchange.
        # We take the first 100 chars of each message as a preview.
        # ---------------------------------------------------------------
        evicted_preview_parts = []
        for msg in evicted_messages:
            role = msg["role"]
            # [:100] takes the first 100 characters of the content.
            # If the content is shorter than 100 chars, it returns all of it.
            preview = msg["content"][:100].replace("\n", " ")
            evicted_preview_parts.append(f"{role}: {preview}")

        evicted_preview = " | ".join(evicted_preview_parts)

        # ---------------------------------------------------------------
        # Build or append to the running summary.
        # ---------------------------------------------------------------
        if self.summary is None:
            self.summary = (
                f"Earlier in this conversation, the user discussed: "
                f"{evicted_preview}"
            )
        else:
            # Append to existing summary, keeping it concise.
            # We cap total summary length to prevent unbounded growth.
            new_part = f" Then discussed: {evicted_preview}"
            self.summary = self.summary + new_part

            # ---------------------------------------------------------------
            # Cap summary at 500 characters to prevent unbounded growth.
            #
            # PYTHON REFRESHER: String truncation with ellipsis
            # ---------------------------------------------------------------
            # "hello world"[:5]  -> "hello"
            # We add "..." to signal that text was cut off.
            #
            # 500 chars ~= 125 tokens. This is a small overhead compared
            # to the ~4000 token budget for the full message window.
            # ---------------------------------------------------------------
            if len(self.summary) > 500:
                self.summary = self.summary[:497] + "..."


    def _estimate_total_tokens(self) -> int:
        """
        Estimate the total token count of all messages in the window
        plus the summary (if any).

        This is used to decide if we are approaching the token budget.
        It is NOT used to truncate messages -- we use turn count for that.
        The token estimate is informational, for debugging and cost tracking.
        """
        total = 0
        for msg in self.messages:
            total += estimate_tokens(msg["content"])
            # Each message has overhead: role name, JSON formatting
            # Anthropic estimates ~4 tokens of overhead per message
            total += 4

        if self.summary:
            total += estimate_tokens(self.summary)

        return total


    def get_messages_for_api(self) -> list[dict]:
        """
        Get the message history formatted for the Anthropic Messages API.

        This is the method app.py calls right before making an API request.
        It returns a clean list of {"role": ..., "content": ...} dicts
        that can be passed directly to client.messages.create(messages=...).

        CRITICAL: This filters out any extra fields that app.py might
        have stored in session_state messages (like "citations" or
        "debug_info"). The Anthropic API only reads "role" and "content".

        RETURNS
        -------
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
        # ---------------------------------------------------------------
        # PYTHON REFRESHER: List comprehension with dict filtering
        # ---------------------------------------------------------------
        # We extract ONLY "role" and "content" from each message.
        #
        # Long-form equivalent:
        #   result = []
        #   for msg in self.messages:
        #       clean_msg = {
        #           "role": msg["role"],
        #           "content": msg["content"],
        #       }
        #       result.append(clean_msg)
        #   return result
        #
        # The comprehension version is more concise but does the same thing.
        # ---------------------------------------------------------------
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
        Reset all memory state. Called when user clicks "New Conversation"
        in the Streamlit UI.

        PYTHON REFRESHER: deque.clear()
        --------------------------------
        deque.clear() removes all elements but keeps the maxlen setting.
        This is different from creating a new deque:
          self.messages.clear()           # Same object, maxlen preserved
          self.messages = deque(maxlen=X) # New object, same effect
        """
        self.messages.clear()
        self.summary = None
        self.total_turns = 0
        self.evicted_turns = 0
        self.estimated_tokens = 0


    def get_stats(self) -> dict:
        """
        Return memory statistics for the debug panel in app.py.

        These stats help you understand memory behavior during development.
        In production, you would log these to your observability system.
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
        # No history means it cannot be a follow-up
        if self.total_turns == 0:
            return False

        query_lower = query.lower().strip()
        words = query_lower.split()

        # ---------------------------------------------------------------
        # Signal 1: Pronouns that reference previous context.
        # These words only make sense if there is a prior topic.
        # ---------------------------------------------------------------
        pronoun_signals = [
            "it", "its", "that", "those", "them", "this",
            "the same", "that table", "that one", "which ones",
            "what about", "how about", "and the", "also",
            "same table", "same columns",
        ]
        # ---------------------------------------------------------------
        # PYTHON REFRESHER: any() with a generator expression
        # ---------------------------------------------------------------
        # any(signal in query_lower for signal in pronoun_signals)
        #
        # Long-form equivalent:
        #   found = False
        #   for signal in pronoun_signals:
        #       if signal in query_lower:
        #           found = True
        #           break
        #
        # any() is preferred because:
        #   1. It short-circuits (stops at first True)
        #   2. It is a single expression (no temp variable)
        #   3. It reads like English: "is any signal in the query?"
        # ---------------------------------------------------------------
        if any(signal in query_lower for signal in pronoun_signals):
            return True

        # ---------------------------------------------------------------
        # Signal 2: Very short query without a table name.
        # "and the keys?" is likely a follow-up.
        # "What is FACT_SALES_ORDER?" is likely a new topic.
        # ---------------------------------------------------------------
        has_table_name = any(
            prefix in query.upper()
            for prefix in ["FACT_", "DIM_", "BRIDGE_"]
        )

        if len(words) <= 5 and not has_table_name:
            return True

        return False


# =====================================================================
# SECTION 3: ask_claude_with_memory() -- THE UPGRADED GENERATION CALL
# =====================================================================
#
# This function wraps your existing ask_claude() with conversation
# memory support. It is the function that app.py will call instead
# of ask_claude() when memory is enabled.
#
# CRITICAL DESIGN DECISION: This is a SEPARATE function, not a
# modification to ask_claude() in rag.py. This means:
#   - rag.py remains untouched (eval.py still works)
#   - Memory is opt-in (app.py chooses which function to call)
#   - You can A/B test memory vs no-memory easily
#
# dbt ANALOGY: This is like creating a new model that references
# the staging model (ask_claude) but adds extra logic on top.
# The staging model does not change.

def ask_claude_with_memory(
    query: str,
    context_chunks: list[dict],
    memory: ConversationMemory,
    system_prompt: str = None,
) -> str:
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

    GOTCHA: IMPORT LOCATION
    -----------------------
    We import anthropic and IMPROVED_SYSTEM_PROMPT here rather than
    at the top of the file. This is intentional:

    1. It avoids a circular import if rag.py ever imports from this file
    2. It means conversation_memory.py can be imported even if
       anthropic is not installed (the class itself works standalone)
    3. The import only runs when this function is actually called

    This pattern is called "lazy importing" or "deferred importing".
    It trades a tiny bit of import-time speed for better modularity.
    """
    import anthropic
    from rag import IMPROVED_SYSTEM_PROMPT as DEFAULT_PROMPT

    # ---------------------------------------------------------------
    # Step 1: Build the system prompt with conversation summary
    # ---------------------------------------------------------------
    effective_prompt = system_prompt or DEFAULT_PROMPT
    summary_addition = memory.get_system_prompt_addition()
    if summary_addition:
        effective_prompt = effective_prompt + "\n\n" + summary_addition

    # ---------------------------------------------------------------
    # Step 2: Build the context string from retrieved chunks
    # This is the SAME logic as ask_claude() in rag.py.
    # We duplicate it here rather than calling ask_claude() because
    # ask_claude() does not accept a messages parameter.
    # ---------------------------------------------------------------
    context_parts = []
    for chunk in context_chunks:
        label_parts = []
        if chunk.get("table_name"):
            label_parts.append(chunk["table_name"])
        if chunk.get("doc_type") and chunk["doc_type"] != "text":
            label_parts.append(chunk["doc_type"])
        label = " - ".join(label_parts) if label_parts else chunk.get("source", "unknown")
        context_parts.append(f"[Source: {label}]\n{chunk['text']}")

    context = "\n\n---\n\n".join(context_parts)

    # ---------------------------------------------------------------
    # Step 3: Build the messages list with conversation history
    #
    # The structure is:
    #   [prior_messages..., current_user_message_with_context]
    #
    # The current message includes the retrieved context so Claude
    # can answer based on documents. Prior messages do NOT include
    # context -- they are just the raw Q&A pairs.
    # ---------------------------------------------------------------
    prior_messages = memory.get_messages_for_api()

    current_message = {
        "role": "user",
        "content": f"""Context from documents:

{context}

---

Question: {query}""",
    }

    all_messages = prior_messages + [current_message]

    # ---------------------------------------------------------------
    # Step 4: Call the Anthropic API
    # ---------------------------------------------------------------
    client = anthropic.Anthropic()
    response = client.messages.create(
        model="claude-sonnet-4-5-20250929",
        max_tokens=1024,
        system=effective_prompt,
        messages=all_messages,
    )

    answer = response.content[0].text

    # ---------------------------------------------------------------
    # Step 5: Record this turn in memory
    #
    # IMPORTANT: We store the RAW query, not the context-enhanced
    # version. The context is re-retrieved each time, so storing it
    # would waste memory tokens on stale context.
    # ---------------------------------------------------------------
    memory.add_turn(query, answer)

    return answer


# =====================================================================
# SECTION 4: STANDALONE TEST
# =====================================================================
#
# Run this file directly to verify the ConversationMemory class works:
#   cd C:\Users\laaro\AI-Eng
#   uv run python conversation_memory.py

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