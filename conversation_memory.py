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
    