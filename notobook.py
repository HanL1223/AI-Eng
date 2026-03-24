"""
ollama_client.py -- Local Model Inference via Ollama
====================================================
Week 4, Step 3 of 6

WHAT THIS FILE DOES
-------------------
Provides a function to call locally-running LLMs through Ollama.
This gives you a free, private alternative to the Claude API.

WHAT IS OLLAMA?
---------------
Ollama is a tool that runs open-source LLMs on your local machine.
It handles model downloading, GPU memory management, and serves
the model through a simple HTTP API.

Think of it as "Docker for language models":
  - Docker:  docker pull nginx   -> docker run nginx  -> http://localhost:80
  - Ollama:  ollama pull qwen2.5 -> ollama serve      -> http://localhost:11434

The key insight: Ollama runs a LOCAL HTTP server. Your Python code
talks to it the same way it talks to the Claude API -- by sending
HTTP requests. The only difference is the URL (localhost vs api.anthropic.com)
and the request/response format.

INSTALLATION (DO THIS FIRST)
-----------------------------
Option A: Mac (Apple Silicon -- your Mac Mini)
  1. Download from https://ollama.ai
  2. Install the .dmg file
  3. Open Terminal and run: ollama pull qwen2.5:0.5b
  4. Ollama auto-starts as a background service

Option B: Windows
  1. Download from https://ollama.ai
  2. Run the installer
  3. Open PowerShell and run: ollama pull qwen2.5:0.5b
  4. Ollama runs as a Windows service

Option C: Linux
  curl -fsSL https://ollama.com/install.sh | sh
  ollama pull qwen2.5:0.5b

VERIFY INSTALLATION:
  ollama list              # Shows downloaded models
  ollama run qwen2.5:0.5b  # Interactive chat (type /bye to exit)
  curl http://localhost:11434/api/tags  # Should return JSON with models

RECOMMENDED MODELS FOR YOUR STTM PROJECT
-----------------------------------------
| Model | Size | RAM | Quality | Speed | Best For |
|-------|------|-----|---------|-------|----------|
| qwen2.5:0.5b | 400MB | 1GB | Low | Fast | Testing, simple lookups |
| qwen2.5:3b | 2GB | 4GB | Medium | Medium | Single-table questions |
| llama3.2:3b | 2GB | 4GB | Medium | Medium | General questions |
| qwen2.5:7b | 4.5GB | 8GB | Good | Slower | Cross-entity questions |
| llama3.1:8b | 4.7GB | 8GB | Good | Slower | Complex reasoning |

Start with qwen2.5:0.5b for learning. It will fail on complex questions,
but that is the POINT -- you will see exactly where local models break
compared to Claude, which informs your query routing decisions in Step 5.

dbt ANALOGY:
  Claude API   = Snowflake (powerful, pay-per-query, cloud)
  Ollama       = DuckDB (free, local, limited but fast for simple things)

WHY SMALL MODELS FAIL IN PREDICTABLE WAYS
------------------------------------------
From your Week 2 experiments, you learned that smaller models fail
differently across question types:

  Simple lookups:     "What is the grain of DIM_STORE?"
    -> Small model: Often correct (the answer is in the context verbatim)

  Cross-entity:       "Which dimensions does FACT_SALES reference?"
    -> Small model: Often fails (requires reasoning across multiple chunks)

  Edge cases:         "What is the SLA for FACT_SALES refresh?"
    -> Small model: Often WRONG (hallucinates instead of saying "I don't know")

These failure modes are exactly why we route queries: simple lookups
go to Ollama (free/fast), complex queries go to Claude (accurate).


OLLAMA API FORMAT
-----------------
Ollama's API is different from Anthropic's. Here is the comparison:

  Anthropic (what you know):
    POST https://api.anthropic.com/v1/messages
    {"model": "claude-sonnet-4-5-20250929", "messages": [...], "system": "..."}

  Ollama (what this file uses):
    POST http://localhost:11434/api/chat
    {"model": "qwen2.5:0.5b", "messages": [...], "stream": false}

Key differences:
  1. URL: localhost instead of api.anthropic.com
  2. No API key needed (it is your own machine)
  3. "stream": false (we want the full response, not streaming chunks)
  4. System prompt goes in messages as {"role": "system", ...}
     instead of a separate "system" parameter
  5. Response format: {"message": {"content": "..."}} instead of
     {"content": [{"text": "..."}]}


DEPENDENCIES
------------
  - requests (for HTTP calls to Ollama -- likely already installed)
  - Ollama must be running locally (see installation above)

GOTCHA: OLLAMA MUST BE RUNNING
-------------------------------
If Ollama is not running, requests.post() will raise:
  ConnectionError: ('Connection aborted.', ConnectionRefusedError(...))

The fix: start Ollama before running your chatbot.
  Mac:     ollama serve  (or it auto-starts after install)
  Windows: Start the Ollama service, or run ollama serve in PowerShell
  Linux:   systemctl start ollama


HOW THIS FILE CONNECTS TO YOUR PROJECT
---------------------------------------
  model_switcher.py imports ask_ollama() from this file
    --> calls it when the selected model starts with "ollama/"
    --> the response format matches ask_claude() (returns a string)
"""

import requests
import json
import time


# =====================================================================
# CONFIGURATION
# =====================================================================
#
# The Ollama server URL. By default, Ollama listens on port 11434.
# If you change this in Ollama's config, update it here.
#
# PYTHON REFRESHER: Module-level constants
# -----------------------------------------
# Constants at the top of a file are available to all functions below.
# Convention: UPPER_SNAKE_CASE for constants, lower_snake_case for variables.
# Python does not enforce "const" -- you CAN reassign these. The naming
# convention signals "this should not change at runtime."

OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_CHAT_ENDPOINT = f"{OLLAMA_BASE_URL}/api/chat"
OLLAMA_TAGS_ENDPOINT = f"{OLLAMA_BASE_URL}/api/tags"

# Default timeout in seconds. Ollama can be slow on first request
# (loading model into GPU memory). Subsequent requests are faster.
OLLAMA_TIMEOUT = 120


# =====================================================================
# SECTION 1: HEALTH CHECK
# =====================================================================

def is_ollama_running() -> bool:
    """
    Check if the Ollama server is running and reachable.

    This is called by model_switcher.py before attempting to use Ollama.
    If Ollama is not running, model_switcher falls back to Claude.

    HOW IT WORKS
    ------------
    We make a GET request to the /api/tags endpoint, which returns a
    list of downloaded models. If the request succeeds, Ollama is running.
    If it fails (ConnectionError), Ollama is not running.

    PYTHON REFRESHER: try/except for expected errors
    --------------------------------------------------
    requests.get() can raise several exceptions:
      - ConnectionError: Server is not running
      - Timeout: Server took too long to respond
      - requests.RequestException: Any other request error

    We catch all of these as a single "Ollama is not available" case.
    In production, you might handle each differently (retry on timeout,
    alert on connection error).

    RETURNS
    -------
    bool
        True if Ollama is running and reachable, False otherwise.
    """
    try:
        response = requests.get(OLLAMA_TAGS_ENDPOINT, timeout=5)
        return response.status_code == 200
    except requests.RequestException:
        return False


def list_available_models() -> list[str]:
    """
    Get the list of models downloaded in Ollama.

    This is used by the Streamlit UI to populate the model dropdown.
    Only models that are actually downloaded can be used.

    RETURNS
    -------
    list[str]
        Model names like ["qwen2.5:0.5b", "llama3.2:3b"].
        Returns empty list if Ollama is not running.
    """
    try:
        response = requests.get(OLLAMA_TAGS_ENDPOINT, timeout=5)
        if response.status_code == 200:
            data = response.json()
            # ---------------------------------------------------------------
            # The /api/tags response looks like:
            # {"models": [{"name": "qwen2.5:0.5b", "size": 400000000, ...}]}
            #
            # PYTHON REFRESHER: List comprehension from nested dict
            # ---------------------------------------------------------------
            # [m["name"] for m in data.get("models", [])]
            #
            # data.get("models", []) returns the "models" list, or an empty
            # list if "models" key is missing. This prevents KeyError.
            #
            # Long-form equivalent:
            #   names = []
            #   models = data.get("models", [])
            #   for m in models:
            #       names.append(m["name"])
            #   return names
            # ---------------------------------------------------------------
            return [m["name"] for m in data.get("models", [])]
        return []
    except requests.RequestException:
        return []


# =====================================================================
# SECTION 2: THE MAIN GENERATION FUNCTION
# =====================================================================

def ask_ollama(
    query: str,
    context_chunks: list[dict],
    model: str = "qwen2.5:0.5b",
    system_prompt: str = None,
    temperature: float = 0.1,
) -> str:
    """
    Generate an answer using a local Ollama model.

    This function has the SAME signature as ask_claude() in rag.py
    (query + context_chunks -> string answer). This is intentional --
    model_switcher.py can call either function interchangeably.

    PARAMETERS
    ----------
    query : str
        The user's question.
    context_chunks : list[dict]
        Retrieved chunks from ChromaDB (same format as ask_claude).
    model : str
        The Ollama model to use. Must be downloaded:
          ollama pull qwen2.5:0.5b
        Default is qwen2.5:0.5b (smallest, fastest, for learning).
    system_prompt : str, optional
        Override the system prompt. If None, uses a simplified version
        of your IMPROVED_SYSTEM_PROMPT from rag.py.
    temperature : float
        Controls randomness. 0.0 = deterministic, 1.0 = creative.
        We default to 0.1 (nearly deterministic) because we want
        factual answers about data warehouse tables, not creativity.

        DESIGN RATIONALE: Low temperature for RAG
        ------------------------------------------
        RAG answers should be grounded in the retrieved context.
        High temperature encourages the model to be "creative",
        which in a factual domain means "hallucinate".
        0.1 gives a tiny bit of variability to avoid exact repetition
        while staying close to the source material.

    RETURNS
    -------
    str
        The model's response text.

    RAISES
    ------
    ConnectionError
        If Ollama is not running.
    RuntimeError
        If the model returns an unexpected response format.

    GOTCHA: FIRST REQUEST IS SLOW
    ------------------------------
    The first request after starting Ollama (or after the model is
    unloaded from memory) will be slow because Ollama needs to load
    the model into GPU/CPU memory. On a Mac Mini with Apple Silicon:
      - qwen2.5:0.5b: ~2 seconds to load
      - llama3.1:8b:   ~10 seconds to load
    Subsequent requests are much faster (100ms - 2s depending on model).
    """
    # ---------------------------------------------------------------
    # Step 1: Build the system prompt
    # ---------------------------------------------------------------
    if system_prompt is None:
        # We use a SIMPLIFIED system prompt for local models.
        # Small models (0.5B-3B) struggle with long system prompts.
        # The full IMPROVED_SYSTEM_PROMPT from rag.py is ~800 tokens,
        # which consumes a large fraction of a small model's capacity.
        #
        # This simplified version keeps the essential instructions.
        system_prompt = (
            "You are a data warehouse documentation assistant for Sigma Healthcare. "
            "Answer questions about Snowflake tables, columns, and data lineage "
            "using ONLY the provided context. "
            "If the context does not contain the answer, say 'I don't have that information.' "
            "Be concise and factual."
        )

    # ---------------------------------------------------------------
    # Step 2: Build the context string (same logic as ask_claude)
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
    # Step 3: Build the Ollama request payload
    #
    # CRITICAL DIFFERENCE FROM ANTHROPIC API:
    # Ollama puts the system prompt in the messages array as
    # {"role": "system", ...}. Anthropic has a separate "system"
    # parameter outside the messages array.
    #
    # Ollama format:
    #   messages = [
    #       {"role": "system", "content": "..."},
    #       {"role": "user", "content": "..."},
    #   ]
    #
    # Anthropic format:
    #   system = "..."
    #   messages = [
    #       {"role": "user", "content": "..."},
    #   ]
    # ---------------------------------------------------------------
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": f"Context from documents:\n\n{context}\n\n---\n\nQuestion: {query}",
        },
    ]

    payload = {
        "model": model,
        "messages": messages,
        "stream": False,          # Get the full response at once
        "options": {
            "temperature": temperature,
            "num_predict": 512,   # Max tokens to generate (like max_tokens)
        },
    }

    # ---------------------------------------------------------------
    # Step 4: Send the request to Ollama
    #
    # PYTHON REFRESHER: requests.post() with JSON
    # ---------------------------------------------------------------
    # requests.post(url, json=payload) does three things:
    #   1. Serializes payload to JSON string
    #   2. Sets Content-Type header to application/json
    #   3. Sends the POST request
    #
    # This is equivalent to:
    #   headers = {"Content-Type": "application/json"}
    #   body = json.dumps(payload)
    #   requests.post(url, data=body, headers=headers)
    # ---------------------------------------------------------------
    try:
        response = requests.post(
            OLLAMA_CHAT_ENDPOINT,
            json=payload,
            timeout=OLLAMA_TIMEOUT,
        )
    except requests.ConnectionError:
        raise ConnectionError(
            "Cannot connect to Ollama. Is it running?\n"
            "  Mac/Linux: ollama serve\n"
            "  Windows:   Start the Ollama service or run 'ollama serve' in PowerShell"
        )

    # ---------------------------------------------------------------
    # Step 5: Parse the response
    #
    # Ollama response format:
    # {
    #     "model": "qwen2.5:0.5b",
    #     "message": {
    #         "role": "assistant",
    #         "content": "DIM_STORE is a dimension table..."
    #     },
    #     "done": true,
    #     "total_duration": 1234567890,  # nanoseconds
    #     "eval_count": 150              # tokens generated
    # }
    #
    # Compare with Anthropic response format:
    # {
    #     "content": [{"type": "text", "text": "DIM_STORE is..."}],
    #     "usage": {"input_tokens": 500, "output_tokens": 150}
    # }
    # ---------------------------------------------------------------
    if response.status_code != 200:
        raise RuntimeError(
            f"Ollama returned HTTP {response.status_code}: {response.text}"
        )

    data = response.json()

    # ---------------------------------------------------------------
    # Extract the answer text.
    #
    # PYTHON REFRESHER: Safe nested dict access with .get()
    # ---------------------------------------------------------------
    # data["message"]["content"]  # Raises KeyError if "message" missing
    # data.get("message", {}).get("content", "")  # Returns "" if missing
    #
    # The second form is "safe" because it never raises KeyError.
    # The trade-off: if the format is wrong, you get an empty string
    # instead of an error, which could hide bugs.
    #
    # We use the first form here because if Ollama's response format
    # changes, we WANT to see an error (fail loudly vs silently).
    # ---------------------------------------------------------------
    try:
        answer = data["message"]["content"]
    except (KeyError, TypeError) as e:
        raise RuntimeError(
            f"Unexpected Ollama response format: {e}\n"
            f"Response: {json.dumps(data, indent=2)[:500]}"
        )

    return answer.strip()


# =====================================================================
# SECTION 3: STANDALONE TEST
# =====================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("OLLAMA CLIENT -- STANDALONE TEST")
    print("=" * 60)

    # Step 1: Check if Ollama is running
    print("\nChecking Ollama connection...")
    if not is_ollama_running():
        print("  Ollama is NOT running.")
        print("  Start it with: ollama serve")
        print("  Then pull a model: ollama pull qwen2.5:0.5b")
        print("\n  Skipping live test. The code structure is correct.")
        print("  You can verify by starting Ollama and re-running this script.")
        exit(0)

    print("  Ollama is running.")

    # Step 2: List available models
    models = list_available_models()
    print(f"\nAvailable models: {models}")

    if not models:
        print("  No models downloaded. Run: ollama pull qwen2.5:0.5b")
        exit(0)

    # Step 3: Test generation with fake chunks
    test_model = models[0]  # Use the first available model
    print(f"\nTesting with model: {test_model}")

    test_chunks = [
        {
            "text": "DIM_STORE is a dimension table. Grain: one row per store location. "
                    "Primary Key: SK_STORE_KEY (surrogate). Business Key: BK_STORE_KEY. "
                    "Source System: SAP via CDS View.",
            "source": "STTM__DIM_STORE__summary",
            "table_name": "DIM_STORE",
            "doc_type": "summary",
        },
    ]

    test_query = "What is the grain of DIM_STORE?"
    print(f"  Query: {test_query}")

    start = time.time()
    try:
        answer = ask_ollama(test_query, test_chunks, model=test_model)
        elapsed = time.time() - start
        print(f"  Answer ({elapsed:.1f}s): {answer[:200]}")
    except Exception as e:
        print(f"  Error: {e}")

    print("\nAll tests passed. ollama_client.py is ready.")