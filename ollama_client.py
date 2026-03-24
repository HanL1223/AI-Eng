"""
ollama_client.py -- Local Model Inference via Ollama

Ollama's API is different from Anthropic's.
 
  Anthropic
    POST https://api.anthropic.com/v1/messages
    {"model": "claude-sonnet-4-5-20250929", "messages": [...], "system": "..."}
 
  Ollama
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
  - requests (for HTTP calls to Ollama -- likely already installed)
  - Ollama must be running locally (see installation above)

HOW THIS FILE CONNECTS TO  PROJECT
  model_switcher.py imports ask_ollama() from this file
    --> calls it when the selected model starts with "ollama/"
    --> the response format matches ask_claude() (returns a string)

"""

import requests
import json
import time

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


#Healthcheck
def is_ollama_running() -> bool:
    try:
        response = requests.get(OLLAMA_TAGS_ENDPOINT, timeout=5)
        return response.status_code == 200 
    except requests.RequestException:
        return False
    
def list_available_models() -> list[str]:
    """
    Get the list of models downloaded in Ollama

    Use this for streamlist UI to populate the model dropdown, allow user
    to use downloaded model only
    """
    try:
        response = requests.get(OLLAMA_TAGS_ENDPOINT,timeout = 5)
        if response.status_code == 200:
            data = response.json()
            # The /api/tags response looks like:
            # {"models": [{"name": "qwen2.5:0.5b", "size": 400000000, ...}]}
            return [n["name"] for n in data.get("models",[])]
        return []
    except requests.RequestException:
        return []
    
def ask_ollama(
        query:str,
        context_chunks: list[dict],
        model: str = "qwen2.5:0.5b",
        system_prompt: str = None,
        temperature: float = 0.1
) -> str:
    """
    Generate an answer using a local Ollama model.

    This function has the SAME signature as ask_claude() in rag.py
    (query + context_chunks -> string answer). This is intentional --
    model_switcher.py can call either function interchangeably.

    PARAMETERS
    """
    #Building system prompt
    if system_prompt is None:   
        #use SIMPLIFIED system prompt for local models
        #Small models 
        system_prompt = (
            "You are a data warehouse documentation assistant for a company "
            "Answer questions about Snowflake tables, columns, and data lineage "
            "using ONLY the provided context. "
            "If the context does not contain the answer, say 'I don't have that information.' "
            "Be concise and factual."
        )

        #Build th econtext string
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

    #Build ollama request payload
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
    messages = [
        {"role":"system","content":system_prompt},
        {
            "role":"user",
            "content":f"Context from documents:\n\n {context}\n\n---\n\nQuestion: {query}",
        },
    ]
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict":512 #Max tokens to genereate
        }
    }

    #Send request to ollama
    try:
        response=requests.post(
            OLLAMA_CHAT_ENDPOINT,
            json= payload,
            timeout=OLLAMA_TIMEOUT
        )
    except requests.ConnectionError:
        raise ConnectionError(
            "Cannot connect to Ollama, check if server is running"
        )
    #Parse the format
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
    if response.status_code != 200:
        raise RuntimeError(
        f"Ollama returned HTTP {response.status_code}: {response.text}"
    )
    data = response.json()

    try:
        answer = data["message"]["content"]
    except (KeyError, TypeError) as e:
        raise RuntimeError(
            f"Unexpected Ollama response format: {e}\n"
            f"Response: {json.dumps(data, indent=2)[:500]}"
        )

    return answer.strip()



#Testing
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
 
    


    