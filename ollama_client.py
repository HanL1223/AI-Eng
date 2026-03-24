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
 
    


    