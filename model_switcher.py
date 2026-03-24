"""
model_switcher.py -- Model Abstraction Layer (Strategy Pattern)

rovides a single generate() function that works with ANY model backend.
Instead of the code needing to know whether it is talking to Claude
or Ollama, it just calls:

    answer = generate(query, chunks, model="claude")
    answer = generate(query, chunks, model="ollama/qwen2.5:0.5b")

The model_switcher handles all the differences (API format, endpoint,
authentication, response parsing) internally.

THE STRATEGY PATTERN
Without this abstraction, model-specific code spreads everywhere:

    # BAD: Model logic scattered across app.py, eval.py, etc.
    if model_choice == "claude":
        import anthropic
        client = anthropic.Anthropic()
        response = client.messages.create(model="claude-sonnet-4-5-20250929", ...)
        answer = response.content[0].text
    elif model_choice == "ollama":
        import requests
        response = requests.post("http://localhost:11434/api/chat", ...)
        answer = response.json()["message"]["content"]

This is problematic because:
  1. Every new model requires changes in multiple files
  2. Testing is hard (you cannot mock one model without touching all code)
  3. Configuration is scattered (URLs, model names, timeouts everywhere)

The Strategy Pattern solves this by putting all model-specific logic
behind a SINGLE interface:

    # GOOD: One function, any model
    answer = generate(query, chunks, model="claude")

Adding a new model (e.g., GPT-4, Gemini) means adding ONE new function
in THIS file, not touching app.py or eval.py at all.

DESIGN PATTERN: STRATEGY PATTERN
The Strategy Pattern defines a family of algorithms (Claude, Ollama),
encapsulates each one, and makes them interchangeable.

In classical OOP,  use classes:
    class ModelStrategy(ABC):
        def generate(self, query, chunks) -> str: ...

    class ClaudeStrategy(ModelStrategy): ...
    class OllamaStrategy(ModelStrategy): ...

We use a SIMPLER approach: a dictionary mapping model names to functions.
This is more Pythonic -- Python's first-class functions eliminate the
need for a class hierarchy.

    STRATEGIES = {
        "claude": _generate_claude,
        "ollama": _generate_ollama,
    }
    answer = STRATEGIES[model_name](query, chunks)

PYTHON REFRESHER: Functions as first-class objects
In Python, functions are objects. You can:
  - Assign them to variables: greet = print
  - Store them in dicts: {"say_hi": print}
  - Pass them as arguments: map(print, [1, 2, 3])

This is what makes the dictionary-based Strategy Pattern possible.
In Java, you would need interfaces and classes for the same thing.

Example:
    def add(a, b): return a + b
    def mul(a, b): return a * b

    ops = {"add": add, "multiply": mul}
    result = ops["add"](3, 4)  # -> 7
    result = ops["multiply"](3, 4)  # -> 12

    # The dict lookup returns the FUNCTION OBJECT, then () calls it.


HOW THIS FILE CONNECTS TO YOUR PROJECT
---------------------------------------
  query_router.py imports generate() from this file
    --> decides which model to use based on query complexity
    --> calls generate(query, chunks, model=chosen_model)

  app.py offers a model dropdown in the sidebar
    --> user selects "Claude Sonnet" or "Ollama/qwen2.5"
    --> app.py passes the selection to generate()

  eval.py can be extended to test models side-by-side:
    --> uv run python eval.py --model claude --tag claude_sonnet
    --> uv run python eval.py --model ollama/qwen2.5:0.5b --tag qwen_0.5b
    --> uv run python eval.py --compare eval_results/claude.csv eval_results/qwen.csv


DEPENDENCIES
------------
  - anthropic (for Claude -- already installed)
  - requests (for Ollama -- already installed)
  - conversation_memory.py (for memory-aware generation)
  - ollama_client.py (for Ollama integration)
"""


def _generate_claude(
        query:str,
        context_chunks:list[dict],
        memory = None,
        system_prompt:str = None,
        **kwarg
) ->str:
    """
    Generate using Claude API (via  existing ask_claude or ask_claude_with_memory).

    PRIVATE FUNCTION (underscore prefix). External code should call
    generate() instead, which dispatches to this function.

    PYTHON REFRESHER: **kwargs (keyword arguments)
    -----------------------------------------------
    **kwargs collects any extra keyword arguments into a dictionary.

        def f(a, b, **kwargs):
            print(kwargs)

        f(1, 2, color="red", size=5)
        # kwargs = {"color": "red", "size": 5}

    We use **kwargs here so that model-specific parameters (like
    Ollama's temperature) can be passed through without causing errors.
    Claude does not use temperature from kwargs
    """
    if memory is not None:
        #use memory aware generation
        from conversation_memory import ask_claude_with_memory
        return ask_claude_with_memory(
            query = query,
            context_chunks=context_chunks,
            memory=memory,
            system_prompt=system_prompt
        )
    else:
        #uss original statelss genreation
        from rag import ask_claude
        return ask_claude(query=query,context_chunks=context_chunks)

def _generate_ollama(
        query:str,
        context_chunks: list[dict],
        model_name: str = "qwen2.5:0.5b",
        system_prompt:str = None
) -> str:
    """
    
    """
    from ollama_client import ask_ollama
    return ask_ollama(
        query=query,
        context_chunks=context_chunks,
        model=model_name,
        system_prompt=system_prompt
    )
    
#MODEL REGISTRY
# To add a new model backend (e.g., OpenAI GPT-4):
#   1. Write _generate_openai() with the same signature
#   2. Add "openai": _generate_openai to this dict
#   3. Done -- generate() automatically supports it
MODEL_REGISTRY = {
    "claude": _generate_claude,
}

#PUBLIC GENERATE FUNCTION
def generate(
    query: str,
    context_chunks: list[dict],
    model: str = "claude",
    memory=None,
    system_prompt: str = None,
    **kwargs,
):
    if model == "claude" or model.startswith("claude-"):
        return _generate_claude(
            query=query,
            context_chunks=context_chunks,
            memory=memory,
            system_prompt=system_prompt,
            **kwargs,
        )
    
    if model.startswith("ollama/"):
        # Extract the Ollama model name after the "ollama/" prefix
        ollama_model_name = model.split("/", 1)[1]

        # Check if Ollama is running before attempting
        from ollama_client import is_ollama_running
        if not is_ollama_running():
            raise ConnectionError(
                f"Ollama is not running. Cannot use model '{model}'.\n"
                f"Start Ollama with: ollama serve\n"
                f"Falling back to Claude is handled by query_router.py."
            )

        return _generate_ollama(
            query=query,
            context_chunks=context_chunks,
            model_name=ollama_model_name,
            system_prompt=system_prompt,
            **kwargs,
        )
    if model in MODEL_REGISTRY:
        return MODEL_REGISTRY[model](
            query=query,
            context_chunks=context_chunks,
            memory=memory,
            system_prompt=system_prompt,
            **kwargs,
        )
    raise ValueError(
        f"Unknown model: '{model}'. "
        f"Supported: 'claude', 'ollama/<model_name>'. "
        f"Available Ollama models: run 'ollama list' to check."
    )
def list_models() -> list[dict]:
    """
    List all available models across all backends.

    This is used by the Streamlit sidebar to populate the model dropdown.

    RETURNS
    -------
    list[dict]
        Each dict has:
          "id": str -- the model identifier to pass to generate()
          "name": str -- human-readable display name
          "backend": str -- "claude" or "ollama"
          "available": bool -- whether the model can be used right now
    """
    models = [
        {
            "id": "claude",
            "name": "Claude Sonnet (API)",
            "backend": "claude",
            "available": True,  # Always available if API key is set
        },
    ]
    try:
        from ollama_client import is_ollama_running, list_available_models

        if is_ollama_running():
            for ollama_model in list_available_models():
                models.append({
                    "id": f"ollama/{ollama_model}",
                    "name": f"Ollama: {ollama_model} (local)",
                    "backend": "ollama",
                    "available": True,
                })
        else:
            # Ollama not running -- show it as unavailable
            models.append({
                "id": "ollama/unavailable",
                "name": "Ollama (not running)",
                "backend": "ollama",
                "available": False,
            })
    except ImportError:
        # ollama_client.py not found -- skip Ollama models
        pass

    return models


if __name__ == "__main__":
    import os

    print("=" * 60)
    print("MODEL SWITCHER -- STANDALONE TEST")
    print("=" * 60)

    # Show available models
    print("\nAvailable models:")
    for m in list_models():
        status = "READY" if m["available"] else "UNAVAILABLE"
        print(f"  [{status}] {m['id']} -> {m['name']}")

    # Test with fake chunks
    test_chunks = [
        {
            "text": "DIM_STORE is a dimension table. Grain: one row per store. "
                    "Keys: SK_STORE_KEY (surrogate), BK_STORE_KEY (business).",
            "source": "STTM__DIM_STORE__summary",
            "table_name": "DIM_STORE",
            "doc_type": "summary",
        },
    ]
    test_query = "What is DIM_STORE?"

    # Test Claude (requires API key)
    print(f"\n--- Testing Claude ---")
    if os.environ.get("ANTHROPIC_API_KEY"):
        try:
            answer = generate(test_query, test_chunks, model="claude")
            print(f"  Answer: {answer[:150]}...")
        except Exception as e:
            print(f"  Error: {e}")
    else:
        # Load .env manually for testing
        env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
        if os.path.exists(env_path):
            with open(env_path) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        key, _, value = line.partition("=")
                        os.environ[key.strip()] = value.strip()
            try:
                answer = generate(test_query, test_chunks, model="claude")
                print(f"  Answer: {answer[:150]}...")
            except Exception as e:
                print(f"  Error: {e}")
        else:
            print("  Skipped (no ANTHROPIC_API_KEY).")

    # Test Ollama (requires Ollama running)
    print(f"\n--- Testing Ollama ---")
    from ollama_client import is_ollama_running, list_available_models
    if is_ollama_running():
        available = list_available_models()
        if available:
            model_id = f"ollama/{available[0]}"
            print(f"  Using model: {model_id}")
            try:
                answer = generate(test_query, test_chunks, model=model_id)
                print(f"  Answer: {answer[:150]}...")
            except Exception as e:
                print(f"  Error: {e}")
        else:
            print("  Ollama running but no models downloaded.")
            print("  Run: ollama pull qwen2.5:0.5b")
    else:
        print("  Skipped (Ollama not running).")

    # Test unknown model (should raise ValueError)
    print(f"\n--- Testing unknown model ---")
    try:
        generate(test_query, test_chunks, model="gpt-4-turbo")
        print("  ERROR: Should have raised ValueError!")
    except ValueError as e:
        print(f"  Correctly raised ValueError: {str(e)[:100]}")

    print("\nAll tests passed. model_switcher.py is ready.")

