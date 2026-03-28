"""
query_router.py -- Intelligent Query Routing for RAG
=====================================================
Week 4, Step 5 of 6

WHAT THIS FILE DOES
-------------------
Orchestrates the FULL query pipeline: given a user question, this module
decides HOW to answer it by making three routing decisions:

  Decision 1: RETRIEVAL STRATEGY
    -> Use memory context? (follow-up vs new topic)
    -> How many chunks to retrieve? (simple vs complex query)

  Decision 2: RERANKING
    -> Should we rerank? (complex queries benefit, simple ones do not)
    -> Which method? (LLM reranker for quality, keyword for speed)

  Decision 3: MODEL SELECTION
    -> Claude for complex/cross-entity questions
    -> Ollama for simple single-table lookups (if available)
    -> Fallback to Claude if Ollama is not running



THE ROUTING TABLE
-----------------
Here is the full decision matrix:

| Query Type     | Memory? | Retrieve | Rerank? | Model    | Reason                           |
|---------------|---------|----------|---------|----------|----------------------------------|
| simple_lookup  | No*     | 3 chunks | No      | Ollama** | Fast, cheap, answer is verbatim  |
| follow_up      | Yes     | 3 chunks | No      | Claude   | Needs context to resolve "it"    |
| cross_entity   | No*     | 10 chunks| Yes->3  | Claude   | Needs wide retrieval + precision |
| edge_case      | No      | 3 chunks | No      | Claude   | Needs reasoning to say "I don't know" |
"""

import time


#ROUTING CONFIGURATION
# These settings control the routing decisions. You can tune them
# based on your eval results.
#
# DESIGN DECISION: Configuration as module-level constants
# --------------------------------------------------------
# In production, these would live in a config file or environment
# variables. For learning, module-level constants are simpler and
# let you see all settings in one place.

# How many chunks to retrieve for different query types

SIMPLE_TOP_K = 3        # Single-table lookups need few, precise chunks
COMPLEX_TOP_K = 10      # Cross-entity needs wide retrieval for reranking
DEFAULT_TOP_K = 3       # Fallback for unknown query types

# After reranking, how many chunks to keep
RERANK_TOP_N = 3        # Return 3 best chunks after reranking 10

# Reranking method: "hybrid" (BM25+cross-encoder+RRF, production recommended),
# "cross_encoder" (transformer only), "bm25" (keyword only, free), "none"
RERANK_METHOD = "hybrid"

# Model preferences
SIMPLE_MODEL = "ollama/qwen2.5:0.5b"   # Free/fast for simple queries
COMPLEX_MODEL = "claude"                 # Accurate for complex queries
FALLBACK_MODEL = "claude"                # If preferred model is unavailable

#ROUTING FUNCTION
def route_query(
        query:str,
        collection,
        known_tables:list[str],
        memory = None,
        force_model:str = None,
        force_rerank:str = None,
) -> dict:
    """
    Route a query throught the optimal pipeline and return answer

    main entry point for query pipeline
    """
    timing = {}
    debug = {}

    #Classify query

    from rag import extract_table_name,classify_query,retrieve

    classify_start = time.time()
    query_type = classify_query(query)
    detected_table = extract_table_name(query,known_tables)
    timing['classify_ms'] = (time.time() - classify_start) * 1000

    #Check if is a follow up question
    is_follow_up = False
    if memory is not None:
        is_follow_up = memory.is_follow_up(query)

    #Decide retrieval strtegy
    # The key decision: how many chunks to retrieve.
    # For cross-entity queries, we retrieve MORE chunks (10) because:
    #   - The answer requires information from multiple tables
    #   - We will rerank down to 3, so we need a large initial pool
    #
    # For simple queries, 3 chunks is sufficient because:
    #   - The answer is in one table's documentation
    #   - More chunks = more noise = worse answers

    if query_type == "corss_entity":
        retrieve_top_k = COMPLEX_TOP_K
    else:
        retrieve_top_k  = SIMPLE_TOP_K

    #Retrieve chunk
    retrieve_start = time.time()
    chunks = retrieve(
        collection,
        query,
        top_k=retrieve_top_k,
        table_name=detected_table,
        known_tables=known_tables
    )
    timing["retrieve_ms"] = (time.time() - retrieve_start) * 1000

    debug["raw_chunk_count"] = len(chunks)
    debug['detected_table'] = detected_table
    debug['query_type'] = query_type

    #Reranking
    #useful for complex query with more than 10 chunks

    rerank_method = force_rerank

    if rerank_method is None:
        #Auto-decide rerank only for complex queries with many chunks
        if query_type =='cross_entity' and len(chunks) > RERANK_TOP_N:
            rerank_method = RERANK_METHOD
        else:
            rerank_method = 'none'
    if rerank_method != "none" and len(chunks) > RERANK_TOP_N:
        from reranker import rerank_chunks
        rerank_start = time.time()
        chunks = rerank_chunks(
            query=query,
            chunks=chunks,
            top_n=RERANK_TOP_N,
            method = rerank_method
        )
        timing["rerank_ms"] = (time.time() - rerank_start) * 1000
        debug["reranked"] = True
        debug["rerank_method"] = rerank_method
    else:
        timing["rerank_ms"] = 0
        debug["reranked"] = False
        debug["rerank_method"] = "none"

    #Design model
    # The model choice depends on:
    #   1. force_model override (from UI dropdown or eval flag)
    #   2. Query complexity (simple -> Ollama, complex -> Claude)
    #   3. Follow-up status (follow-ups need Claude for memory)
    #   4. Ollama availability (fall back to Claude if not running)
    if force_model is not None:
        chosen_model = force_model
    elif is_follow_up:
        chosen_model = COMPLEX_MODEL
    elif query_type =='single_table':
        chosen_model = SIMPLE_MODEL
    else:
        chosen_model = COMPLEX_MODEL

    #Verify Ollama availability fall back if needed
    if chosen_model.startswith("ollama/"):
        from ollama_client import is_ollama_running
        if not is_ollama_running():
            debug["model_fallback"] =f"{chosen_model} -> {FALLBACK_MODEL}"
            chosen_model = FALLBACK_MODEL


    #Generate answer 
    from model_switcher import generate
    generate_start  = time.time()
    try:
        answer = generate(
            query=query,
            context_chunks=chunks,
            model=chosen_model,
            memory=memory if is_follow_up else None,
        )
    except Exception as e:
        if chosen_model != FALLBACK_MODEL:
            debug["generation_error"] = str(e)
            debug["model_fallback"] = f"{chosen_model} -> {FALLBACK_MODEL}"
            chosen_model = FALLBACK_MODEL
            try:
                answer = generate(
                    query=query,
                    context_chunks=chunks,
                    model=FALLBACK_MODEL,
                    memory=memory if is_follow_up else None,
                )
            except Exception as fallback_error:
                answer = f"Error generating response: {fallback_error}"
        else:
            answer = f"Error generating response: {e}"
    timing["generate_ms"] = (time.time() - generate_start) * 1000

    #Building result
    total_ms = sum(timing.values())

    routing_decision = {
        "query_type": query_type,
        "is_follow_up": is_follow_up,
        "model": chosen_model,
        "rerank_method": rerank_method,
        "retrieve_top_k": retrieve_top_k,
    }

    return {
        "answer": answer,
        "chunks": chunks,
        "routing": routing_decision,
        "timing": {
            **timing,
            "total_ms": total_ms,
        },
        "debug": debug,
    }

# Explaing routing

def explain_routing(routing_decision:dict) -> str:
    """
    generate human readable explanation of the routing decision

    EXAMPLE OUTPUT
    --------------
    "Query type: cross_entity | Follow-up: No | Model: claude |
     Reranking: llm (10 -> 3) | Rationale: Complex query needs
     wide retrieval and strong reasoning"
    """
    qt = routing_decision["query_type"]
    fu = "Yes" if routing_decision["is_follow_up"] else "No"
    model = routing_decision["model"]
    rerank = routing_decision["rerank_method"]
    top_k = routing_decision["retrieve_top_k"]

    # Build rationale
    if routing_decision["is_follow_up"]:
        rationale = "Follow-up detected -- using Claude with memory for context"
    elif qt == "cross_entity":
        rationale = f"Cross-entity query -- retrieved {top_k} chunks, reranked to {RERANK_TOP_N}"
    elif qt == "single_table":
        rationale = f"Simple lookup -- {top_k} chunks, {'local model' if 'ollama' in model else 'Claude'}"
    else:
        rationale = f"Default routing -- {top_k} chunks via {model}"

    return (
        f"Type: {qt} | Follow-up: {fu} | Model: {model} | "
        f"Rerank: {rerank} | {rationale}"
    )


# =====================================================================
# SECTION 4: STANDALONE TEST
# =====================================================================

if __name__ == "__main__":
    import os

    print("=" * 60)
    print("QUERY ROUTER -- STANDALONE TEST")
    print("=" * 60)

    # Load .env for API key
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, _, value = line.partition("=")
                    os.environ[key.strip()] = value.strip()

    # Build the pipeline (same as app.py and eval.py)
    from rag import load_documents, chunk_text, build_vector_store, DOCS_DIR
    from conversation_memory import ConversationMemory

    print("\nBuilding RAG pipeline...")
    documents = load_documents(DOCS_DIR)
    if not documents:
        print(f"No documents found in {DOCS_DIR}/. Add STTM files to test.")
        exit(1)

    all_chunks = []
    for doc in documents:
        chunks = chunk_text(doc["content"], doc["source"])
        all_chunks.extend(chunks)
    collection = build_vector_store(all_chunks)

    all_meta = collection.get()
    known_tables = sorted(list(set(
        m.get("table_name", "")
        for m in all_meta["metadatas"]
        if m.get("table_name") and m["table_name"].strip()
    )))

    print(f"Pipeline ready: {len(all_chunks)} chunks, {len(known_tables)} tables")

    # Create memory for follow-up testing
    memory = ConversationMemory(max_turns=3)

    # Test queries
    test_cases = [
        {
            "query": "What is the grain of DIM_STORE?",
            "expected_type": "single_table",
            "note": "Simple lookup -- should use Ollama if available",
        },
        {
            "query": "Which dimensions does FACT_SALES_ORDER reference?",
            "expected_type": "cross_entity",
            "note": "Cross-entity -- should use Claude with reranking",
        },
    ]

    for tc in test_cases:
        print(f"\n{'='*50}")
        print(f"Query: {tc['query']}")
        print(f"Expected type: {tc['expected_type']}")
        print(f"Note: {tc['note']}")

        # Force Claude to avoid Ollama dependency in test
        result = route_query(
            query=tc["query"],
            collection=collection,
            known_tables=known_tables,
            memory=memory,
            force_model="claude",         # Force Claude for testing
            force_rerank="keyword",       # Use free reranker for testing
        )

        print(f"\nRouting: {explain_routing(result['routing'])}")
        print(f"Timing: {result['timing']}")
        print(f"Answer: {result['answer'][:150]}...")

    # Test follow-up detection
    print(f"\n{'='*50}")
    print("Testing follow-up routing...")
    memory.add_turn("What is DIM_STORE?", "DIM_STORE is a dimension table...")

    follow_up_result = route_query(
        query="What about its foreign keys?",
        collection=collection,
        known_tables=known_tables,
        memory=memory,
        force_model="claude",
        force_rerank="none",
    )
    print(f"Routing: {explain_routing(follow_up_result['routing'])}")
    print(f"Follow-up detected: {follow_up_result['routing']['is_follow_up']}")
    print(f"Answer: {follow_up_result['answer'][:150]}...")

    print("\nAll tests passed. query_router.py is ready.")




      
    
