"""
Wrap rag pipeline behind a REST API

ARCHITECTURE
-------------
  HTTP Client (curl, browser, Slack bot, mobile app)
       |
       | HTTP POST /api/query  {"query": "What is DIM_STORE?"}
       v
  ┌─────────────────────────────────────────────────────────┐
  │  api_server.py (THIS FILE)                              │
  │                                                          │
  │  1. Uvicorn receives HTTP request                        │
  │  2. FastAPI parses JSON -> QueryRequest (Pydantic)       │
  │  3. Handler calls query_router.route_query()             │
  │  4. route_query() orchestrates:                          │
  │     - rag.retrieve() -> chunks from ChromaDB             │
  │     - reranker.rerank_chunks() -> refined chunks         │
  │     - model_switcher.generate() -> LLM answer            │
  │  5. Handler builds QueryResponse (Pydantic)              │
  │  6. FastAPI serializes -> JSON                           │
  │  7. Uvicorn sends HTTP response                          │
  └─────────────────────────────────────────────────────────┘
       |
       | HTTP 200  {"answer": "DIM_STORE is a dimension table..."}
       v
  HTTP Client receives the response

  KEY FASTAPI CONCEPTS USED IN THIS FILE
-----------------------------------------

1. ROUTES (also called "endpoints" or "path operations"):
   A route maps a URL + HTTP method to a Python function.

   @app.get("/api/health")
   def health_check():
       return {"status": "healthy"}

   This says: when someone sends GET /api/health, call health_check().
   The decorator (@app.get) registers the route with FastAPI.

2. REQUEST BODY:
   For POST endpoints, FastAPI reads JSON from the request body and
   converts it to a Pydantic model:

   @app.post("/api/query")
   def query(request: QueryRequest):
       # request.query is already validated as a string
       # request.top_k is already validated as an int between 1-20

3. LIFESPAN:
   Code that runs when the server STARTS and STOPS. We use this to
   build the RAG pipeline once at startup, rather than on every request.

   @asynccontextmanager
   async def lifespan(app: FastAPI):
       # STARTUP: build pipeline
       yield
       # SHUTDOWN: cleanup

4. DEPENDENCY INJECTION:
   FastAPI can automatically provide shared objects to route handlers.
   We use this to pass the RAG pipeline state to every endpoint.

5. RESPONSE MODELS:
   response_model=QueryResponse tells FastAPI the SHAPE of the response.
   This enables automatic documentation and serialization.

"""

import os
import time
import traceback
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from api_models import (
    QueryRequest,
    QueryResponse,
    SourceChunk,
    RoutingInfo,
    TimingInfo,
    HealthResponse,
    TablesResponse,
    StatsResponse,
    ErrorResponse,
    FeedbackRequest,
)



if not os.environ.get("ANTHROPIC_API_KEY"):
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),".env")
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and "=" in line:
                    key,_,value = line.partition("=")
                    os.environ[key.strip()] = value.strip()


pipeline_state = {
    "collection":None,
    "known_tables":[],
    "documents_count":0,
    "memories":{}, # session_id -> ConversationMemory instance
    "ready":False
}

#Lifespan
@asynccontextmanager
async def lifespan(app:FastAPI):
    """
    Build RAG pipeline once at server startup
    """
    print("STARTING RAG API SERVER")
    print("=" * 60)

    from rag import (
        load_documents,
        chunk_text,
        build_vector_store,
        extract_table_name,
        DOCS_DIR
    )

    print("\n[1/3] Loading documents...")
    startup_start = time.time()
    documents = load_documents(DOCS_DIR)

    if not documents:
        print(f"WARNING: No documents found in {DOCS_DIR}/")
        print("The API will start but /api/query will return empty results.")
        print(f"Add .xlsx, .pdf, or .txt files to {DOCS_DIR}/ and restart.")
    else:
        print(f"Loaded {len(documents)} documents")

    print("[2/3] Chunking and building vector store...")
    all_chunks = []
    for doc in documents:
        chunks = chunk_text(doc["content"],doc["source"])
        all_chunks.extend(chunks)
    print(f"Created {len(all_chunks)} chunks")

    collection = build_vector_store(all_chunks)
    print(f"Vector store ready")

    #Extract known table name
    print("[3/3] Extracting table names...")
    all_names = []
    for doc in documents:
        parts = doc["source"].split("__")
        if len(parts) >= 2:
            all_names.append(parts[1].strip().upper())

    known_tables = sorted(set(name for name in all_names if name))

    print(f"  Found {len(known_tables)} tables: {known_tables[:5]}...")
    startup_time = time.time() - startup_start
    print(f"\nStartup complete in {startup_time:.1f}s")
    print(f"  Documents: {len(documents)}")
    print(f"  Chunks:    {len(all_chunks)}")
    print(f"  Tables:    {len(known_tables)}")
    print("=" * 60)

    #Populate the shared state
    pipeline_state["collection"] = collection
    pipeline_state["known_tables"] = known_tables
    pipeline_state["documents_count"] = len(documents)
    pipeline_state["ready"] = True

    #yield = server is now running and handling requests ──
    yield

    print("\nShutting down RAG API server...")
    pipeline_state["ready"] = False



#FASTAPI APPLICATION
# The FastAPI() constructor creates the application instance.
# All routes are registered on this instance via decorators.
#
# PARAMETERS EXPLAINED:
#   title:       Appears in the /docs page header
#   description: Appears below the title in /docs
#   version:     API version (semantic versioning)
#   lifespan:    The startup/shutdown handler (Section 3)
#
# PYTHON REFRESHER: Decorators
# ------------------------------
# @app.get("/api/health") is a decorator. It modifies the function
# below it by registering it as a route handler. Conceptually:
#
#   @app.get("/api/health")
#   def health_check(): ...
#
# is equivalent to:
#   def health_check(): ...
#   health_check = app.get("/api/health")(health_check)
#
# The decorator pattern is used everywhere in Python frameworks.

app = FastAPI(
    title = "STTM RAG Assistant API",
    description=(
        "REST API for querying Sigma Healthcare's Snowflake data warehouse "
        "documentation (STTM). Wraps a RAG pipeline with ChromaDB vector "
        "store, BM25/cross-encoder reranking, and Claude/Ollama generation."
    ),
    version="0.1.0",
    lifespan=lifespan,
)


#CORS MIDDLEWAR
# CORS (Cross-Origin Resource Sharing) controls which websites can
# call your API from a browser.
#
# Without CORS, a browser running code from http://localhost:8501
# (your Streamlit app) would be BLOCKED from calling your API at
# http://localhost:8000. This is a browser security feature called
# the Same-Origin Policy.
#
# CORS middleware tells the browser: "it is OK for these origins to
# call my API."
#
# SECURITY NOTE: In production, replace ["*"] with specific origins:
#   allow_origins=["https://your-app.example.com"]
# Using ["*"] allows ANY website to call your API, which is fine for
# development but not for production with sensitive data.
#
# WHY THIS ONLY AFFECTS BROWSERS:
# curl, Postman, Python requests, and mobile apps are NOT affected by
# CORS. It is purely a browser security feature. Your test_api.py
# script will work fine without CORS configuration.


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # Allow all origins (dev only)
    allow_credentials=True,
    allow_methods=["*"],       # Allow all HTTP methods
    allow_headers=["*"],       # Allow all headers
)


# =====================================================================
# SECTION 6: ROUTE HANDLERS
# =====================================================================
# Each route handler is a function decorated with @app.get() or
# @app.post(). FastAPI reads the function's type hints to:
#   1. Parse the request body (Pydantic model in function params)
#   2. Serialize the response (response_model= in decorator)
#   3. Generate /docs page entries
#
# NAMING CONVENTION:
#   /api/...  prefix distinguishes API routes from potential web pages.
#   This is a common REST API convention (also: /v1/... for versioning).
# =====================================================================


# ── GET /api/health ──
# Health check endpoint. Used by Docker, load balancers, and monitoring.
# Should be FAST -- no heavy computation, just report readiness.

@app.get(
    "/api/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Returns the API's current health status and pipeline statistics.",
    tags=["System"],
)
def health_check():
    """
    Retrun a server's health status

    For simple operations (reading from a dict), sync is fine and simpler.
    We use async for the /api/query endpoint because it calls the LLM API,
    which is an I/O-bound operation that benefits from async.

    RULE OF THUMB:
      - Quick dict lookups, math -> sync (def)
      - Network calls, file I/O -> async (async def)
      - Not sure? -> sync is always safe (FastAPI handles it)
    """

    if not pipeline_state["ready"]:
        #Return 503
        raise HTTPException(
            status_code = 503,
            detail = "Pipelien is still loading"
        )
    return HealthResponse(
        status = 'healthy',
        documents_loaded  = pipeline_state["documents_count"],
        tables_available = len(pipeline_state["known_tables"]),
        version = "0.1.0"
    )



# ── GET /api/tables ──
# List all known table names. Useful for autocomplete in clients.
@app.get(
    "/api/tables",
    response_model = TablesResponse,
    summary="List available tables",
    description="Returns all table names the chatbot knows about.",
    tags=["System"],
)
def list_tables():
    """Return all tables names extracted from loaded documents"""
    tables = pipeline_state["known_tables"]
    return TablesResponse(
        tables = tables,
        count = len(tables)
    )

# ── GET /api/stats ──
# Query analytics from the JSONL log.
@app.get(
    "/api/stats",
    response_model=StatsResponse,
    summary="Query statistics",
    description="Returns aggregated query analytics from the log.",
    tags=["System"],
)
def get_stats():
    """
    Return query statistics from the JSONL query log.

    This reuses your existing query_logger.py analytics functions.
    If no logs exist, returns zeroed-out stats.
    """
    from query_logger import load_logs, analyze_logs

    entries = load_logs()
    if not entries:
        return StatsResponse(
            total_queries=0,
            avg_latency_ms=0.0,
            p95_latency_ms=0.0,
            total_cost_usd=0.0,
            model_distribution={},
        )

    analytics = analyze_logs(entries)

    # Extract fields from your existing analytics structure.
    # analytics has keys: volume, cost, latency, models
    volume = analytics.get("volume", {})
    cost = analytics.get("cost", {})
    latency = analytics.get("latency", {}).get("total_ms", {})
    models = analytics.get("models", {})

    return StatsResponse(
        total_queries=volume.get("total_queries", 0),
        avg_latency_ms=latency.get("mean", 0.0),
        p95_latency_ms=latency.get("p95", 0.0),
        total_cost_usd=cost.get("total_usd", 0.0),
        model_distribution=models.get("distribution", {}),
    )


# ── POST /api/query ──
# The main endpoint. Receives a question, returns an answer.
# This is where the RAG pipeline is invoked.

@app.post(
    "/api/query",
    response_model = QueryResponse,
    summary = "Ask a question",
    description = (
        "Submit a question about STTM data and receive an answer "
        "with source citations, routing details, and timing information."
    ),
    tags = ['Query'],
    responses={
        422: {"model": ErrorResponse, "description": "Validation error"},
        500: {"model": ErrorResponse, "description": "Pipeline error"},
        503: {"model": ErrorResponse, "description": "Service not ready"},
    },
)

def handle_query(request:QueryRequest):
    """
    Process a RAG query and return answer

    """
    if not pipeline_state["ready"]:
        raise HTTPException(
            status_code = 503,
            detail = "Pipeline is still loading"
        )
    #Get or create conversation memory
    from conversation_memory import ConversationMemory

    session_id = request.session_id or "default"
    if session_id not in pipeline_state["memories"]:
        pipeline_state["memories"][session_id] = ConversationMemory(
            max_turns=5
        )
    memory = pipeline_state["memories"][session_id]

    # ── Step 2: Call the query router ──
    # route_query() is the SAME function your app.py uses.
    # It handles: classification, retrieval, reranking, model selection,
    # generation, and timing measurement.
    try:
        from query_router import route_query
        result = route_query(query = request.query,
                             collection=pipeline_state['collection'],
                             known_tables=pipeline_state["known_tables"],
                             memory=memory)
    except Exception as e:
        print(f"Error in route_query: {e}")
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Pipeline error: {str(e)}",
        )
    

    #Extract results
    answer = result.get("answer", "No answer generated.")
    chunks = result.get("chunks", [])
    routing = result.get("routing", {})
    timing = result.get("timing", {})

    #Update conversation memory
    memory.add_turn(request.query, answer)


    #Log the query
    try:
        from query_logger import log_query,generate_session_id
        log_query(
            query=request.query,
            answer=answer,
            routing=routing,
            timing=timing,
            chunks=chunks,
            session_id=session_id,
        )
    except Exception as e:
        # Logging failure should NOT crash the request
        print(f"WARNING: Query logging failed: {e}")

    #Build response
    #Convert internal chunk dict to Pydantic Sourcehunk models
    source_chunks = []
    if request.include_sources:
        for chunk in chunks:
            source_chunks.append(
                SourceChunk(
                    text=chunk.get("text", "")[:500],
                    source=chunk.get("source", ""),
                    table_name=chunk.get("table_name", ""),
                    doc_type=chunk.get("doc_type", ""),
                    relevance_score=chunk.get("rerank_score"),
                )
            )
    routing_info = RoutingInfo(
        query_type=routing.get("query_type", "unknown"),
        model=routing.get("model", "unknown"),
        rerank_method=routing.get("rerank_method", "none"),
        is_follow_up=routing.get("is_follow_up", False),
    )

    timing_info = TimingInfo(
        retrieve_ms=timing.get("retrieve_ms", 0.0),
        rerank_ms=timing.get("rerank_ms", 0.0),
        generate_ms=timing.get("generate_ms", 0.0),
        total_ms=timing.get("total_ms", 0.0),
    )

    return QueryResponse(
        answer=answer,
        sources=source_chunks,
        routing=routing_info,
        timing=timing_info,
    )

# ── POST /api/feedback ──
# Accept user feedback on responses.

@app.post(
    "/api/feedback",
    summary="Submit feedback",
    description="Rate a response for quality tracking.",
    tags=["Query"],
)
def submit_feedback(feedback: FeedbackRequest):
    """
    Log user feedback for future evaluation.

    This endpoint appends feedback to the query log. In a production
    system, this data would feed into your DPO training pipeline
    (Phase 4 of your roadmap) to improve model quality.

    DESIGN DECISION: Why a separate endpoint?
    -------------------------------------------
    You could include a rating field in QueryRequest and have the
    client send feedback with the next query. But that couples the
    rating to the next interaction. A separate endpoint lets the
    client send feedback at any time (or never).
    """
    from query_logger import log_query

    try:
        log_query(
            query=feedback.query,
            answer=feedback.answer,
            routing={"feedback": True, "rating": feedback.rating},
            timing={"total_ms": 0},
            chunks=[],
            extra={"rating": feedback.rating, "comment": feedback.comment},
        )
        return {"status": "ok", "message": "Feedback recorded."}
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to log feedback: {str(e)}",
        )


# =====================================================================
# SECTION 7: STANDALONE EXECUTION
# =====================================================================
# When you run `python api_server.py` directly, it starts the server.
# This is convenient for development. In production, you run:
#   uvicorn api_server:app --host 0.0.0.0 --port 8000
#
# The difference:
#   python api_server.py       -> uses uvicorn.run() below
#   uvicorn api_server:app     -> uvicorn imports the `app` object directly
#
# Both are valid. The uvicorn command gives you more control over
# workers, host binding, and other server options.

# =====================================================================

if __name__ == "__main__":
    import uvicorn

    print("Starting STTM RAG API server...")
    print("  Docs:   http://localhost:8000/docs")
    print("  ReDoc:  http://localhost:8000/redoc")
    print("  Health: http://localhost:8000/api/health")
    print()

    uvicorn.run(
        "api_server:app",
        # EXPLANATION: "api_server:app" is a string, not the app object.
        # This tells uvicorn to import the module "api_server" and use
        # the object named "app". This is required for --reload to work
        # (uvicorn needs to re-import the module on file changes).
        host="0.0.0.0",
        # EXPLANATION: 0.0.0.0 means "listen on all network interfaces".
        # localhost/127.0.0.1 = only accessible from this machine
        # 0.0.0.0 = accessible from other machines on the network
        # In Docker, you MUST use 0.0.0.0 or the container will not
        # receive requests from outside.
        port=8000,
        reload=True,
        # EXPLANATION: reload=True watches for file changes and restarts
        # the server automatically. Great for development. NEVER use in
        # production (it is slower and uses more memory).
        log_level="info",
    )

