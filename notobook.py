"""
api_models.py -- Pydantic Request/Response Schemas for RAG API
================================================================
Week 6, Step 1 of 6

WHAT THIS FILE DOES
-------------------
Defines the SHAPE of every request and response your API handles.
These are Pydantic models -- Python classes that automatically validate
data coming in and going out of your API.

Think of this file as a SCHEMA DEFINITION, exactly like a dbt schema.yml
that defines columns, types, and constraints for your data models.


WHY SEPARATE MODELS FROM THE SERVER?
--------------------------------------
You could define Pydantic models inline in api_server.py, but separating
them has three benefits:

  1. REUSABILITY: The same models can be used by the test client,
     documentation generators, and other services.

  2. CLARITY: api_server.py focuses on LOGIC (what to do with requests).
     api_models.py focuses on SHAPE (what requests look like).

  3. TESTING: You can unit-test models independently -- create one,
     check validation, verify serialization -- without starting a server.

dbt ANALOGY:
  This is like separating schema.yml (column definitions, tests) from
  the SQL model file (the actual transformation logic). The schema
  file is the CONTRACT; the SQL file is the IMPLEMENTATION.


PYDANTIC IN 5 MINUTES
-----------------------
Pydantic is a data validation library built on Python type hints. When
you define a Pydantic model, you get:

  1. VALIDATION: If incoming data has wrong types, Pydantic raises an error
  2. SERIALIZATION: .model_dump() converts to dict, .model_dump_json() to JSON
  3. DEFAULTS: Fields with default values are optional in the input
  4. DOCUMENTATION: FastAPI reads the model and generates API docs

Here is a minimal example to understand the syntax:

  from pydantic import BaseModel

  class User(BaseModel):
      name: str               # Required string field
      age: int                # Required integer field
      email: str = ""         # Optional string, defaults to ""

  # Valid:
  user = User(name="Test", age=30)
  print(user.name)    # "Test"
  print(user.email)   # ""  (default applied)

  # Invalid (raises ValidationError):
  user = User(name="Test", age="not a number")
  # Error: age - Input should be a valid integer

  # From JSON dict:
  data = {"name": "Test", "age": 30, "email": "test@example.com"}
  user = User(**data)      # ** unpacks dict as keyword arguments
  user = User.model_validate(data)  # Same thing, more explicit


PYDANTIC vs PYTHON DATACLASSES
--------------------------------
You may have seen Python's built-in dataclasses:

  from dataclasses import dataclass

  @dataclass
  class User:
      name: str
      age: int

Pydantic models look similar but do MORE:
  - dataclass: stores data, no validation
  - Pydantic:  stores data + validates types + serializes to JSON

  dataclass:  User(name="Test", age="oops")  -> stores "oops" as age
  Pydantic:   User(name="Test", age="oops")  -> raises ValidationError

For APIs, you ALWAYS want validation. A malformed request should be
rejected immediately with a clear error, not silently accepted and
then crash somewhere deep in your pipeline.


FIELD VALIDATORS AND CONSTRAINTS
----------------------------------
Pydantic supports validation beyond basic types:

  from pydantic import BaseModel, Field

  class QueryRequest(BaseModel):
      query: str = Field(..., min_length=1, max_length=1000)
      top_k: int = Field(default=3, ge=1, le=20)

  Field(...) means "required" (the ... is Python's Ellipsis literal).
  min_length=1 means the string must have at least 1 character.
  ge=1 means "greater than or equal to 1".
  le=20 means "less than or equal to 20".

  If someone sends top_k=0, they get:
    422 Unprocessable Entity
    {"detail": [{"msg": "Input should be greater than or equal to 1"}]}

This is AUTOMATIC -- you never write if/else validation code.


HOW THIS FILE CONNECTS TO YOUR PROJECT
-----------------------------------------
  api_server.py imports these models:
    from api_models import QueryRequest, QueryResponse, HealthResponse, ...

  FastAPI uses them to:
    1. Parse and validate incoming JSON -> QueryRequest
    2. Serialize outgoing data -> QueryResponse (as JSON)
    3. Generate /docs page with all field descriptions

  test_api.py uses them to:
    1. Construct valid requests
    2. Parse and verify responses


DEPENDENCIES
-------------
  pydantic (already installed -- it comes with FastAPI)
"""

from pydantic import BaseModel, Field
from typing import Optional


# =====================================================================
# SECTION 1: REQUEST MODELS
# =====================================================================
# These define what the CLIENT sends to the API.
# Every field has a type, a default (if optional), and a description.
#
# DESIGN DECISION: Why descriptions on every field?
# --------------------------------------------------
# FastAPI reads these descriptions and includes them in the auto-
# generated API documentation at /docs. Without descriptions, your
# team would have to guess what each field means. With descriptions,
# the API is self-documenting.
#
# PYTHON REFRESHER: Field() vs plain defaults
# ---------------------------------------------
# These two are equivalent for setting defaults:
#   top_k: int = 3
#   top_k: int = Field(default=3)
#
# But Field() lets you add constraints and metadata:
#   top_k: int = Field(default=3, ge=1, le=20, description="...")
#
# Rule: Use Field() when you need validation or description.
#       Use plain defaults for simple fields.
# =====================================================================


class QueryRequest(BaseModel):
    """
    Request body for the POST /api/query endpoint.

    This is what the client sends when asking a question.
    FastAPI automatically validates every field against
    these type annotations and constraints.

    EXAMPLE REQUEST (what the client sends as JSON):
    {
        "query": "What is the grain of DIM_STORE?",
        "top_k": 3,
        "rerank": true,
        "model": null,
        "include_sources": true,
        "session_id": "abc-123"
    }
    """

    query: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description=(
            "The user's question about STTM data. "
            "Must be between 1 and 2000 characters. "
            "Example: 'What is the grain of DIM_STORE?'"
        ),
    )
    # PYTHON REFRESHER: Field(...)
    # The first argument `...` is Python's Ellipsis literal.
    # In Pydantic, Ellipsis means "this field is required".
    # If the client omits `query`, they get a 422 error.
    #
    # You could also write: query: str (no Field)
    # But then you lose the constraints (min_length, max_length)
    # and the description for the auto-generated docs.

    top_k: int = Field(
        default=3,
        ge=1,
        le=20,
        description=(
            "Number of chunks to retrieve from the vector store. "
            "Higher values give more context but may include noise. "
            "Default: 3. Range: 1-20."
        ),
    )
    # GOTCHA: ge/le are Pydantic constraint names.
    #   ge = Greater than or Equal to
    #   le = Less than or Equal to
    #   gt = Greater Than (strict)
    #   lt = Less Than (strict)
    #
    # These map to JSON Schema's minimum/maximum fields.

    rerank: bool = Field(
        default=True,
        description=(
            "Whether to apply reranking to retrieved chunks. "
            "When true, the query router decides the reranking method "
            "(BM25, cross-encoder, or none) based on query complexity."
        ),
    )

    model: Optional[str] = Field(
        default=None,
        description=(
            "Override the model selection. "
            "If null, the query router selects the model automatically. "
            "Examples: 'claude-sonnet-4-5-20250929', 'claude-haiku-4-5-20251001', "
            "'ollama/qwen2.5:0.5b'"
        ),
    )
    # PYTHON REFRESHER: Optional[str]
    # Optional[str] is equivalent to: str | None
    # It means: this field can be a string OR None.
    #
    # In Pydantic:
    #   Optional[str] = Field(default=None)
    # means: if the client does not send this field, it is None.
    #
    # vs:
    #   str = Field(...)
    # means: the client MUST send this field (required).

    include_sources: bool = Field(
        default=True,
        description=(
            "Whether to include source chunk details in the response. "
            "Set to false for simpler responses (e.g., in a Slack bot)."
        ),
    )

    session_id: Optional[str] = Field(
        default=None,
        max_length=100,
        description=(
            "Session identifier for conversation memory. "
            "If provided, the server maintains conversation context "
            "across multiple queries in the same session."
        ),
    )

    # MODEL CONFIG: Controls serialization behavior.
    # This is Pydantic v2 syntax (model_config dict).
    # Pydantic v1 used a nested class Config.
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "query": "What is the grain of DIM_STORE?",
                    "top_k": 3,
                    "rerank": True,
                    "model": None,
                    "include_sources": True,
                    "session_id": "session-001",
                }
            ]
        }
    }
    # DESIGN DECISION: json_schema_extra
    # ------------------------------------
    # This provides example values that appear in the /docs page.
    # Your team can click "Try it out" and see a pre-filled example.


class FeedbackRequest(BaseModel):
    """
    Request body for the POST /api/feedback endpoint.

    Allows users to rate a response as helpful or not.
    This data feeds into your evaluation pipeline.

    EXAMPLE REQUEST:
    {
        "query": "What is the grain of DIM_STORE?",
        "answer": "The grain is one row per store.",
        "rating": 5,
        "comment": "Correct and clear"
    }
    """

    query: str = Field(
        ...,
        description="The original question that was asked.",
    )
    answer: str = Field(
        ...,
        description="The answer that was returned.",
    )
    rating: int = Field(
        ...,
        ge=1,
        le=5,
        description="Rating from 1 (terrible) to 5 (perfect).",
    )
    comment: Optional[str] = Field(
        default=None,
        max_length=500,
        description="Optional free-text comment about the response quality.",
    )


# =====================================================================
# SECTION 2: RESPONSE MODELS
# =====================================================================
# These define what the SERVER sends back to the client.
# They serve two purposes:
#   1. FastAPI uses them to serialize (convert to JSON)
#   2. FastAPI uses them to document the response shape in /docs
#
# DESIGN DECISION: Why not just return a plain dict?
# ---------------------------------------------------
# You COULD write: return {"answer": "...", "sources": [...]}
# FastAPI would serialize it to JSON just fine.
#
# But Pydantic models give you:
#   1. TYPE SAFETY: Your editor catches typos (e.g., "anwser")
#   2. DOCUMENTATION: /docs shows the exact response shape
#   3. CONSISTENCY: Every response follows the same structure
#   4. NESTED MODELS: SourceChunk inside QueryResponse
#
# In production, response models prevent you from accidentally
# returning internal data (e.g., raw embeddings, API keys).
# =====================================================================


class SourceChunk(BaseModel):
    """
    A single retrieved chunk included in the response.

    This is a NESTED MODEL -- it appears inside QueryResponse's
    sources list. Pydantic handles nested models automatically.
    """

    text: str = Field(
        description="The chunk text content (truncated to 500 chars)."
    )
    source: str = Field(
        description="Source document identifier (e.g., 'STTM__DIM_STORE__summary')."
    )
    table_name: str = Field(
        default="",
        description="Extracted table name (e.g., 'DIM_STORE')."
    )
    doc_type: str = Field(
        default="",
        description="Document type (e.g., 'summary', 'column_mapping')."
    )
    relevance_score: Optional[float] = Field(
        default=None,
        description=(
            "Relevance score from retrieval or reranking. "
            "Higher is more relevant. Scale depends on the method used."
        ),
    )


class RoutingInfo(BaseModel):
    """
    Details about how the query was routed internally.

    Exposed so the client can understand WHY a particular model
    or reranking method was used. This is valuable for debugging
    and for building smarter client-side logic.
    """

    query_type: str = Field(
        description=(
            "Classified query type: 'single_table', 'cross_entity', "
            "'edge_case', or 'follow_up'."
        ),
    )
    model: str = Field(
        description="The model that generated the answer."
    )
    rerank_method: str = Field(
        description="Reranking method used: 'bm25', 'cross_encoder', 'llm', or 'none'."
    )
    is_follow_up: bool = Field(
        description="Whether this query was classified as a follow-up."
    )


class TimingInfo(BaseModel):
    """
    Latency breakdown for the query pipeline.

    Each phase is measured independently so you can identify bottlenecks.
    """

    retrieve_ms: float = Field(
        description="Time to retrieve chunks from ChromaDB (milliseconds)."
    )
    rerank_ms: float = Field(
        default=0.0,
        description="Time spent reranking (0 if reranking was skipped)."
    )
    generate_ms: float = Field(
        description="Time for the LLM to generate the answer."
    )
    total_ms: float = Field(
        description="Total end-to-end latency."
    )


class QueryResponse(BaseModel):
    """
    Response body for the POST /api/query endpoint.

    This is the complete response the client receives after
    submitting a question.

    EXAMPLE RESPONSE:
    {
        "answer": "The grain of DIM_STORE is one row per store location.",
        "sources": [
            {
                "text": "DIM_STORE: Grain = one row per store...",
                "source": "STTM__DIM_STORE__summary",
                "table_name": "DIM_STORE",
                "doc_type": "summary",
                "relevance_score": 0.92
            }
        ],
        "routing": {
            "query_type": "single_table",
            "model": "claude-sonnet-4-5-20250929",
            "rerank_method": "none",
            "is_follow_up": false
        },
        "timing": {
            "retrieve_ms": 42.5,
            "rerank_ms": 0,
            "generate_ms": 1150.3,
            "total_ms": 1192.8
        }
    }
    """

    answer: str = Field(
        description="The generated answer to the user's question."
    )
    sources: list[SourceChunk] = Field(
        default_factory=list,
        description=(
            "Retrieved source chunks. Empty if include_sources was false "
            "or if no relevant chunks were found."
        ),
    )
    # PYTHON REFRESHER: default_factory
    # -----------------------------------
    # default_factory=list means "call list() to create a NEW empty list
    # for each instance". This is important because:
    #
    #   default=[]
    #   This creates ONE list object shared by ALL instances (mutable default bug).
    #
    #   default_factory=list
    #   This creates a NEW list for each instance (correct).
    #
    # You saw this same issue in Week 3 with Streamlit session state.
    # The rule: NEVER use a mutable object as a default value.

    routing: Optional[RoutingInfo] = Field(
        default=None,
        description="How the query was routed (model, reranking, type)."
    )
    timing: Optional[TimingInfo] = Field(
        default=None,
        description="Latency breakdown for each pipeline phase."
    )


class HealthResponse(BaseModel):
    """
    Response for the GET /api/health endpoint.

    Health checks are used by:
      - Docker: to know if the container is ready
      - Load balancers: to route traffic to healthy instances
      - Monitoring: to alert when the service is down

    A health check should be FAST (no heavy computation) and
    verify that critical dependencies are available.
    """

    status: str = Field(
        description="'healthy' if the service is operational."
    )
    documents_loaded: int = Field(
        description="Number of documents currently in the vector store."
    )
    tables_available: int = Field(
        description="Number of distinct tables recognized."
    )
    version: str = Field(
        default="0.1.0",
        description="API version string."
    )


class TablesResponse(BaseModel):
    """
    Response for the GET /api/tables endpoint.

    Lists all table names the chatbot knows about.
    Useful for clients that want to show a dropdown or autocomplete.
    """

    tables: list[str] = Field(
        description="List of table names (e.g., ['DIM_STORE', 'FACT_SALES_ORDER'])."
    )
    count: int = Field(
        description="Number of tables."
    )


class StatsResponse(BaseModel):
    """
    Response for the GET /api/stats endpoint.

    Returns query analytics from the JSONL log.
    """

    total_queries: int = Field(
        description="Total number of queries logged."
    )
    avg_latency_ms: float = Field(
        description="Average end-to-end latency in milliseconds."
    )
    p95_latency_ms: float = Field(
        description="95th percentile latency in milliseconds."
    )
    total_cost_usd: float = Field(
        description="Total estimated API cost in USD."
    )
    model_distribution: dict[str, int] = Field(
        default_factory=dict,
        description="Count of queries per model."
    )


class ErrorResponse(BaseModel):
    """
    Standard error response body.

    FastAPI returns 422 errors automatically for validation failures.
    This model is for YOUR custom errors (404, 500, etc.).
    """

    error: str = Field(
        description="Error type (e.g., 'not_found', 'pipeline_error')."
    )
    message: str = Field(
        description="Human-readable error description."
    )
    detail: Optional[str] = Field(
        default=None,
        description="Additional technical details (shown in debug mode)."
    )


# =====================================================================
# SECTION 3: STANDALONE TEST
# =====================================================================
# Verify that models validate correctly without starting a server.
# This is a UNIT TEST for your data contracts.
# =====================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("API MODELS -- STANDALONE TEST")
    print("=" * 60)

    # --- Test 1: Valid QueryRequest ---
    print("\n--- Test 1: Valid QueryRequest ---")
    req = QueryRequest(
        query="What is the grain of DIM_STORE?",
        top_k=5,
        rerank=True,
    )
    print(f"  query: {req.query}")
    print(f"  top_k: {req.top_k}")
    print(f"  model: {req.model}")  # Should be None (default)
    print(f"  session_id: {req.session_id}")  # Should be None

    # Convert to dict (what FastAPI does internally):
    req_dict = req.model_dump()
    print(f"  model_dump(): {req_dict}")

    # Convert to JSON string:
    req_json = req.model_dump_json()
    print(f"  JSON: {req_json[:80]}...")
    print("  PASS")

    # --- Test 2: Validation catches bad input ---
    print("\n--- Test 2: Validation catches bad input ---")
    from pydantic import ValidationError

    test_cases = [
        ({"query": ""}, "empty query (min_length=1)"),
        ({"query": "x" * 2001}, "query too long (max_length=2000)"),
        ({}, "missing required field 'query'"),
        ({"query": "ok", "top_k": 0}, "top_k below minimum (ge=1)"),
        ({"query": "ok", "top_k": 21}, "top_k above maximum (le=20)"),
        ({"query": "ok", "top_k": "not_a_number"}, "top_k wrong type"),
    ]

    for data, description in test_cases:
        try:
            QueryRequest(**data)
            print(f"  FAIL: {description} -- should have raised error")
        except ValidationError as e:
            # Count the number of validation errors
            error_count = len(e.errors())
            print(f"  PASS: {description} -- caught {error_count} error(s)")

    # --- Test 3: QueryResponse with nested models ---
    print("\n--- Test 3: QueryResponse with nested models ---")
    resp = QueryResponse(
        answer="The grain of DIM_STORE is one row per store location.",
        sources=[
            SourceChunk(
                text="DIM_STORE: Grain = one row per store...",
                source="STTM__DIM_STORE__summary",
                table_name="DIM_STORE",
                doc_type="summary",
                relevance_score=0.92,
            )
        ],
        routing=RoutingInfo(
            query_type="single_table",
            model="claude-sonnet-4-5-20250929",
            rerank_method="none",
            is_follow_up=False,
        ),
        timing=TimingInfo(
            retrieve_ms=42.5,
            rerank_ms=0.0,
            generate_ms=1150.3,
            total_ms=1192.8,
        ),
    )
    resp_dict = resp.model_dump()
    print(f"  answer: {resp.answer[:50]}...")
    print(f"  sources: {len(resp.sources)} chunk(s)")
    print(f"  routing.model: {resp.routing.model}")
    print(f"  timing.total_ms: {resp.timing.total_ms}")
    print(f"  JSON size: {len(resp.model_dump_json())} bytes")
    print("  PASS")

    # --- Test 4: HealthResponse ---
    print("\n--- Test 4: HealthResponse ---")
    health = HealthResponse(
        status="healthy",
        documents_loaded=42,
        tables_available=15,
    )
    print(f"  {health.model_dump_json()}")
    print("  PASS")

    # --- Test 5: FeedbackRequest validation ---
    print("\n--- Test 5: FeedbackRequest validation ---")
    try:
        FeedbackRequest(query="test", answer="test", rating=6)
        print("  FAIL: rating=6 should have been rejected (le=5)")
    except ValidationError:
        print("  PASS: rating=6 correctly rejected")

    try:
        fb = FeedbackRequest(query="test", answer="test", rating=4)
        print(f"  PASS: valid feedback created: rating={fb.rating}")
    except ValidationError:
        print("  FAIL: valid feedback was rejected")

    print("\n" + "=" * 60)
    print("All model tests passed. api_models.py is ready.")
    print("=" * 60)