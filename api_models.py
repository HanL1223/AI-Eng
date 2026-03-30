"""
Pydantic Request response shcemas for rag api

WHAT THIS FILE DOES
-------------------
Defines the SHAPE of every request and response your API handles.
These are Pydantic models -- Python classes that automatically validate
data coming in and going out of the API.

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
"""

from pydantic import BaseModel,Field
from typing import Optional

# Section 1 request models
#These define what the client sends to the API
# Every filed has a type a default and a description 


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
    query:str = Field(
        ...,
        min_length = 1,
        max_length = 2000,
        description = (
            "The user's question about STTM data. "
            "Must be between 1 and 2000 characters. "
            "Example: 'What is the grain of DIM_STORE?'"
        ),
    )

    top_k:str = Field(default = 3,
                      ge = 1,
                      le = 20,
                      description = (
                          "Number of chunks to retrieve from the vector store. "
            "Higher values give more context but may include noise. "
            "Default: 3. Range: 1-20."
                      ),
                      )
    rerank:bool = Field(
        default = True,
        description  = (
            "Whether to apply reranking to retrieved chunks. "
            "When true, the query router decides the reranking method "
            "(BM25, cross-encoder, or none) based on query complexity."
        ),
    )
    model:Optional[str] = Field(
        default = None,
        description=(
            "Override the model selection. "
            "If null, the query router selects the model automatically. "
            "Examples: 'claude-sonnet-4-5-20250929', 'claude-haiku-4-5-20251001', "
            "'ollama/qwen2.5:0.5b'"),
    )
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
    model_config = {
        "json_schema_extra":{
            "example":[
                {"query": "what is the grain of Dim_Store?",
                 "top_k":3,
                 "rerank":True,
                 "model":None,
                 "include_sources":True,
                 "session_id":"session-001"}
            ]
        }
    }

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