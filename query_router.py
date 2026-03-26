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


def route_query()