"""
model_comparison.py -- LLM Model A/B Testing for RAG
======================================================
Week 5, Step 4 of 4

WHAT THIS FILE DOES
-------------------
Runs your 20-question eval with different Claude models and
compares quality vs cost. The central question:

  "Is Claude Haiku good enough for my use case, or do I need Sonnet?"

This is a COST vs QUALITY tradeoff analysis. For each model, we
measure:
  - Answer quality (keyword score + LLM-as-judge)
  - Latency per query
  - Estimated cost per query
  - Score by question category (simple_lookup, cross_entity, edge_case)


WHY THIS COMPARISON MATTERS
────────────────────────────
Your current setup uses Claude Sonnet for ALL answer generation.
Claude pricing (as of early 2026, approximate):

  Claude Haiku:
    Input:  $0.001 / 1K tokens
    Output: $0.005 / 1K tokens
    Per query estimate: ~$0.002

  Claude Sonnet:
    Input:  $0.003 / 1K tokens
    Output: $0.015 / 1K tokens
    Per query estimate: ~$0.006

At 100 queries/day:
  Sonnet: $0.60/day = $18/month
  Haiku:  $0.20/day = $6/month
  Savings: $12/month (67% cheaper)

But if Haiku answers 30% of questions wrong, the savings are worthless.
This comparison tells you the EXACT quality tradeoff.


EXPECTED RESULTS
─────────────────
Based on experience with STTM domain data:

  Simple lookup ("What is the grain of DIM_STORE?"):
    Both models should score similarly. The answer is verbatim in
    the context, so even a weaker model extracts it correctly.

  Cross-entity ("Which dimensions does FACT_SALES_ORDER reference?"):
    Sonnet should score higher. These require synthesizing information
    across multiple chunks, which benefits from stronger reasoning.

  Edge cases ("What is the SLA for data refresh?"):
    Mixed results. Haiku may be MORE conservative (more "I don't know"
    answers), which could be either better or worse depending on
    whether the correct answer is indeed "I don't know".


THE ROUTING IMPLICATION
────────────────────────
If Haiku scores well on simple_lookup but poorly on cross_entity,
this VALIDATES your query_router.py architecture:

  simple_lookup  -> Haiku (or Ollama) = cheaper, fast, good enough
  cross_entity   -> Sonnet = more expensive but necessary
  edge_case      -> Sonnet = better reasoning for "I don't know"

This is why we build the router in Week 4 and measure in Week 5.
The router's value depends on HOW MUCH quality drops per model tier.

dbt ANALOGY:
  This is like comparing dbt Cloud vs dbt Core for a specific
  project. You measure: build time, cost, features needed. If the
  project only uses basic models, dbt Core is sufficient. If it
  needs dbt Cloud's features (orchestration, CI), the extra cost
  is justified. Same logic: simple queries do not need Sonnet's
  reasoning power.


HOW THIS FILE CONNECTS TO YOUR PROJECT
───────────────────────────────────────
  This file uses your existing eval.py framework.
  It runs the SAME 20 questions with DIFFERENT models and compares.

  It does NOT modify eval.py. Instead, it:
    1. Imports the pipeline building functions from rag.py
    2. Imports scoring functions from eval.py
    3. Calls ask_claude() with different model parameters
    4. Produces a comparison CSV and terminal report

  After running this, you update query_router.py's SIMPLE_MODEL
  and COMPLEX_MODEL constants based on the results.

"""

import os
import csv
import time
import json
from pathlib import Path
from datetime import datetime


MODELS_TO_COMPARE = [
    {
        "id": "claude-haiku-4-5-20251001",
        "name": "Claude Haiku 4.5",
        "tier": "haiku",
        "estimated_cost_per_query": 0.002,
    },
    {
        "id": "claude-sonnet-4-5-20250929",
        "name": "Claude Sonnet 4.5",
        "tier": "sonnet",
        "estimated_cost_per_query": 0.006,
    },
]


def run_model_eval(
        model_id :str,
        questions: list[dict],
        collection,
        known_tables:list[str],
        top_k:int = 3,
        system_prompt:str = None,
) -> list[dict]:
    """
    Run the eval questions against a specific Claude model.

    This is similar to eval.py's run_evaluation(), but:
    1. It takes an explicit model_id parameter
    2. It records timing per question
    3. It does not use LLM-as-judge (to avoid circular bias)

    We avoid LLM-as-judge here because the judge IS Claude.
    Using Sonnet to judge Sonnet creates a bias. Keyword scoring
    is objective and model-independent.

    PARAMETERS
    ----------
    model_id : str
        Anthropic model identifier (e.g., "claude-haiku-4-5-20251001")
    questions : list[dict]
        Eval questions from eval_questions.csv
    collection : chromadb.Collection
        Your ChromaDB collection (shared across models -- same retrieval)
    known_tables : list[str]
        Table names for extract_table_name()
    top_k : int
        Number of chunks to retrieve
    system_prompt : str
        System prompt to use (defaults to IMPROVED_SYSTEM_PROMPT from rag.py)

    RETURNS
    -------
    list[dict]
        Per-question results with scores, timing, and model info.
    """
    import anthropic
    from rag import retrieve,extract_table_name, IMPROVED_SYSTEM_PROMPT
    if system_prompt is None:
        system_prompt = IMPROVED_SYSTEM_PROMPT
    client = anthropic.Anthropic()
    results = []

    for i,q in enumerate(questions):
        question_text = q.get("question", "")
        question_id = q.get("question_id", f"Q{i+1:02d}")
        category = q.get("category", "unknown")
        expected_keywords = q.get("expected_keywords", [])

        if not expected_keywords:
            raw = q.get("expected_answer_keywords","")
            expected_keywords = [kw.strip().upper() for kw in raw.split(",") if kw.strip()]
        detected_table = extract_table_name(question_text,known_tables)
        chunks = retrieve(collection, question_text,table_name=detected_table,known_tables=known_tables,top_k=top_k)

        #Build context string
        #Same as ask_claude in rag.py
        context_parts = []
        for chunk in chunks:
            label_parts = []
            if chunk.get("table_name"):
                label_parts.append(chunk['table_name'])


