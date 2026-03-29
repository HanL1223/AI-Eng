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


DEPENDENCIES
────────────
  None beyond what you already have (anthropic, rag.py, eval.py).
"""

import os
import csv
import time
import json
from pathlib import Path
from datetime import datetime


# =====================================================================
# SECTION 1: MODEL CONFIGURATIONS
# =====================================================================

# Models to compare. Each entry has:
#   id:     The Anthropic model string (passed to client.messages.create)
#   name:   Human-readable label
#   tier:   Cost tier for grouping
#   cost:   Estimated cost per query (input + output tokens)
#
# GOTCHA: Model IDs change when Anthropic releases new versions.
# Check https://docs.anthropic.com/en/docs/about-claude/models
# for the latest model strings.
#
# At the time of writing (March 2026):
#   claude-haiku-4-5-20251001  = latest Haiku
#   claude-sonnet-4-5-20250929 = latest Sonnet (your current default)

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


# =====================================================================
# SECTION 2: SINGLE-MODEL EVALUATION
# =====================================================================

def run_model_eval(
    model_id: str,
    questions: list[dict],
    collection,
    known_tables: list[str],
    top_k: int = 3,
    system_prompt: str = None,
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
    from rag import retrieve, extract_table_name, IMPROVED_SYSTEM_PROMPT

    if system_prompt is None:
        system_prompt = IMPROVED_SYSTEM_PROMPT

    client = anthropic.Anthropic()
    results = []

    for i, q in enumerate(questions):
        question_text = q.get("question", "")
        question_id = q.get("question_id", f"Q{i+1:02d}")
        category = q.get("category", "unknown")
        expected_keywords = q.get("expected_keywords", [])

        if not expected_keywords:
            raw = q.get("expected_answer_keywords", "")
            expected_keywords = [
                kw.strip().upper() for kw in raw.split(",") if kw.strip()
            ]

        # Retrieve chunks (same retrieval for all models).
        detected_table = extract_table_name(question_text, known_tables)
        chunks = retrieve(
            collection, question_text,
            table_name=detected_table,
            known_tables=known_tables,
            top_k=top_k,
        )

        # Build context string (same as ask_claude in rag.py).
        context_parts = []
        for chunk in chunks:
            label_parts = []
            if chunk.get("table_name"):
                label_parts.append(chunk["table_name"])
            if chunk.get("doc_type") and chunk["doc_type"] != "text":
                label_parts.append(chunk["doc_type"])
            label = (
                " -- ".join(label_parts)
                if label_parts
                else chunk.get("source", "unknown")
            )
            context_parts.append(f"[Source: {label}]\n{chunk['text']}")

        context = "\n\n---\n\n".join(context_parts)
        user_message = (
            f"Context from documents:\n\n{context}\n\n---\n\n"
            f"Question: {question_text}"
        )

        # Call the model with timing.
        start_time = time.time()
        try:
            response = client.messages.create(
                model=model_id,
                max_tokens=1024,
                system=system_prompt,
                messages=[{"role": "user", "content": user_message}],
            )
            answer = response.content[0].text
            input_tokens = response.usage.input_tokens
            output_tokens = response.usage.output_tokens
        except Exception as e:
            answer = f"ERROR: {e}"
            input_tokens = 0
            output_tokens = 0

        elapsed_ms = (time.time() - start_time) * 1000

        # Score with keyword matching (objective, model-independent).
        from eval import score_keyword, score_edge_case

        if category == "edge_case":
            score_result = score_edge_case(answer)
            keyword_score = score_result["score"]
        else:
            score_result = score_keyword(answer, expected_keywords)
            keyword_score = score_result["score"]

        results.append({
            "question_id": question_id,
            "question": question_text,
            "category": category,
            "model_id": model_id,
            "answer": answer,
            "keyword_score": keyword_score,
            "matched_keywords": score_result.get("matched", []),
            "missed_keywords": score_result.get("missed", []),
            "latency_ms": round(elapsed_ms, 1),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        })

        # Progress indicator.
        print(f"    [{question_id}] score={keyword_score:.2f} "
              f"latency={elapsed_ms:.0f}ms "
              f"tokens={input_tokens}+{output_tokens}")

    return results


# =====================================================================
# SECTION 3: COMPARISON RUNNER
# =====================================================================

def run_comparison(
    questions: list[dict],
    collection,
    known_tables: list[str],
    models: list[dict] = None,
    top_k: int = 3,
) -> dict:
    """
    Run the full A/B comparison across all specified models.

    RETURNS
    -------
    dict mapping model_id -> {results, summary}
    """
    if models is None:
        models = MODELS_TO_COMPARE

    all_results = {}

    for model_config in models:
        model_id = model_config["id"]
        model_name = model_config["name"]

        print(f"\n{'='*60}")
        print(f"Running eval: {model_name} ({model_id})")
        print(f"{'='*60}")

        results = run_model_eval(
            model_id=model_id,
            questions=questions,
            collection=collection,
            known_tables=known_tables,
            top_k=top_k,
        )

        # Compute summary statistics.
        scores = [r["keyword_score"] for r in results]
        latencies = [r["latency_ms"] for r in results]
        total_input = sum(r["input_tokens"] for r in results)
        total_output = sum(r["output_tokens"] for r in results)

        # Score by category.
        category_scores = {}
        for r in results:
            cat = r["category"]
            if cat not in category_scores:
                category_scores[cat] = []
            category_scores[cat].append(r["keyword_score"])

        category_avgs = {
            cat: round(sum(s) / len(s), 4)
            for cat, s in category_scores.items()
        }

        summary = {
            "model_name": model_name,
            "model_id": model_id,
            "avg_score": round(sum(scores) / len(scores), 4) if scores else 0,
            "avg_latency_ms": round(sum(latencies) / len(latencies), 1),
            "total_input_tokens": total_input,
            "total_output_tokens": total_output,
            "estimated_cost": round(
                model_config["estimated_cost_per_query"] * len(questions), 4
            ),
            "category_scores": category_avgs,
            "num_questions": len(questions),
        }

        all_results[model_id] = {
            "results": results,
            "summary": summary,
        }

    return all_results


# =====================================================================
# SECTION 4: COMPARISON OUTPUT
# =====================================================================

def print_comparison(all_results: dict) -> None:
    """Print a formatted comparison report."""
    print("\n" + "=" * 70)
    print("MODEL COMPARISON RESULTS")
    print("=" * 70)

    # Overall summary table.
    print(f"\n{'Model':<25} {'Avg Score':>10} {'Avg Latency':>12} {'Est. Cost':>10}")
    print("-" * 60)

    for model_id, data in all_results.items():
        s = data["summary"]
        print(
            f"{s['model_name']:<25} "
            f"{s['avg_score']:>10.4f} "
            f"{s['avg_latency_ms']:>10.1f}ms "
            f"${s['estimated_cost']:>8.4f}"
        )

    # Category breakdown.
    print(f"\n--- Score by Category ---")
    categories = set()
    for data in all_results.values():
        categories.update(data["summary"]["category_scores"].keys())

    header = f"{'Category':<15}"
    for data in all_results.values():
        header += f" {data['summary']['model_name']:>15}"
    print(header)
    print("-" * (15 + 16 * len(all_results)))

    for cat in sorted(categories):
        row = f"{cat:<15}"
        for data in all_results.values():
            score = data["summary"]["category_scores"].get(cat, 0)
            row += f" {score:>15.4f}"
        print(row)

    # Per-question delta (show where models diverge).
    model_ids = list(all_results.keys())
    if len(model_ids) >= 2:
        m1_id, m2_id = model_ids[0], model_ids[1]
        m1_name = all_results[m1_id]["summary"]["model_name"]
        m2_name = all_results[m2_id]["summary"]["model_name"]
        r1 = all_results[m1_id]["results"]
        r2 = all_results[m2_id]["results"]

        print(f"\n--- Per-Question Comparison: {m1_name} vs {m2_name} ---")
        print(f"{'QID':<6} {'Category':<15} {m1_name:>12} {m2_name:>12} {'Delta':>8}")
        print("-" * 58)

        divergent_count = 0
        for q1, q2 in zip(r1, r2):
            s1 = q1["keyword_score"]
            s2 = q2["keyword_score"]
            delta = s1 - s2
            marker = ""
            if abs(delta) > 0.1:
                divergent_count += 1
                marker = " <<<"
            print(
                f"{q1['question_id']:<6} "
                f"{q1['category']:<15} "
                f"{s1:>12.2f} "
                f"{s2:>12.2f} "
                f"{delta:>+8.2f}{marker}"
            )

        print(f"\nDivergent questions (delta > 0.1): {divergent_count}/{len(r1)}")

    # Recommendation.
    print(f"\n--- RECOMMENDATION ---")
    best_id = max(all_results, key=lambda k: all_results[k]["summary"]["avg_score"])
    cheapest_id = min(all_results, key=lambda k: all_results[k]["summary"]["estimated_cost"])

    best = all_results[best_id]["summary"]
    cheapest = all_results[cheapest_id]["summary"]

    if best_id == cheapest_id:
        print(f"  {best['model_name']} wins on BOTH quality and cost.")
    else:
        score_gap = best["avg_score"] - cheapest["avg_score"]
        cost_savings = best["estimated_cost"] - cheapest["estimated_cost"]
        print(f"  Best quality: {best['model_name']} (score={best['avg_score']:.4f})")
        print(f"  Cheapest:     {cheapest['model_name']} (cost=${cheapest['estimated_cost']:.4f})")
        print(f"  Quality gap:  {score_gap:+.4f} ({score_gap/max(best['avg_score'],0.01)*100:+.1f}%)")
        print(f"  Cost savings: ${abs(cost_savings):.4f} per eval run")

        if score_gap < 0.05:
            print(f"\n  VERDICT: Quality gap is small (<5%). Use {cheapest['model_name']} "
                  f"for simple queries to save cost.")
        elif score_gap < 0.15:
            print(f"\n  VERDICT: Moderate quality gap. Route simple queries to "
                  f"{cheapest['model_name']}, complex queries to {best['model_name']}.")
        else:
            print(f"\n  VERDICT: Significant quality gap. Use {best['model_name']} "
                  f"for all queries until retrieval quality improves.")


def save_comparison(all_results: dict, output_dir: str = "eval_results") -> str:
    """Save comparison results to a CSV file."""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(output_dir, f"model_comparison_{timestamp}.csv")

    all_rows = []
    for model_id, data in all_results.items():
        for r in data["results"]:
            all_rows.append(r)

    if not all_rows:
        print("No results to save.")
        return ""

    fieldnames = list(all_rows[0].keys())
    # Convert lists to strings for CSV
    for row in all_rows:
        for key in ["matched_keywords", "missed_keywords"]:
            if key in row and isinstance(row[key], list):
                row[key] = "; ".join(row[key])

    with open(filepath, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\nResults saved to: {filepath}")
    return filepath


# =====================================================================
# SECTION 5: STANDALONE TEST
# =====================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("MODEL COMPARISON -- STANDALONE TEST")
    print("=" * 60)

    # Load .env for API key
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if not os.path.exists(env_path):
        env_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"
        )
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, _, value = line.partition("=")
                    os.environ[key.strip()] = value.strip()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("\nANTHROPIC_API_KEY not set. Cannot run model comparison.")
        print("Set it in your .env file and re-run.")
        exit(1)

    # Build the RAG pipeline (same as eval.py).
    from rag import load_documents, chunk_text, build_vector_store, DOCS_DIR
    from eval import load_eval_questions

    print("\nBuilding RAG pipeline...")
    documents = load_documents(DOCS_DIR)
    if not documents:
        print(f"No documents in {DOCS_DIR}/. Add STTM files to run comparison.")
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

    # Load eval questions.
    questions = load_eval_questions("eval_questions.csv")
    if not questions:
        print("No eval questions found. Check eval_questions.csv.")
        exit(1)

    # Run comparison.
    print(f"\nComparing {len(MODELS_TO_COMPARE)} models on {len(questions)} questions...")
    print("This will cost approximately $0.16 total.")

    all_results = run_comparison(
        questions=questions,
        collection=collection,
        known_tables=known_tables,
    )

    # Print and save results.
    print_comparison(all_results)
    save_comparison(all_results)

    print("\nComparison complete.")
    print("Use the category breakdown to decide your routing strategy.")