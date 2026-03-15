"""
experiment_runner.py — Automated RAG Quality Experiments
========================================================
Runs your eval.py with different configurations automatically,
then compares all results to find the optimal settings.

This is Week 2's core tool: the scientific method applied to RAG.

WHAT THIS DOES
──────────────
Instead of manually editing rag.py, running eval.py, editing again,
running again, and comparing... this script does it all in one go:

  For each configuration (e.g., CHUNK_SIZE = 200, 500, 800, 1000, 1500):
      1. Override the setting
      2. Rebuild the entire pipeline (re-chunk, re-embed, re-store)
      3. Run all 20 eval questions
      4. Save results with a descriptive tag
      5. Print a comparison table at the end

HOW TO RUN
──────────
  uv run python experiment_runner.py --experiment chunk_size
  uv run python experiment_runner.py --experiment top_k
  uv run python experiment_runner.py --experiment cross_entity_fix
  uv run python experiment_runner.py --experiment all

PYTHON REFRESHER: if __name__ == "__main__"
──────────────────────────────────────────
When you run `python experiment_runner.py`, Python sets __name__ to "__main__".
When another file does `import experiment_runner`, __name__ is "experiment_runner".
This pattern ensures main() only runs when the script is executed directly,
not when it's imported as a library.
"""

import os
import csv
import time
import glob
import argparse
from datetime import datetime


# ═══════════════════════════════════════════════════════════════
# SECTION 1: IMPORTS FROM YOUR EXISTING PIPELINE
# ═══════════════════════════════════════════════════════════════
# We import the building blocks from rag.py — the same functions
# your chatbot uses. This ensures experiments test the REAL code.
#
# PYTHON REFRESHER: Importing specific names vs modules
# ─────────────────────────────────────────────────────
# from rag import load_documents   → brings load_documents into our namespace
# import rag                        → brings the whole module, access as rag.load_documents
#
# We use "from X import Y" because we want to call these directly
# without the "rag." prefix everywhere.

from rag import (
    load_documents,
    chunk_text,
    build_vector_store,
    retrieve,
    ask_claude,
    extract_table_name,
    DOCS_DIR,
    TOP_K,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)

from eval import (
    load_eval_questions,
    score_keyword,
    score_edge_case,
    run_evaluation,
    save_results,
    print_summary,
    compare_runs,
)


# ═══════════════════════════════════════════════════════════════
# SECTION 2: CONFIGURATION — WHAT WE'RE TESTING
# ═══════════════════════════════════════════════════════════════
#
# Each experiment defines a list of values to try.
# We test ONE variable at a time (ablation study).
#
# WHY THESE SPECIFIC VALUES?
# ──────────────────────────
# CHUNK_SIZE values:
#   200  → Very small. Tests if your STTM summaries are short enough
#          that small chunks capture complete facts. Good for simple lookups,
#          bad for context-heavy questions.
#   500  → Small-medium. Each chunk is about 2-3 paragraphs. Might capture
#          a full table summary but not the columns too.
#   800  → Your current default. A reasonable starting point.
#   1000 → Medium-large. Should capture most table summaries + some columns.
#   1500 → Large. Might include an entire table summary + column mapping
#          in one chunk, but embeddings become very diluted.
#
# TOP_K values:
#   1 → Minimum. Only the single best chunk. High precision if retrieval
#       is perfect, total failure if retrieval makes a mistake.
#   2 → One backup chunk. Slightly more resilient.
#   3 → Your current default. Good balance.
#   5 → More context. Risk of noise from irrelevant chunks.
#   7 → Lots of context. Likely includes chunks from wrong tables.

CHUNK_SIZE_VALUES = [200, 500, 800, 1000, 1500]
TOP_K_VALUES = [1, 2, 3, 5, 7]


# ═══════════════════════════════════════════════════════════════
# SECTION 3: PIPELINE BUILDER
# ═══════════════════════════════════════════════════════════════
#
# This function rebuilds the entire RAG pipeline with a given
# chunk_size. It's the equivalent of:
#   1. Delete the old ChromaDB
#   2. Re-chunk all documents with new chunk_size
#   3. Re-embed and store in a fresh ChromaDB
#
# PYTHON REFRESHER: Default parameter values
# ──────────────────────────────────────────
# def func(x: int = 800) means "if caller doesn't provide x, use 800"
# This lets us call:
#   build_pipeline()          → uses chunk_size=800
#   build_pipeline(500)       → uses chunk_size=500

def build_pipeline(chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
    """
    Build a fresh RAG pipeline from scratch with the given chunk_size.

    WHY REBUILD FROM SCRATCH?
    ─────────────────────────
    When you change chunk_size, EVERYTHING downstream changes:
    - Different chunk boundaries → different text fragments
    - Different text fragments → different embeddings (vectors)
    - Different embeddings → different retrieval results
    - Different retrieval results → different answers

    You CAN'T just change chunk_size and keep the old ChromaDB.
    The old embeddings were computed from old chunks — they'd be
    searching with mismatched vectors.

    ANALOGY:
    It's like changing the page size of a book and expecting the
    old table of contents to still work. The chapters moved!

    RETURNS
    ───────
    tuple of (collection, known_tables, num_chunks, num_docs)
    
    PYTHON REFRESHER: Returning multiple values (tuples)
    ────────────────────────────────────────────────────
    Python lets you return multiple values separated by commas:
        return a, b, c
    The caller unpacks them:
        x, y, z = build_pipeline(500)
    Under the hood, Python packs them into a tuple: (a, b, c)
    """

    # Step 1: Load raw documents (same regardless of chunk_size)
    # ─────────────────────────────────────────────────────────
    # This reads your STTM Excel files and .txt/.md files.
    # The content doesn't change — only how we SPLIT it changes.
    documents = load_documents(DOCS_DIR)
    if not documents:
        print(f"  ERROR: No documents found in {DOCS_DIR}/")
        return None, None, 0, 0

    # Step 2: Chunk with the EXPERIMENTAL chunk_size
    # ─────────────────────────────────────────────────────────
    # THIS is where the experiment variable takes effect.
    # chunk_text() is called with our experimental value instead
    # of the default from rag.py's CHUNK_SIZE constant.
    #
    # PYTHON REFRESHER: Passing keyword arguments
    # ────────────────────────────────────────────
    # chunk_text(doc["content"], doc["source"], chunk_size=500)
    # The chunk_size=500 OVERRIDES the default parameter value
    # defined in chunk_text's function signature.
    all_chunks = []
    for doc in documents:
        chunks = chunk_text(
            doc["content"],
            doc["source"],
            chunk_size=chunk_size,    # ← THE EXPERIMENTAL VARIABLE
            overlap=overlap,
        )
        all_chunks.extend(chunks)

    # Step 3: Build fresh vector store
    # ─────────────────────────────────────────────────────────
    # This creates a NEW ChromaDB collection, embeds all chunks,
    # and stores them. The old collection is deleted first
    # (that happens inside build_vector_store).
    collection = build_vector_store(all_chunks)

    # Step 4: Extract known table names (for extract_table_name)
    # ─────────────────────────────────────────────────────────
    # PYTHON REFRESHER: set() for unique values
    # ────────────────────────────────────────────
    # set() removes duplicates:
    #   list(set(["a", "b", "a"])) → ["a", "b"]
    # sorted() puts them in alphabetical order for consistent output.
    all_meta = collection.get()
    known_tables = sorted(list(set(
        m.get("table_name", "")
        for m in all_meta["metadatas"]
        if m.get("table_name") and m["table_name"].strip()
    )))

    return collection, known_tables, len(all_chunks), len(documents)


# ═══════════════════════════════════════════════════════════════
# SECTION 4: CHUNK_SIZE EXPERIMENT
# ═══════════════════════════════════════════════════════════════
#
# THE EXPERIMENT DESIGN
# ─────────────────────
# Independent variable: CHUNK_SIZE (what we change)
# Dependent variable:   eval scores (what we measure)
# Controls:             TOP_K=3, same system prompt, same eval questions
#
# PYTHON REFRESHER: Type hints for complex types
# ──────────────────────────────────────────────
# results_by_size: dict[int, list[dict]]
# This means: "a dictionary where keys are integers (chunk sizes)
# and values are lists of dictionaries (eval result rows)"
# Example: {200: [{question_id: "Q01", score: 0.8, ...}, ...], 500: [...]}

def run_chunk_size_experiment(
    questions: list[dict],
    chunk_sizes: list[int] = None,
    top_k: int = TOP_K,
) -> dict[int, list[dict]]:
    """
    Run the full eval for each chunk_size, keeping TOP_K constant.

    THE EXPERIMENT LOOP
    ───────────────────
    For each chunk_size in [200, 500, 800, 1000, 1500]:
        1. Print what we're testing (so you can follow progress)
        2. Rebuild the pipeline with this chunk_size
        3. Run all eval questions through it
        4. Save results to CSV with a descriptive tag
        5. Store results in memory for comparison

    PARAMETERS
    ──────────
    questions:    Your 20 eval questions (loaded from CSV)
    chunk_sizes:  List of sizes to test (defaults to CHUNK_SIZE_VALUES)
    top_k:        Keep constant during this experiment (default=3)

    RETURNS
    ───────
    dict mapping each chunk_size → its list of eval results
    Example: {200: [results...], 500: [results...], ...}

    GOTCHA: This function is EXPENSIVE
    ───────────────────────────────────
    Each iteration:
      - Rebuilds ChromaDB (re-embeds all chunks)
      - Calls Claude API for each of the 20 questions
    With 5 chunk sizes × 20 questions = 100 API calls minimum.
    Budget ~$1-2 for keyword-only scoring, ~$2-4 with LLM judge.
    Runtime: ~15-25 minutes total.
    """
    if chunk_sizes is None:
        chunk_sizes = CHUNK_SIZE_VALUES

    results_by_size = {}
    result_files = []

    print("\n" + "=" * 60)
    print("EXPERIMENT: CHUNK_SIZE")
    print("=" * 60)
    print(f"  Testing: {chunk_sizes}")
    print(f"  Constant: TOP_K={top_k}")
    print(f"  Questions: {len(questions)}")
    print(f"  Estimated API calls: {len(chunk_sizes) * len(questions)}")
    print()

    for i, size in enumerate(chunk_sizes):
        print(f"\n{'─' * 60}")
        print(f"  RUN {i + 1}/{len(chunk_sizes)}: CHUNK_SIZE = {size}")
        print(f"{'─' * 60}")

        # Step 1: Build pipeline with this chunk size
        # ─────────────────────────────────────────────
        # This completely rebuilds ChromaDB. Takes 5-15 seconds
        # depending on document count and chunk size.
        start_time = time.time()
        collection, known_tables, num_chunks, num_docs = build_pipeline(
            chunk_size=size
        )
        build_time = time.time() - start_time

        if collection is None:
            print(f"  SKIPPED: Pipeline build failed for chunk_size={size}")
            continue

        print(f"  Pipeline: {num_chunks} chunks from {num_docs} docs "
              f"(built in {build_time:.1f}s)")

        # Step 2: Run evaluation
        # ─────────────────────────────────────────────
        # This calls the SAME run_evaluation from eval.py,
        # using the freshly-built collection.
        results = run_evaluation(
            questions=questions,
            collection=collection,
            known_tables=known_tables,
            use_llm_judge=False,     # Keyword only (faster + cheaper)
            top_k=top_k,
        )

        # Step 3: Save results
        # ─────────────────────────────────────────────
        # The tag includes both the chunk_size and "experiment"
        # so you can distinguish experiment runs from manual runs.
        tag = f"chunk{size}"
        filepath = save_results(results, tag=tag)
        result_files.append((size, filepath))

        # Step 4: Print quick summary for this run
        # ─────────────────────────────────────────────
        keyword_scores = [r["keyword_score"] for r in results]
        avg = sum(keyword_scores) / len(keyword_scores) if keyword_scores else 0
        print(f"  → Average keyword score: {avg:.3f}")

        # Store for comparison
        results_by_size[size] = results

    # ═══════════════════════════════════════════════════════════
    # COMPARISON TABLE
    # ═══════════════════════════════════════════════════════════
    # After ALL runs complete, print a side-by-side comparison.
    #
    # PYTHON REFRESHER: f-string alignment
    # ─────────────────────────────────────
    # f"{'text':<20}" → left-align in 20 chars:  "text                "
    # f"{value:>8.3f}" → right-align in 8 chars:  "   0.400"
    # The : separates the variable from the format spec.

    print("\n\n" + "=" * 60)
    print("CHUNK_SIZE EXPERIMENT — COMPARISON")
    print("=" * 60)

    # Header row
    # ──────────
    header = f"  {'Metric':<25}"
    for size in sorted(results_by_size.keys()):
        header += f" {'CS=' + str(size):>10}"
    print(header)
    print("  " + "─" * (25 + 11 * len(results_by_size)))

    # Overall average
    # ──────────────
    row = f"  {'Overall avg':.<25}"
    for size in sorted(results_by_size.keys()):
        scores = [r["keyword_score"] for r in results_by_size[size]]
        avg = sum(scores) / len(scores) if scores else 0
        row += f" {avg:>10.3f}"
    print(row)

    # Per-category averages
    # ────────────────────
    categories = ["simple_lookup", "cross_entity", "edge_case"]
    for cat in categories:
        row = f"  {cat:.<25}"
        for size in sorted(results_by_size.keys()):
            cat_results = [r for r in results_by_size[size] if r["category"] == cat]
            if cat == "edge_case":
                # Edge cases use their own scoring
                cat_scores = [
                    r["edge_case_score"] for r in cat_results
                    if r["edge_case_score"] != ""
                ]
            else:
                cat_scores = [r["keyword_score"] for r in cat_results]
            avg = sum(cat_scores) / len(cat_scores) if cat_scores else 0
            row += f" {avg:>10.3f}"
        print(row)

    # Chunk count (how many chunks were created)
    # ──────────────────────────────────────────
    row = f"  {'Num chunks':.<25}"
    for size in sorted(results_by_size.keys()):
        # All results in a run have the same chunk count
        # We stored it during pipeline build
        num = len(results_by_size[size][0].get("retrieved_sources", "[]")) if results_by_size[size] else 0
        # Actually, let's count chunks by rebuilding (we already have them)
        row += f" {'—':>10}"
    print(row)

    # Best performer
    # ─────────────
    best_size = max(
        results_by_size.keys(),
        key=lambda s: sum(r["keyword_score"] for r in results_by_size[s]) / len(results_by_size[s])
    )
    print(f"\n  ★ BEST CHUNK_SIZE: {best_size}")
    print(f"    Recommendation: Set CHUNK_SIZE = {best_size} in rag.py")

    # List saved files
    # ───────────────
    print(f"\n  Saved result files:")
    for size, filepath in result_files:
        print(f"    chunk_size={size}: {filepath}")

    print(f"\n  To compare any two runs:")
    if len(result_files) >= 2:
        print(f"    uv run python eval.py --compare {result_files[0][1]} {result_files[-1][1]}")

    return results_by_size


# ═══════════════════════════════════════════════════════════════
# SECTION 5: TOP_K EXPERIMENT
# ═══════════════════════════════════════════════════════════════
#
# SAME PATTERN as chunk_size experiment, but varying TOP_K instead.
#
# KEY DIFFERENCE: We don't rebuild the pipeline for each TOP_K value!
# TOP_K only affects retrieval (how many chunks we pull from ChromaDB),
# not how the chunks are created or embedded.
#
# This means TOP_K experiments are MUCH faster:
# - Build pipeline ONCE
# - Run eval with different top_k values
# - Each run only costs API calls, not re-embedding

def run_top_k_experiment(
    questions: list[dict],
    top_k_values: list[int] = None,
    chunk_size: int = CHUNK_SIZE,
) -> dict[int, list[dict]]:
    """
    Run the full eval for each TOP_K value, keeping chunk_size constant.

    WHY THIS IS FASTER THAN CHUNK_SIZE EXPERIMENTS
    ───────────────────────────────────────────────
    Chunk size experiments rebuild ChromaDB each time (expensive).
    TOP_K experiments reuse the SAME ChromaDB — they just change
    how many results we pull from it.

    Think of it like a library:
    - Chunk size experiment = reorganizing all the bookshelves
    - TOP_K experiment = deciding how many books to check out

    PARAMETERS
    ──────────
    questions:     Your 20 eval questions
    top_k_values:  List of TOP_K values to try (default: [1, 2, 3, 5, 7])
    chunk_size:    Keep constant (use your current default or best from chunk experiment)
    """
    if top_k_values is None:
        top_k_values = TOP_K_VALUES

    results_by_k = {}
    result_files = []

    print("\n" + "=" * 60)
    print("EXPERIMENT: TOP_K")
    print("=" * 60)
    print(f"  Testing: {top_k_values}")
    print(f"  Constant: CHUNK_SIZE={chunk_size}")
    print(f"  Questions: {len(questions)}")
    print()

    # Build pipeline ONCE (same for all TOP_K values)
    # ─────────────────────────────────────────────────
    print("Building pipeline once (reused for all TOP_K values)...")
    collection, known_tables, num_chunks, num_docs = build_pipeline(
        chunk_size=chunk_size
    )
    if collection is None:
        print("  ERROR: Pipeline build failed")
        return {}
    print(f"  Pipeline: {num_chunks} chunks from {num_docs} docs\n")

    for i, k in enumerate(top_k_values):
        print(f"\n{'─' * 60}")
        print(f"  RUN {i + 1}/{len(top_k_values)}: TOP_K = {k}")
        print(f"{'─' * 60}")

        # Run evaluation with this TOP_K
        # ─────────────────────────────────────────────
        # The collection is the SAME — only top_k changes.
        # run_evaluation passes top_k to retrieve(), which
        # passes it to ChromaDB's query(n_results=top_k).
        results = run_evaluation(
            questions=questions,
            collection=collection,
            known_tables=known_tables,
            use_llm_judge=False,
            top_k=k,                    # ← THE EXPERIMENTAL VARIABLE
        )

        tag = f"topk{k}"
        filepath = save_results(results, tag=tag)
        result_files.append((k, filepath))

        keyword_scores = [r["keyword_score"] for r in results]
        avg = sum(keyword_scores) / len(keyword_scores) if keyword_scores else 0
        print(f"  → Average keyword score: {avg:.3f}")

        results_by_k[k] = results

    # Comparison table (same format as chunk_size experiment)
    # ─────────────────────────────────────────────────────
    print("\n\n" + "=" * 60)
    print("TOP_K EXPERIMENT — COMPARISON")
    print("=" * 60)

    header = f"  {'Metric':<25}"
    for k in sorted(results_by_k.keys()):
        header += f" {'K=' + str(k):>10}"
    print(header)
    print("  " + "─" * (25 + 11 * len(results_by_k)))

    row = f"  {'Overall avg':.<25}"
    for k in sorted(results_by_k.keys()):
        scores = [r["keyword_score"] for r in results_by_k[k]]
        avg = sum(scores) / len(scores) if scores else 0
        row += f" {avg:>10.3f}"
    print(row)

    categories = ["simple_lookup", "cross_entity", "edge_case"]
    for cat in categories:
        row = f"  {cat:.<25}"
        for k in sorted(results_by_k.keys()):
            cat_results = [r for r in results_by_k[k] if r["category"] == cat]
            if cat == "edge_case":
                cat_scores = [
                    r["edge_case_score"] for r in cat_results
                    if r["edge_case_score"] != ""
                ]
            else:
                cat_scores = [r["keyword_score"] for r in cat_results]
            avg = sum(cat_scores) / len(cat_scores) if cat_scores else 0
            row += f" {avg:>10.3f}"
        print(row)

    best_k = max(
        results_by_k.keys(),
        key=lambda k: sum(r["keyword_score"] for r in results_by_k[k]) / len(results_by_k[k])
    )
    print(f"\n  ★ BEST TOP_K: {best_k}")
    print(f"    Recommendation: Set TOP_K = {best_k} in rag.py")

    print(f"\n  Saved result files:")
    for k, filepath in result_files:
        print(f"    TOP_K={k}: {filepath}")

    return results_by_k


# ═══════════════════════════════════════════════════════════════
# SECTION 6: MAIN — EXPERIMENT ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════
#
# PYTHON REFRESHER: argparse for CLI arguments
# ─────────────────────────────────────────────
# argparse turns command-line flags into Python variables:
#   python experiment_runner.py --experiment chunk_size
# Inside the script:
#   args.experiment == "chunk_size"
#
# This is the same pattern used in eval.py for --tag, --category, etc.
# The argparse library handles:
#   - Parsing the command line
#   - Type checking (--experiment must be a valid choice)
#   - Generating --help text automatically

def main():
    parser = argparse.ArgumentParser(
        description="RAG Quality Experiments — Systematic tuning with your eval framework",
        # ─────────────────────────────────────────────────────
        # PYTHON REFRESHER: Raw strings (r"...")
        # ─────────────────────────────────────────────────────
        # r"..." treats backslashes as literal characters, not escape codes.
        # Useful for text with lots of special characters.
        # Without r: "\n" = newline. With r: r"\n" = literal backslash + n.
        # We're not using r-strings here, but good to know for regex patterns.
        epilog="""
Examples:
  python experiment_runner.py --experiment chunk_size
  python experiment_runner.py --experiment top_k
  python experiment_runner.py --experiment all
  python experiment_runner.py --experiment chunk_size --chunk-sizes 400 600 800
  python experiment_runner.py --experiment top_k --top-k-values 2 3 5
        """,
        # ─────────────────────────────────────────────────────
        # PYTHON REFRESHER: RawDescriptionHelpFormatter
        # ─────────────────────────────────────────────────────
        # By default, argparse reformats your help text (wraps lines,
        # removes whitespace). RawDescriptionHelpFormatter preserves
        # your formatting exactly as written. Good for examples.
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--experiment",
        choices=["chunk_size", "top_k", "all"],
        required=True,
        help="Which experiment to run",
    )

    # Optional overrides for experiment values
    # ─────────────────────────────────────────
    # PYTHON REFRESHER: nargs="+" means "one or more values"
    # ─────────────────────────────────────────────────────
    # --chunk-sizes 200 500 800 → args.chunk_sizes = [200, 500, 800]
    # nargs="+" requires at least one value.
    # nargs="*" allows zero values.
    # nargs=2 requires exactly two values (used in eval.py's --compare).
    parser.add_argument(
        "--chunk-sizes",
        type=int,
        nargs="+",
        default=None,
        help=f"Override chunk sizes to test (default: {CHUNK_SIZE_VALUES})",
    )
    parser.add_argument(
        "--top-k-values",
        type=int,
        nargs="+",
        default=None,
        help=f"Override TOP_K values to test (default: {TOP_K_VALUES})",
    )
    parser.add_argument(
        "--questions",
        default="eval_questions.csv",
        help="Path to eval questions CSV file",
    )

    args = parser.parse_args()

    # ─── Load environment (same pattern as rag.py and eval.py) ───
    # ─────────────────────────────────────────────────────────────
    # PYTHON REFRESHER: DRY principle (Don't Repeat Yourself)
    # ─────────────────────────────────────────────────────────
    # This .env loading code appears in rag.py, eval.py, and now here.
    # In production, you'd extract it into a shared utility function.
    # For learning, having it visible in each file is actually helpful
    # because you see the pattern repeated and internalize it.
    if not os.environ.get("ANTHROPIC_API_KEY"):
        env_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), ".env"
        )
        if os.path.exists(env_path):
            with open(env_path) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        key, _, value = line.partition("=")
                        os.environ[key.strip()] = value.strip()

    # ─── Load eval questions ───
    print("EXPERIMENT RUNNER")
    print("=" * 60)
    questions = load_eval_questions(args.questions)
    if not questions:
        print("No questions found! Check your CSV file path.")
        return

    # ─── Run the requested experiment ───
    # ─────────────────────────────────────────────────────────
    # PYTHON REFRESHER: Early returns vs nested if/elif
    # ─────────────────────────────────────────────────────────
    # Some people write:
    #   if experiment == "chunk_size":
    #       ...
    #   elif experiment == "top_k":
    #       ...
    # We use that pattern here because argparse ensures only valid
    # choices reach this point. No need for an "else: error" branch.

    experiment_start = time.time()

    if args.experiment in ("chunk_size", "all"):
        run_chunk_size_experiment(
            questions=questions,
            chunk_sizes=args.chunk_sizes,
        )

    if args.experiment in ("top_k", "all"):
        run_top_k_experiment(
            questions=questions,
            top_k_values=args.top_k_values,
        )

    total_time = time.time() - experiment_start

    # ─── Final summary ───
    print("\n" + "=" * 60)
    print("EXPERIMENT COMPLETE")
    print("=" * 60)
    print(f"  Total time: {total_time / 60:.1f} minutes")
    print(f"  Results saved in: eval_results/")
    print()
    print("  NEXT STEPS:")
    print("  1. Look at the comparison tables above")
    print("  2. Set the winning values in rag.py")
    print("  3. Run: uv run python eval.py --tag 'optimized'")
    print("  4. Compare: uv run python eval.py --compare eval_results/baseline.csv eval_results/optimized.csv")
    print()
    print("  THEN apply the cross-entity fix from week2_improvements.py")
    print("  and run eval again to measure its impact separately.")


if __name__ == "__main__":
    main()