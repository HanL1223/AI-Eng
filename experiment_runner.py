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


CHUNK_SIZE_VALUES = [200, 500, 800, 1000, 1500]
TOP_K_VALUES = [1, 2, 3, 5, 7]

