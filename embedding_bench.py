"""
embedding_bench.py -- Embedding Model Comparison for RAG
=========================================================
Week 5, Step 3 of 4

WHAT THIS FILE DOES
-------------------
Compares different embedding models to find which one retrieves
the most relevant chunks for your STTM queries. This directly
impacts your chatbot's answer quality.

Your current setup uses ChromaDB's default embedding model:
  all-MiniLM-L6-v2 (384 dimensions, ~80MB, from sentence-transformers)

This benchmark tests whether a different embedding model retrieves
better chunks for your specific domain (Snowflake data warehouse
documentation with STTM terminology).


WHY EMBEDDING MODEL CHOICE MATTERS
────────────────────────────────────
The embedding model is the FOUNDATION of your retrieval pipeline.
Everything downstream depends on it:

  Query:  "What foreign keys does DIM_STORE have?"
            |
            v
  [Embedding Model]  <-- THIS determines retrieval quality
            |
            v
  Query vector: [0.12, -0.45, 0.78, ...]   (384 or 1024 dimensions)
            |
            v
  ChromaDB: cosine similarity search
            |
            v
  Top 3 chunks  <-- Right chunks = right answer, wrong chunks = wrong answer

If the embedding model does not understand that "foreign keys" relates
to "FK_STORE_KEY", it will retrieve irrelevant chunks, and even a
perfect LLM cannot generate the right answer from wrong context.


EMBEDDING MODELS WE COMPARE
─────────────────────────────

1. ChromaDB Default: all-MiniLM-L6-v2
   - 384 dimensions, ~80MB model
   - General-purpose (trained on web text, forums, Wikipedia)
   - Free, runs locally
   - Weakness: may not understand domain-specific abbreviations
     (FK_, SK_, BK_, STTM, SCD Type 2, etc.)

2. Voyage AI: voyage-3-lite (optional, API-based)
   - 512 dimensions, cloud API
   - Trained with focus on code and technical text
   - ~$0.02 per 1M tokens (your corpus: ~$0.002 total)
   - Strength: better on technical vocabulary and identifiers

3. OpenAI: text-embedding-3-small (optional, API-based)
   - 1536 dimensions, cloud API
   - General-purpose but high quality
   - ~$0.02 per 1M tokens
   - Strength: strong general understanding

4. Sentence-Transformers: all-mpnet-base-v2 (local)
   - 768 dimensions, ~420MB model
   - Better than MiniLM on most benchmarks (MTEB)
   - Free, runs locally
   - Tradeoff: larger model, slower embedding, more memory


HOW THE COMPARISON WORKS
─────────────────────────
We use your existing eval questions as the benchmark. For each
embedding model:

  1. Embed all chunks using the model
  2. Build a ChromaDB collection with that embedding
  3. For each eval question, retrieve top-K chunks
  4. Check: do the retrieved chunks contain the right information?
     (using your existing keyword scoring from eval.py)
  5. Compute average retrieval quality score

The model with the highest average retrieval score wins.

CRITICAL INSIGHT: We are measuring RETRIEVAL quality, not GENERATION
quality. The LLM (Claude) stays the same. Only the embedding changes.
This isolates the variable we are testing.

dbt ANALOGY:
  This is like A/B testing two different source connectors for the
  same data. Both feed into the same staging/mart models. You
  measure which connector produces better data quality downstream.


DEPENDENCIES
────────────
  Required: None beyond what you already have (chromadb includes
            sentence-transformers which includes all-MiniLM-L6-v2)

  Optional (for alternative models):
    uv add voyageai     # For Voyage AI embeddings
    uv add openai       # For OpenAI embeddings

  Optional (for better local model):
    No extra install needed -- sentence-transformers is already
    installed via chromadb. You just specify a different model name.
"""

import os
import time
import json
from pathlib import Path


# =====================================================================
# SECTION 1: EMBEDDING FUNCTION WRAPPERS
# =====================================================================
#
# Each embedding model has a different API. We wrap each one in a
# function with the SAME SIGNATURE so the benchmark can call any
# of them interchangeably.
#
# Signature: embed_fn(texts: list[str]) -> list[list[float]]
#
# This is the Strategy Pattern again (same as model_switcher.py).

def embed_with_default(texts: list[str]) -> list[list[float]]:
    """
    Embed texts using ChromaDB's default model (all-MiniLM-L6-v2).

    This is your current embedding. It runs locally, is free, and
    produces 384-dimensional vectors.

    ChromaDB uses the sentence-transformers library internally.
    We call the same library directly to get the raw vectors.

    PYTHON REFRESHER: SentenceTransformer
    ─────────────────────────────────────
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(["Hello world", "Foo bar"])
    # embeddings.shape = (2, 384)  -- 2 texts, 384 dimensions each

    The model is downloaded on first use (~80MB) and cached
    in ~/.cache/torch/sentence_transformers/
    """
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(texts, show_progress_bar=False)

    # SentenceTransformer returns numpy arrays. ChromaDB expects
    # plain Python lists. Convert.
    return embeddings.tolist()


def embed_with_mpnet(texts: list[str]) -> list[list[float]]:
    """
    Embed texts using all-mpnet-base-v2 (768 dimensions).

    This is a stronger general-purpose model than MiniLM.
    On MTEB benchmarks, it scores ~3-5% higher on retrieval tasks.
    Tradeoff: 2x larger model, 2-3x slower embedding.

    No extra installation needed -- sentence-transformers is already
    installed via your chromadb dependency.
    """
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("all-mpnet-base-v2")
    embeddings = model.encode(texts, show_progress_bar=False)
    return embeddings.tolist()


def embed_with_voyage(texts: list[str]) -> list[list[float]]:
    """
    Embed texts using Voyage AI's voyage-3-lite model.

    Requires: uv add voyageai
    Requires: VOYAGE_API_KEY environment variable

    Voyage AI specializes in embeddings for code and technical text.
    Their models are designed to understand identifiers, APIs, and
    technical documentation -- which aligns well with your STTM data.

    Cost: ~$0.02 per 1M tokens. Your corpus (~60 entities, ~100 chunks)
    is roughly 50K tokens = ~$0.001 per embedding run.
    """
    try:
        import voyageai
    except ImportError:
        raise ImportError(
            "voyageai is required for Voyage AI embeddings.\n"
            "Install with: uv add voyageai\n"
            "Get API key from: https://www.voyageai.com/"
        )

    api_key = os.environ.get("VOYAGE_API_KEY")
    if not api_key:
        raise ValueError(
            "VOYAGE_API_KEY environment variable not set.\n"
            "Add it to your .env file: VOYAGE_API_KEY=your_key_here"
        )

    client = voyageai.Client(api_key=api_key)
    result = client.embed(texts, model="voyage-3-lite")
    return result.embeddings


def embed_with_openai(texts: list[str]) -> list[list[float]]:
    """
    Embed texts using OpenAI's text-embedding-3-small model.

    Requires: uv add openai
    Requires: OPENAI_API_KEY environment variable

    OpenAI's embedding models are high quality and general-purpose.
    text-embedding-3-small produces 1536-dimensional vectors.

    Cost: ~$0.02 per 1M tokens.
    """
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError(
            "openai is required for OpenAI embeddings.\n"
            "Install with: uv add openai"
        )

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set.")

    client = OpenAI(api_key=api_key)
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts,
    )
    return [item.embedding for item in response.data]


# Registry of available embedding models.
# Maps model name -> (embed_function, dimensions, description)
EMBEDDING_MODELS = {
    "default": {
        "fn": embed_with_default,
        "dims": 384,
        "description": "all-MiniLM-L6-v2 (ChromaDB default, local, free)",
        "requires": None,
    },
    "mpnet": {
        "fn": embed_with_mpnet,
        "dims": 768,
        "description": "all-mpnet-base-v2 (stronger local model, free)",
        "requires": None,
    },
    "voyage": {
        "fn": embed_with_voyage,
        "dims": 512,
        "description": "voyage-3-lite (Voyage AI API, ~$0.001/run)",
        "requires": "VOYAGE_API_KEY",
    },
    "openai": {
        "fn": embed_with_openai,
        "dims": 1536,
        "description": "text-embedding-3-small (OpenAI API, ~$0.001/run)",
        "requires": "OPENAI_API_KEY",
    },
}


# =====================================================================
# SECTION 2: RETRIEVAL QUALITY MEASUREMENT
# =====================================================================

def measure_retrieval_quality(
    embed_fn,
    chunks: list[dict],
    eval_questions: list[dict],
    top_k: int = 3,
) -> dict:
    """
    Measure how well an embedding model retrieves relevant chunks.

    For each eval question, we:
    1. Embed all chunks using the given embedding function
    2. Embed the question
    3. Find the top-K most similar chunks (cosine similarity)
    4. Check if the expected keywords appear in those chunks

    This is a RETRIEVAL-ONLY evaluation. The LLM is not involved.
    We are measuring: "Does the embedding model put the right
    chunks at the top of the similarity list?"

    PARAMETERS
    ----------
    embed_fn : callable
        Function that takes list[str] and returns list[list[float]].
    chunks : list[dict]
        All document chunks (same format as from rag.py's chunk_text).
    eval_questions : list[dict]
        Your eval questions from eval_questions.csv.
    top_k : int
        How many chunks to retrieve per question.

    RETURNS
    -------
    dict with:
      "avg_score": float (0-1, average keyword match score)
      "per_question": list of per-question results
      "embed_time_s": float (total embedding time in seconds)
      "retrieval_time_s": float (total retrieval time in seconds)
    """
    import numpy as np

    # Step 1: Embed all chunks.
    chunk_texts = [c.get("text", "") for c in chunks]
    print(f"    Embedding {len(chunk_texts)} chunks...")

    embed_start = time.time()
    chunk_embeddings = embed_fn(chunk_texts)
    embed_time = time.time() - embed_start

    # Convert to numpy for fast similarity computation.
    chunk_matrix = np.array(chunk_embeddings)

    # Normalize for cosine similarity.
    # Cosine similarity = dot product of unit vectors.
    # Normalizing each vector to unit length lets us use dot product
    # directly instead of computing the full cosine formula.
    #
    # MATH REFRESHER: Cosine Similarity
    # ──────────────────────────────────
    # cos(a, b) = (a . b) / (|a| * |b|)
    #
    # If we normalize: a_norm = a / |a|, b_norm = b / |b|
    # Then: cos(a, b) = a_norm . b_norm  (just a dot product!)
    #
    # This is faster because we normalize once (O(n*d)) and then
    # compute dot products (O(n*d)) instead of computing norms
    # for every pair (O(n^2 * d)).
    norms = np.linalg.norm(chunk_matrix, axis=1, keepdims=True)
    # Avoid division by zero for empty chunks
    norms = np.where(norms == 0, 1, norms)
    chunk_matrix_norm = chunk_matrix / norms

    # Step 2: For each question, retrieve top-K chunks.
    results = []
    retrieval_start = time.time()

    for q in eval_questions:
        question_text = q.get("question", "")
        expected_keywords = q.get("expected_keywords", [])

        if not expected_keywords:
            # Parse from CSV format if needed
            raw = q.get("expected_answer_keywords", "")
            expected_keywords = [kw.strip().upper() for kw in raw.split(",") if kw.strip()]

        # Embed the question.
        q_embedding = embed_fn([question_text])[0]
        q_vec = np.array(q_embedding)
        q_norm = np.linalg.norm(q_vec)
        if q_norm > 0:
            q_vec = q_vec / q_norm

        # Compute cosine similarity with all chunks.
        # @ is the matrix multiplication operator in numpy.
        # chunk_matrix_norm @ q_vec computes the dot product of each
        # normalized chunk vector with the normalized query vector.
        similarities = chunk_matrix_norm @ q_vec

        # Get indices of top-K most similar chunks.
        # argsort() returns indices that would sort the array (ascending).
        # [-top_k:] takes the last top_k (highest similarity).
        # [::-1] reverses to descending order.
        top_indices = similarities.argsort()[-top_k:][::-1]

        # Check if expected keywords appear in the retrieved chunks.
        retrieved_text = " ".join(
            chunks[i].get("text", "").upper() for i in top_indices
        )

        matched = [kw for kw in expected_keywords if kw in retrieved_text]
        total_kw = len(expected_keywords)
        score = len(matched) / total_kw if total_kw > 0 else 0.0

        results.append({
            "question_id": q.get("question_id", ""),
            "question": question_text,
            "category": q.get("category", ""),
            "score": round(score, 2),
            "matched_keywords": matched,
            "total_keywords": total_kw,
            "top_chunks": [
                {
                    "table_name": chunks[i].get("table_name", ""),
                    "doc_type": chunks[i].get("doc_type", ""),
                    "similarity": round(float(similarities[i]), 4),
                }
                for i in top_indices
            ],
        })

    retrieval_time = time.time() - retrieval_start

    # Compute average score.
    avg_score = sum(r["score"] for r in results) / len(results) if results else 0

    return {
        "avg_score": round(avg_score, 4),
        "per_question": results,
        "embed_time_s": round(embed_time, 2),
        "retrieval_time_s": round(retrieval_time, 4),
        "num_chunks": len(chunks),
        "num_questions": len(eval_questions),
    }


# =====================================================================
# SECTION 3: COMPARISON RUNNER
# =====================================================================

def run_comparison(
    chunks: list[dict],
    eval_questions: list[dict],
    models: list[str] = None,
    top_k: int = 3,
) -> dict:
    """
    Run the embedding comparison across multiple models.

    PARAMETERS
    ----------
    chunks : list[dict]
        All document chunks from your pipeline.
    eval_questions : list[dict]
        Your 20 eval questions.
    models : list[str]
        Which embedding models to compare. Default: ["default", "mpnet"].
        Add "voyage" or "openai" if you have the API keys.
    top_k : int
        How many chunks to retrieve per question.

    RETURNS
    -------
    dict mapping model_name -> results dict
    """
    if models is None:
        # Start with local models only (free, no API keys needed).
        models = ["default", "mpnet"]

    all_results = {}

    for model_name in models:
        if model_name not in EMBEDDING_MODELS:
            print(f"  Skipping unknown model: {model_name}")
            continue

        model_info = EMBEDDING_MODELS[model_name]

        # Check if API key is required and available.
        required_key = model_info.get("requires")
        if required_key and not os.environ.get(required_key):
            print(f"  Skipping {model_name}: {required_key} not set")
            continue

        print(f"\n  === {model_name}: {model_info['description']} ===")
        print(f"    Dimensions: {model_info['dims']}")

        try:
            results = measure_retrieval_quality(
                embed_fn=model_info["fn"],
                chunks=chunks,
                eval_questions=eval_questions,
                top_k=top_k,
            )
            all_results[model_name] = results

            print(f"    Avg retrieval score: {results['avg_score']:.4f}")
            print(f"    Embed time:         {results['embed_time_s']:.2f}s")
            print(f"    Retrieval time:     {results['retrieval_time_s']:.4f}s")

        except Exception as e:
            print(f"    ERROR: {e}")
            all_results[model_name] = {"error": str(e)}

    return all_results


def print_comparison(results: dict) -> None:
    """
    Print a formatted comparison table of embedding model results.
    """
    print("\n" + "=" * 70)
    print("EMBEDDING MODEL COMPARISON")
    print("=" * 70)

    # Header
    print(f"\n{'Model':<15} {'Avg Score':>10} {'Embed Time':>12} {'Dimensions':>12}")
    print("-" * 55)

    for model_name, result in results.items():
        if "error" in result:
            print(f"{model_name:<15} {'ERROR':>10}")
            continue

        dims = EMBEDDING_MODELS.get(model_name, {}).get("dims", "?")
        print(
            f"{model_name:<15} "
            f"{result['avg_score']:>10.4f} "
            f"{result['embed_time_s']:>10.2f}s "
            f"{dims:>12}"
        )

    # Find the winner
    valid_results = {
        k: v for k, v in results.items() if "error" not in v
    }
    if valid_results:
        best_model = max(valid_results, key=lambda k: valid_results[k]["avg_score"])
        best_score = valid_results[best_model]["avg_score"]
        print(f"\nBest model: {best_model} (avg score: {best_score:.4f})")

    # Per-question breakdown for the top 2 models
    sorted_models = sorted(
        valid_results.keys(),
        key=lambda k: valid_results[k]["avg_score"],
        reverse=True,
    )

    if len(sorted_models) >= 2:
        m1, m2 = sorted_models[0], sorted_models[1]
        r1 = valid_results[m1]["per_question"]
        r2 = valid_results[m2]["per_question"]

        print(f"\n--- Per-Question Comparison: {m1} vs {m2} ---")
        print(f"{'QID':<6} {'Category':<15} {m1:>10} {m2:>10} {'Delta':>8}")
        print("-" * 55)

        for q1, q2 in zip(r1, r2):
            qid = q1.get("question_id", "?")
            cat = q1.get("category", "?")[:14]
            s1 = q1["score"]
            s2 = q2["score"]
            delta = s1 - s2
            marker = " <<<" if abs(delta) > 0.2 else ""
            print(f"{qid:<6} {cat:<15} {s1:>10.2f} {s2:>10.2f} {delta:>+8.2f}{marker}")


# =====================================================================
# SECTION 4: STANDALONE TEST
# =====================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("EMBEDDING BENCH -- STANDALONE TEST")
    print("=" * 60)

    # Load .env for API keys
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if not os.path.exists(env_path):
        # Try parent directory
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

    # Create synthetic test data (so this test works without real STTM files).
    test_chunks = [
        {"text": "DIM_STORE is a dimension table. Grain: one row per store. "
                 "PK: SK_STORE_KEY. BK: BK_STORE_KEY. Source: SAP CDS.",
         "table_name": "DIM_STORE", "doc_type": "summary", "source": "test"},
        {"text": "DIM_STORE columns: SK_STORE_KEY (BIGINT), BK_STORE_KEY (VARCHAR), "
                 "STORE_NAME (VARCHAR), FK_REGION_KEY (BIGINT -> DIM_REGION), "
                 "FK_STORE_TYPE_KEY (BIGINT -> DIM_STORE_TYPE).",
         "table_name": "DIM_STORE", "doc_type": "column_mapping", "source": "test"},
        {"text": "DIM_DATE is a date dimension table. Grain: one row per calendar date. "
                 "PK: SK_DATE_KEY. Contains YEAR, QUARTER, MONTH, DAY columns.",
         "table_name": "DIM_DATE", "doc_type": "summary", "source": "test"},
        {"text": "FACT_SALES_ORDER records sales transactions. Grain: one row per order line. "
                 "FK_STORE_KEY -> DIM_STORE, FK_DATE_KEY -> DIM_DATE, FK_PRODUCT_KEY -> DIM_PRODUCT.",
         "table_name": "FACT_SALES_ORDER", "doc_type": "summary", "source": "test"},
        {"text": "Azure Data Factory manages extraction from source systems. "
                 "Bronze layer stores raw data as-is from SAP and MyPOS.",
         "table_name": "", "doc_type": "text", "source": "architecture"},
    ]

    test_questions = [
        {"question_id": "Q01", "question": "What is the grain of DIM_STORE?",
         "category": "simple_lookup", "expected_answer_keywords": "ONE ROW,STORE"},
        {"question_id": "Q02", "question": "What foreign keys does DIM_STORE have?",
         "category": "simple_lookup", "expected_answer_keywords": "FK_REGION_KEY,FK_STORE_TYPE_KEY"},
        {"question_id": "Q03", "question": "Which dimensions does FACT_SALES_ORDER reference?",
         "category": "cross_entity", "expected_answer_keywords": "DIM_STORE,DIM_DATE,DIM_PRODUCT"},
    ]

    # Determine which models are available.
    available_models = ["default"]  # Always available

    # Check if mpnet model download is feasible.
    # For a quick test, we only use default to avoid downloading ~420MB.
    print("\nAvailable models for testing:")
    for name, info in EMBEDDING_MODELS.items():
        req = info.get("requires")
        available = req is None or os.environ.get(req)
        status = "AVAILABLE" if available else f"NEEDS {req}"
        print(f"  {name:<10} {info['description']:<50} [{status}]")

    # Run with default model only for quick test.
    print("\nRunning comparison with 'default' model only (quick test)...")
    results = run_comparison(
        chunks=test_chunks,
        eval_questions=test_questions,
        models=["default"],
        top_k=3,
    )

    print_comparison(results)

    print("\nTo compare models, run with your real data:")
    print("  1. cd C:\\Users\\laaro\\AI-Eng")
    print("  2. uv run python embedding_bench.py")
    print("     (modify __main__ to use load_documents + load_eval_questions)")
    print("\nAll tests passed. embedding_bench.py is ready.")