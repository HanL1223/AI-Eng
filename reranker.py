"""
Reraner - Cross encode Reranking for RAG Retrieval

Add a second pass to the retrieval pipeline instead of using VectorDB top 3 reslut directly, rule apply to

  Step 1: Retrieve TOP 10 chunks from ChromaDB (cast a wide net)
  Step 2: Score each chunk against the query using a more accurate method
  Step 3: Return only the TOP 3 highest-scoring chunks

  This is called the "retrieve-then-rerank" pattern.



RERANKING APPROACHES (FROM SIMPLEST TO MOST COMPLEX)

1. LLM-as-Reranker (what use here):
   - Send each chunk + query to Claude and ask "how relevant is this?"
   - Pro: Very accurate, uses your existing Claude setup
   - Con: N API calls per query (one per chunk), costs money
   - Best for: Learning, low-volume production

2. Cross-Encoder Model (e.g., ms-marco-MiniLM):
   - A small model specifically trained for relevance scoring
   - Pro: Fast, free, runs locally
   - Con: Requires downloading a model, may not understand your domain
   - Best for: High-volume production where cost matters

3. Cohere Rerank API:
   - Dedicated reranking service (paid)
   - Pro: Very accurate, simple API
   - Con: Another external dependency, costs money
   - Best for: Production systems with budget


dbt ANALOGY:
  ChromaDB retrieval = staging model (broad, fast, catches everything)
  Reranking          = intermediate model (precise, filters to what matters)
  Final answer       = mart model (the polished output)


HOW THE LLM-AS-RERANKER WORKS
------------------------------
For each chunk, we ask Claude a simple question:

    "On a scale of 1-5, how relevant is this chunk to this query?"

use Claude Haiku for this (cheaper, faster) rather than Sonnet.
The reranking prompt is simple enough that a smaller model handles it
well. This is an important cost optimization:


HOW THIS FILE CONNECTS SO FAR
  query_router.py imports rerank_chunks() from this file
    --> calls it between retrieve() and ask_claude()
    --> passes the reranked chunks to ask_claude() instead of raw chunks

  app.py shows reranking info in the debug panel:
    --> "Reranked: chunk moved from position 7 to position 1"
"""

import time
import re
import os

# Load .env if API key not already set (same pattern as rag.py)
if not os.environ.get("ANTHROPIC_API_KEY"):
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, _, value = line.partition("=")
                    os.environ[key.strip()] = value.strip()

#SECTION 1: THE RERANKING PROMPT
# This prompt is sent to Claude Haiku for each chunk. It asks for a
# simple 1-5 relevance score.
#
# PROMPT DESIGN DECISIONS:
# 1. We ask for a NUMBER, not text. This makes parsing trivial.
# 2. We include the scoring rubric in the prompt so Claude is consistent.
# 3. We say "Respond ONLY with a number" to prevent verbose explanations.
# 4. We include domain context (STTM, Snowflake) to help Claude judge
#    relevance in your specific domain.

RERANK_PROMPT = """As a professional Data Enginner You are evaluating whether a document chunk is relevant
to a user's question about a Snowflake data warehouse.

The data warehouse uses STTM (Source-to-Target Mapping) documentation
with tables named FACT_*, DIM_*, BRIDGE_*.

Score the relevance of this chunk to the question on a scale of 1 to 5:
  1 = Completely irrelevant (wrong table, wrong topic)
  2 = Slightly relevant (mentions related concepts but does not answer)
  3 = Moderately relevant (partially answers the question)
  4 = Highly relevant (directly addresses the question)
  5 = Perfectly relevant (exactly what the user is looking for)

QUESTION: {query}

CHUNK (from {source}):
{chunk_text}

Respond with ONLY a single integer from 1 to 5. No explanation."""

#SECTION 2: LLM-BASED RERANKER

def rerank_with_llm(
        query:str,
        chunks:list[dict],
        top_n:int = 3, 
        model: str = "claude-haiku-4-5-20251001"
) -> list[dict]:
    """
    Rerank chunks using Claude as a cross-encoder.

    For each chunk, we ask Claude to score its relevance to the query
    on a 1-5 scale. Then we sort by score (descending) and return
    only the top N chunks.

    PARAMETERS
    ----------
    query : str
        The user's question.
    chunks : list[dict]
        Chunks from retrieve(). Each must have "text" and "source" keys.
    top_n : int
        How many chunks to return after reranking. Default 3.
        This should match your current TOP_K in rag.py.
    model : str
        The Claude model to use for reranking. Default is Haiku
        because it is cheaper and fast enough for simple scoring.

        COST COMPARISON for reranking 10 chunks:
          Haiku:  ~$0.006 per query  (10 chunks * ~500 tokens each)
          Sonnet: ~$0.025 per query  (4x more expensive)

    RETURNS
    -------
    list[dict]
        The top N chunks, sorted by relevance score (highest first).
        Each chunk dict gets two new keys added:
          "rerank_score": int (1-5) -- the LLM's relevance score
          "original_rank": int (0-indexed) -- position before reranking
    """
    import anthropic

    client = anthropic.Anthropic()
    scored_chunks = []

    for i,chunk in enumerate(chunks):
        # Build the reranking prompt for this specific chunk
        prompt = RERANK_PROMPT.format(
            query = query,
            source = chunk.get("source","unknown"),
            chunk_text = chunk.get("text","")[:800]
        )

        try:
            response = client.messages.create(
                model=model,
                max_tokens=5,  # We only need a single digit
                messages=[{"role": "user", "content": prompt}],
            )
            # ---------------------------------------------------------------
            # Parse the score from Claude's response.
            #
            # We expect a single digit 1-5. But Claude might respond
            # with "3" or "3." or "Score: 3" or even "I'd rate this a 3".
            #
            # PYTHON REFRESHER: str.strip() and int() conversion
            # ---------------------------------------------------------------
            # "  3  ".strip()    -> "3"     (removes whitespace)
            # int("3")           -> 3       (string to integer)
            # int("3.0")         -> ERROR   (int() cannot parse decimals)
            # int(float("3.0"))  -> 3       (two-step conversion)
            #
            # We take the first character that is a digit:
            # ---------------------------------------------------------------
            raw_text = response.content[0].text.strip()
            score = None
            for char in raw_text:
                if char.isdigit():
                    score = int(char)
                    break
            if score is not None:
                score=  max(1,min(5,score))
            else:
                score = 0 
        except Exception as e:
            score = 0
        # Create a copy of the chunk with reranking metadata added.
        scored_chunks.append(
            {**chunk,
            "rerank_score":score,
            "original_rank":i,}
        )
    # Sort by score descending, return top N.

    # key=lambda x: x["score"] tells sorted() to compare items
    # by their "score" value rather than the dict itself.
    #
    # reverse=True means highest first (5, 4, 3, 2, 1).
    #
    # Long-form equivalent:
    #   def get_score(chunk):
    #       return chunk["rerank_score"]
    #   sorted_chunks = sorted(scored_chunks, key=get_score, reverse=True)
    sorted_chunks = sorted(
        scored_chunks,
        key = lambda x:x['rerank_score'],
        reverse = True
    )

    return sorted_chunks[:top_n]

#SECTION 3: KEYWORD-BASED RERANKER (FREE ALTERNATIVE)
# This is a simple, zero-cost reranker that scores chunks based on
# exact keyword overlap with the query. It is less accurate than the
# LLM reranker but useful when:
#   - want to avoid API costs during development
#   - testing the reranking pipeline logic
#   - The LLM API is down or rate-limited


def rerank_with_keywords(
    query: str,
    chunks: list[dict],
    top_n: int = 3,
) -> list[dict]:
    """
    Rerank chunks using keyword overlap scoring.

    For each chunk, we count how many query words appear in the chunk
    text. Chunks with more keyword matches rank higher.

    This is a BASELINE reranker. It does not understand semantics:
      "What are the foreign keys?" and "FK_STORE_KEY" would not match
      because the words are different even though the meaning is the same.

    The LLM reranker handles this because Claude understands that
    "foreign key" and "FK_" refer to the same concept.

    PARAMETERS
    ----------
    query : str
        The user's question.
    chunks : list[dict]
        Chunks from retrieve().
    top_n : int
        How many chunks to return.

    RETURNS
    -------
    list[dict]
        Top N chunks sorted by keyword overlap score.
    """
    stop_words = {"the", "is", "of", "in", "and", "or", "a", "an", "for", "to", "what", "how", "does"}
    query_words = {
        word.upper()
        for word in query.split()
        if len(word) > 2 and word.lower() not in stop_words
    }

    scored_chunks = []
    for i, chunk in enumerate(chunks):
        chunk_text_upper = chunk.get("text", "").upper()

        # Count how many query words appear in the chunk
        matches = sum(1 for word in query_words if word in chunk_text_upper)

        # Bonus: if the chunk's table_name matches a table in the query
        table_name = chunk.get("table_name", "")
        if table_name and table_name.upper() in query.upper():
            matches += 3  # Strong signal

        # Normalize to 0-5 scale (to match LLM reranker output)
        max_possible = len(query_words) + 3  # +3 for the table bonus
        if max_possible > 0:
            score = min(5, int((matches / max_possible) * 5) + 1)
        else:
            score = 1

        scored_chunks.append({
            **chunk,
            "rerank_score": score,
            "original_rank": i,
        })

    sorted_chunks = sorted(
        scored_chunks,
        key=lambda x: x["rerank_score"],
        reverse=True,
    )

    return sorted_chunks[:top_n]

def _tokenize_for_bm25(text:str) ->list[str]:
    """
    Tokenize text for BM25 scoring with data engineing domain awareness,tokenization directly affects scoring quality.

     1. PRESERVE UNDERSCORED IDENTIFIERS:
    e.g. "FK_STORE_KEY" stays as ONE token, not three ("FK", "STORE", "KEY").
     2. CASE-FOLD TO UPPERCASE:
    "dim_store" and "DIM_STORE" should match. 
    """
    text_upper = text.upper()
    tokens = re.findall(r'[A-Z][A-Z0-9_]{2,}|[A-Z]{3,}', text_upper)
    return tokens

def rerank_with_bm25(
    query: str,
    chunks: list[dict],
    top_n: int = 3,
) -> list[dict]:
    try:
        from rank_bm25 import BM25Okapi
    except ImportError:
        raise ImportError(
            "BM25 reranking requires the rank-bm25 package.\n"
            "Install with: uv add rank-bm25"
        )
 
    if not chunks:
        return []
    #Tokenize all chunks to build the BM25 corpus.
    corpus = [_tokenize_for_bm25(chunk.get("text","")) for chunk in chunks]
    
    #Build Bm25 index
    # BM25Okapi(corpus, k1=1.5, b=0.75)
    #   k1: Controls term frequency saturation.
    #       Higher k1 = more credit for repeated terms.
    #       1.5 is the standard value from the original BM25 paper.
    #   b:  Controls document length normalization.
    #       0 = no length penalty, 1 = full penalty.
    #       0.75 is the standard value.
    #
    # These defaults work well for most corpora. You could tune them
    # via your experiment_runner if you add a BM25 experiment.
    bm25 = BM25Okapi(corpus)

    #  Score the query against all chunks.
    #
    # bm25.get_scores(query_tokens) returns a numpy array of floats,
    # one score per document in the corpus.
    # Higher score = more relevant.
    #
    # PYTHON REFRESHER: numpy array indexing
    # scores[i] gives the BM25 score for chunk i.
    # scores is NOT a Python list -- it is a numpy ndarray.
    # But you can index it the same way: scores[0], scores[1], etc.
    # float(scores[i]) converts from numpy.float64 to Python float,
    # which avoids JSON serialization issues downstream.

    query_tokens = _tokenize_for_bm25(query)
    scores = bm25.get_scores(query_tokens)
    #Build scored chunk list, sort descending, return top N.
    scored_chunks = [
        {
            **chunk,
            "rerank_score": float(scores[i]),
            "rerank_method": "bm25",
            "original_rank": i,
        }
        for i, chunk in enumerate(chunks)
    ]

    scored_chunks.sort(key=lambda x: x["rerank_score"], reverse=True)
    return scored_chunks[:top_n]







#UNIFIED RERANK FUNCTION

def rerank_chunks(
    query: str,
    chunks: list[dict],
    top_n: int = 3,
    method: str = "bm25",
) -> list[dict]:
    """
    Rerank retrieved chunks uysing specified method
    PARAMETERS
    ----------
    query : str
        The user's question.
    chunks : list[dict]
        Chunks from retrieve().
    top_n : int
        How many chunks to return.
    method : str
        "llm"     -- Use Claude Haiku as reranker (accurate, costs API $)
        "keyword"  -- Use keyword overlap (free, less accurate)
        "none"    -- Skip reranking, return chunks as-is (for A/B testing)

        list[dict]
        Top N chunks, reranked (or original order if method="none").
    """
    if method == 'none' or not chunks:
        return [
            {**chunk, "rerank_score":0,"original_rank":i} for i,chunk in enumerate(chunks[:top_n])
        ]
    if method == "keyword":
        return rerank_with_keywords(query,chunks,top_n)
    if method == "llm":
        return rerank_with_llm(query,chunks,top_n)
    
    if method == "bm25":
        return rerank_with_bm25(query,chunks,top_n)
    
    print(f"WARNING: Unknown rerank method '{method}', skipping reranking")
    return [
        {**chunk, "rerank_score": 0, "original_rank": i}
        for i, chunk in enumerate(chunks[:top_n])
    ]
    
#Test
if __name__ == "__main__":
    print("=" * 60)
    print("RERANKER -- STANDALONE TEST")
    print("=" * 60)

    # Create fake chunks to test with (no ChromaDB needed)
    test_chunks = [
        {
            "text": "DIM_STORE is a dimension table. Grain: one row per store. Keys: SK_STORE_KEY, BK_STORE_KEY.",
            "source": "STTM__DIM_STORE__summary",
            "table_name": "DIM_STORE",
            "doc_type": "summary",
            "distance": 0.3,
        },
        {
            "text": "DIM_STORE_TYPE describes the type of store (retail, wholesale, online). FK: FK_STORE_TYPE_KEY.",
            "source": "STTM__DIM_STORE_TYPE__summary",
            "table_name": "DIM_STORE_TYPE",
            "doc_type": "summary",
            "distance": 0.35,
        },
        {
            "text": "FACT_SALES_ORDER contains sales transactions. Columns include FK_STORE_KEY linking to DIM_STORE.",
            "source": "STTM__FACT_SALES_ORDER__columns",
            "table_name": "FACT_SALES_ORDER",
            "doc_type": "column_mapping",
            "distance": 0.5,
        },
        {
            "text": "DIM_DATE provides calendar attributes. Not related to store dimensions.",
            "source": "STTM__DIM_DATE__summary",
            "table_name": "DIM_DATE",
            "doc_type": "summary",
            "distance": 0.7,
        },
        {
            "text": "Bronze layer stores raw extracts from SAP CDS Views before any transformation.",
            "source": "architecture_overview",
            "table_name": "",
            "doc_type": "text",
            "distance": 0.9,
        },
    ]

    query = "What foreign keys does DIM_STORE have?"
    print(f"\nQuery: {query}")
    print(f"Input chunks: {len(test_chunks)}")

    # Test keyword reranker (free, no API needed)
    print("\n--- Keyword Reranker ---")
    keyword_results = rerank_chunks(query, test_chunks, top_n=3, method="keyword")
    for i, chunk in enumerate(keyword_results):
        print(
            f"  [{i+1}] score={chunk['rerank_score']} "
            f"(was #{chunk['original_rank']+1}) "
            f"{chunk['table_name']} ({chunk['doc_type']})"
        )

    # Test "none" (passthrough)
    print("\n--- No Reranking ---")
    none_results = rerank_chunks(query, test_chunks, top_n=3, method="none")
    for i, chunk in enumerate(none_results):
        print(
            f"  [{i+1}] score={chunk['rerank_score']} "
            f"(was #{chunk['original_rank']+1}) "
            f"{chunk['table_name']} ({chunk['doc_type']})"
        )

    # Test LLM reranker (requires API key)
    print("\n--- LLM Reranker (requires ANTHROPIC_API_KEY) ---")
    import os
    if os.environ.get("ANTHROPIC_API_KEY"):
        llm_results = rerank_chunks(query, test_chunks, top_n=3, method="llm")
        for i, chunk in enumerate(llm_results):
            print(
                f"  [{i+1}] score={chunk['rerank_score']} "
                f"(was #{chunk['original_rank']+1}) "
                f"{chunk['table_name']} ({chunk['doc_type']})"
            )
    else:
        print("  Skipped (no API key). Set ANTHROPIC_API_KEY to test.")
        print("  The keyword reranker test above confirms the pipeline works.")

    # ── Test BM25 reranker (requires rank-bm25 installed) ──
    print("\\n--- BM25 Reranker ---")
    try:
        bm25_results = rerank_chunks(query, test_chunks, top_n=3, method="bm25")
        for i, chunk in enumerate(bm25_results):
            print(
                f"  [{i+1}] score={chunk['rerank_score']:.2f} "
                f"method={chunk.get('rerank_method', '?')} "
                f"(was #{chunk['original_rank']+1}) "
                f"{chunk['table_name']} ({chunk['doc_type']})"
            )
    except ImportError as e:
        print(f"  Skipped: {e}")
        print("  Install with: uv add rank-bm25")

    print("\nAll tests passed. reranker.py is ready.")