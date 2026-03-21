"""
app.py — Streamlit Web UI for Your STTM RAG Chatbot
=====================================================
Week 3 deliverable: your terminal chatbot, now as a web app.

HOW TO RUN
──────────
  cd C:\\Users\\laaro\\AI-Eng
  uv run streamlit run app.py
  → Opens http://localhost:8501 in your browser

WHAT THIS FILE DOES
───────────────────
Wraps your existing rag.py pipeline in a Streamlit web interface.
Your rag.py functions (load_documents, chunk_text, build_vector_store,
retrieve, improved_ask_claude, extract_table_name) are imported
and used EXACTLY as they are — no changes needed to rag.py.

This means your eval.py still tests the same code path. The web UI
is just a different "front door" to the same pipeline.

ARCHITECTURE DIAGRAM
────────────────────
  ┌─────────────┐     ┌──────────┐     ┌──────────┐
  │  Browser UI  │────▸│  app.py  │────▸│  rag.py  │
  │  (Streamlit) │◂────│  (glue)  │◂────│(pipeline)│
  └─────────────┘     └──────────┘     └──────────┘
                                             │
                                             ▼
                                      ┌──────────┐
                                      │ ChromaDB │
                                      └──────────┘

FUTURE CONNECTIONS (how this file grows across weeks):
──────────────────────────────────────────────────────
  Week 4: + conversation_memory.py  → multi-turn follow-ups
  Week 4: + reranker.py             → reranked results badge in debug panel
  Week 4: + model_switcher.py       → model dropdown in sidebar
  Week 4: + ollama_client.py        → local model option
  Week 5: + query logging           → JSONL output for observability
  Week 6: + FastAPI backend         → app.py calls API instead of importing rag.py


PYTHON REFRESHER: How Streamlit Differs From Normal Python
──────────────────────────────────────────────────────────
Normal Python: runs top-to-bottom ONCE, then exits.
Streamlit:     runs top-to-bottom EVERY TIME something changes.

Every button click, text input, or widget interaction causes Streamlit
to re-execute the ENTIRE script from line 1. This is called the
"rerun model."

This has HUGE implications:
  - Variables you create are LOST between reruns
  - Solution: st.session_state (a persistent dictionary)
  - Expensive operations (loading docs, building ChromaDB) must be CACHED
  - Solution: @st.cache_resource (compute once, reuse forever)

dbt ANALOGY:
  st.session_state    = dbt variable that survives incremental runs
  @st.cache_resource  = materialized='table' (compute once, read many)
  A Streamlit rerun   = a dbt run that re-evaluates all models

GOTCHA: Streamlit has its own web server built in. You do NOT need
Flask, Django, or FastAPI. Just `streamlit run app.py` and it handles
everything: HTTP server, WebSocket for live updates, static file serving.

DEPENDENCIES (already in your pyproject.toml)
─────────────────────────────────────────────
  streamlit, anthropic, chromadb, openpyxl
"""

import os
import time
import streamlit as st

# ═══════════════════════════════════════════════════════════════
# SECTION 0: IMPORTS FROM YOUR EXISTING PIPELINE
# ═══════════════════════════════════════════════════════════════
#
# These are the SAME functions your terminal chatbot and eval.py use.
# We're just calling them from a web context instead of a terminal.
#
# CRITICAL: We ONLY import what exists in YOUR rag.py right now.
# ──────────────────────────────────────────────────────────────
# Your actual rag.py contains these functions:
#   load_documents, chunk_text, build_vector_store, retrieve,
#   extract_table_name, improved_ask_claude
#
# It also exports these constants:
#   DOCS_DIR, TOP_K, CHUNK_SIZE, CHUNK_OVERLAP, MODEL,
#   IMPROVED_SYSTEM_PROMPT
#
# Things that are NOT in your current rag.py (yet):
#   classify_query      → Will be added in Week 4 query routing
#   conversation_memory → Will be a separate file in Week 4
#   reranker            → Will be a separate file in Week 4
#
# PYTHON REFRESHER: ImportError debugging
# ───────────────────────────────────────
# If you see: ImportError: cannot import name 'X' from 'rag'
# It means your rag.py doesn't have a function/variable called X.
# Check: does rag.py define X? Is it spelled exactly the same?
# Common causes:
#   - Function was renamed (ask_claude → improved_ask_claude)
#   - Function exists in generated code but wasn't copied to local file
#   - Typo in import name

from rag import (
    load_documents,           # Loads .txt, .md, .xlsx from docs/
    chunk_text,               # Splits documents into overlapping chunks
    build_vector_store,       # Stores chunks in ChromaDB with embeddings
    retrieve,                 # Queries ChromaDB for relevant chunks
    improved_ask_claude,      # Sends query + context to Claude API
    extract_table_name,       # Detects table names in questions
    DOCS_DIR,                 # "docs" — where your STTM files live
    TOP_K,                    # How many chunks to retrieve (3)
    CHUNK_SIZE,               # Characters per chunk (800)
    CHUNK_OVERLAP,            # Overlap between chunks (100)
    IMPROVED_SYSTEM_PROMPT,   # Your domain-specific system prompt
)


# ═══════════════════════════════════════════════════════════════
# SECTION 1: PAGE CONFIGURATION
# ═══════════════════════════════════════════════════════════════
#
# st.set_page_config() MUST be the first Streamlit command in the script.
# If ANY st.* call comes before it, you get an error:
#   "set_page_config() can only be called once per app, and must be
#    called as the first Streamlit command in your script."
#
# PYTHON REFRESHER: Named arguments for readability
# ──────────────────────────────────────────────────
# Instead of positional args like func("STTM", "wide", "📊"),
# named args make the code self-documenting:
#   func(page_title="STTM", layout="wide", page_icon="📊")
# You can read what each argument does without checking the docs.

st.set_page_config(
    page_title="STTM Assistant",     # Browser tab title
    page_icon="📊",                   # Browser tab icon (emoji works)
    layout="wide",                    # "wide" = full browser width
                                       # "centered" = narrow ~700px column
    initial_sidebar_state="expanded", # Sidebar open by default
)


# ═══════════════════════════════════════════════════════════════
# SECTION 2: CACHING THE RAG PIPELINE
# ═══════════════════════════════════════════════════════════════
#
# THE PROBLEM
# ───────────
# Streamlit reruns this script on EVERY interaction. Without caching:
#   1. User types a question → entire script reruns from line 1
#   2. load_documents() reads all Excel files from disk (~2 seconds)
#   3. chunk_text() splits into chunks (~0.5 seconds)
#   4. build_vector_store() rebuilds ChromaDB (~5 seconds)
#   5. THEN we can actually answer the question
#
# That's ~8 seconds of wasted work EVERY TIME.
#
# THE SOLUTION: @st.cache_resource
# ─────────────────────────────────
# This decorator tells Streamlit: "Run this function ONCE. Save the
# result. On every future rerun, return the saved result instead."
#
# PYTHON REFRESHER: Decorators
# ────────────────────────────
# A decorator is a function that wraps another function with extra
# behavior. Writing:
#
#   @st.cache_resource
#   def init_pipeline():
#       ...
#
# Is syntactic sugar (shorthand) for:
#   def init_pipeline():
#       ...
#   init_pipeline = st.cache_resource(init_pipeline)
#
# When you call init_pipeline(), you're actually calling the WRAPPER.
# The wrapper checks its internal cache:
#   - Cache empty? → Run the real function, save result, return it
#   - Cache has result? → Return saved result (skip the real function)
#
# dbt ANALOGY:
#   @st.cache_resource = materialized='table'
#   No decorator        = materialized='view' (recomputes every time)
#
# THERE ARE TWO CACHING DECORATORS IN STREAMLIT:
# ───────────────────────────────────────────────
# @st.cache_resource → For heavy objects you create ONCE and reuse:
#     database connections, ML models, ChromaDB collections
#     SHARED across all users of the app.
#
# @st.cache_data → For data that depends on INPUT arguments:
#     API responses, dataframe transformations
#     Each unique input gets its own cached result.
#
# We use @st.cache_resource because our pipeline is a "resource":
# expensive to build, doesn't change between user interactions,
# and should be shared across all browser tabs.
#
# GOTCHA: Cached resources persist until the Streamlit server restarts
# or you explicitly clear the cache. If you add new files to docs/,
# you need to clear the cache (we handle this in the file upload section).
#
# show_spinner= parameter shows a loading message while the function
# runs for the first time. On cached reruns, no spinner appears.

@st.cache_resource(show_spinner="Loading documents and building vector store...")
def init_pipeline():
    """
    Build the RAG pipeline ONCE and cache it for all future requests.

    This is identical to the startup sequence in rag.py's main() and
    eval.py's main() — same functions, same order, same result.

    RETURNS
    ───────
    tuple of (collection, known_tables, num_chunks, num_docs)
      collection:   ChromaDB collection with all your STTM chunks
      known_tables: Sorted list of table names for extract_table_name()
      num_chunks:   Total chunks created (for sidebar stats)
      num_docs:     Total documents loaded (for sidebar stats)

    PYTHON REFRESHER: Returning multiple values (tuple packing)
    ───────────────────────────────────────────────────────────
    return collection, known_tables, len(all_chunks), len(documents)

    Python automatically packs these into a tuple:
    return (collection, known_tables, len(all_chunks), len(documents))

    The caller unpacks them:
    coll, tables, chunks, docs = init_pipeline()

    If the counts don't match (e.g., 4 values returned but only 3
    variables on the left), Python raises ValueError.
    """

    # ─── Step 1: Load documents from docs/ folder ───
    # Same as eval.py and rag.py main()
    documents = load_documents(DOCS_DIR)

    if not documents:
        # st.error() displays a red banner in the web UI.
        # st.stop() halts execution — like sys.exit(1) but for Streamlit.
        # The page renders up to this point but nothing after runs.
        st.error(f"No documents found in `{DOCS_DIR}/` folder. "
                 f"Add .xlsx, .txt, or .md files and restart.")
        st.stop()

    # ─── Step 2: Chunk all documents ───
    # PYTHON REFRESHER: list.extend() vs list.append()
    # ─────────────────────────────────────────────────
    # append() adds ONE item:     [1,2].append([3,4]) → [1,2,[3,4]]
    # extend() adds EACH item:    [1,2].extend([3,4]) → [1,2,3,4]
    # chunk_text() returns a LIST of chunks, so we extend.
    all_chunks = []
    for doc in documents:
        chunks = chunk_text(doc["content"], doc["source"])
        all_chunks.extend(chunks)

    # ─── Step 3: Build vector store ───
    collection = build_vector_store(all_chunks)

    # ─── Step 4: Extract known table names ───
    # Query ChromaDB for ALL metadata, then extract unique table names.
    #
    # PYTHON REFRESHER: Generator expression inside set()
    # ───────────────────────────────────────────────────
    # set(x for x in items if condition) creates a set (unique values)
    # from a generator expression. The generator is lazy — it doesn't
    # create a list in memory, it yields one item at a time.
    #
    # Long-form equivalent:
    #   table_names = set()
    #   for m in all_meta["metadatas"]:
    #       name = m.get("table_name", "")
    #       if name and name.strip():
    #           table_names.add(name)
    #   known_tables = sorted(list(table_names))
    all_meta = collection.get()
    known_tables = sorted(list(set(
        m.get("table_name", "")
        for m in all_meta["metadatas"]
        if m.get("table_name") and m["table_name"].strip()
    )))

    return collection, known_tables, len(all_chunks), len(documents)


# ═══════════════════════════════════════════════════════════════
# SECTION 3: SESSION STATE INITIALIZATION
# ═══════════════════════════════════════════════════════════════
#
# THE PROBLEM WITH STREAMLIT'S RERUN MODEL
# ──────────────────────────────────────────
# In your terminal chatbot, variables persist naturally:
#   messages = []
#   while True:
#       query = input("> ")
#       messages.append(query)  # messages grows over time
#
# In Streamlit, EVERY interaction reruns the script:
#   messages = []               # ← This runs EVERY TIME!
#   # messages is always empty — your chat history is lost
#
# THE SOLUTION: st.session_state
# ───────────────────────────────
# st.session_state is a dictionary that PERSISTS across reruns
# for a single user's browser session. It survives when:
#   - User types a message (script reruns) ← SURVIVES
#   - User clicks a button (script reruns)  ← SURVIVES
#   - User uploads a file (script reruns)   ← SURVIVES
#
# It does NOT survive when:
#   - User closes the browser tab           ← LOST
#   - User opens a new tab (new session)    ← SEPARATE session
#   - You restart the Streamlit server      ← ALL sessions LOST
#
# Pattern: "Initialize once, use forever"
#   if "key" not in st.session_state:   # Only on FIRST run
#       st.session_state["key"] = value
#
# PYTHON REFRESHER: `not in` operator with dictionaries
# ──────────────────────────────────────────────────────
# "messages" in st.session_state     → True if key exists
# "messages" not in st.session_state → True if key does NOT exist
# Even if st.session_state["messages"] = None, the key EXISTS
# so `in` returns True. The check is about key presence, not value.
#
# DESIGN DECISION: Message format
# ────────────────────────────────
# Each message is a dict: {"role": "user"|"assistant", "content": "text"}
#
# This matches the Anthropic API's message format EXACTLY.
# When we add conversation memory in Week 4, we can pass
# st.session_state["messages"] to the API with minimal transformation.
# Planning ahead saves refactoring later.

if "messages" not in st.session_state:
    st.session_state["messages"] = []

if "debug_mode" not in st.session_state:
    st.session_state["debug_mode"] = False

if "query_count" not in st.session_state:
    st.session_state["query_count"] = 0

# ─── Source citations for each assistant message ───
# Key:   message index in st.session_state["messages"]
# Value: list of chunk dicts from retrieve()
#
# WHY A DICT, NOT A LIST?
# ───────────────────────
# We only have sources for ASSISTANT messages, not user messages.
# A dict lets us use the message index as the key, naturally
# skipping user messages. If we used a list, we'd need to track
# which indices are user vs assistant — more error-prone.
#
# PYTHON REFRESHER: Dict with integer keys
# ─────────────────────────────────────────
# Python dicts can use ANY hashable type as keys — strings, ints,
# tuples, etc. Using integer keys is perfectly valid:
#   d = {0: "first", 2: "third"}  # key 1 is missing — that's fine
if "sources_log" not in st.session_state:
    st.session_state["sources_log"] = {}


# ═══════════════════════════════════════════════════════════════
# SECTION 4: INITIALIZE THE PIPELINE (CACHED)
# ═══════════════════════════════════════════════════════════════
#
# This line calls init_pipeline(). Because of @st.cache_resource,
# it only actually executes the function body on the FIRST run.
# Every subsequent rerun returns the cached result instantly (<1ms).

collection, known_tables, num_chunks, num_docs = init_pipeline()


# ═══════════════════════════════════════════════════════════════
# SECTION 5: HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════

def extract_citations(chunks: list[dict]) -> list[dict]:
    """
    Convert raw retrieval chunks into display-friendly citation objects.

    Each chunk from retrieve() looks like:
    {
        "text": "Table: Dim_Store\\nGrain: ...",
        "source": "STTM__Dim_Store__summary",
        "table_name": "DIM_STORE",
        "doc_type": "summary",
        "distance": 0.342,
    }

    We transform this into a citation for the UI:
    {
        "table": "DIM_STORE",
        "type": "summary",
        "source": "STTM__Dim_Store__summary",
        "relevance": "🟢 High",
        "distance": 0.342,
        "preview": "Table: Dim_Store\\nGrain: ..."
    }

    DISTANCE INTERPRETATION
    ───────────────────────
    ChromaDB returns cosine distance (0 = identical, 2 = opposite).
    In practice for your STTM data, distances cluster between 0.2–1.5:
      < 0.5  = Strong match (very relevant)  → 🟢 High
      0.5–1.0 = Moderate match               → 🟡 Medium
      > 1.0  = Weak match (probably noise)    → 🔴 Low

    These thresholds are heuristic — they came from observing your
    Week 2 eval experiments. When distance was < 0.5, answers were
    almost always correct. Above 1.0, they were unreliable.

    GOTCHA: distance might be None if the chunk didn't come from
    a ChromaDB query (e.g., manually injected context).
    The `or 999` handles this: None or 999 → 999 → "🔴 Low"

    PYTHON REFRESHER: `or` with non-boolean values
    ───────────────────────────────────────────────
    Python's `or` returns the first "truthy" value:
      None or 999     → 999  (None is falsy, 999 is truthy)
      0.5 or 999      → 0.5  (0.5 is truthy, so it wins)
      0 or 999        → 999  (0 is falsy!)
      "" or "default"  → "default"

    GOTCHA: 0 is falsy in Python! If distance could legitimately be 0.0
    (meaning perfect match), `distance or 999` would incorrectly return 999.
    For our use case, a distance of exactly 0.0 is extremely unlikely
    (it would mean the query IS one of the chunks verbatim).
    If this ever became a real concern, use:
      distance if distance is not None else 999
    """
    citations = []
    for chunk in chunks:
        distance = chunk.get("distance") or 999

        if distance < 0.5:
            relevance = "🟢 High"
        elif distance < 1.0:
            relevance = "🟡 Medium"
        else:
            relevance = "🔴 Low"

        citations.append({
            "table": chunk.get("table_name", "Unknown"),
            "type": chunk.get("doc_type", "text"),
            "source": chunk.get("source", "unknown"),
            "relevance": relevance,
            "distance": round(distance, 3) if distance != 999 else None,
            "preview": chunk.get("text", "")[:200] + "...",
        })

    return citations


def format_citation_badges(citations: list[dict]) -> str:
    """
    Create a compact markdown string showing source citations.

    Example output:
        📎 Sources: **DIM_STORE** (summary, 🟢 High) · **DIM_PRODUCT** (columns, 🟡 Medium)

    This goes BELOW Claude's answer so the user sees where info came from.

    WHY MARKDOWN AND NOT st.columns()?
    ──────────────────────────────────
    Streamlit's st.chat_message() only renders markdown content.
    You can't nest st.columns() inside a chat message. So we build
    a markdown string and include it in the message rendering.

    PYTHON REFRESHER: str.join()
    ────────────────────────────
    " · ".join(["A", "B", "C"]) → "A · B · C"
    The string BEFORE .join() is the separator.
    The list INSIDE .join() is what gets joined.

    Long-form equivalent:
        result = ""
        for i, part in enumerate(badge_parts):
            if i > 0:
                result += " · "
            result += part
    """
    if not citations:
        return ""

    badge_parts = []
    for c in citations:
        badge_parts.append(f"**{c['table']}** ({c['type']}, {c['relevance']})")

    return "Sources: " + " · ".join(badge_parts)


def render_sources_detail(chunks: list[dict]):
    """
    Render expandable source citation detail under an assistant message.

    WHY SHOW SOURCES?
    ─────────────────
    In data engineering, trust is everything. If the chatbot says
    "FACT_STORE_INVENTORY_INTRA joins to DIM_DATE via SK_DATE_KEY",
    your team needs to verify that against the actual STTM document.

    Source citations provide:
      1. TRANSPARENCY  — Which documents were used
      2. VERIFIABILITY — User can check the original source
      3. DEBUGGING     — When an answer is wrong, you see WHY
                         (wrong chunks retrieved vs wrong interpretation)

    This is the same principle as your eval.py's "retrieved_sources"
    column — but now visible to the end user, not just in a CSV file.

    PARAMETERS
    ──────────
    chunks: List of chunk dicts from retrieve(), each containing:
            text, source, table_name, doc_type, distance
    """
    if not chunks:
        return

    with st.expander(f"🔍 Source Details ({len(chunks)} chunks)", expanded=False):
        for j, chunk in enumerate(chunks):
            table = chunk.get("table_name", "Unknown")
            doc_type = chunk.get("doc_type", "unknown")
            distance = chunk.get("distance")

            # Convert distance to relevance score
            # ChromaDB: 0.0 = identical, 2.0 = opposite (cosine)
            # Relevance: 1.0 = identical, 0.0 = opposite
            #
            # PYTHON REFRESHER: Ternary expression
            # ─────────────────────────────────────
            # value = x if condition else y
            # Long form:
            #   if distance is not None:
            #       relevance = round(1 - distance, 3)
            #   else:
            #       relevance = None
            relevance = round(1 - distance, 3) if distance is not None else None
            relevance_str = f" | Relevance: {relevance}" if relevance is not None else ""

            st.markdown(f"**Chunk {j+1}:** `{table}` — {doc_type}{relevance_str}")

            # st.code() renders text in a monospace box — perfect for
            # document excerpts without markdown formatting interfering.
            #
            # DESIGN DECISION: Truncate to 300 chars
            # ───────────────────────────────────────
            # Full chunks can be 800+ characters. 300 chars gives enough
            # context to identify the source without overwhelming the UI.
            preview = chunk.get("text", "")[:300]
            if len(chunk.get("text", "")) > 300:
                preview += "..."
            st.code(preview, language=None)


# ═══════════════════════════════════════════════════════════════
# SECTION 6: SIDEBAR
# ═══════════════════════════════════════════════════════════════
#
# STREAMLIT LAYOUT MODEL
# ──────────────────────
# Streamlit has two main areas:
#   1. Main area (center) — where your chat lives
#   2. Sidebar (left)     — for controls, settings, info
#
# `with st.sidebar:` is a context manager that puts everything
# inside it into the sidebar panel. When you de-indent, you're
# back to the main area.
#
# PYTHON REFRESHER: Context managers (`with` statement)
# ────────────────────────────────────────────────────
# `with X:` runs X.__enter__() at the start and X.__exit__() at
# the end. Streamlit uses this to "switch" the rendering target:
#   with st.sidebar:        # __enter__: switch to sidebar
#       st.write("sidebar") # renders in sidebar
#   # __exit__: switch back to main
#   st.write("main")        # renders in main area

with st.sidebar:
    st.title("📊 STTM Assistant")
    st.caption("Data Warehouse Documentation")

    # ─── Pipeline statistics ───
    # st.metric() creates a stat card with label + large number.
    # st.columns() arranges them side by side.
    #
    # STREAMLIT REFRESHER: st.columns()
    # ──────────────────────────────────
    # st.columns(2) returns two column objects.
    # col1, col2 = st.columns(2)  ← equal width
    # col1, col2 = st.columns([3, 1])  ← col1 is 3x wider
    st.subheader("Pipeline Stats")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Documents", num_docs)
        st.metric("Queries", st.session_state["query_count"])
    with col2:
        st.metric("Chunks", num_chunks)
        st.metric("Tables", len(known_tables))

    st.divider()

    # ─── Debug Toggle ───
    # st.toggle() returns True/False and re-renders on change.
    #
    # GOTCHA: We store the toggle state in session_state so it
    # persists between reruns. The `value=` parameter sets the
    # initial visual state from our stored value.
    st.session_state["debug_mode"] = st.toggle(
        "Show Debug Info",
        value=st.session_state["debug_mode"],
        help="Show retrieved chunks, distances, and timing for each query",
    )

    st.divider()

    # ─── File Upload ───
    # st.file_uploader() creates a drag-and-drop zone.
    #
    # STREAMLIT REFRESHER: File uploader returns
    # ──────────────────────────────────────────
    # Returns None if no file uploaded (most reruns).
    # Returns UploadedFile object when a file is dropped.
    # The object has .name (filename), .getvalue() (raw bytes).
    #
    # The `type` parameter restricts accepted file extensions.
    #
    # GOTCHA: The uploaded file is in MEMORY, not on disk.
    # To use it with load_documents() (which reads from disk),
    # we save it to docs/ first, then rebuild the pipeline.
    st.subheader("Upload Documents")
    uploaded_file = st.file_uploader(
        "Add STTM or documentation files",
        type=["xlsx", "txt", "md"],
        help="Upload .xlsx (STTM workbooks), .txt, or .md files. "
             "File is saved to docs/ and the pipeline rebuilds automatically.",
    )

    if uploaded_file is not None:
        save_path = os.path.join(DOCS_DIR, uploaded_file.name)

        # ─── Save the file to docs/ ───
        # PYTHON REFRESHER: Writing binary files
        # ──────────────────────────────────────
        # open(path, "wb") opens in Write Binary mode.
        # "wb" because .xlsx files are binary (ZIP archives).
        # Text files (.txt, .md) also work with "wb" mode — the
        # raw bytes are identical to the text content for UTF-8.
        # If we used "w" (text mode) for Excel files, it would
        # corrupt them by attempting character encoding.
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getvalue())

        st.success(f"Saved {uploaded_file.name} to {DOCS_DIR}/")

        # ─── Force pipeline rebuild ───
        # st.cache_resource.clear() nukes ALL cached resources.
        # st.rerun() immediately restarts the script from line 1.
        #
        # On the next run, init_pipeline() has no cache → it rebuilds
        # the vector store including the new file.
        #
        # GOTCHA: st.rerun() immediately stops current execution.
        # Nothing after st.rerun() runs — like a return + restart.
        st.cache_resource.clear()
        st.rerun()

    st.divider()

    # ─── Table list ───
    # st.expander() creates a collapsible section — good for long
    # lists that would clutter the sidebar.
    with st.expander(f"📋 Indexed Tables ({len(known_tables)})", expanded=False):
        for i, table in enumerate(known_tables):
            st.text(f"{i+1:2d}. {table}")

    st.divider()

    # ─── Control Buttons ───
    col1, col2 = st.columns(2)

    # STREAMLIT REFRESHER: Button return values
    # ──────────────────────────────────────────
    # st.button() returns True ONCE — on the rerun triggered by click.
    # On all other reruns, it returns False.
    # So `if st.button("X"):` code runs exactly once per click.
    if col1.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state["messages"] = []
        st.session_state["sources_log"] = {}
        st.session_state["query_count"] = 0
        st.rerun()

    if col2.button("🔄 Re-index", use_container_width=True,
                    help="Rebuild vector store from docs/ folder"):
        st.cache_resource.clear()
        st.rerun()

    st.divider()

    # ─── Current config display ───
    st.subheader("Current Config")
    st.code(
        f"CHUNK_SIZE = {CHUNK_SIZE}\n"
        f"CHUNK_OVERLAP = {CHUNK_OVERLAP}\n"
        f"TOP_K = {TOP_K}",
        language="python",
    )


# ═══════════════════════════════════════════════════════════════
# SECTION 7: MAIN CHAT AREA
# ═══════════════════════════════════════════════════════════════

st.title("💬 STTM Assistant")
st.caption(
    "Ask questions about Sigma Healthcare's data warehouse tables, "
    "columns, mappings, and data pipelines. "
    "Source citations shown under each answer."
)


# ═══════════════════════════════════════════════════════════════
# SECTION 8: RENDER CHAT HISTORY
# ═══════════════════════════════════════════════════════════════
#
# On every rerun, we re-render ALL past messages from session_state.
# Streamlit doesn't "remember" what it drew last time — it starts
# with a blank page and re-draws everything top-to-bottom.
#
# This is like a SQL dashboard that runs SELECT * FROM messages
# on every page load — the data is in the database (session_state),
# and the rendering is stateless.

for i, msg in enumerate(st.session_state["messages"]):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        # Show citation badges below assistant messages
        if msg["role"] == "assistant" and msg.get("citations"):
            citation_text = format_citation_badges(msg["citations"])
            if citation_text:
                st.caption(citation_text)

        # Show debug panel if debug mode is on
        if (
            st.session_state["debug_mode"]
            and msg["role"] == "assistant"
            and msg.get("debug_info")
        ):
            debug = msg["debug_info"]
            with st.expander("🔍 Debug: Retrieved Chunks", expanded=False):
                # Timing
                if debug.get("timing"):
                    t = debug["timing"]
                    st.text(
                        f"⏱ Retrieval: {t['retrieval']:.2f}s | "
                        f"Generation: {t['generation']:.2f}s | "
                        f"Total: {t['total']:.2f}s"
                    )
                # Detected table
                if debug.get("detected_table"):
                    st.text(f"🎯 Detected table: {debug['detected_table']}")
                # Each chunk
                for j, chunk in enumerate(debug.get("chunks", [])):
                    dist_str = f" (d={chunk['distance']:.3f})" if chunk.get("distance") else ""
                    st.text(f"[{j+1}] {chunk.get('table_name', '?')} "
                            f"({chunk.get('doc_type', '?')}){dist_str}")
                    st.code(chunk.get("text", "")[:300], language=None)

        # Show detailed sources panel if it exists
        if msg["role"] == "assistant" and i in st.session_state["sources_log"]:
            render_sources_detail(st.session_state["sources_log"][i])


# ═══════════════════════════════════════════════════════════════
# SECTION 9: HANDLE NEW USER INPUT
# ═══════════════════════════════════════════════════════════════
#
# st.chat_input() creates the text input box at the BOTTOM of
# the page. When the user presses Enter:
#   1. Streamlit captures the text
#   2. Triggers a full script rerun
#   3. On this rerun, st.chat_input() returns the user's text
#   4. On all other reruns, it returns None
#
# GOTCHA: st.chat_input() can only be called ONCE per script.
# Multiple chat inputs cause an error.

query = st.chat_input("Ask about your data warehouse...")

if query:
    # ─── Step 1: Display the user's message ───
    # We render it IMMEDIATELY so it appears before the (slow) API call.
    st.session_state["messages"].append({
        "role": "user",
        "content": query,
    })

    with st.chat_message("user"):
        st.markdown(query)

    # ─── Step 2: Run the RAG pipeline ───
    # This is the EXACT SAME sequence as rag.py's main() loop
    # and eval.py's run_evaluation(). Same functions, same order.
    with st.chat_message("assistant"):
        with st.spinner("Searching documents and generating answer..."):

            # ─── 2a: Detect table name ───
            detected = extract_table_name(query, known_tables)

            # ─── 2b: Retrieve relevant chunks ───
            retrieval_start = time.time()
            chunks = retrieve(
                collection,
                query,
                table_name=detected,
                known_tables=known_tables,
            )
            retrieval_time = time.time() - retrieval_start

            # ─── 2c: Generate answer via Claude ───
            # We use improved_ask_claude() from rag.py — the same
            # function that eval.py calls. One source of truth.
            #
            # WEEK 4 UPGRADE NOTE:
            # ─────────────────────
            # Right now, each question is independent — Claude has
            # no memory of previous questions in this session.
            # In Week 4, we'll replace this with ask_claude_with_memory()
            # that passes st.session_state["messages"] to the API,
            # enabling follow-up questions like:
            #   "Tell me about DIM_STORE"
            #   "What about its foreign keys?"  ← Claude knows "its" = DIM_STORE
            generation_start = time.time()
            try:
                answer = improved_ask_claude(query, chunks)
            except Exception as e:
                answer = f"❌ Error generating response: {e}"
            generation_time = time.time() - generation_start

            total_time = retrieval_time + generation_time

        # ─── Step 3: Extract citations and display ───
        citations = extract_citations(chunks)
        citation_text = format_citation_badges(citations)

        # Display the answer
        st.markdown(answer)

        # Display citation badges below the answer
        if citation_text:
            st.caption(citation_text)

        # ─── Step 4: Build debug info ───
        debug_info = {
            "detected_table": detected,
            "timing": {
                "retrieval": retrieval_time,
                "generation": generation_time,
                "total": total_time,
            },
            "chunks": chunks,
        }

        # Show debug panel if enabled
        if st.session_state["debug_mode"]:
            with st.expander("🔍 Debug: Retrieved Chunks", expanded=False):
                t = debug_info["timing"]
                st.text(
                    f"⏱ Retrieval: {t['retrieval']:.2f}s | "
                    f"Generation: {t['generation']:.2f}s | "
                    f"Total: {t['total']:.2f}s"
                )
                if detected:
                    st.text(f"🎯 Detected table: {detected}")
                for j, chunk in enumerate(chunks):
                    dist_str = (
                        f" (d={chunk['distance']:.3f})"
                        if chunk.get("distance")
                        else ""
                    )
                    st.text(
                        f"[{j+1}] {chunk.get('table_name', '?')} "
                        f"({chunk.get('doc_type', '?')}){dist_str}"
                    )
                    st.code(chunk.get("text", "")[:300], language=None)

        # Show detailed source panel
        render_sources_detail(chunks)

        # Show timing as caption
        table_info = f" | Table: {detected}" if detected else ""
        st.caption(
            f" Retrieval: {retrieval_time:.2f}s | "
            f"Generation: {generation_time:.2f}s | "
            f"Total: {total_time:.2f}s"
            f"{table_info}"
        )

    # ─── Step 5: Save to session state ───
    # Store answer + citations + debug info together so they
    # render correctly on future reruns.
    #
    # DESIGN DECISION: Storing extra fields with each message
    # ────────────────────────────────────────────────────────
    # The Anthropic API only reads "role" and "content" — it
    # ignores "citations" and "debug_info". So storing extra
    # fields is safe for future Week 4 integration.
    #
    # When you add conversation memory in Week 4, you'll filter
    # messages to only send {"role", "content"} to the API:
    #   api_messages = [
    #       {"role": m["role"], "content": m["content"]}
    #       for m in st.session_state["messages"]
    #   ]
    st.session_state["messages"].append({
        "role": "assistant",
        "content": answer,
        "citations": citations,
        "debug_info": debug_info,
    })

    # Save source chunks for detailed source panel on reruns
    msg_index = len(st.session_state["messages"]) - 1
    st.session_state["sources_log"][msg_index] = chunks

    # Update query counter
    st.session_state["query_count"] += 1