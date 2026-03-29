"""
Streamlit Web UI for STTM RAG Chatbot


Wraps your existing rag.py pipeline in a Streamlit web interface.
Your rag.py functions (load_documents, chunk_text, build_vector_store,
retrieve, ask_claude, extract_table_name) are imported
and used EXACTLY as they are -- no changes needed to rag.py.

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
"""

import os
import time
import streamlit as st
from pathlib import Path
from conversation_memory import ConversationMemory
from query_router import route_query, explain_routing
from model_switcher import list_models
from query_logger import log_query, generate_session_id, load_logs, analyze_logs

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
    IMPROVED_SYSTEM_PROMPT,
)

if not os.environ.get("ANTHROPIC_API_KEY"):
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, _, value = line.partition("=")
                    os.environ[key.strip()] = value.strip()


# =====================================================================
# SECTION 1: PAGE CONFIGURATION
# =====================================================================

st.set_page_config(
    page_title="STTM Assistant",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =====================================================================
# SECTION 2: CACHING RAG PIPELINE
# =====================================================================

@st.cache_resource(show_spinner="Loading documents and building vector store")
def init_pipeline():
    """
    Build the RAG pipeline ONCE and cache it for all future requests.
    """
    documents = load_documents(DOCS_DIR)

    if not documents:
        st.error(
            f"No documents found in `{DOCS_DIR}/` folder. "
            f"Add .xlsx, .txt, or .md files and restart."
        )
        st.stop()

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

    return collection, known_tables, len(all_chunks), len(documents)


# =====================================================================
# SECTION 3: SESSION STATE INITIALISATION
# =====================================================================

if "messages" not in st.session_state:
    st.session_state["messages"] = []

if "debug_mode" not in st.session_state:
    st.session_state["debug_mode"] = False

if "query_count" not in st.session_state:
    st.session_state["query_count"] = 0

if "sources_log" not in st.session_state:
    st.session_state["sources_log"] = {}

if "memory" not in st.session_state:
    st.session_state["memory"] = ConversationMemory(max_turns=5)

if "selected_model" not in st.session_state:
    st.session_state["selected_model"] = None

if "rerank_method" not in st.session_state:
    st.session_state["rerank_method"] = "auto"

if "routing_log" not in st.session_state:
    st.session_state["routing_log"] = {}

if "session_id" not in st.session_state:
    st.session_state["session_id"] = generate_session_id()


# =====================================================================
# SECTION 4: BUILD PIPELINE
# =====================================================================

collection, known_tables, num_chunks, num_docs = init_pipeline()


# =====================================================================
# SECTION 5: HELPER FUNCTIONS
# =====================================================================

def extract_citation(chunks: list[dict]) -> list[dict]:
    """
    Convert raw retrieval chunks into display-friendly citation objects.

    DISTANCE INTERPRETATION (ChromaDB cosine distance):
      < 0.5  = Strong match   -> High
      0.5-1.0 = Moderate match -> Medium
      > 1.0  = Weak match     -> Low
    """
    citations = []
    for chunk in chunks:
        distance = chunk.get("distance") or 999
        if distance < 0.5:
            relevance = "High"
        elif distance < 1.0:
            relevance = "Medium"
        else:
            relevance = "Low"

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
        Sources: **DIM_STORE** (summary, High) . **DIM_PRODUCT** (columns, Medium)
    """
    if not citations:
        return ""
    badge_parts = []
    for c in citations:
        badge_parts.append(f"**{c['table']}** ({c['type']}, {c['relevance']})")
    return "Sources: " + " . ".join(badge_parts)


def render_sources_detail(chunks: list[dict]):
    """
    Render expandable source citation detail under an assistant message.
    """
    if not chunks:
        return
    with st.expander(f"Source Detail ({len(chunks)} chunks)", expanded=False):
        for j, chunk in enumerate(chunks):
            table = chunk.get("table_name", "Unknown")
            doc_type = chunk.get("doc_type", "Unknown")
            distance = chunk.get("distance")

            relevance = round(1 - distance, 3) if distance is not None else None
            relevance_str = f" | Relevance:{relevance}" if relevance is not None else ""
            st.markdown(f"**Chunk {j+1}:** `{table}` - {doc_type}{relevance_str}")

            preview = chunk.get("text", "")[:300]
            if len(chunk.get("text", "")) > 300:
                preview += "..."
            st.code(preview, language=None)


# =====================================================================
# SECTION 6: SIDEBAR
# =====================================================================

with st.sidebar:
    st.title("STTM Assistant")
    st.caption("Source to Target Mapping Reference Tool")

    # ── Pipeline statistics ──
    st.subheader("Source Document Stats")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Documents", num_docs)
        st.metric("Queries", st.session_state["query_count"])
    with col2:
        st.metric("Chunks", num_chunks)
        st.metric("Tables", len(known_tables))

    st.divider()

    # ── Debug toggle ──
    st.session_state["debug_mode"] = st.toggle(
        "Show Debug Info",
        value=st.session_state["debug_mode"],
        help="Show retrieved chunks, distances, timing for each query",
    )

    st.divider()

    # ── Model Selection ──
    available_models = list_models()
    model_options = {m["name"]: m["id"] for m in available_models if m["available"]}

    if model_options:
        display_options = ["Auto (Router Decides)"] + list(model_options.keys())
        selected_display = st.selectbox(
            "Model",
            display_options,
            index=0,
            help="Auto lets the query router choose based on query complexity.",
        )
        if selected_display == "Auto (Router Decides)":
            st.session_state["selected_model"] = None
        else:
            st.session_state["selected_model"] = model_options[selected_display]

    # ── Reranking Control ──
    rerank_options = {
        "Auto (Router Decides)": "auto",
        "BM25 (Free)": "bm25",
        "LLM Reranker (Haiku)": "llm",
        "Off": "none",
    }
    selected_rerank = st.selectbox(
        "Reranking",
        list(rerank_options.keys()),
        index=0,
        help="Auto enables reranking for complex queries only.",
    )
    st.session_state["rerank_method"] = rerank_options[selected_rerank]

    # ── Memory Stats ──
    memory = st.session_state["memory"]
    stats = memory.get_stats()
    st.caption(
        f"Memory: {stats['window_turns']}/{stats['max_turns']} turns | "
        f"~{stats['estimated_tokens']} tokens"
    )

    st.divider()

    # ── File Upload ──
    st.subheader("Upload Documents")
    uploaded_file = st.file_uploader(
        "Add STTM files here",
        type=["xlsx"],
        help="Upload .xlsx (STTM workbooks). "
             "File is saved to docs/ and the pipeline rebuilds automatically.",
    )

    if uploaded_file is not None:
        save_path = os.path.join(DOCS_DIR, uploaded_file.name)

        with open(save_path, "wb") as f:
            f.write(uploaded_file.getvalue())
        st.success(f"Saved {uploaded_file.name} to {DOCS_DIR}/")

        st.cache_resource.clear()
        st.rerun()

    st.divider()

    # ── Indexed Tables ──
    with st.expander(f"Indexed Tables ({len(known_tables)})", expanded=False):
        for i, table in enumerate(known_tables):
            st.text(f"{i+1:2d}. {table}")

    st.divider()

    # ── Action Buttons ──
    col1, col2 = st.columns(2)

    if col1.button("Clear Chat", use_container_width=True):
        st.session_state["messages"] = []
        st.session_state["sources_log"] = {}
        st.session_state["routing_log"] = {}
        st.session_state["query_count"] = 0
        st.session_state["memory"] = ConversationMemory(max_turns=5)
        st.rerun()

    if col2.button("Re-index", use_container_width=True,
                   help="Rebuild vector store from docs"):
        st.cache_resource.clear()
        st.rerun()

    st.divider()

    # ── Query Logs (Week 5) ──
    st.subheader("Query Logs")
    log_path = Path("logs/queries.jsonl")
    if log_path.exists():
        log_entries = load_logs()
        st.caption(
            f"Log: {len(log_entries)} queries "
            f"({log_path.stat().st_size / 1024:.1f} KB)"
        )
        if st.button("Show Analytics"):
            analytics = analyze_logs(log_entries)
            if "error" not in analytics:
                vol = analytics["volume"]
                cost = analytics["cost"]
                lat = analytics["latency"]["total_ms"]
                st.metric("Total Queries", vol["total_queries"])
                st.metric("Total Cost", f"${cost['total_usd']:.4f}")
                st.metric("p50 Latency", f"{lat['p50']:.0f}ms")
                st.metric("p95 Latency", f"{lat['p95']:.0f}ms")
    else:
        st.caption("No logs yet. Ask some questions first.")

    st.divider()

    # ── Current Config ──
    st.subheader("Current Config")
    st.code(
        f"CHUNK_SIZE = {CHUNK_SIZE}\n"
        f"CHUNK_OVERLAP = {CHUNK_OVERLAP}\n"
        f"TOP_K = {TOP_K}",
        language="python",
    )


# =====================================================================
# SECTION 7: MAIN CHAT AREA
# =====================================================================

st.title("STTM Assistant")
st.caption(
    "Ask questions about F06 STTM "
    "columns, mappings, and data pipelines. "
    "Source citations shown under each answer."
)


# =====================================================================
# SECTION 8: RENDER CHAT HISTORY
# =====================================================================
# On every rerun, we re-render ALL past messages from session_state.
# Streamlit does not "remember" what it drew last time -- it starts
# with a blank page and re-draws everything top-to-bottom.

for i, msg in enumerate(st.session_state["messages"]):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        # Show citation badges for assistant messages
        if msg["role"] == "assistant" and msg.get("citations"):
            citation_text = format_citation_badges(msg["citations"])
            if citation_text:
                st.caption(citation_text)

        # Show debug panel for historical messages.
        #
        # BUG FIX (Week 5): The original code referenced msg["debug_info"]
        # which was never stored in the message dict. Messages only contain
        # "role", "content", and "citations". The routing data is stored
        # separately in st.session_state["routing_log"].
        #
        # This fix reads from routing_log (which IS populated) instead of
        # msg["debug_info"] (which does NOT exist and would raise KeyError).
        if (
            st.session_state["debug_mode"]
            and msg["role"] == "assistant"
            and i in st.session_state["routing_log"]
        ):
            routing_info = st.session_state["routing_log"][i]
            with st.expander("Debug: Routing", expanded=False):
                st.text(f"Routing: {explain_routing(routing_info)}")

        # Show source detail for historical messages
        if msg["role"] == "assistant" and i in st.session_state["sources_log"]:
            render_sources_detail(st.session_state["sources_log"][i])


# =====================================================================
# SECTION 9: HANDLE NEW USER INPUT
# =====================================================================
# st.chat_input() creates the text input box at the BOTTOM of the page.
# It returns the user's text on the rerun triggered by pressing Enter,
# and None on all other reruns.

query = st.chat_input("Ask about Mapping information")

if query:
    # ── Step 1: Save user message to session state ──
    st.session_state["messages"].append({
        "role": "user",
        "content": query,
    })

    # ── Step 2: Display user message ──
    with st.chat_message("user"):
        st.markdown(query)

    # ── Step 3: Run RAG pipeline and display response ──
    with st.chat_message("assistant"):
        with st.spinner("Searching document and generating answer..."):
            force_model = st.session_state["selected_model"]
            force_rerank = (
                None if st.session_state["rerank_method"] == "auto"
                else st.session_state["rerank_method"]
            )

            result = route_query(
                query=query,
                collection=collection,
                known_tables=known_tables,
                memory=st.session_state["memory"],
                force_model=force_model,
                force_rerank=force_rerank,
            )

            answer = result["answer"]
            chunks = result["chunks"]
            routing = result["routing"]
            timing = result["timing"]

        # ── Step 4: Extract citations and display ──
        citations = extract_citation(chunks)
        citation_text = format_citation_badges(citations)

        st.markdown(answer)

        if citation_text:
            st.caption(citation_text)

        # ── Step 5: Show debug panel if enabled ──
        if st.session_state["debug_mode"]:
            with st.expander("Debug: Routing + Retrieval", expanded=False):
                st.text(f"Routing: {explain_routing(routing)}")
                st.text(
                    f"Timing: classify={timing.get('classify_ms', 0):.0f}ms | "
                    f"retrieve={timing.get('retrieve_ms', 0):.0f}ms | "
                    f"rerank={timing.get('rerank_ms', 0):.0f}ms | "
                    f"generate={timing.get('generate_ms', 0):.0f}ms | "
                    f"total={timing.get('total_ms', 0):.0f}ms"
                )
                mem_stats = st.session_state["memory"].get_stats()
                st.text(
                    f"Memory: {mem_stats['window_turns']}/{mem_stats['max_turns']} turns | "
                    f"Follow-up: {routing['is_follow_up']}"
                )
                debug = result.get("debug", {})
                if debug.get("detected_table"):
                    st.text(f"Detected table: {debug['detected_table']}")
                for j, chunk in enumerate(chunks):
                    rerank_info = ""
                    if chunk.get("rerank_score") is not None:
                        rerank_info = f" rerank={chunk['rerank_score']:.2f}"
                    dist_str = (
                        f" d={chunk['distance']:.3f}"
                        if chunk.get("distance")
                        else ""
                    )
                    st.text(
                        f"[{j+1}] {chunk.get('table_name', '?')} "
                        f"({chunk.get('doc_type', '?')}){dist_str}{rerank_info}"
                    )
                    st.code(chunk.get("text", "")[:300], language=None)

        # ── Step 6: Show source detail panel ──
        render_sources_detail(chunks)

        # ── Step 7: Show timing caption ──
        total_s = timing.get("total_ms", 0) / 1000
        st.caption(f"Model: {routing['model']} | Total: {total_s:.2f}s")

    # ── Step 8: Save assistant message to session state ──
    st.session_state["messages"].append({
        "role": "assistant",
        "content": answer,
        "citations": citations,
    })

    # ── Step 9: Save debug data for historical re-rendering ──
    msg_index = len(st.session_state["messages"]) - 1
    st.session_state["sources_log"][msg_index] = chunks
    st.session_state["routing_log"][msg_index] = routing

    # ── Step 10: Update query counter ──
    st.session_state["query_count"] += 1

    # ── Step 11: Log query to JSONL (Week 5) ──
    try:
        log_query(
            query=query,
            answer=answer,
            routing=routing,
            timing=timing,
            chunks=chunks,
            session_id=st.session_state["session_id"],
        )
    except Exception as e:
        print(f"WARNING: Query logging failed: {e}")