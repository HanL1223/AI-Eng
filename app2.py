"""
Streamlit Web UI for STTM RAG Chatbot


Wraps your existing rag.py pipeline in a Streamlit web interface.
Your rag.py functions (load_documents, chunk_text, build_vector_store,
retrieve, ask_claude, extract_table_name) are imported
and used EXACTLY as they are — no changes needed to rag.py.

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

from rag import (
    load_documents,           # Loads .txt, .md, .xlsx from docs/
    chunk_text,               # Splits documents into overlapping chunks
    build_vector_store,       # Stores chunks in ChromaDB with embeddings
    retrieve,                 # Queries ChromaDB for relevant chunks
    ask_claude,      # Sends query + context to Claude API
    extract_table_name,       # Detects table names in questions
    DOCS_DIR,                 # "docs" — where your STTM files live
    TOP_K,                    # How many chunks to retrieve (3)
    CHUNK_SIZE,               # Characters per chunk (800)
    CHUNK_OVERLAP,            # Overlap between chunks (100)
    IMPROVED_SYSTEM_PROMPT,   # Your domain-specific system prompt
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

#Section 1 PAGE CONFIGURATION

# st.set_page_config() MUST be the first Streamlit command in the script.
# If ANY st.* call comes before it, you get an error:
#   "set_page_config() can only be called once per app, and must be
#    called as the first Streamlit command in your script."

st.set_page_config(
    page_title = "STTM Assistant",
    page_icon ='📊',
    layout = 'wide',
    initial_sidebar_state= 'expanded',
)

#Section 2 caching rag pipeline 
# Streamlit reruns this script on EVERY interaction. Without caching:
#   1. User types a question → entire script reruns from line 1
#   2. load_documents() reads all Excel files from disk (~2 seconds)
#   3. chunk_text() splits into chunks (~0.5 seconds)
#   4. build_vector_store() rebuilds ChromaDB (~5 seconds)
#   5. THEN we can actually answer the question

@st.cache_resource(show_spinner="Loading documents and building vector store")
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
   
   #Step 1 load document from folder
   documents = load_documents(DOCS_DIR)

   if not documents:
      st.error(f"No documents found in `{DOCS_DIR}/` folder. "
                 f"Add .xlsx, .txt, or .md files and restart.")
      st.stop()

    #Step 2 chunking all documents
   all_chunks = []
   for doc in documents:
      chunks = chunk_text(doc['content'],doc['source'])
      all_chunks.extend(chunks)
      #Build vector store
   collection = build_vector_store(all_chunks)

   #Extract known table name
   all_meta = collection.get()
   known_tables = sorted(list(set(
      m.get('table_name',"")
      for m in all_meta['metadatas']
      if m.get('table_name') and m['table_name'].strip()
   )))

   return collection, known_tables, len(all_chunks), len(documents)

#Section 3 Session state initialisation
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

if "sources_log" not in st.session_state:
    st.session_state["sources_log"] = {} 

collection, known_tables, num_chunks, num_docs = init_pipeline()

#Section 5 Helper function
def extract_citation(chunks:list[dict]) ->list[dict]:
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
    In practice for  STTM data, distances cluster between 0.2–1.5:
      < 0.5  = Strong match (very relevant)  → 🟢 High
      0.5–1.0 = Moderate match               → 🟡 Medium
      > 1.0  = Weak match (probably noise)    → 🔴 Low

    distance might be None if the chunk didn't come from
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

def render_sources_detail(chunks:list[dict]):
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
    with st.expander(f"Source Detail ({len(chunks)} chunks)",expanded=False):
        for j,chunk in enumerate(chunks):
            table = chunk.get("table_name",'Unknown')
            doc_type = chunk.get("doc_type",'Unknown')
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
            relevance = round(1 - distance,3) if distance is not None else None
            relevance_str = f" | Relevance:{relevance}" if relevance is not None else ""
            st.markdown(f"**Chunk {j+1}:** `{table}` - {doc_type}{relevance_str}")

            #st.code() render text in monospace box - suitable for document
            #or code display
            preview = chunk.get("text","")[:300]
            if len(chunk.get("text", "")) > 300:
                preview += "..."
            st.code(preview,language=None)


# SECTION 6: SIDEBAR
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
    st.title("STTM Assistant")
    st.caption("Source to Target Mapping Reference Tool")
    # ─── Pipeline statistics ───
    # st.metric() creates a stat card with label + large number.
    # st.columns() arranges them side by side.
    #
    # STREAMLIT REFRESHER: st.columns()
    # ──────────────────────────────────
    # st.columns(2) returns two column objects.
    # col1, col2 = st.columns(2)  ← equal width
    # col1, col2 = st.columns([3, 1])  ← col1 is 3x wider
    st.subheader("Source Document Stats")
    col1,col2 = st.columns(2)
    with col1:
        st.metric("Documents",num_docs)
        st.metric("Queries",st.session_state['query_count'])
    with col2:
        st.metric("Chunks",num_chunks)
        st.metric("Tables",len(known_tables))

    st.divider()

    st.session_state["debug_mode"] = st.toggle(
        "Show Debug Info",
        value = st.session_state['debug_mode'],
        help = 'Show retrieved chunk, distances, timing for each query',
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
    st.subheader("Upload Documents")
    uploaded_file = st.file_uploader(
        "Add STTM files here",
        type = ['xlsx'],
        help ="Upload .xlsx (STTM workbooks)"
             "File is saved to docs/ and the pipeline rebuilds automatically.",
    )

    if uploaded_file is not None:
        save_path = os.path.join(DOCS_DIR,uploaded_file.name)

        with open(save_path,'wb') as f:
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

    with st.expander(f"Indexed Tables ({len(known_tables)})",expanded=False):
        for i,table in enumerate(known_tables):
            st.text(f"{i+1:2d}.{table}")

    st.divider()

    col1,col2 = st.columns(2)

    if col1.button("Clear Chat", use_container_width=True):
        st.session_state["messages"] = []
        st.session_state["sources_log"] = {}
        st.session_state["query_count"] = 0
        st.rerun()

    if col2.button("re-index",use_container_width=True,
                   help = 'Rebuild vector store from docs'):
        st.cache_resource.clear()
        st.rerun()
    st.divider()


    st.subheader("Current Config")
    st.code(
        f"CHUNK_SIZE = {CHUNK_SIZE}\n"
        f"CHUNK_OVERLAP = {CHUNK_OVERLAP}\n"
        f"TOP_K = {TOP_K}",
        language="python",
    )

#Main Chat Area
st.title("STTM Assistant")
st.caption(
    "Ask questions about F06 STTM "
    "columns, mappings, and data pipelines. "
    "Source citations shown under each answer."
)


# SECTION 8: RENDER CHAT HISTORY
# On every rerun, we re-render ALL past messages from session_state.
# Streamlit doesn't "remember" what it drew last time — it starts
# with a blank page and re-draws everything top-to-bottom.
#
# This is like a SQL dashboard that runs SELECT * FROM messages
# on every page load — the data is in the database (session_state),
# and the rendering is stateless.

for i,msg in enumerate(st.session_state['messages']):
    with st.chat_message(msg['role']):
        st.markdown(msg['content'])

        if msg["role"] =='assistant' and msg.get("citations"):
            citation_text = format_citation_badges(msg['citations'])
            if citation_text:
                st.caption(citation_text)

        # Show debug panel if debug mode is on
        if (
            st.session_state['debug_mode']
            and msg['role'] =='assistant'
            and msg.get('debug_info')
        ):
            debug = msg["debug_info"]
            with st.expander("Debug: Retrieved Chunks", expanded=False):
                if debug.get("timing"):
                    t = debug["timing"]
                    st.text(
                        f"Retrieval: {t['retrieval']:.2f}s | "
                        f"Generation: {t['generation']:.2f}s | "
                        f"Total: {t['total']:.2f}s"
                    )
                if debug.get("detected_table"):
                    st.text(f"Detected table: {debug['detected_table']}")
                #Each Chunk
                for j,chunk in enumerate(debug.get("chunks",[])):
                    dist_str = f" (d={chunk['distance']:.3f})" if chunk.get("distance") else ""
                    st.text(f"[{j+1}] {chunk.get('table_name', '?')} "
                            f"({chunk.get('doc_type', '?')}){dist_str}")
                    st.code(chunk.get("text", "")[:300], language=None)
        if msg["role"] == "assistant" and i in st.session_state['sources_log']:
            render_sources_detail(st.session_state["sources_log"][i])


# SECTION 9: HANDLE NEW USER INPUT
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
query = st.chat_input("Ask about Mapping information")


if query:
    st.session_state["messages"].append({
        "role": "user",
        "content": query,
    })

    with st.chat_message("user"):
        st.markdown(query)

    #Run RAG pipeline
    with st.chat_message("assistant"):
        with st.spinner("Searching document and generating answer..."):
            #detech table name
            detected = extract_table_name(query,known_tables)

            #Retrieve relevant chunks
            retrieval_start = time.time()
            chunks = retrieve(
                collection,
                query,
                table_name=detected,
                known_tables=known_tables,
            )
            retrieval_time = time.time() - retrieval_start

            #Generate answer via LLM
            # Right now, each question is independent — Claude has
            # no memory of previous questions in this session.
            # In Week 4, we'll replace this with ask_claude_with_memory()
            # that passes st.session_state["messages"] to the API,
            # enabling follow-up questions like:
            #   "Tell me about DIM_STORE"
            #   "What about its foreign keys?"  ← Claude knows "its" = DIM_STORE
            generation_start = time.time()
            try:
                answer = ask_claude(query,chunks)
            except Exception as e:
                answer = f"Error generation response: {e}"
            generation_time = time.time() - generation_start

            total_time = retrieval_time + generation_time

        #Extract citations and display
        citations = extract_citation(chunks)
        citation_text = format_citation_badges(citations)

        #Display answer
        st.markdown(answer)

        if citation_text:
            st.caption(citation_text)
        
        debug_info = {
            "detected_table":detected,
            "timing":{
                "retrieval":retrieval_time,
                "generation":generation_time,
                "total":total_time
            },
            "chunks": chunks,
        }
        #Show debug panel if enable
        if st.session_state['debug_mode']:
            with st.expander("Debug:Retrieved Chunks",expanded=False):
                t = debug_info['timing']
                st.text(
                    f"Retrieval: {t['retrieval']:.2f}s | "
                    f"Generation: {t['generation']:.2f}s | "
                    f"Total: {t['total']:.2f}s"
                )
                if detected:
                    st.text(f"Detected table: {detected}")
                for j, chunk in enumerate(chunks):
                    dist_str = (
                        f" (d={chunk['distance']:.3f})"
                        if chunk.get('distance')
                        else ""

                    )
                    st.text(
                        f"[{j+1}] {chunk.get('table_name','?')}"
                        f"({chunk.get('doc_type',"?")}){dist_str}"
                    )
                    st.code(chunk.get("text","")[:300],language=None)
        #Show detailed source panel                   
        render_sources_detail(chunks)

        #show timing as caption
        table_info = f" | Table: {detected}" if detected else ""
        st.caption(
            f" Retrieval: {retrieval_time:.2f}s | "
            f"Generation: {generation_time:.2f}s | "
            f"Total: {total_time:.2f}s"
            f"{table_info}"
        )


    #Save to session state
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

        

        
            



    








