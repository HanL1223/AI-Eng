"""
rag.py — RAG Chatbot with STTM-Aware Loading
==============================================
Drop this + sttm_loader.py into your rag-chatbot/ folder.
Put your STTM Excel file in docs/
Run: python rag.py

Changes from v2:
- Uses sttm_loader.py for smart Excel extraction
- Each entity becomes 2 docs (summary + columns) instead of 1 messy blob
- Still loads .txt and .md files normally
"""

import os
import glob
import anthropic
import chromadb

# ─────────────────────────────────────────────
# PYTHON REFRESHER: Importing from your own file
# ─────────────────────────────────────────────
# This imports the function from sttm_loader.py
# The file must be in the same folder as rag.py
from sttm_loader import load_sttm_workbook

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
DOCS_DIR = "docs"
CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "my_docs"
CHUNK_SIZE = 800                     # Bigger now — STTM docs are pre-structured
CHUNK_OVERLAP = 100
TOP_K = 3
MODEL = "claude-sonnet-4-5-20250929"


# ─────────────────────────────────────────────
# STEP 1: Load all documents
# ─────────────────────────────────────────────
def load_documents(docs_dir: str) -> list[dict]:
    """
    Load .txt, .md files normally.
    Load .xlsx files using the smart STTM loader.
    
    PYTHON REFRESHER: list.extend() vs list.append()
    ─────────────────────────────────────────────────
    append() adds ONE item:     [1,2].append(3)     → [1,2,3]
    extend() adds MANY items:   [1,2].extend([3,4]) → [1,2,3,4]
    
    load_sttm_workbook returns a LIST of documents,
    so we use extend() to add them all at once.
    """
    documents = []

    # --- Text and Markdown files ---
    for pattern in ["*.txt", "*.md"]:
        for filepath in glob.glob(os.path.join(docs_dir, pattern)):
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
            documents.append({
                "content": content,
                "source": os.path.basename(filepath),
            })
            print(f"  📄 Loaded: {os.path.basename(filepath)} ({len(content)} chars)")

    # --- Excel files (using smart STTM loader) ---
    for filepath in glob.glob(os.path.join(docs_dir, "*.xlsx")):
        print(f"\n  📊 Loading Excel: {os.path.basename(filepath)}")
        excel_docs = load_sttm_workbook(filepath)
        documents.extend(excel_docs)  # extend, not append!

    return documents


# ─────────────────────────────────────────────
# STEP 2: Chunk documents
# ─────────────────────────────────────────────
def chunk_text(text: str, source: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[dict]:
    """
    Split text into overlapping chunks with metadata.
    
    PYTHON REFRESHER: Parsing info from strings
    ─────────────────────────────────────────────
    We extract table_name and doc_type from the source string.
    The STTM loader creates sources like:
        "STTM.xlsx__Fact_Store_Inventory_Intra__summary"
        "STTM.xlsx__Fact_Store_Inventory_Intra__columns"
    
    str.split("__") breaks on double underscore:
        "a__b__c".split("__") → ["a", "b", "c"]
    """
    # Parse metadata from source name
    parts = source.split("__")
    
    if len(parts) >= 3:
        # From STTM loader: "filename__SheetName__summary" or "__columns"
        table_name = parts[1].strip()
        doc_type = parts[2]  # "summary" or "columns"
    elif len(parts) == 2:
        table_name = parts[1].strip()
        doc_type = "unknown"
    else:
        # Regular text file: "DIM_PRODUCT.txt"
        table_name = os.path.splitext(source)[0]
        doc_type = "text"

    # Guess table type from naming convention
    table_type = "unknown"
    upper = table_name.upper()
    # ─────────────────────────────────────────────────────────
    # PYTHON REFRESHER: elif chains
    # ─────────────────────────────────────────────────────────
    # Python doesn't have switch/case (well, it has match/case in 3.10+
    # but elif chains are still more common in practice)
    if upper.startswith("FACT") or upper.startswith("FACT_"):
        table_type = "fact"
    elif upper.startswith("DIM") or upper.startswith("DIM_"):
        table_type = "dimension"
    elif upper.startswith("BRIDGE") or upper.startswith("BRIDG"):
        table_type = "bridge"
    elif "control" in upper.lower():
        table_type = "control"

    # ─────────────────────────────────────────────────────────
    # Chunking logic
    # ─────────────────────────────────────────────────────────
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]

        # Try to break at a clean boundary
        if end < len(text):
            # ─────────────────────────────────────────────────
            # PYTHON REFRESHER: rfind() vs find()
            # ─────────────────────────────────────────────────
            # find() searches LEFT to RIGHT → first match
            # rfind() searches RIGHT to LEFT → last match
            # We want the LAST newline in the chunk so we break
            # at the end of a line, not the beginning
            last_newline = chunk.rfind("\n")
            last_period = chunk.rfind(". ")
            break_point = max(last_newline, last_period)
            if break_point > chunk_size * 0.3:
                chunk = chunk[: break_point + 1]
                end = start + break_point + 1

        stripped = chunk.strip()
        if stripped:  # Don't create empty chunks
            chunks.append({
                "text": stripped,
                "source": source,
                "chunk_index": len(chunks),
                "table_name": table_name.upper(),
                "table_type": table_type,
                "doc_type": doc_type,   # "summary", "columns", or "text"
            })

        start = end - overlap

    return chunks


# ─────────────────────────────────────────────
# STEP 3: Store in ChromaDB
# ─────────────────────────────────────────────
def build_vector_store(chunks: list[dict]) -> chromadb.Collection:
    """Store chunks with metadata in ChromaDB."""
    client = chromadb.PersistentClient(path=CHROMA_DIR)

    # ─────────────────────────────────────────────────────────
    # PYTHON REFRESHER: try/except (exception handling)
    # ─────────────────────────────────────────────────────────
    # try: runs the code
    # except Exception: catches ANY error and runs the fallback
    # We use this to delete the old collection (it might not exist yet)
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass  # "pass" means "do nothing" — it's a no-op placeholder

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    # ─────────────────────────────────────────────────────────
    # PYTHON REFRESHER: Building parallel lists
    # ─────────────────────────────────────────────────────────
    # ChromaDB wants three separate lists: ids, documents, metadatas
    # Each position corresponds to the same chunk:
    #   ids[0], documents[0], metadatas[0] → chunk 0
    #   ids[1], documents[1], metadatas[1] → chunk 1
    ids = []
    documents = []
    metadatas = []

    for i, chunk in enumerate(chunks):
        ids.append(f"chunk_{i}")
        documents.append(chunk["text"])
        metadatas.append({
            "source": chunk["source"],
            "chunk_index": chunk["chunk_index"],
            "table_name": chunk["table_name"],
            "table_type": chunk["table_type"],
            "doc_type": chunk["doc_type"],
        })

    collection.add(ids=ids, documents=documents, metadatas=metadatas)
    print(f"  💾 Stored {len(chunks)} chunks in ChromaDB")
    return collection


# ─────────────────────────────────────────────
# STEP 4: Retrieve
# ─────────────────────────────────────────────
def retrieve(collection: chromadb.Collection, query: str, top_k: int = TOP_K,
             table_name: str = None, table_type: str = None) -> list[dict]:
    """Find relevant chunks, optionally filtered."""

    # ─────────────────────────────────────────────────────────
    # PYTHON REFRESHER: Building dicts conditionally
    # ─────────────────────────────────────────────────────────
    # Instead of a chain of if/elif, we build the filter dict
    # only if we have values to filter on
    where_filter = None
    if table_name and table_type:
        where_filter = {"$and": [
            {"table_name": {"$eq": table_name}},
            {"table_type": {"$eq": table_type}},
        ]}
    elif table_name:
        where_filter = {"table_name": {"$eq": table_name}}
    elif table_type:
        where_filter = {"table_type": {"$eq": table_type}}

    # ─────────────────────────────────────────────────────────
    # PYTHON REFRESHER: ** (dictionary unpacking)
    # ─────────────────────────────────────────────────────────
    # function(**{"a": 1, "b": 2}) is the same as function(a=1, b=2)
    # We build query_params as a dict then unpack it
    query_params = {"query_texts": [query], "n_results": top_k}
    if where_filter:
        query_params["where"] = where_filter

    try:
        results = collection.query(**query_params)
    except Exception:
        # Fallback to unfiltered if filter fails
        results = collection.query(query_texts=[query], n_results=top_k)

    # ─────────────────────────────────────────────────────────
    # PYTHON REFRESHER: range(len()) vs enumerate()
    # ─────────────────────────────────────────────────────────
    # Both work for indexed iteration. range(len()) when you
    # need the index to access multiple parallel lists:
    retrieved = []
    for i in range(len(results["ids"][0])):
        retrieved.append({
            "text": results["documents"][0][i],
            "source": results["metadatas"][0][i]["source"],
            "table_name": results["metadatas"][0][i].get("table_name", ""),
            "table_type": results["metadatas"][0][i].get("table_type", ""),
            "doc_type": results["metadatas"][0][i].get("doc_type", ""),
            "distance": results["distances"][0][i] if results.get("distances") else None,
        })

    return retrieved


# ─────────────────────────────────────────────
# STEP 5: Auto-detect table name
# ─────────────────────────────────────────────
def extract_table_name(query: str, known_tables: list[str]) -> str | None:
    """
    Detect table names mentioned in the user's question.
    
    PYTHON REFRESHER: sorted() with key= parameter
    ─────────────────────────────────────────────────
    sorted([3,1,2]) → [1,2,3]
    sorted(["b","a"], key=len) → sorts by string length
    reverse=True → longest first
    
    We sort by length descending so "FACT_STORE_INVENTORY_INTRA"
    matches before "FACT_STORE_INVENTORY"
    """
    query_upper = query.upper()
    # ─────────────────────────────────────────────────────────
    # PYTHON REFRESHER: Also try without underscores
    # ─────────────────────────────────────────────────────────
    # Users might type "fact store inventory" instead of "FACT_STORE_INVENTORY"
    query_no_spaces = query_upper.replace(" ", "_")

    for table in sorted(known_tables, key=len, reverse=True):
        if table in query_upper or table in query_no_spaces:
            return table
    return None


# ─────────────────────────────────────────────
# STEP 6: Ask Claude
# ─────────────────────────────────────────────
def ask_claude(query: str, context_chunks: list[dict]) -> str:
    """Send query + retrieved context to Claude."""
    # ─────────────────────────────────────────────────────────
    # PYTHON REFRESHER: List comprehension to build strings
    # ─────────────────────────────────────────────────────────
    context_parts = []
    for chunk in context_chunks:
        # Build a useful label from metadata
        label_parts = []
        if chunk.get("table_name"):
            label_parts.append(chunk["table_name"])
        if chunk.get("doc_type") and chunk["doc_type"] != "text":
            label_parts.append(chunk["doc_type"])
        label = " — ".join(label_parts) if label_parts else chunk["source"]

        context_parts.append(f"[Source: {label}]\n{chunk['text']}")

    context = "\n\n---\n\n".join(context_parts)

    client = anthropic.Anthropic()

    message = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        system="""You are a data warehouse documentation assistant for Sigma Healthcare.

You answer questions about tables, columns, mappings, and data pipelines 
in Sigma's Snowflake data warehouse.

Context:
- The warehouse follows a Bronze → Platinum → Gold layer architecture
- Source systems include SAP (via CDS Views), MyPOS, and other operational systems
- Data flows: Source → Azure Data Factory → ADLS Gen2 → dbt → Snowflake
- Tables follow naming conventions: DIM_ (dimensions), FACT_ (facts), BRIDGE_ (bridges)
- Documents include table summaries (grain, source, refresh) and column mappings (column names, types, source columns)

Rules:
- ONLY use information from the provided context documents to answer
- If the context doesn't contain the answer, say "I don't have that information in the loaded documents"
- Always cite which table/source the information came from
- When describing a table, include: grain, key columns, source system, and refresh cadence if available
- When listing columns, format them clearly with their types and sources
- Be concise and direct — the user is a data engineer""",
        messages=[
            {
                "role": "user",
                "content": f"""Context from documents:

{context}

---

Question: {query}""",
            }
        ],
    )

    return message.content[0].text


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    print("\n🔧 RAG Chatbot — STTM-Aware Version")
    print("=" * 45)

    # ─────────────────────────────────────────────────────────
    # PYTHON REFRESHER: os.environ and .env loading
    # ─────────────────────────────────────────────────────────
    # os.environ is a dict of all environment variables
    # .get() returns None instead of raising KeyError
    if not os.environ.get("ANTHROPIC_API_KEY"):
        env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
        if os.path.exists(env_path):
            with open(env_path) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        # ─────────────────────────────────────
                        # PYTHON REFRESHER: str.partition()
                        # ─────────────────────────────────────
                        # "KEY=VALUE".partition("=") → ("KEY", "=", "VALUE")
                        # Better than split("=") when value contains "="
                        key, _, value = line.partition("=")
                        os.environ[key.strip()] = value.strip()
        else:
            print("\n❌ No ANTHROPIC_API_KEY found!")
            print("   export ANTHROPIC_API_KEY='sk-ant-...'")
            return

    if not os.path.exists(DOCS_DIR):
        print(f"\n❌ No '{DOCS_DIR}/' folder found! mkdir {DOCS_DIR}")
        return

    # Load
    print("\n📂 Loading documents...")
    documents = load_documents(DOCS_DIR)
    if not documents:
        print(f"  ⚠️  No files found in {DOCS_DIR}/")
        return
    print(f"\n  📚 Total: {len(documents)} documents loaded")

    # Chunk
    print("\n✂️  Chunking...")
    all_chunks = []
    for doc in documents:
        chunks = chunk_text(doc["content"], doc["source"])
        all_chunks.extend(chunks)
    print(f"  📦 {len(all_chunks)} chunks from {len(documents)} documents")

    # Store
    print("\n🗄️  Building vector store...")
    collection = build_vector_store(all_chunks)

    # Get known tables
    # ─────────────────────────────────────────────────────────
    # PYTHON REFRESHER: set() for unique values
    # ─────────────────────────────────────────────────────────
    # set() removes duplicates: set([1,2,2,3]) → {1,2,3}
    # list(set(...)) converts back to a list
    all_meta = collection.get()
    known_tables = sorted(list(set(
        m.get("table_name", "")
        for m in all_meta["metadatas"]
        if m.get("table_name") and m["table_name"].strip()
    )))
    print(f"  📋 {len(known_tables)} tables indexed")

    # Interactive loop
    print("\n✅ Ready! Ask questions about your STTM documents.")
    print("   Commands: 'debug' | 'tables' | 'quit'\n")

    debug_mode = False

    while True:
        try:
            query = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 Bye!")
            break

        if not query:
            continue
        if query.lower() == "quit":
            print("👋 Bye!")
            break
        if query.lower() == "debug":
            debug_mode = not debug_mode
            print(f"  🔍 Debug: {'ON' if debug_mode else 'OFF'}")
            continue
        if query.lower() == "tables":
            # ─────────────────────────────────────────────────
            # PYTHON REFRESHER: Formatted printing in columns
            # ─────────────────────────────────────────────────
            # f"{var:<30}" left-aligns in a 30-char field
            for i, t in enumerate(known_tables):
                print(f"  {i+1:2d}. {t}")
            continue

        # Auto-detect table
        detected = extract_table_name(query, known_tables)
        if debug_mode and detected:
            print(f"  🎯 Detected: {detected}")

        # Retrieve
        chunks = retrieve(collection, query, table_name=detected)

        if debug_mode:
            print(f"\n  📎 {len(chunks)} chunks retrieved:")
            for i, c in enumerate(chunks):
                dist = f" d={c['distance']:.3f}" if c["distance"] else ""
                print(f"    [{i+1}] {c['table_name']} ({c['doc_type']}){dist}")
                print(f"        {c['text'][:100]}...")
            print()

        # Answer
        try:
            answer = ask_claude(query, chunks)
            print(f"\nClaude: {answer}\n")
        except anthropic.AuthenticationError:
            print("\n❌ Invalid API key.")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")


if __name__ == "__main__":
    main()