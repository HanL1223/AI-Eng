"""ragcopy.py — RAG Chatbot with STTM-Aware Loading
Drop this + sttm_loader.py into your rag-chatbot/ folder.
Put  STTM Excel file in docs/
Run: python rag.py
"""

IMPROVED_SYSTEM_PROMPT = """
You are a data warehouse documentation assistant for Sigma Healthcare.
You answer questions about tables, columns, mappings, and data pipelines
in Sigma's Snowflake data warehouse.
 
ARCHITECTURE CONTEXT:
─────────────────────
The warehouse follows a layered architecture:
- Source Systems: SAP (via CDS Views), MyPOS, PDB08, PDB15, TDB08AX2012
- Extraction: Azure Data Factory pulls from sources to ADLS Gen2
- Bronze Layer: Raw data landed 1:1 from source (minimal transformation)
- Platinum Layer: Cleaned, conformed, business-rule-applied via dbt
- Gold Layer: Star schema for reporting (facts, dimensions, bridges)
- Reporting: Power BI reads from Gold layer in Snowflake
 
TABLE NAMING CONVENTIONS:
─────────────────────────
- FACT_* : Fact tables (transactional data, measures, foreign keys to dimensions)
- DIM_*  : Dimension tables (descriptive attributes, surrogate keys SK_*)
- BRIDGE_* : Bridge tables (resolve many-to-many relationships)
- Surrogate keys: SK_*_KEY (e.g., SK_STORE_KEY, SK_PRODUCT_KEY)
- Foreign keys in facts point to dimension surrogate keys
 
ANSWERING RULES:
────────────────
1. ONLY use information from the provided context documents to answer.
2. If the context doesn't contain the answer, say:
   "I don't have that information in the loaded documents."
   Do NOT guess, infer, or make up information.
3. Always cite which table/document the information came from.
4. When describing a table, include: grain, key columns, source system,
   and refresh cadence if available in the context.
5. When listing columns, use compact format:
   ColumnName (Type) ← SourceSystem.SourceTable.SourceColumn
6. Be concise and direct — the user is a data engineer.
 
CROSS-TABLE REASONING:
──────────────────────
- When asked about relationships between tables, look for:
  * Foreign key columns (SK_*_KEY, FK_*) that suggest joins
  * Shared source systems or source tables
  * Columns with matching names across tables
- When asked "which tables use X", scan ALL provided chunks for references
  to X, not just chunks explicitly tagged as X.
- Fact tables connect to dimensions via surrogate keys (SK_ prefix).
- If you see a column like SK_STORE_KEY in a fact table, it joins to DIM_STORE.
"""

import os
import glob
import anthropic
import chromadb
import re
# Import sttm loading function to handle complex excel
from sttm_loader import load_sttm_workbook


# CONFIG
DOCS_DIR = "docs"
CHROMA_DIR = "chroma_db"
COLLECTION_NAME = 'my_docs'
CHUNK_SIZE = 800    
CHUNK_OVERLAP = 100
TOP_K = 3
MODEL = "claude-sonnet-4-5-20250929"




#Loading all documents

def load_documents(docs_dir:str) -> list[str]:
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

    for patterns in ["*.txt","*.md"]:
        for filepath in glob.glob(os.path.join(docs_dir,patterns)):
            with open(filepath,'r',encoding='utf-8') as f:
                content = f.read()
            documents.append(
                {
                    "content":content,
                    "source":os.path.basename(filepath),
                }
            )
            print(f"Loaded {os.path.basename(filepath)}: ({len(content)} chars)")
    #Excel files using sttm loader
    for filepath in glob.glob(os.path.join(docs_dir,"*.xlsx")):
        print(f"Loading Excel: {os.path.basename(filepath)})")
        excel_docs = load_sttm_workbook(filepath)
        documents.extend(excel_docs)

    return documents

#Chunk Documents
def chunk_text(text:str,source:str,chunk_size:int=CHUNK_SIZE,overlap:int = CHUNK_OVERLAP) -> list[dict]:
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
    #parsee metdata from source name
    parts = source.split("__")

    if len(parts) >=3:
        #From STTM loader: "filename__SheetName__summary" or "__columns"
        table_name = parts[1].strip()
        doc_type = parts[2] #summary or columns
    elif len(parts) == 2:
        table_name = parts[1].strip()
        doc_type = "unknown"
    else:
        table_name = os.path.splitext(source)[0]
        doc_type = "text"

    # Guess table type from naming convention
    # Sttm source only got fact, dim and bridge tables
    table_type = "unknown"
    upper = table_name.upper() #short for table_name.upper()

    if upper.startswith("FACT") or upper.startswith("FACT_"):
        table_type = "fact"
    elif upper.startswith("DIM") or upper.startswith("DIM_"):
        table_type = "dimension"
    elif upper.startswith("BRIDGE") or upper.startswith("BRIDG"):
        table_type = "bridge"
    elif "control" in upper.lower():
        table_type = "control"

    #Chunking logic
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        

        #Try to break at a clean boundary,to increase accuracy
        if end < len(text):

            # We want the LAST newline in the chunk so we break
            # at the end of a line, not the beginning to find the end boundary 
            last_newline = chunk.rfind("\n")
            last_period = chunk.rfind(". ")
            #rfine return location in str, breakpoint find the last match from left
            break_point = max(last_newline,last_period)
            #we use 0.3 because if the newline is too early, you would get tiny chunks, which is bad for embeddings.
            if break_point > chunk_size * 0.3:
                    chunk = chunk[: break_point + 1]
                    end = start + break_point + 1
        stripped = chunk.strip()
        if stripped:
            chunks.append(
                {
                    "text":stripped,
                    "source":source,
                    "chunk_index":len(chunks),
                    "table_name":table_name.upper(),
                    "table_type":table_type,
                    "doc_type":doc_type,

                }
            )
        start = end - overlap
    return chunks


#Store in ChromaDB/Building vector store

def build_vector_store(chunks:list[dict])->chromadb.Collection:
    """
    Store chunks with metadata in Chromadb
    """
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.get_or_create_collection(
        name = COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    ## Each position corresponds to the same chunk,e.g.:
    #   ids[0], documents[0], metadatas[0] → chunk 0
    #   ids[1], documents[1], metadatas[1] → chunk 1
    ids =[]
    documents = []
    metadatas = []

    for i,chunk in enumerate(chunks):
        ids.append(f"chunk_{i}")
        documents.append(chunk["text"])
        metadatas.append({
            "source": chunk["source"],
            "chunk_index": chunk["chunk_index"],
            "table_name": chunk["table_name"],
            "table_type": chunk["table_type"],
            "doc_type": chunk["doc_type"],
        })
    collection.add(ids = ids,documents=documents,metadatas=metadatas)
    print(f"Stored {len(chunks)} chunks in ChromaDB")
    return collection


#Retrieve
def retrieve(
    collection,       # Your ChromaDB collection
    query: str,                           # The user's question
    top_k: int = 3,                       # How many chunks to retrieve
    table_name: str = None,               # Detected table name (from extract_table_name)
    table_type: str = None,               # Optional table type filter
    known_tables: list = None,            # List of all known table names
) -> list[dict]:
    """
    Find relevant chunks with SMART filtering based on query type.
 
    This is the upgraded version of retrieve() that handles both
    single-table and cross-entity questions correctly.
 
    THE KEY INSIGHT
    ───────────────
    Your original retrieve() always applied the table filter when a
    table name was detected. This was GREAT for single-table questions:
      Q: "What is the grain of FACT_STORE_INVENTORY_INTRA?"
      → Filter to FACT_STORE_INVENTORY_INTRA chunks → precise results ✅
 
    But it BROKE cross-entity questions:
      Q: "Which dimensions are referenced by FACT_STORE_INVENTORY_INTRA?"
      → Filter to FACT_STORE_INVENTORY_INTRA chunks → only sees that table ❌
      → Claude can't see DIM_STORE, DIM_PRODUCT, etc.
 
    The fix: classify the query FIRST, then decide whether to filter.
 
    PYTHON REFRESHER: Optional parameters with None default
    ───────────────────────────────────────────────────────
    table_name: str = None means "this parameter is optional".
    Inside the function, we check `if table_name:` to see if it was provided.
    None is "falsy" in Python — `if None:` evaluates to False.
 
    PARAMETERS
    ──────────
    collection:    ChromaDB collection (your vector store)
    query:         The user's question text
    top_k:         Number of chunks to retrieve
    table_name:    Table name detected by extract_table_name() (or None)
    table_type:    Optional: "fact", "dimension", "bridge"
    known_tables:  List of all table names (used for multi-table retrieval)
 
    RETURNS
    ───────
    List of chunk dicts, each with:
      text, source, table_name, table_type, doc_type, distance, query_type
    """
 
    # Step 1: Classify the query
    # ─────────────────────────────────────────────
    query_type = classify_query(query)
 
    # Step 2: Build the retrieval strategy based on classification
    # ─────────────────────────────────────────────
    where_filter = None
 
    if query_type == "single_table":
        # ─────────────────────────────────────────
        # SINGLE TABLE: Apply filter (current behavior)
        # ─────────────────────────────────────────
        # This is your existing logic — unchanged.
        # When we know the user wants info about ONE table,
        # filtering makes retrieval faster and more precise.
        if table_name and table_type:
            where_filter = {"$and": [
                {"table_name": {"$eq": table_name}},
                {"table_type": {"$eq": table_type}},
            ]}
        elif table_name:
            where_filter = {"table_name": {"$eq": table_name}}
        elif table_type:
            where_filter = {"table_type": {"$eq": table_type}}
 
    elif query_type == "cross_entity":
        # ─────────────────────────────────────────
        # CROSS ENTITY: NO filter — let semantic search decide
        # ─────────────────────────────────────────
        # By setting where_filter = None, ChromaDB searches
        # ALL chunks based on vector similarity to the query.
        #
        # "Which dimensions are referenced by FACT_STORE_INVENTORY?"
        # → ChromaDB finds chunks that are semantically similar:
        #   - FACT_STORE_INVENTORY summary (mentions dimension keys)
        #   - FACT_STORE_INVENTORY columns (lists SK_STORE_KEY, SK_PRODUCT_KEY)
        #   - DIM_STORE summary (if relevant)
        #   - DIM_PRODUCT summary (if relevant)
        #
        # We also INCREASE top_k for cross-entity questions because
        # we need chunks from MULTIPLE tables, not just one.
        #
        # DESIGN DECISION: top_k * 2 for cross-entity
        # ─────────────────────────────────────────────
        # If the user's TOP_K is 3, a single-table query gets 3 chunks
        # (all about one table — usually enough). But a cross-entity
        # query might need 2 chunks per table × 3 tables = 6 chunks
        # to have enough context. So we double it.
        #
        # TRADEOFF: More chunks = more context but more cost and noise.
        # We cap at 10 to avoid excessive API costs.
        top_k = min(top_k * 2, 10)
        where_filter = None  # ← THE KEY CHANGE: no filter
 
        # OPTIONAL: Print debug info so you can see the classification
        print(f"  [SMART RETRIEVE] Query classified as CROSS_ENTITY")
        print(f"  [SMART RETRIEVE] Removed table filter, TOP_K boosted to {top_k}")
 
    # Step 3: Execute the ChromaDB query
    # ─────────────────────────────────────────────
    # This is the same query logic as your original retrieve(),
    # but now where_filter might be None for cross-entity questions.
    #
    # PYTHON REFRESHER: ** dictionary unpacking
    # ─────────────────────────────────────────
    # query_params = {"query_texts": [...], "n_results": 3}
    # collection.query(**query_params)
    # is equivalent to:
    # collection.query(query_texts=[...], n_results=3)
    #
    # The ** "unpacks" the dictionary into keyword arguments.
    # This lets us conditionally add the 'where' key only when
    # we have a filter.
    query_params = {
        "query_texts": [query],
        "n_results": top_k,
    }
    if where_filter:
        query_params["where"] = where_filter
 
    try:
        results = collection.query(**query_params)
    except Exception:
        # ─────────────────────────────────────────────
        # FALLBACK: If filtered query fails (e.g., no matching docs),
        # retry WITHOUT the filter. This handles edge cases like:
        # - Table name detected but not in ChromaDB
        # - Filter + top_k combination returns nothing
        # ─────────────────────────────────────────────
        print(f"  [SMART RETRIEVE] Filtered query failed, retrying without filter")
        results = collection.query(
            query_texts=[query],
            n_results=top_k,
        )
 
    # Step 4: Build the results list
    # ─────────────────────────────────────────────
    # Same as your original retrieve() — extract text, metadata,
    # and distance from ChromaDB's response format.
    #
    # ADDITION: We include "query_type" in each result so you can
    # see in the eval CSV whether the query was classified correctly.
    retrieved = []
    for i in range(len(results["ids"][0])):
        retrieved.append({
            "text": results["documents"][0][i],
            "source": results["metadatas"][0][i]["source"],
            "table_name": results["metadatas"][0][i].get("table_name", ""),
            "table_type": results["metadatas"][0][i].get("table_type", ""),
            "doc_type": results["metadatas"][0][i].get("doc_type", ""),
            "distance": results["distances"][0][i] if results.get("distances") else None,
            "query_type": query_type,   # ← NEW: helps debug in eval results
        })
 
    return retrieved

def classify_query(query: str) -> str:
    """
    Classify whether a question needs single-table or cross-entity retrieval.
 
    PYTHON REFRESHER: String pattern matching with 'in' operator
    ─────────────────────────────────────────────────────────────
    "which tables" in "Which tables use DIM_DATE?" → True
    
    We check for phrases that SIGNAL cross-entity intent.
    This is a simple keyword approach — in production, you might
    use an LLM to classify (query routing), but keywords work well
    for a structured domain like data warehousing.
 
    RETURNS
    ───────
    "cross_entity" → DON'T filter by table name during retrieval
    "single_table" → DO filter by table name (current behavior)
 
    DESIGN DECISION: Conservative classification
    ─────────────────────────────────────────────
    If we're unsure, we default to "single_table" (current behavior).
    False positives (classifying single as cross) just remove the filter,
    which makes retrieval slightly less precise but still works.
    False negatives (classifying cross as single) BREAK cross-entity
    questions entirely — so we'd rather have false positives.
    
    ANALOGY
    ───────
    Think of this like a librarian deciding how to search:
    - "What's the grain of FACT_SALES?" → Go to the FACT_SALES section
    - "Which tables reference DIM_DATE?" → Search the WHOLE library
    """
    query_lower = query.lower()
 
    # ─────────────────────────────────────────────────────────
    # CROSS-ENTITY SIGNALS
    # ─────────────────────────────────────────────────────────
    # These phrases indicate the user wants information ACROSS tables.
    #
    # PYTHON REFRESHER: Lists of strings as lookup tables
    # ─────────────────────────────────────────────────────
    # We use a list instead of a long if/elif chain because:
    #   1. Easier to add new signals (just append to the list)
    #   2. Easier to read (all signals in one place)
    #   3. The any() call below checks all of them in one line
    cross_entity_signals = [
        # Relationship questions
        "which tables",         # "Which tables use DIM_DATE?"
        "which fact",           # "Which fact tables reference..."
        "which dim",            # "Which dimension tables..."
        "which bridge",         # "Which bridge tables..."
        "referenced by",        # "...dimensions referenced by FACT_X"
        "used by",              # "...used by which tables"
        "related to",           # "How is X related to Y?"
        "relationship",         # "What is the relationship between..."
        "foreign key",          # "What foreign keys does X have?"
        "joins to",             # "What does X join to?"
        "linked to",            # "Which tables are linked to..."
 
        # Comparison questions
        "compare",              # "Compare the grain of X and Y"
        "difference between",   # "What's the difference between X and Y?"
        "different from",       # "How is X different from Y?"
        "vs",                   # "FACT_X vs FACT_Y"
        "versus",               # "FACT_X versus FACT_Y"
 
        # Lineage/flow questions
        "lineage",              # "What is the data lineage for..."
        "data flow",            # "What is the data flow..."
        "pipeline",             # "Describe the pipeline for..."
        "upstream",             # "What are the upstream sources..."
        "downstream",           # "What downstream tables use..."
 
        # Aggregation questions
        "all tables",           # "List all tables that..."
        "every table",          # "Does every table have..."
        "across",               # "...across all dimensions"
        "how many tables",      # "How many tables use SAP?"
    ]
 
    # ─────────────────────────────────────────────────────────
    # PYTHON REFRESHER: any() with generator expression
    # ─────────────────────────────────────────────────────────
    # any(signal in query_lower for signal in cross_entity_signals)
    #
    # This is equivalent to:
    #   found = False
    #   for signal in cross_entity_signals:
    #       if signal in query_lower:
    #           found = True
    #           break
    #
    # any() short-circuits: it stops as soon as it finds a True.
    # This makes it efficient for long lists — no need to check
    # all 20+ signals if the first one matches.
 
    if any(signal in query_lower for signal in cross_entity_signals):
        return "cross_entity"
 
    # ─────────────────────────────────────────────────────────
    # MULTI-TABLE DETECTION
    # ─────────────────────────────────────────────────────────
    # If the question mentions TWO OR MORE table names,
    # it's probably a comparison or relationship question.
    #
    # Examples:
    #   "Compare FACT_SALES_ORDER and FACT_STORE_INVENTORY"
    #   → Mentions 2 tables → cross_entity
    #
    # We count how many table-like patterns appear in the query.
    # Table patterns: FACT_*, DIM_*, BRIDGE_*
    #
    # PYTHON REFRESHER: re.findall() returns all matches
    # ──────────────────────────────────────────────────
    # re.findall(r"pattern", text) → ["match1", "match2", ...]
    # The r"" is a raw string (backslashes are literal).
    # \b means "word boundary" — ensures we match whole words:
    #   \bFACT\b matches "FACT" but not "FACTORY"
    # \w+ means "one or more word characters" (letters, digits, underscore)
 
    table_pattern = r'\b(?:FACT|DIM|BRIDGE)[_\s]\w+'
    table_mentions = re.findall(table_pattern, query.upper())
 
    # ─────────────────────────────────────────────────────────
    # PYTHON REFRESHER: set() to count unique items
    # ─────────────────────────────────────────────────────────
    # set(["FACT_SALES", "FACT_SALES", "DIM_DATE"]) → {"FACT_SALES", "DIM_DATE"}
    # len(set(...)) gives the count of UNIQUE tables mentioned.
    # We use >= 2 because a comparison needs at least two tables.
    unique_tables = set(table_mentions)
    if len(unique_tables) >= 2:
        return "cross_entity"
 
    # Default: single table question
    return "single_table"



#Table Name auto detection

def extract_table_name(query:str, known_tables:list[str] ) -> str | None :
    """
    Detect table names mentioned in the user's question.
    
    Sort by length descending so "FACT_STORE_INVENTORY_INTRA"
    matches before "FACT_STORE_INVENTORY"

    """
    query_upper = query.upper()
    query_no_spaces = query_upper.replace(" ","_") #for understand input like fact sales order vs fact_sales_order

    for table in sorted(known_tables,key=len,reverse=True):
        if table in query_upper or table in query_no_spaces:
            return table
    return None


#Claude API

#First input to Claude

def ask_claude(query: str, context_chunks: list[dict]) -> str:
    """
    Send query and retrieved context to Claude with the IMPROVED system prompt.
 
    WHAT CHANGED FROM YOUR ORIGINAL ask_claude()
    ─────────────────────────────────────────────
    1. Uses IMPROVED_SYSTEM_PROMPT (module-level constant)
       instead of an inline prompt string
    2. Adds query_type info to the context (so Claude knows
       whether this is a single-table or cross-entity question)
    3. Same API call structure — no other changes
 
    PYTHON REFRESHER: import at function level (conditional import)
    ──────────────────────────────────────────────────────────────
    We import anthropic here. In your rag.py, it's already imported
    at the top of the file. When you copy this function into rag.py,
    you won't need the import line — it's already available.
    """
    import anthropic
 
    # Build context string from chunks (same as your original)
    # ─────────────────────────────────────────────────────────
    context_parts = []
    for chunk in context_chunks:
        label_parts = []
        if chunk.get("table_name"):
            label_parts.append(chunk["table_name"])
        if chunk.get("doc_type") and chunk["doc_type"] != "text":
            label_parts.append(chunk["doc_type"])
        label = " - ".join(label_parts) if label_parts else chunk["source"]
        context_parts.append(f"[Source: {label}]\n{chunk['text']}")
 
    context = "\n\n---\n\n".join(context_parts)
 
    # API call with improved prompt
    # ─────────────────────────────────────────────────────────
    client = anthropic.Anthropic()
    message = client.messages.create(
        model="claude-sonnet-4-5-20250929",
        max_tokens=1024,
        system=IMPROVED_SYSTEM_PROMPT,     # ← THE KEY CHANGE
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


#Main
def main():
    print("STTM Assistant")
    print("="*45)

    if not os.environ.get("ANTHROPIC_API_KEY"):
        env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
        if os.path.exists(env_path):
            with open(env_path) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        key,_,value = line.partition("=")
                        os.environ[key.strip()] = value.strip()
        else:
            print("API Key Not Found")
            print("exprot/$env = ANTHROPIC_API_KEY = 'API KEY HERE' ")
            return

    if not os.path.exists(DOCS_DIR):
        print(f"\n No '{DOCS_DIR}/' folder found! mkdir {DOCS_DIR}")
        return
        
    #Load Document/file
    print("Loading raw documents")
    documents = load_documents(DOCS_DIR)
    if not documents:
        print(f"No file found in {DOCS_DIR}")
        return
    print(f"Total {len(documents)} documents loaded")

    #Chunk
    print("Chunking")
    all_chunks = []
    for doc in documents:
        chunks = chunk_text(doc["content"],doc['source'])
        all_chunks.extend(chunks)
    print(f"{len(all_chunks)} chunk from {len(documents)} documents")

    #Store
    print("Building vector store")
    collection = build_vector_store(all_chunks)


    #Get known table
    all_meta = collection.get()
    known_tables = sorted(list(set(
        m.get("table_name","")
        for m in all_meta["metadatas"]
        if m.get("table_name") and m['table_name'].strip()
    )))
    print(f"  {len(known_tables)} tables indexed")

    #Interactive loop/UI
    print("\n Ready! Ask questions about your STTM documents.")
    print("   Commands: 'debug' | 'tables' | 'quit'\n")

    debug_mode = False

    while True:
        try:
            query = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n Bye!")
            break
        if not query:
            continue
        if query.lower() == "quit":
            break
        if query.lower == 'debug':
            debug_mode = not debug_mode
            print(f"Debug {'ON' if debug_mode else 'OFF'}")
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
        detected = extract_table_name(query,known_tables)
        if debug_mode and detected:
            print(f"Detected: {detected}")
        chunks = retrieve(collection,query,table_name=detected,known_tables=known_tables)

        if debug_mode:
            print(f"\n {len(chunks)} chunks retrieved:")
            for i, c in enumerate(chunks):
                dist = f" d={c['distance']:.3f}" if c["distance"] else ""
                print(f"    [{i+1}] {c['table_name']} ({c['doc_type']}){dist}")
                print(f"        {c['text'][:100]}...")
            print()

        try:
            answer = improved_ask_claude(query,chunks)
            print(f"\nClaude: {answer}\n")
        except anthropic.AuthenticationError:
            print("\n Invalid API key.")
            break
        except Exception as e:
            print(f"\n Error: {e}")


if __name__ == "__main__":
    main()




    



        


