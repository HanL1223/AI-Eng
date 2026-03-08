"""rag.py — RAG Chatbot with STTM-Aware Loading
Drop this + sttm_loader.py into your rag-chatbot/ folder.
Put  STTM Excel file in docs/
Run: python rag.py
"""

import os
import glob
import anthropic
import chromadb
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
    for filepath in glob.glob(os.path.join(docs_dir,".xlsx")):
        print(f"Loading Excel: {os.path.basename(filepath)}: ({len(content)} chars)")
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



        


