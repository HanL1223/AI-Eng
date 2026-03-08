import os
import glob
import anthropic
import chromadb

#CONFIG
DOCS_DIR = "docs"
CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "my_docs"
CHUNK_SIZE = 500                     # TRY: 300, 500, 800, 1200 — see what works
CHUNK_OVERLAP = 50
TOP_K = 3                            # TRY: 1, 3, 5 — fewer = precise, more = broad
MODEL = "claude-sonnet-4-5-20250929"



def load_documents(docs_dir:str) -> list[dict]:
    """Read all .txt, .md, and .xlsx files from a directory."""
    documents = []

    for pattern in [""]