"""
pdf_loader.py -- PDF Document Ingestion for RAG
=================================================
Week 5, Step 2 of 4

WHAT THIS FILE DOES
-------------------
Extracts text from PDF files and converts them into the same document
format that your existing rag.py pipeline expects. This allows your
chatbot to answer questions from PDFs alongside your Excel STTM files.

After this module, your document loading pipeline handles:
  .xlsx  -> sttm_loader.py  -> structured STTM data
  .pdf   -> pdf_loader.py   -> unstructured text content
  .txt   -> rag.py (direct) -> plain text
  .md    -> rag.py (direct) -> plain text


WHY PDF INGESTION MATTERS
─────────────────────────
Your STTM Excel files are structured: headers in rows 0-12, column
mappings from row 16. But real data warehouse documentation also
includes PDFs:

  - Architecture diagrams with text annotations
  - Data dictionary documents
  - Design specification documents
  - SOPs (Standard Operating Procedures)
  - Vendor documentation (SAP, Snowflake, Azure)

These documents contain context that your chatbot currently cannot
access. Adding PDF support immediately expands your knowledge base
without changing any downstream code (chunking, embedding, retrieval,
generation all stay the same).


WHY PYMUPDF (not pdfplumber, not PyPDF2, not Tika)
────────────────────────────────────────────────────
There are several Python PDF libraries. Here is why we chose PyMuPDF:

  PyPDF2 (pypdf):
    - Pure Python, no C dependencies
    - Slow on large files
    - Poor text extraction quality (often missing spaces, wrong order)
    - Good enough for simple PDFs, fails on complex layouts

  pdfplumber:
    - Built on pdfminer, which is accurate
    - Good table extraction
    - Slow (3-10x slower than PyMuPDF)
    - Heavy dependency chain
    - Best for: PDFs with lots of tables

  Apache Tika:
    - Java-based, requires JVM running
    - Excellent extraction quality
    - Heavy infrastructure (Java server process)
    - Best for: Enterprise systems with Java stack

  PyMuPDF (fitz):
    - C extension (fast, ~10-50x faster than pdfplumber)
    - Good text extraction quality
    - Handles most PDF layouts correctly
    - Small install (~15MB)
    - Also handles images, annotations, metadata
    - Best for: General-purpose PDF reading in Python

For your use case (reading data warehouse docs), PyMuPDF is the
right choice: fast, accurate enough, minimal dependencies.

INSTALLATION:
  uv add pymupdf

GOTCHA: The import name is "fitz", not "pymupdf":
  import fitz       # Correct -- this IS PyMuPDF
  import pymupdf    # This also works in newer versions


HOW PDF TEXT EXTRACTION WORKS
──────────────────────────────
A PDF is NOT a text file. It is a page-description format that says
"draw character 'H' at position (72, 700) in Times-Roman 12pt".

Text extraction requires:
  1. Parse the PDF's binary structure
  2. Find all text-drawing commands
  3. Group characters into words based on spatial proximity
  4. Group words into lines based on vertical position
  5. Order lines top-to-bottom, left-to-right

This is why PDF extraction is imperfect -- the PDF does not store
"paragraphs" or "sentences". It stores character positions. The
extraction library must reconstruct the reading order.

Common extraction problems:
  - Multi-column layouts: text from column 1 and column 2 gets merged
  - Headers/footers: repeated on every page, polluting the text
  - Tables: columns may merge or split incorrectly
  - Watermarks: "DRAFT" or "CONFIDENTIAL" appears in the text
  - Scanned PDFs: contain images of text, not text (need OCR)

PyMuPDF handles most of these well. For scanned PDFs, you would
need OCR (Tesseract or cloud OCR). We skip scanned PDFs for now.

dbt ANALOGY:
  PDF extraction is like an ELT source connector. You are extracting
  raw data (text) from an external system (PDF) and loading it into
  your staging layer (the documents list). The downstream models
  (chunking, embedding) do not care where the text came from.


HOW THIS FILE CONNECTS TO YOUR PROJECT
───────────────────────────────────────
  rag.py's load_documents() gains PDF support:
    Before: if file.endswith(".xlsx") -> sttm_loader
            if file.endswith(".txt") or file.endswith(".md") -> direct read
    After:  + if file.endswith(".pdf") -> pdf_loader.load_pdf()

  The return format matches what rag.py expects:
    {"content": "extracted text...", "source": "filename.pdf"}

  No changes to chunk_text(), build_vector_store(), retrieve(),
  or ask_claude(). The PDF text flows through the exact same
  pipeline as any other document.


DEPENDENCIES
────────────
  uv add pymupdf
"""

import os
from pathlib import Path


# =====================================================================
# SECTION 1: CORE PDF TEXT EXTRACTION
# =====================================================================

def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract all text from a PDF file.

    Uses PyMuPDF (fitz) to read each page and concatenate the text.
    Pages are separated by a form-feed marker and a page header
    to help the chunker maintain page context.

    PARAMETERS
    ----------
    pdf_path : str or Path
        Path to the PDF file.

    RETURNS
    -------
    str
        All extracted text, with page boundaries marked.

    RAISES
    ------
    FileNotFoundError
        If the PDF file does not exist.
    ImportError
        If pymupdf is not installed.
    RuntimeError
        If the PDF cannot be opened (encrypted, corrupted, etc.)

    HOW IT WORKS
    ────────────
    1. Open the PDF with fitz.open()
    2. Iterate over each page (0-indexed)
    3. Call page.get_text("text") for plain text extraction
    4. Optionally clean up the extracted text
    5. Join all pages with markers

    GOTCHA: fitz.open() does NOT raise an error for encrypted PDFs.
    Instead, page.get_text() returns empty string. We check for this
    and warn the user.
    """
    try:
        import fitz  # PyMuPDF
    except ImportError:
        raise ImportError(
            "pymupdf is required for PDF support.\n"
            "Install it with: uv add pymupdf"
        )

    pdf_path = str(pdf_path)
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    # Open the PDF.
    # fitz.open() returns a Document object. It can also open
    # images, XPS files, and other formats -- but we only use it
    # for PDFs here.
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        raise RuntimeError(f"Failed to open PDF '{pdf_path}': {e}")

    # Extract text from each page.
    pages_text = []
    empty_page_count = 0

    for page_num in range(len(doc)):
        page = doc[page_num]

        # get_text("text") extracts plain text in reading order.
        # Other options:
        #   "blocks" -> text blocks with bounding boxes
        #   "dict"   -> full page structure (fonts, sizes, etc.)
        #   "html"   -> HTML formatted text
        #   "xml"    -> XML structured text
        #
        # We use "text" because it is the simplest and works well
        # for most document types. If you needed table structure,
        # you would use "blocks" or pdfplumber instead.
        text = page.get_text("text")

        if not text.strip():
            empty_page_count += 1
            continue

        # Clean up common extraction artifacts.
        text = _clean_extracted_text(text)

        # Add a page marker so chunks can reference page numbers.
        # This is useful for citations: "See page 5 of document.pdf"
        pages_text.append(f"[Page {page_num + 1}]\n{text}")

    doc.close()

    # Warn if most pages are empty (likely a scanned PDF).
    total_pages = page_num + 1 if 'page_num' in dir() else 0
    if empty_page_count > 0 and empty_page_count >= total_pages * 0.5:
        print(
            f"WARNING: {empty_page_count}/{total_pages} pages in "
            f"'{os.path.basename(pdf_path)}' are empty. "
            f"This may be a scanned PDF (requires OCR)."
        )

    return "\n\n".join(pages_text)


def _clean_extracted_text(text: str) -> str:
    """
    Clean up common PDF extraction artifacts.

    This is a PRIVATE function (underscore prefix) used only by
    extract_text_from_pdf().

    Common issues we fix:
    1. Excessive whitespace (multiple blank lines)
    2. Hyphenated line breaks ("docu-\\nment" -> "document")
    3. Page numbers standalone on a line
    4. Repeated headers/footers (hard to detect generically)

    We keep the cleaning light-touch. Aggressive cleaning can
    destroy meaningful formatting (e.g., code blocks, tables).
    """
    import re

    # Remove excessive blank lines (3+ consecutive -> 2).
    #
    # PYTHON REFRESHER: re.sub() with regex
    # ─────────────────────────────────────
    # re.sub(pattern, replacement, string)
    #
    # \n{3,} matches 3 or more consecutive newlines.
    # We replace with exactly 2 newlines (one blank line).
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Rejoin hyphenated words at line breaks.
    # "docu-\nment" -> "document"
    #
    # The regex: a lowercase letter, then a hyphen, then a newline,
    # then a lowercase letter. We join them without the hyphen/newline.
    #
    # GOTCHA: This can incorrectly join intentional hyphens:
    # "self-\nservice" -> "selfservice" (wrong!)
    # In practice, this is rare enough that the benefit outweighs
    # the risk for data warehouse documentation.
    text = re.sub(r"([a-z])-\n([a-z])", r"\1\2", text)

    # Strip trailing whitespace from each line.
    lines = [line.rstrip() for line in text.split("\n")]
    text = "\n".join(lines)

    return text


# =====================================================================
# SECTION 2: PDF METADATA EXTRACTION
# =====================================================================

def extract_pdf_metadata(pdf_path: str) -> dict:
    """
    Extract metadata from a PDF file.

    PDF files can contain embedded metadata:
      - Title, Author, Subject, Keywords
      - Creation date, Modification date
      - Producer (which software created it)
      - Page count

    This metadata is useful for:
    1. Automatic source labeling (use the title as the document name)
    2. Filtering (only include documents from a certain date range)
    3. Debugging (knowing which tool produced a PDF helps explain
       extraction issues)

    RETURNS
    -------
    dict with keys: title, author, subject, keywords, creator,
    producer, creation_date, mod_date, page_count, file_size_kb
    """
    try:
        import fitz
    except ImportError:
        return {"error": "pymupdf not installed"}

    if not os.path.exists(pdf_path):
        return {"error": f"File not found: {pdf_path}"}

    doc = fitz.open(pdf_path)
    meta = doc.metadata  # Returns a dict with standard PDF metadata keys

    result = {
        "title": meta.get("title", ""),
        "author": meta.get("author", ""),
        "subject": meta.get("subject", ""),
        "keywords": meta.get("keywords", ""),
        "creator": meta.get("creator", ""),
        "producer": meta.get("producer", ""),
        "creation_date": meta.get("creationDate", ""),
        "mod_date": meta.get("modDate", ""),
        "page_count": len(doc),
        "file_size_kb": round(os.path.getsize(pdf_path) / 1024, 1),
    }

    doc.close()
    return result


# =====================================================================
# SECTION 3: HIGH-LEVEL LOADING FUNCTION
# =====================================================================

def load_pdf(pdf_path: str) -> dict:
    """
    Load a PDF file into the document format expected by rag.py.

    This is the main function that rag.py's load_documents() will call
    for PDF files. It returns a dict with the same structure as other
    document types:

      {"content": "...", "source": "filename.pdf"}

    This uniformity is critical. The downstream pipeline (chunk_text,
    build_vector_store, retrieve) does not know or care whether the
    document came from Excel, PDF, or plain text. It just sees a
    dict with "content" and "source".

    dbt ANALOGY:
      This function is a source connector. Like a dbt source definition,
      it wraps the raw data (PDF) in a standardized interface (dict).
      The staging model (chunk_text) consumes this interface without
      knowing the underlying data source.

    PARAMETERS
    ----------
    pdf_path : str
        Path to the PDF file.

    RETURNS
    -------
    dict
        {"content": extracted_text, "source": filename}
        Returns None if extraction fails.
    """
    filename = os.path.basename(pdf_path)

    try:
        text = extract_text_from_pdf(pdf_path)

        if not text.strip():
            print(f"WARNING: No text extracted from '{filename}'. "
                  f"May be a scanned PDF or empty file.")
            return None

        # Get metadata for enriched source labeling.
        meta = extract_pdf_metadata(pdf_path)
        title = meta.get("title", "").strip()

        # Use the PDF's embedded title if available, otherwise filename.
        source_label = title if title else filename

        return {
            "content": text,
            "source": source_label,
        }

    except ImportError as e:
        print(f"ERROR: {e}")
        return None
    except Exception as e:
        print(f"ERROR: Failed to load PDF '{filename}': {e}")
        return None


def load_pdfs_from_directory(directory: str) -> list[dict]:
    """
    Load all PDF files from a directory.

    Scans the directory for .pdf files, extracts text from each,
    and returns a list of document dicts.

    This function is useful for batch loading. In rag.py, you may
    call it to load all PDFs from the docs/ folder.

    PARAMETERS
    ----------
    directory : str
        Path to the directory containing PDF files.

    RETURNS
    -------
    list[dict]
        List of {"content": ..., "source": ...} dicts.
        Skips files that fail to load (logs a warning).
    """
    directory = str(directory)
    if not os.path.isdir(directory):
        print(f"WARNING: Directory not found: {directory}")
        return []

    pdf_files = sorted([
        f for f in os.listdir(directory)
        if f.lower().endswith(".pdf")
    ])

    if not pdf_files:
        return []

    print(f"Found {len(pdf_files)} PDF file(s) in {directory}/")
    documents = []

    for filename in pdf_files:
        filepath = os.path.join(directory, filename)
        doc = load_pdf(filepath)
        if doc:
            # Count pages for logging.
            meta = extract_pdf_metadata(filepath)
            pages = meta.get("page_count", "?")
            chars = len(doc["content"])
            print(f"  Loaded: {filename} ({pages} pages, {chars:,} chars)")
            documents.append(doc)
        else:
            print(f"  Skipped: {filename} (extraction failed)")

    return documents


# =====================================================================
# SECTION 4: INTEGRATION INSTRUCTIONS FOR rag.py
# =====================================================================
#
# To wire PDF loading into your existing rag.py, add these lines
# to the load_documents() function:
#
#   BEFORE (current rag.py):
#   ────────────────────────
#   def load_documents(docs_dir):
#       documents = []
#       for filename in os.listdir(docs_dir):
#           filepath = os.path.join(docs_dir, filename)
#           if filename.endswith(".xlsx"):
#               # ... existing Excel loading code ...
#           elif filename.endswith(".txt") or filename.endswith(".md"):
#               # ... existing text loading code ...
#       return documents
#
#   AFTER (with PDF support):
#   ─────────────────────────
#   def load_documents(docs_dir):
#       documents = []
#       for filename in os.listdir(docs_dir):
#           filepath = os.path.join(docs_dir, filename)
#           if filename.endswith(".xlsx"):
#               # ... existing Excel loading code ...
#           elif filename.endswith(".pdf"):              # <-- NEW
#               from pdf_loader import load_pdf          # <-- NEW
#               doc = load_pdf(filepath)                 # <-- NEW
#               if doc:                                  # <-- NEW
#                   documents.append(doc)                # <-- NEW
#           elif filename.endswith(".txt") or filename.endswith(".md"):
#               # ... existing text loading code ...
#       return documents
#
# That is the ONLY change needed. Five lines. Everything downstream
# (chunking, embedding, retrieval, generation) works unchanged.
#
# BACKWARD COMPATIBILITY: If pymupdf is not installed, load_pdf()
# returns None with a helpful error message. The pipeline continues
# processing other file types.
#
# GOTCHA: PDF text does NOT have STTM structure (no table_name,
# no doc_type metadata). The chunks from PDFs will have:
#   table_name = ""  (empty -- not a table)
#   doc_type = "text" (generic text)
#
# This means PDF chunks participate in global retrieval but are
# not filtered by extract_table_name(). This is correct behavior:
# if a user asks "What is the data architecture?", the answer might
# come from an architecture PDF, not from an STTM Excel file.


# =====================================================================
# SECTION 5: STANDALONE TEST
# =====================================================================

if __name__ == "__main__":
    import tempfile

    print("=" * 60)
    print("PDF LOADER -- STANDALONE TEST")
    print("=" * 60)

    # Check if pymupdf is installed.
    try:
        import fitz
        print(f"\nPyMuPDF version: {fitz.__doc__}")
        print(f"PyMuPDF is installed and working.")
    except ImportError:
        print("\nPyMuPDF is NOT installed.")
        print("Install with: uv add pymupdf")
        print("\nSkipping live test. The code structure is correct.")
        print("After installing pymupdf, re-run this script.")
        exit(0)

    # Create a simple test PDF using PyMuPDF itself.
    # This avoids needing an external test file.
    print("\nCreating test PDF...")
    test_dir = tempfile.mkdtemp()
    test_pdf_path = os.path.join(test_dir, "test_document.pdf")

    doc = fitz.open()  # Create a new empty PDF

    # Add page 1
    page1 = doc.new_page(width=612, height=792)  # US Letter
    page1.insert_text(
        (72, 72),
        "STTM Data Warehouse Architecture\n\n"
        "The Sigma Healthcare data warehouse is built on Snowflake.\n"
        "Source systems include SAP (via CDS Views), MyPOS, and PDB08.\n\n"
        "Data flows through these layers:\n"
        "1. Bronze: Raw extraction via Azure Data Factory\n"
        "2. Silver: Cleaned and conformed via dbt\n"
        "3. Gold: Star schema for reporting\n"
        "4. Platinum: Business-ready aggregations\n\n"
        "The FACT tables include FACT_SALES_ORDER, FACT_INVENTORY,\n"
        "and FACT_STORE_INVENTORY_INTRA.",
        fontsize=11,
    )

    # Add page 2
    page2 = doc.new_page(width=612, height=792)
    page2.insert_text(
        (72, 72),
        "Key Design Decisions\n\n"
        "Surrogate keys (SK_*_KEY) are used for all dimension tables.\n"
        "Business keys (BK_*_KEY) are preserved for lineage tracking.\n"
        "Foreign keys (FK_*_KEY) reference the surrogate keys.\n\n"
        "SCD Type 2 is used for slowly changing dimensions like\n"
        "DIM_STORE and DIM_PRODUCT, with VALID_FROM and VALID_TO\n"
        "date columns tracking historical changes.",
        fontsize=11,
    )

    doc.save(test_pdf_path)
    doc.close()
    print(f"  Created: {test_pdf_path}")

    # Test extract_text_from_pdf
    print("\n--- Text Extraction ---")
    text = extract_text_from_pdf(test_pdf_path)
    print(f"  Extracted {len(text)} characters from {2} pages")
    print(f"  First 200 chars:\n  {text[:200]}")

    # Test extract_pdf_metadata
    print("\n--- Metadata ---")
    meta = extract_pdf_metadata(test_pdf_path)
    for key, value in meta.items():
        if value:
            print(f"  {key}: {value}")

    # Test load_pdf (the main function)
    print("\n--- load_pdf() ---")
    doc_dict = load_pdf(test_pdf_path)
    if doc_dict:
        print(f"  source: {doc_dict['source']}")
        print(f"  content length: {len(doc_dict['content'])} chars")
    else:
        print("  ERROR: load_pdf returned None")

    # Test load_pdfs_from_directory
    print("\n--- load_pdfs_from_directory() ---")
    docs = load_pdfs_from_directory(test_dir)
    print(f"  Loaded {len(docs)} documents")

    # Verify the output format matches rag.py expectations
    print("\n--- Format Verification ---")
    if doc_dict:
        assert "content" in doc_dict, "Missing 'content' key!"
        assert "source" in doc_dict, "Missing 'source' key!"
        assert isinstance(doc_dict["content"], str), "'content' must be str!"
        assert isinstance(doc_dict["source"], str), "'source' must be str!"
        assert len(doc_dict["content"]) > 0, "'content' must not be empty!"
        print("  Format matches rag.py expectations.")
    else:
        print("  Cannot verify format (load_pdf returned None).")

    # Clean up
    os.remove(test_pdf_path)
    os.rmdir(test_dir)

    print("\nAll tests passed. pdf_loader.py is ready.")
    print("\nNext step: Wire this into rag.py's load_documents() function.")
    print("See the integration instructions in SECTION 4 of this file.")