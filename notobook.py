"""
tools.py -- Tool Definitions and Execution for STTM ReAct Agent
================================================================
Week 7, Step 1 of 4

WHAT THIS FILE DOES
-------------------
Defines the TOOLS that the ReAct agent can use and implements the
Python functions that EXECUTE each tool when the agent requests it.

There are two distinct concepts here that must not be confused:

  1. TOOL DEFINITIONS: JSON Schema objects that tell Claude what tools
     exist, what they do, and what inputs they accept. Claude reads
     these definitions and decides which tool to call. These are
     DECLARATIVE -- they describe the interface, not the implementation.

  2. TOOL IMPLEMENTATIONS: Python functions that actually perform the
     work when Claude requests a tool call. These are IMPERATIVE --
     they run your code (ChromaDB queries, list operations, etc.)
     and return results as strings.

The separation matters: Claude only sees the definitions. Your code
only runs the implementations. The agent loop (agent.py) connects
the two by matching tool names to functions.

dbt ANALOGY:
  Tool definitions = schema.yml (declares what a model produces,
    column names, types, descriptions, tests)
  Tool implementations = the SQL model file (the actual logic)
  Agent loop = dbt run (the orchestrator that connects schema to logic)


HOW THIS FILE CONNECTS TO YOUR EXISTING CODE
---------------------------------------------
The tool implementations call functions from your existing rag.py:

  retrieve()           -> Used by retrieve_table_info and search_across_tables
  extract_table_name() -> Used internally for validation
  build_vector_store() -> Called at init to get the collection

NOTHING in rag.py needs to change. This file IMPORTS from it.

The tools receive the ChromaDB collection and known_tables list as
parameters (dependency injection), NOT as globals. This makes testing
straightforward -- you can pass a mock collection in tests.


FILE STRUCTURE
--------------
  Section 1: Tool Definitions (JSON Schema for Claude)
  Section 2: Tool Implementations (Python functions)
  Section 3: Tool Registry (maps names to functions)
  Section 4: Standalone Test
"""

import json
import logging
import time
from typing import Optional

# =====================================================================
# STRUCTURED LOGGING SETUP
# =====================================================================
# DESIGN DECISION: Use Python's logging module with a JSON formatter
# instead of print(). This produces machine-parseable log lines that
# can be ingested by log aggregators (Datadog, CloudWatch, Azure
# Monitor, ELK stack).
#
# In production, you would configure the handler in a central
# logging config (e.g., logging.dictConfig). For this learning
# project, we configure it at the module level.
#
# PYTHON REFRESHER: logging.getLogger(__name__)
# -----------------------------------------------
# __name__ is the module name ("tools" when imported, "__main__" when
# run directly). Using __name__ creates a logger specific to this
# module, so log messages include which module produced them.
# This is the standard Python logging pattern.

logger = logging.getLogger(__name__)


# =====================================================================
# SECTION 1: TOOL DEFINITIONS
# =====================================================================
# These are the JSON Schema objects passed to Claude in the `tools`
# parameter of the Messages API. Claude reads these to understand:
#   - What tools are available
#   - What each tool does (description)
#   - What inputs each tool accepts (input_schema)
#   - Which inputs are required vs optional
#
# CRITICAL: The quality of the description directly affects how well
# Claude chooses the right tool. Vague descriptions = wrong tool calls.
# Each description should answer: WHAT does this tool do? WHEN should
# Claude use it? WHAT does it return?
#
# PYTHON REFRESHER: Type annotation for list of dicts
# ----------------------------------------------------
# TOOL_DEFINITIONS: list[dict] declares that this is a list where
# each element is a dictionary. Python does not enforce this at
# runtime (it is a hint for humans and type checkers like mypy).
# =====================================================================

TOOL_DEFINITIONS: list[dict] = [
    # ── Tool 1: retrieve_table_info ──
    # The most frequently used tool. Handles single-table lookups.
    # This wraps your existing retrieve() function with metadata
    # filtering to focus on a specific table.
    {
        "name": "retrieve_table_info",
        "description": (
            "Retrieve documentation for a specific table in the  "
            "Snowflake data warehouse (STTM). Use this when you need information "
            "about ONE specific table -- its grain, description, source systems, "
            "columns, mappings, or business area. "
            "Returns the relevant documentation text extracted from the STTM Excel "
            "workbooks. If the table is not found, returns a message indicating "
            "no documentation is available."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "table_name": {
                    "type": "string",
                    "description": (
                        "The exact table name in uppercase, e.g. 'FACT_SALES_ORDER', "
                        "'DIM_STORE', 'BRIDGE_CUSTOMER'. Must match the table names "
                        "in the STTM documentation."
                    ),
                },
                "info_type": {
                    "type": "string",
                    "enum": ["summary", "columns","all"],
                    "description": (
                        "What kind of information to retrieve. "
                        "'summary' = table grain, description, source systems, business area. "
                        "'columns' = column names, data types, descriptions. "
                        "'all' = everything available for the table. "
                        "Defaults to 'summary' if not specified."
                    ),
                },
            },
            "required": ["table_name"],
        },
    },

    # ── Tool 2: list_tables ──
    # Used when the agent needs to know what tables exist.
    # Essential for questions like "which fact tables reference DIM_PRODUCT?"
    # where the agent first needs to get a list, then iterate.
    {
        "name": "list_tables",
        "description": (
            "List all tables available in the STTM documentation, optionally "
            "filtered by table type (fact, dimension, bridge). Use this when "
            "you need to enumerate tables before looking up details on each one, "
            "or when answering questions like 'which fact tables exist' or "
            "'how many dimension tables are there'. "
            "Returns a list of table names."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "table_type": {
                    "type": "string",
                    "enum": ["fact", "dimension", "bridge", "all"],
                    "description": (
                        "Filter by table type. "
                        "'fact' = FACT_ prefixed tables (transactional measures). "
                        "'dimension' = DIM_ prefixed tables (descriptive attributes). "
                        "'bridge' = BRIDGE_ prefixed tables (many-to-many relationships). "
                        "'all' = all tables regardless of type. "
                        "Defaults to 'all' if not specified."
                    ),
                },
            },
            "required": [],
        },
    },

    # ── Tool 3: search_across_tables ──
    # The "broad search" tool. Uses RAG retrieval without table filtering.
    # For questions that do not target a specific table, or where the
    # user does not know which table contains the answer.
    {
        "name": "search_across_tables",
        "description": (
            "Search across ALL tables in the STTM documentation using semantic "
            "search. Use this when the question does not target a specific table, "
            "or when you need to find which tables contain information about a "
            "specific topic (e.g., 'inventory', 'SAP CDS', 'store hierarchy'). "
            "Returns the most relevant text chunks from any table, with source "
            "table names indicated."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "The search query. Be specific about what you are looking for. "
                        "Examples: 'tables with PDB15 as data source', "
                        "'inventory snapshot grain', 'SAP CDS view mappings'."
                    ),
                },
                "top_k": {
                    "type": "integer",
                    "description": (
                        "Number of results to return. Use 3-5 for focused searches, "
                        "7-10 for broad exploratory searches. Defaults to 5."
                    ),
                },
            },
            "required": ["query"],
        },
    },

    # ── Tool 4: get_table_relationships ──
    # For questions about which tables reference which.
    # This searches for foreign key columns and dimension references.
    {
        "name": "get_table_relationships",
        "description": (
            "Find relationships between tables. For a fact table, returns which "
            "dimension tables it references (via foreign key columns ending in _KEY "
            "or _SK). For a dimension table, returns which fact tables reference it. "
            "Use this when answering questions about table dependencies, star schema "
            "relationships, or data lineage between tables."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "table_name": {
                    "type": "string",
                    "description": (
                        "The table name to find relationships for, "
                        "e.g. 'FACT_SALES_ORDER' or 'DIM_PRODUCT'."
                    ),
                },
                "direction": {
                    "type": "string",
                    "enum": ["references", "referenced_by"],
                    "description": (
                        "'references' = find tables THIS table points to "
                        "(e.g., which dims does this fact reference). "
                        "'referenced_by' = find tables that point to THIS table "
                        "(e.g., which facts reference this dim). "
                        "Defaults to 'references'."
                    ),
                },
            },
            "required": ["table_name"],
        },
    },
]


# =====================================================================
# SECTION 2: TOOL IMPLEMENTATIONS
# =====================================================================
# Each function below is the ACTUAL CODE that runs when Claude
# requests a tool call. The agent loop (agent.py) calls these
# functions and returns their string output to Claude.
#
# DESIGN DECISION: Every tool function follows this contract:
#   - Accepts a ChromaDB collection + known_tables + tool-specific params
#   - Returns a STRING (Claude needs text, not Python objects)
#   - Handles errors gracefully (returns error message, never raises)
#   - Logs its execution (structured JSON via logging module)
#
# DESIGN DECISION: Why pass collection and known_tables as parameters
# instead of importing them directly from rag.py?
#
#   Importing: from rag import collection  <- This would require rag.py
#     to have a module-level collection object, coupling tools.py to
#     rag.py's initialization order. If tools.py is imported before
#     rag.py builds the collection, you get None.
#
#   Parameter passing: The agent loop builds the collection at startup
#     and passes it to each tool call. This is dependency injection --
#     the same pattern FastAPI uses with Depends() (Week 6).
#
# dbt ANALOGY: This is like ref('dim_store') in dbt. The model declares
# its dependency; the framework resolves it at runtime. The model does
# not reach into another model's internals to grab data.
# =====================================================================


def execute_retrieve_table_info(
    collection,
    known_tables: list[str],
    table_name: str,
    info_type: str = "summary",
) -> str:
    """
    Retrieve documentation for a specific table.

    This wraps your existing retrieve() function from rag.py with
    metadata filtering to focus on a specific table and doc_type.

    PARAMETERS
    ----------
    collection : ChromaDB Collection
        The vector store containing embedded STTM documentation.
    known_tables : list[str]
        All table names in the system. Used for fuzzy matching.
    table_name : str
        The table to look up, e.g. "FACT_SALES_ORDER".
    info_type : str
        What to retrieve: "summary", "columns", "mappings", or "all".

    RETURNS
    -------
    str
        The retrieved documentation as formatted text, or an error
        message if the table is not found.
    """
    from rag import retrieve

    start_time = time.time()

    # Normalise table name (Claude might send mixed case)
    table_name = table_name.strip().upper()

    # Validate table exists
    # DESIGN DECISION: We check against known_tables rather than
    # just querying ChromaDB. This gives a clear "table not found"
    # message instead of returning irrelevant chunks from other tables.
    if table_name not in known_tables:
        # Attempt fuzzy match -- the user might say "SALES_ORDER"
        # instead of "FACT_SALES_ORDER"
        matches = [t for t in known_tables if table_name in t]
        if len(matches) == 1:
            logger.info(
                json.dumps({
                    "event": "tool_fuzzy_match",
                    "tool": "retrieve_table_info",
                    "input_name": table_name,
                    "matched_name": matches[0],
                })
            )
            table_name = matches[0]
        elif len(matches) > 1:
            return (
                f"Ambiguous table name '{table_name}'. "
                f"Multiple matches found: {', '.join(matches)}. "
                f"Please specify the exact table name."
            )
        else:
            return (
                f"Table '{table_name}' not found in the STTM documentation. "
                f"Available tables include: {', '.join(known_tables[:10])}... "
                f"({len(known_tables)} tables total)."
            )

    # Map info_type to doc_type filter for ChromaDB metadata filtering
    # DESIGN DECISION: The info_type parameter maps to the doc_type
    # metadata that sttm_loader.py assigns to each chunk. This lets
    # us retrieve only summary chunks, or only column chunks, etc.
    doc_type_filter = None
    if info_type == "summary":
        doc_type_filter = "summary"
    elif info_type == "columns":
        doc_type_filter = "columns"
    elif info_type == "mappings":
        doc_type_filter = "mappings"
    # "all" -> no filter, retrieve everything for this table

    # Build a targeted query
    query = f"{table_name} {info_type}"
    top_k = 5 if info_type == "all" else 3

    # Use existing retrieve() with table_name filtering
    chunks = retrieve(
        collection,
        query,
        table_name=table_name,
        known_tables=known_tables,
        top_k=top_k,
    )

    # Filter by doc_type if specified
    if doc_type_filter and chunks:
        filtered = [c for c in chunks if c.get("doc_type") == doc_type_filter]
        # Fall back to unfiltered if doc_type filter removes everything
        if filtered:
            chunks = filtered

    duration_ms = (time.time() - start_time) * 1000

    if not chunks:
        logger.info(
            json.dumps({
                "event": "tool_no_results",
                "tool": "retrieve_table_info",
                "table_name": table_name,
                "info_type": info_type,
                "duration_ms": round(duration_ms, 1),
            })
        )
        return (
            f"No {info_type} documentation found for table '{table_name}'. "
            f"The table exists but may not have {info_type} information "
            f"in the loaded STTM workbooks."
        )

    # Format results as readable text for Claude
    # DESIGN DECISION: Return formatted text, not JSON. Claude needs
    # to REASON about this content, and natural text is easier for an
    # LLM to process than nested JSON structures.
    result_parts = [f"=== {table_name} ({info_type}) ==="]
    for chunk in chunks:
        text = chunk.get("text", "").strip()
        doc_type = chunk.get("doc_type", "unknown")
        result_parts.append(f"[{doc_type}] {text}")

    result = "\n\n".join(result_parts)

    logger.info(
        json.dumps({
            "event": "tool_executed",
            "tool": "retrieve_table_info",
            "table_name": table_name,
            "info_type": info_type,
            "chunks_returned": len(chunks),
            "result_length": len(result),
            "duration_ms": round(duration_ms, 1),
        })
    )

    return result


def execute_list_tables(
    known_tables: list[str],
    table_type: str = "all",
) -> str:
    """
    List available tables, optionally filtered by type.

    PARAMETERS
    ----------
    known_tables : list[str]
        All table names extracted from the STTM documentation.
    table_type : str
        Filter: "fact", "dimension", "bridge", or "all".

    RETURNS
    -------
    str
        A formatted list of table names.
    """
    start_time = time.time()

    # Apply filter based on naming convention
    # DESIGN DECISION: STTM tables follow strict naming conventions:
    #   FACT_*     -> fact tables (transactional data)
    #   DIM_*      -> dimension tables (descriptive attributes)
    #   BRIDGE_*   -> bridge tables (many-to-many relationships)
    # This convention is enforced by the data engineering team and
    # is reliable for filtering.
    if table_type == "fact":
        filtered = [t for t in known_tables if t.startswith("FACT_")]
    elif table_type == "dimension":
        filtered = [t for t in known_tables if t.startswith("DIM_")]
    elif table_type == "bridge":
        filtered = [t for t in known_tables if t.startswith("BRIDGE_")]
    else:
        filtered = known_tables

    duration_ms = (time.time() - start_time) * 1000

    logger.info(
        json.dumps({
            "event": "tool_executed",
            "tool": "list_tables",
            "table_type": table_type,
            "count": len(filtered),
            "duration_ms": round(duration_ms, 1),
        })
    )

    if not filtered:
        return f"No {table_type} tables found in the STTM documentation."

    # Format as numbered list
    lines = [f"Found {len(filtered)} {table_type} table(s):"]
    for i, name in enumerate(filtered, 1):
        lines.append(f"  {i}. {name}")

    return "\n".join(lines)


def execute_search_across_tables(
    collection,
    known_tables: list[str],
    query: str,
    top_k: int = 5,
) -> str:
    """
    Search across all tables using semantic retrieval (no table filter).

    This is the "broad search" tool for exploratory questions where
    the user does not know which table contains the answer.

    PARAMETERS
    ----------
    collection : ChromaDB Collection
        The vector store.
    known_tables : list[str]
        Available table names (used for context in results).
    query : str
        The search query text.
    top_k : int
        Number of results to return.

    RETURNS
    -------
    str
        Formatted search results with source table names.
    """
    from rag import retrieve

    start_time = time.time()

    # Clamp top_k to reasonable bounds
    top_k = max(1, min(top_k, 10))

    # Use retrieve() WITHOUT table_name filtering
    # This searches across all chunks in the vector store
    chunks = retrieve(
        collection,
        query,
        table_name=None,
        known_tables=known_tables,
        top_k=top_k,
    )

    duration_ms = (time.time() - start_time) * 1000

    if not chunks:
        logger.info(
            json.dumps({
                "event": "tool_no_results",
                "tool": "search_across_tables",
                "query": query,
                "top_k": top_k,
                "duration_ms": round(duration_ms, 1),
            })
        )
        return f"No results found for query: '{query}'"

    # Format results showing which table each chunk came from
    result_parts = [f"Search results for '{query}' ({len(chunks)} results):"]
    for i, chunk in enumerate(chunks, 1):
        table_name = chunk.get("table_name", "UNKNOWN")
        doc_type = chunk.get("doc_type", "unknown")
        text = chunk.get("text", "").strip()
        # Truncate long text to keep tool output manageable
        # DESIGN DECISION: Limit each chunk to 500 chars in the tool
        # output. The full text is in ChromaDB if Claude needs more
        # detail -- it can call retrieve_table_info for the specific
        # table. This keeps tool results focused and reduces token
        # usage in the agent's context window.
        if len(text) > 500:
            text = text[:500] + "..."
        result_parts.append(
            f"\n--- Result {i} [Table: {table_name}, Type: {doc_type}] ---\n{text}"
        )

    result = "\n".join(result_parts)

    logger.info(
        json.dumps({
            "event": "tool_executed",
            "tool": "search_across_tables",
            "query": query,
            "top_k": top_k,
            "chunks_returned": len(chunks),
            "unique_tables": len(set(c.get("table_name", "") for c in chunks)),
            "duration_ms": round(duration_ms, 1),
        })
    )

    return result


def execute_get_table_relationships(
    collection,
    known_tables: list[str],
    table_name: str,
    direction: str = "references",
) -> str:
    """
    Find relationships between tables by searching for foreign key
    references in the documentation.

    IMPLEMENTATION NOTE
    -------------------
    This tool does NOT query a real database catalog. It searches
    the STTM documentation for references to key columns. The STTM
    workbooks document foreign key relationships in the column
    detail sheets (columns ending in _KEY or _SK typically indicate
    foreign keys to dimension tables).

    In a production system, you would query Snowflake's
    INFORMATION_SCHEMA or the DBT manifest.json for actual FK
    relationships. This documentation-based approach is a reasonable
    approximation for the learning project.

    PARAMETERS
    ----------
    collection : ChromaDB Collection
    known_tables : list[str]
    table_name : str
        The table to find relationships for.
    direction : str
        "references" = what does this table point TO?
        "referenced_by" = what points TO this table?

    RETURNS
    -------
    str
        Formatted list of related tables and relationship info.
    """
    from rag import retrieve

    start_time = time.time()

    table_name = table_name.strip().upper()

    if direction == "references":
        # Find what this table references (its FKs pointing to dims)
        query = f"{table_name} foreign key dimension reference KEY SK"
        chunks = retrieve(
            collection,
            query,
            table_name=table_name,
            known_tables=known_tables,
            top_k=7,
        )
    else:
        # Find what references this table (other tables' FKs pointing here)
        query = f"references {table_name} foreign key"
        chunks = retrieve(
            collection,
            query,
            table_name=None,  # Search across all tables
            known_tables=known_tables,
            top_k=10,
        )

    duration_ms = (time.time() - start_time) * 1000

    if not chunks:
        logger.info(
            json.dumps({
                "event": "tool_no_results",
                "tool": "get_table_relationships",
                "table_name": table_name,
                "direction": direction,
                "duration_ms": round(duration_ms, 1),
            })
        )
        return (
            f"No relationship information found for '{table_name}' "
            f"(direction: {direction}). The table may not have documented "
            f"foreign key relationships in the STTM workbooks."
        )

    # Extract referenced table names from chunk text
    # Look for patterns like DIM_*, FACT_*, BRIDGE_* in the text
    referenced_tables = set()
    for chunk in chunks:
        text = chunk.get("text", "").upper()
        for known in known_tables:
            if known != table_name and known in text:
                referenced_tables.add(known)

    # Format output
    result_parts = [
        f"=== Relationships for {table_name} (direction: {direction}) ==="
    ]

    if referenced_tables:
        result_parts.append(f"\nRelated tables ({len(referenced_tables)}):")
        for rt in sorted(referenced_tables):
            result_parts.append(f"  - {rt}")

    result_parts.append(f"\nSource documentation:")
    for i, chunk in enumerate(chunks[:5], 1):
        source_table = chunk.get("table_name", "UNKNOWN")
        text = chunk.get("text", "").strip()
        if len(text) > 300:
            text = text[:300] + "..."
        result_parts.append(f"\n[{i}] From {source_table}:\n{text}")

    result = "\n".join(result_parts)

    logger.info(
        json.dumps({
            "event": "tool_executed",
            "tool": "get_table_relationships",
            "table_name": table_name,
            "direction": direction,
            "related_tables_found": len(referenced_tables),
            "chunks_returned": len(chunks),
            "duration_ms": round(duration_ms, 1),
        })
    )

    return result


# =====================================================================
# SECTION 3: TOOL REGISTRY
# =====================================================================
# The registry maps tool NAMES (as they appear in Claude's tool_use
# response) to Python FUNCTIONS that execute them.
#
# DESIGN DECISION: Why a registry dict instead of if/elif?
#
#   if block.name == "retrieve_table_info":
#       result = execute_retrieve_table_info(...)
#   elif block.name == "list_tables":
#       result = execute_list_tables(...)
#   elif ...
#
# This works but violates the Open/Closed Principle: adding a new
# tool requires modifying the if/elif chain. With a registry dict,
# you add a new entry without touching existing code.
#
# This is the same Strategy Pattern you used in model_switcher.py
# (Week 4) -- a dictionary dispatch instead of branching logic.
#
# PYTHON REFRESHER: Functions as values
# --------------------------------------
# In Python, functions are first-class objects. You can store them
# in variables, put them in dictionaries, pass them as arguments.
#
#   def greet(): return "hello"
#   func = greet          # func now points to the same function
#   func()                # returns "hello"
#   d = {"greet": greet}  # store in a dict
#   d["greet"]()          # returns "hello"
#
# The registry below stores function REFERENCES (no parentheses).
# The agent loop calls them with parentheses when needed.
# =====================================================================


def execute_tool(
    tool_name: str,
    tool_input: dict,
    collection,
    known_tables: list[str],
) -> str:
    """
    Execute a tool by name and return its string result.

    This is the single entry point called by the agent loop (agent.py).
    It dispatches to the appropriate tool function based on the name.

    PARAMETERS
    ----------
    tool_name : str
        The tool name from Claude's tool_use block.
    tool_input : dict
        The input arguments from Claude's tool_use block.
    collection : ChromaDB Collection
        The vector store (passed through to tool functions).
    known_tables : list[str]
        Available table names (passed through to tool functions).

    RETURNS
    -------
    str
        The tool's output as a string, or an error message if the
        tool is unknown or execution fails.

    GOTCHA: This function NEVER raises an exception. Tool failures
    are returned as error strings that Claude can see and reason about.
    If a tool crashes, Claude might decide to try a different approach.
    If we raised an exception, the entire agent loop would crash.
    """
    try:
        if tool_name == "retrieve_table_info":
            return execute_retrieve_table_info(
                collection=collection,
                known_tables=known_tables,
                table_name=tool_input.get("table_name", ""),
                info_type=tool_input.get("info_type", "summary"),
            )
        elif tool_name == "list_tables":
            return execute_list_tables(
                known_tables=known_tables,
                table_type=tool_input.get("table_type", "all"),
            )
        elif tool_name == "search_across_tables":
            return execute_search_across_tables(
                collection=collection,
                known_tables=known_tables,
                query=tool_input.get("query", ""),
                top_k=tool_input.get("top_k", 5),
            )
        elif tool_name == "get_table_relationships":
            return execute_get_table_relationships(
                collection=collection,
                known_tables=known_tables,
                table_name=tool_input.get("table_name", ""),
                direction=tool_input.get("direction", "references"),
            )
        else:
            logger.warning(
                json.dumps({
                    "event": "tool_unknown",
                    "tool_name": tool_name,
                })
            )
            return f"Unknown tool: '{tool_name}'. Available tools: retrieve_table_info, list_tables, search_across_tables, get_table_relationships."

    except Exception as e:
        logger.error(
            json.dumps({
                "event": "tool_execution_error",
                "tool_name": tool_name,
                "tool_input": tool_input,
                "error": str(e),
                "error_type": type(e).__name__,
            })
        )
        return f"Tool '{tool_name}' failed with error: {str(e)}"


# =====================================================================
# SECTION 4: STANDALONE TEST
# =====================================================================
# Run this file directly to test each tool:
#   uv run python tools.py
#
# This builds the RAG pipeline and exercises every tool.
# =====================================================================

if __name__ == "__main__":
    import os
    import sys

    # ── Configure structured JSON logging for standalone mode ──
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    # ── Load .env ──
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, _, value = line.partition("=")
                    os.environ[key.strip()] = value.strip()

    # ── Build pipeline ──
    from rag import load_documents, chunk_text, build_vector_store, DOCS_DIR

    logger.info(json.dumps({"event": "test_start", "module": "tools"}))

    documents = load_documents(DOCS_DIR)
    if not documents:
        logger.error(
            json.dumps({
                "event": "test_abort",
                "reason": f"No documents found in {DOCS_DIR}/",
            })
        )
        sys.exit(1)

    all_chunks = []
    for doc in documents:
        all_chunks.extend(chunk_text(doc["content"], doc["source"]))

    collection = build_vector_store(all_chunks)

    # Extract known tables
    all_names = []
    for doc in documents:
        parts = doc["source"].split("__")
        if len(parts) >= 2:
            all_names.append(parts[1].strip().upper())
    known_tables = sorted(set(name for name in all_names if name))

    logger.info(
        json.dumps({
            "event": "pipeline_ready",
            "documents": len(documents),
            "chunks": len(all_chunks),
            "tables": len(known_tables),
        })
    )

    # ── Test each tool ──
    test_cases = [
        ("retrieve_table_info", {"table_name": "DIM_STORE", "info_type": "summary"}),
        ("list_tables", {"table_type": "fact"}),
        ("search_across_tables", {"query": "inventory snapshot grain", "top_k": 3}),
        ("get_table_relationships", {"table_name": "FACT_SALES_ORDER", "direction": "references"}),
        ("retrieve_table_info", {"table_name": "NONEXISTENT_TABLE"}),
        ("unknown_tool", {}),
    ]

    passed = 0
    failed = 0

    for tool_name, tool_input in test_cases:
        logger.info(
            json.dumps({
                "event": "test_case_start",
                "tool": tool_name,
                "input": tool_input,
            })
        )
        result = execute_tool(tool_name, tool_input, collection, known_tables)
        is_ok = result and len(result) > 10

        if is_ok:
            passed += 1
        else:
            failed += 1

        logger.info(
            json.dumps({
                "event": "test_case_result",
                "tool": tool_name,
                "status": "PASS" if is_ok else "FAIL",
                "result_preview": result[:200] if result else "EMPTY",
            })
        )

    logger.info(
        json.dumps({
            "event": "test_complete",
            "passed": passed,
            "failed": failed,
            "total": len(test_cases),
        })
    )