"""
Tool Definitions and execution for STTM React Agent

Define the tools that the ReAct agent can use and omplements the python function that
Execute each tool when the agent requests it

Tool Definitions:Json object tell LLM what tools exists, what they do and what inputs they accept, claude
read the definitions and decides which tool to call
these describe the interface not the implementation

Tool Implementation: Python functions that actually perform the work when LLM require a tool call, there are 
imperative and return results as string

agent loopconnect the 2 by matching tool name to function

The tool implementations call functions from existing rag.py:

  retrieve()           -> Used by retrieve_table_info and search_across_tables
  extract_table_name() -> Used internally for validation
  build_vector_store() -> Called at init to get the collection

The tools receive the ChromaDB collection and known_tables list as
parameters (dependency injection), NOT as globals.


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

logger = logging.getLogger(__name__)



#Tool definitions
"""
Json schema object passed to Claude in the tools parameter of a message API to tell
 - what tools are available
 - what each tool does
 - What inputs each tool accepts
 - which inpouts are required vs optional
"""

TOOL_DEFINITIONS: list[dict] = [
    {
        "name": "retrieve_table_info",
        "description": (
            "Retrieve documentation for a specific table in the healthcare company"
            "Source to Target Mapping spreadsheet "
            "which contain source at column level for each new design data object in snowflake from it's data source which is the on prem DB"
            "Use this when you need information"
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
                    "enum": ["summary","columns","mapping","all"],
                    "description": (
                        "What kind of information to retrieve. "
                        "'summary' = table grain, description, source systems, business area. "
                        "'columns' = column names, data types, descriptions. "
                        "'mappings' = source-to-target column mappings and transformations. "
                        "'all' = everything available for the table. "
                        "Defaults to 'summary' if not specified."
                        ),
                },
            },
            "required": ["table_name"]
        },
    },

    #Tool 2 list_tables
    """
    Used when the agent needs to know what table exist.
    Essential for question like "which fact reference dim_product
    for this question, agent needs to know first what table exist
    """
    {
        "name": "lsist_tables",
        "description": (
            "List all tables available in the sttm documentation, optionally"
            "filtered by table type"
            "User this when you need to enumerate tables before looking up detail on each one"
            "or when asnwering question like 'which fact table exist' or 'how many fact tables are there'"
            "Returns a list of table names"
        ),
        "input_schema":{
            "type": "object",
            "properties": {
                "table_type": {
                    "type":"string",
                    "enum": ["fact","dimension","bridge","all"],
                    "Description": (
                        "Filter by table type. "
                        "'fact' = FACT_ prefixed tables (transactional measures). "
                        "'dimension' = DIM_ prefixed tables (descriptive attributes). "
                        "'bridge' = BRIDGE_ prefixed tables (many-to-many relationships). "
                        "'all' = all tables regardless of type. "
                        "Defaults to 'all' if not specified."
                    ),
                },
            },
            "required": []
        },

    },
    #Tool 3 search_across_tables
    """
    The broad search tool user rag retrieval without table fitlering
    for questions that do not target a specific table
    """
    {
        "name": "search_across_tables",
        "description": (
           " Search across ALL tables in the STTM documentation using semantic "
            "search. Use this when the question does not target a specific table,"
            "or when you need to find which tables contain information about a "
            "specific topic (e.g., 'inventory','store hierarchy'). "
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
                        "'inventory snapshot grain'."
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
            "required": ["query"]
        },
    },

    #Tool 4 get table relationship
    #for questions able which tables reference which
    {
        "name": "get_table_relationships",
        "description": (
            "Find relationships between tables. For a fact table, returns which "
            "dimension tables it references (via foreign key columns ending in _KEY "
            "or _SK). For a dimension table, returns which fact tables reference it. "
            "Use this when answering questions about table dependencies, star schema "
            "relationships, or data lineage between tables."
        ),
        "iunput_schema": {
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
                    "enum": ["references","referenced_by"],
                    "description": (
                        "'references' = find tables THIS table points to "
                        "(e.g., which dims does this fact reference). "
                        "'referenced_by' = find tables that point to THIS table "
                        "(e.g., which facts reference this dim). "
                        "Defaults to 'references'."
                    ),
                },
            },
            "required": ["table_name"]
        },
    },
]

#Tool implementation
"""
Each function is the actual code that runs when claude requested a tool call, the agent.py
call below and returns their string output to LLM
 DESIGN DECISION: Every tool function follows this contract:
   - Accepts a ChromaDB collection + known_tables + tool-specific params
   - Returns a STRING (Claude needs text, not Python objects)
   - Handles errors gracefully (returns error message, never raises)
   - Logs its execution (structured JSON via logging module)
"""



def execute_retrieve_table_info(
        collection,
        known_tables: list[str],
        table_name:str,
        info_type: str = "summary"
):
    """
    Retrieve documentation for a specific table
    This warrping existing retrieve function from rag.py
    with metadata filtering to focus on a specific table
    """

    from rag import retrieve

    start_time = time.time()

    table_name = table_name.strip().upper()
    if table_name not in known_tables:
        #Try to fine the substring 
        #e.g Sales_Order is not in known table but below will return 1 as sales order is a substring of factsalesorder
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
            return(
                f"Table '{table_name}' not found in the STTM documentation. "
                f"Available tables include: {', '.join(known_tables[:10])}... "
                f"({len(known_tables)} tables total)."
            )
    # Map info_type to doc_type filter for ChromaDB metadata filtering
    # DESIGN DECISION: The info_type parameter maps to the doc_type
    # metadata that sttm_loader.py assigns to each chunk. This lets
    # us retrieve only summary chunks, or only column chunks, etc.
    doc_type_filter = None
    if info_type == 'summary':
        doc_type_filter = "summary"
    elif info_type == "columns":
        doc_type_filter = "columns"
    elif info_type == "mappings":
        doc_type_filter = "mappings"

    # Build a targeted query
    query = f"{table_name} {info_type}"
    top_k = 5 if info_type == "all" else 3

    #Use existing retrieve() with table_name filtering
    chunks = retrieve(
        collection,
        query,
        table_name=table_name,
        known_tables = known_tables,
        top_k=top_k
    )

            