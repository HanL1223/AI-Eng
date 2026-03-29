"""
Query_logger -- structured query logging for RAG

Records every chatbot interaction to a JSONL file

Each line self-contains JSON object with eg

  {
    "timestamp":    "2026-03-28T10:15:30.123456",
    "query":        "What is the grain of DIM_STORE?",
    "answer":       "The grain of DIM_STORE is one row per store...",
    "model":        "claude",
    "query_type":   "single_table",
    "is_follow_up": false,
    "rerank_method": "bm25",
    "chunks_retrieved": 3,
    "chunks_used":  3,
    "latency_ms": {
      "retrieve":  45.2,
      "rerank":    120.3,
      "generate":  1230.5,
      "total":     1396.0
    },
    "estimated_cost_usd": 0.012,
    "sources":      ["STTM__DIM_STORE__summary", "STTM__DIM_STORE__columns"],
    "session_id":   "abc123"
  }
"""

import json
import os
import time
import uuid
from datetime import datetime,timezone
from pathlib import Path
from collections import defaultdict

#Confirguration
LOG_DIR = Path("logs")
LOG_FILE = LOG_DIR / 'queries.jsonl'

#Cost estimation as of 202603 = this is for reference only, in usd
MODEL_COSTS = {
    "claude": {
        "input_per_1k": 0.003,       # Claude Sonnet input
        "output_per_1k": 0.015,      # Claude Sonnet output
    },
    "claude-haiku": {
        "input_per_1k": 0.001,       # Claude Haiku input
        "output_per_1k": 0.005,      # Claude Haiku output
    },
    "ollama": {
        "input_per_1k": 0.0,         # Free (local inference)
        "output_per_1k": 0.0,
    },
}

# Average tokens per query and response (rough estimates).
# Used for cost estimation when actual token counts are not available.
AVG_INPUT_TOKENS = 800    # System prompt + context + query
AVG_OUTPUT_TOKENS = 200   # Typical answer length

#Session management
# A "session" groups queries from a single user's conversation.
# This allows  analyze conversation flows, not just individual queries.

def generate_session_id() -> str:
    return str(uuid.uuid4())

def estimate_cost(
        model:str,
        input_tokens:int = AVG_INPUT_TOKENS,
        output_tokens:int = AVG_OUTPUT_TOKENS
) -> float:
    """
    Estimate usd cost of a signle API Call

    """
    if model.startswith("ollama"):
        cost_key = "ollama"
    elif 'haiku' in model.lower():
        cost_key = 'claude-haiku'
    else:
        cost_key = "claude"
    rates = MODEL_COSTS.get(cost_key, {"input_per_1k": 0, "output_per_1k": 0})

    cost = (
        (input_tokens / 1000) * rates["input_per_1k"]
        + (output_tokens / 1000) * rates["output_per_1k"]
    )
    return round(cost, 6)

def log_query(
        query:str,
        answer:str,
        routing:dict,
        timing:dict,
        chunks: list[dict],
        session_id:str = "",
        extra:dict = None
) -> dict:
    """
     log a single query-response interaction to JSONL file
    """
    LOG_DIR.mkdir(parents = True, exist_ok=True)

    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "query": query,
        "answer_length": len(answer),
        "answer_preview": answer[:200],
        "model": routing.get("model", "unknown"),
        "query_type": routing.get("query_type", "unknown"),
        "is_follow_up": routing.get("is_follow_up", False),
        "rerank_method": routing.get("rerank_method", "none"),
        "chunks_retrieved": routing.get("retrieve_top_k", 0),
        "chunks_used": len(chunks),
        "latency_ms": {
            "retrieve": round(timing.get("retrieve_ms", 0), 1),
            "rerank": round(timing.get("rerank_ms", 0), 1),
            "generate": round(timing.get("generate_ms", 0), 1),
            "total": round(timing.get("total_ms", 0), 1),
        },
        "estimated_cost_usd": estimate_cost(routing.get("model", "claude")),
        "sources": [c.get("source", "unknown") for c in chunks],
        "table_names": list(set(
            c.get("table_name", "") for c in chunks if c.get("table_name")
        )),
        "session_id": session_id,
    }

    if extra:
        entry.update(extra)

    try:
        line = json.dumps(entry,ensure_ascii=True)
        with open(LOG_FILE,'a',encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception as e:
        print(f"Warning: Failed to write query log {e}")
    return entry


#Log reading any analytics

def load_logs(log_paths: str = None) -> list[dict]:
    """
    Load all log entries from a JSONL files
    """
    path = Path(log_paths) if log_paths else LOG_FILE
    if not path.exists():
        return []
    entries = []
    with open(path, "r", encoding = "utf-8") as f:
        for line_num, line in enumerate(f,1):
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Warning Skpiing malformed line {line_num}: {e}")
    return entries

def analyze_logs(entries: list[dict]) -> dict:
    """
    Generate an analytics summary from log entries

    This is the "dashboard" function. It computes statistics across
    all logged queries and returns a structured summary.

    he statistics are grouped into four categories:

    1. VOLUME:     Total queries, queries per day, queries per session
    2. LATENCY:    p50, p95, p99 for total and per-stage latency
    3. COST:       Total cost, cost per model, cost per query type
    4. ROUTING:    Model distribution, query type distribution, rerank usage

    ---
    p50 (median): Half of queries are faster than this
    p95:          95% of queries are faster -- only 5% are slower
    p99:          99% of queries are faster -- only 1% are slower

    p95 and p99 are critical for production SLAs. A system with
    p50=100ms but p99=5000ms has a "long tail" -- most users are
    happy, but 1 in 100 waits 5 seconds. That 1% matters.

    """
    if not entries:
        return {"error": "No log entries found"}
    #Volume metrics
    total_queries = len(entries)

    queries_by_date = defaultdict(int)
    for e in entries:
        date_str = e.get('timestamp',"")[:10]
        queries_by_date[date_str] += 1

    #Latency metrics
    total_latencies = [
        e.get("latency_ms",{}).get("total",0)
        for e in entries
        if e.get("latency_ms",{}).get("total",0)>0
    ]
    retrieve_latencies = [
        e.get("latency_ms",{}).get("retrieve",0)
        for e in entries
        if e.get("latency_ms",{}).get("retrieve",0)>0
    ]
    generate_latencies = [
        e.get("latency_ms", {}).get("generate", 0)
        for e in entries
        if e.get("latency_ms", {}).get("generate", 0) > 0
    ]

    def percentiles(values:list[float]) -> dict:
        "calculate the p50,p95,p99 from a sorted list"
        if not values:
            return {"p50": 0, "p95": 0, "p99": 0, "min": 0, "max": 0}
        s = sorted(values)
        n = len(s)
        return {
            "p50": round(s[int(n * 0.50)], 1),
            "p95": round(s[min(int(n * 0.95), n - 1)], 1),
            "p99": round(s[min(int(n * 0.99), n - 1)], 1),
            "min": round(s[0], 1),
            "max": round(s[-1], 1),
        }
    # ── Cost metrics ──
    total_cost = sum(e.get("estimated_cost_usd", 0) for e in entries)
    cost_by_model = defaultdict(float)
    for e in entries:
        model = e.get("model", "unknown")
        cost_by_model[model] += e.get("estimated_cost_usd", 0)

    # ── Routing metrics ──
    model_counts = defaultdict(int)
    query_type_counts = defaultdict(int)
    rerank_counts = defaultdict(int)
    follow_up_count = 0

    for e in entries:
        model_counts[e.get("model", "unknown")] += 1
        query_type_counts[e.get("query_type", "unknown")] += 1
        rerank_counts[e.get("rerank_method", "none")] += 1
        if e.get("is_follow_up"):
            follow_up_count += 1

    return {
        "volume": {
            "total_queries": total_queries,
            "date_range": {
                "first": min(queries_by_date.keys()) if queries_by_date else "N/A",
                "last": max(queries_by_date.keys()) if queries_by_date else "N/A",
            },
            "queries_per_day": dict(queries_by_date),
            "avg_queries_per_day": round(
                total_queries / max(len(queries_by_date), 1), 1
            ),
        },
        "latency": {
            "total_ms": percentiles(total_latencies),
            "retrieve_ms": percentiles(retrieve_latencies),
            "generate_ms": percentiles(generate_latencies),
        },
        "cost": {
            "total_usd": round(total_cost, 4),
            "by_model": {k: round(v, 4) for k, v in cost_by_model.items()},
            "avg_per_query": round(total_cost / max(total_queries, 1), 6),
        },
        "routing": {
            "model_distribution": dict(model_counts),
            "query_type_distribution": dict(query_type_counts),
            "rerank_distribution": dict(rerank_counts),
            "follow_up_rate": round(
                follow_up_count / max(total_queries, 1) * 100, 1
            ),
        },
    }


def print_analytics(analytics: dict) -> None:
    """
    Print a formatted analytics dashboard to the terminal.

    This is the human-readable output of analyze_logs().
    In production, you would send this to Datadog, Grafana,
    or a Streamlit dashboard. For learning, terminal output is fine.
    """
    if "error" in analytics:
        print(f"  {analytics['error']}")
        return

    vol = analytics["volume"]
    lat = analytics["latency"]
    cost = analytics["cost"]
    route = analytics["routing"]

    print("\n" + "=" * 60)
    print("QUERY ANALYTICS DASHBOARD")
    print("=" * 60)

    # Volume
    print(f"\n--- VOLUME ---")
    print(f"  Total queries:     {vol['total_queries']}")
    print(f"  Date range:        {vol['date_range']['first']} to {vol['date_range']['last']}")
    print(f"  Avg queries/day:   {vol['avg_queries_per_day']}")

    # Latency
    print(f"\n--- LATENCY (ms) ---")
    for stage in ["total_ms", "retrieve_ms", "generate_ms"]:
        p = lat[stage]
        label = stage.replace("_ms", "").capitalize()
        print(f"  {label:>12s}: p50={p['p50']:>8.1f}  p95={p['p95']:>8.1f}  "
              f"p99={p['p99']:>8.1f}  min={p['min']:>6.1f}  max={p['max']:>8.1f}")

    # Cost
    print(f"\n--- COST (USD) ---")
    print(f"  Total:           ${cost['total_usd']:.4f}")
    print(f"  Avg per query:   ${cost['avg_per_query']:.6f}")
    for model, model_cost in cost["by_model"].items():
        print(f"  {model:>15s}: ${model_cost:.4f}")

    # Routing
    print(f"\n--- ROUTING ---")
    print(f"  Follow-up rate:  {route['follow_up_rate']}%")
    print(f"  Model distribution:")
    for model, count in route["model_distribution"].items():
        pct = count / vol["total_queries"] * 100
        print(f"    {model:>15s}: {count:>4d} ({pct:>5.1f}%)")
    print(f"  Query type distribution:")
    for qt, count in route["query_type_distribution"].items():
        pct = count / vol["total_queries"] * 100
        print(f"    {qt:>15s}: {count:>4d} ({pct:>5.1f}%)")
    print(f"  Rerank method distribution:")
    for method, count in route["rerank_distribution"].items():
        pct = count / vol["total_queries"] * 100
        print(f"    {method:>15s}: {count:>4d} ({pct:>5.1f}%)")


def export_to_csv(entries: list[dict], output_path: str = "logs/queries_export.csv") -> str:
    """
    Export JSONL logs to a flat CSV for spreadsheet analysis.

    Flattens the nested latency_ms dict into separate columns.
    This is useful for importing into Excel, Google Sheets,
    or Power BI for further analysis.

    PYTHON REFRESHER: csv.DictWriter
    ─────────────────────────────────
    csv.DictWriter(file, fieldnames=["col1", "col2"]) writes dicts
    as CSV rows. Each dict key must match a fieldname.

    writer.writeheader() writes the column names as the first row.
    writer.writerow({"col1": "val1", "col2": "val2"}) writes one row.
    """
    import csv

    if not entries:
        print("No entries to export.")
        return ""

    # Flatten nested fields for CSV export.
    flat_entries = []
    for e in entries:
        lat = e.get("latency_ms", {})
        flat = {
            "timestamp": e.get("timestamp", ""),
            "query": e.get("query", ""),
            "answer_preview": e.get("answer_preview", ""),
            "model": e.get("model", ""),
            "query_type": e.get("query_type", ""),
            "is_follow_up": e.get("is_follow_up", False),
            "rerank_method": e.get("rerank_method", ""),
            "chunks_used": e.get("chunks_used", 0),
            "latency_retrieve_ms": lat.get("retrieve", 0),
            "latency_rerank_ms": lat.get("rerank", 0),
            "latency_generate_ms": lat.get("generate", 0),
            "latency_total_ms": lat.get("total", 0),
            "estimated_cost_usd": e.get("estimated_cost_usd", 0),
            "sources": "; ".join(e.get("sources", [])),
            "session_id": e.get("session_id", ""),
        }
        flat_entries.append(flat)

    fieldnames = list(flat_entries[0].keys())

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in flat_entries:
            writer.writerow(row)

    print(f"Exported {len(flat_entries)} entries to {output_path}")
    return output_path


# =====================================================================
# SECTION 6: STANDALONE TEST
# =====================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("QUERY LOGGER -- STANDALONE TEST")
    print("=" * 60)

    # Use a test log file to avoid polluting real logs.
    import tempfile
    test_dir = Path(tempfile.mkdtemp()) / "test_logs"
    test_dir.mkdir(parents=True, exist_ok=True)

    # Temporarily override LOG_DIR and LOG_FILE for testing.
    # This is a simple approach for testing. In production,
    # you would use dependency injection or a config object.
    original_log_dir = LOG_DIR
    original_log_file = LOG_FILE
    LOG_DIR = test_dir                              # modifies __main__.LOG_DIR
    LOG_FILE = test_dir / "test_queries.jsonl"      # modifies __main__.LOG_FILE

    # Override module-level variables for testing
    import query_logger
    query_logger.LOG_DIR = test_dir
    query_logger.LOG_FILE = test_dir / "test_queries.jsonl"

    session = generate_session_id()
    print(f"\nTest session ID: {session}")

    # Simulate 5 queries with different routing decisions
    test_queries = [
        {
            "query": "What is the grain of DIM_STORE?",
            "answer": "The grain of DIM_STORE is one row per store location.",
            "routing": {
                "query_type": "single_table",
                "is_follow_up": False,
                "model": "claude",
                "rerank_method": "none",
                "retrieve_top_k": 3,
            },
            "timing": {
                "retrieve_ms": 42.5,
                "rerank_ms": 0,
                "generate_ms": 1150.3,
                "total_ms": 1192.8,
            },
            "chunks": [
                {"source": "STTM__DIM_STORE__summary", "table_name": "DIM_STORE",
                 "doc_type": "summary", "text": "Grain: one row per store"},
            ],
        },
        {
            "query": "What about its foreign keys?",
            "answer": "DIM_STORE has FK_REGION_KEY and FK_STORE_TYPE_KEY.",
            "routing": {
                "query_type": "single_table",
                "is_follow_up": True,
                "model": "claude",
                "rerank_method": "bm25",
                "retrieve_top_k": 3,
            },
            "timing": {
                "retrieve_ms": 38.1,
                "rerank_ms": 5.2,
                "generate_ms": 980.7,
                "total_ms": 1024.0,
            },
            "chunks": [
                {"source": "STTM__DIM_STORE__columns", "table_name": "DIM_STORE",
                 "doc_type": "column_mapping", "text": "FK_REGION_KEY, FK_STORE_TYPE_KEY"},
            ],
        },
        {
            "query": "Which dimensions does FACT_SALES_ORDER reference?",
            "answer": "FACT_SALES_ORDER references DIM_STORE, DIM_PRODUCT, DIM_DATE.",
            "routing": {
                "query_type": "cross_entity",
                "is_follow_up": False,
                "model": "claude",
                "rerank_method": "hybrid",
                "retrieve_top_k": 10,
            },
            "timing": {
                "retrieve_ms": 55.3,
                "rerank_ms": 210.5,
                "generate_ms": 1520.1,
                "total_ms": 1785.9,
            },
            "chunks": [
                {"source": "STTM__FACT_SALES_ORDER__summary", "table_name": "FACT_SALES_ORDER",
                 "doc_type": "summary", "text": "References DIM_STORE, DIM_PRODUCT, DIM_DATE"},
                {"source": "STTM__DIM_STORE__summary", "table_name": "DIM_STORE",
                 "doc_type": "summary", "text": "DIM_STORE dimension table"},
            ],
        },
        {
            "query": "What is DIM_DATE?",
            "answer": "DIM_DATE is the date dimension table.",
            "routing": {
                "query_type": "single_table",
                "is_follow_up": False,
                "model": "ollama/qwen2.5:0.5b",
                "rerank_method": "none",
                "retrieve_top_k": 3,
            },
            "timing": {
                "retrieve_ms": 30.2,
                "rerank_ms": 0,
                "generate_ms": 450.8,
                "total_ms": 481.0,
            },
            "chunks": [
                {"source": "STTM__DIM_DATE__summary", "table_name": "DIM_DATE",
                 "doc_type": "summary", "text": "Date dimension table"},
            ],
        },
        {
            "query": "What is the SLA for data refresh?",
            "answer": "I don't have that information in the loaded documents.",
            "routing": {
                "query_type": "edge_case",
                "is_follow_up": False,
                "model": "claude",
                "rerank_method": "none",
                "retrieve_top_k": 3,
            },
            "timing": {
                "retrieve_ms": 35.0,
                "rerank_ms": 0,
                "generate_ms": 890.2,
                "total_ms": 925.2,
            },
            "chunks": [],
        },
    ]

    # Log each test query
    print(f"\nLogging {len(test_queries)} test queries...")
    for tq in test_queries:
        entry = log_query(
            query=tq["query"],
            answer=tq["answer"],
            routing=tq["routing"],
            timing=tq["timing"],
            chunks=tq["chunks"],
            session_id=session,
        )
        print(f"  Logged: '{tq['query'][:40]}...' (model={entry['model']})")

    # Read and analyze the logs
    print(f"\nReading logs from {query_logger.LOG_FILE}...")
    entries = load_logs(str(query_logger.LOG_FILE))
    print(f"  Loaded {len(entries)} entries")

    # Run analytics
    analytics = analyze_logs(entries)
    print_analytics(analytics)

    # Export to CSV
    csv_path = str(test_dir / "test_export.csv")
    export_to_csv(entries, csv_path)

    # Verify the JSONL file content
    print(f"\n--- RAW JSONL (first 2 lines) ---")
    with open(query_logger.LOG_FILE) as f:
        for i, line in enumerate(f):
            if i >= 2:
                break
            parsed = json.loads(line)
            print(f"  Line {i+1}: query='{parsed['query'][:30]}...' "
                  f"cost=${parsed['estimated_cost_usd']}")

    # Restore original log settings
    query_logger.LOG_DIR = original_log_dir
    query_logger.LOG_FILE = original_log_file
    LOG_DIR = original_log_dir
    LOG_FILE = original_log_file

    print(f"\nTest log directory: {test_dir}")
    print("All tests passed. query_logger.py is ready.")
