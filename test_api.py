"""
test_api.py -- Integration Tests for STTM RAG API
====================================================
Week 6, Step 6 of 6

WHAT THIS FILE DOES
-------------------
Sends HTTP requests to your running FastAPI server and verifies the
responses. This is an INTEGRATION TEST -- it tests the entire system
end-to-end, from HTTP request through the RAG pipeline to HTTP response.

Unlike unit tests (which test individual functions), integration tests
verify that all components work together correctly when connected.

PREREQUISITE: The API server must be running before you run this script.
  Start the server:  uv run uvicorn api_server:app --port 8000
  Then run tests:    uv run python test_api.py


HOW HTTP REQUESTS WORK IN PYTHON
----------------------------------
Python's `requests` library sends HTTP requests. It is the most popular
HTTP client library in Python (requests.readthedocs.io).

  import requests

  # GET request (fetch data):
  response = requests.get("http://localhost:8000/api/health")
  print(response.status_code)  # 200
  print(response.json())       # {"status": "healthy", ...}

  # POST request (send data):
  response = requests.post(
      "http://localhost:8000/api/query",
      json={"query": "What is DIM_STORE?"}
  )
  print(response.json()["answer"])

PYTHON REFRESHER: requests.get() vs requests.post()
  GET:   Fetches data. No request body. Data in URL parameters.
  POST:  Sends data. Request body contains JSON payload.
  json=  parameter automatically:
    1. Serializes the dict to a JSON string
    2. Sets Content-Type header to application/json
    3. Encodes the string as UTF-8 bytes


WHAT IS AN INTEGRATION TEST?
-------------------------------
There are three levels of testing:

  Unit Test:
    Tests one function in isolation.
    Example: test that QueryRequest rejects empty queries.
    You did this in api_models.py's __main__ block.

  Integration Test:
    Tests multiple components working together.
    Example: send a query to the API, verify the response has an answer.
    THIS FILE does integration testing.

  End-to-End (E2E) Test:
    Tests the ENTIRE system including UI.
    Example: open the Streamlit app, type a question, verify the answer.
    You would use Selenium or Playwright for E2E tests (not this week).

dbt ANALOGY:
  Unit test = dbt test on a single column (not_null, unique)
  Integration test = dbt test that checks relationships across tables
  E2E test = Testing the Power BI report reads correctly from Gold layer


DEPENDENCIES
-------------
  requests (usually already installed as a transitive dependency)
  If not: uv add requests

  The server must be running at http://localhost:8000.
"""

import sys
import time
import requests


# =====================================================================
# SECTION 1: CONFIGURATION
# =====================================================================

BASE_URL = "http://localhost:8000"
# EXPLANATION: This is the address of your running API server.
# "localhost" means "this machine". Port 8000 matches the port in
# docker-compose.yml and the uvicorn --port flag.
#
# If you run the API on a different port (e.g., 8080):
#   BASE_URL = "http://localhost:8080"
#
# If you run the API on a remote server:
#   BASE_URL = "http://192.168.1.100:8000"
#   BASE_URL = "https://my-rag-api.example.com"


# =====================================================================
# SECTION 2: TEST HELPERS
# =====================================================================

passed = 0
failed = 0


def check(test_name: str, condition: bool, detail: str = ""):
    """
    Record a test result with pass/fail output.

    This is a minimal test framework. In production, you would use
    pytest. For learning, explicit pass/fail output is more transparent.
    """
    global passed, failed
    if condition:
        passed += 1
        print(f"  PASS: {test_name}")
    else:
        failed += 1
        print(f"  FAIL: {test_name}")
        if detail:
            print(f"        {detail}")


def is_server_running() -> bool:
    """Check if the API server is reachable."""
    try:
        r = requests.get(f"{BASE_URL}/api/health", timeout=5)
        return r.status_code in (200, 503)
    except requests.ConnectionError:
        return False


# =====================================================================
# SECTION 3: TEST CASES
# =====================================================================


def test_health_check():
    """Test GET /api/health returns healthy status."""
    print("\n--- Test: Health Check ---")

    r = requests.get(f"{BASE_URL}/api/health")

    check("status code is 200", r.status_code == 200, f"got {r.status_code}")

    data = r.json()
    check("status is 'healthy'", data.get("status") == "healthy",
          f"got '{data.get('status')}'")
    check("documents_loaded >= 0", data.get("documents_loaded", -1) >= 0)
    check("tables_available >= 0", data.get("tables_available", -1) >= 0)
    check("version is present", "version" in data)


def test_list_tables():
    """Test GET /api/tables returns table list."""
    print("\n--- Test: List Tables ---")

    r = requests.get(f"{BASE_URL}/api/tables")

    check("status code is 200", r.status_code == 200, f"got {r.status_code}")

    data = r.json()
    check("tables is a list", isinstance(data.get("tables"), list))
    check("count matches tables length",
          data.get("count") == len(data.get("tables", [])))

    if data.get("tables"):
        # If documents are loaded, we should have table names
        first_table = data["tables"][0]
        check("first table is a string", isinstance(first_table, str))
        print(f"        Found {data['count']} tables: {data['tables'][:5]}...")


def test_simple_query():
    """Test POST /api/query with a simple question."""
    print("\n--- Test: Simple Query ---")

    payload = {
        "query": "What is the grain of DIM_STORE?",
        "top_k": 3,
        "rerank": True,
        "include_sources": True,
    }

    start = time.time()
    r = requests.post(f"{BASE_URL}/api/query", json=payload)
    elapsed = time.time() - start

    check("status code is 200", r.status_code == 200, f"got {r.status_code}")

    data = r.json()
    check("answer is a non-empty string",
          isinstance(data.get("answer"), str) and len(data.get("answer", "")) > 0,
          f"got: {repr(data.get('answer', '')[:50])}")

    check("sources is a list", isinstance(data.get("sources"), list))
    check("routing info present", data.get("routing") is not None)
    check("timing info present", data.get("timing") is not None)

    if data.get("routing"):
        routing = data["routing"]
        check("query_type is classified",
              routing.get("query_type") in ("single_table", "cross_entity",
                                             "edge_case", "follow_up", "unknown"))
        check("model is identified", len(routing.get("model", "")) > 0)

    if data.get("timing"):
        timing = data["timing"]
        check("total_ms > 0", timing.get("total_ms", 0) > 0)

    print(f"        Answer: {data.get('answer', '')[:80]}...")
    print(f"        Latency: {elapsed:.2f}s (HTTP round-trip)")


def test_query_without_sources():
    """Test POST /api/query with include_sources=false."""
    print("\n--- Test: Query Without Sources ---")

    payload = {
        "query": "What tables are in the data warehouse?",
        "include_sources": False,
    }

    r = requests.post(f"{BASE_URL}/api/query", json=payload)

    check("status code is 200", r.status_code == 200)

    data = r.json()
    check("answer is present", len(data.get("answer", "")) > 0)
    check("sources list is empty", len(data.get("sources", ["x"])) == 0)


def test_query_with_session():
    """Test POST /api/query with session_id for conversation memory."""
    print("\n--- Test: Session-Based Conversation ---")

    session_id = f"test-session-{int(time.time())}"

    # First query: establish context
    r1 = requests.post(f"{BASE_URL}/api/query", json={
        "query": "What is DIM_STORE?",
        "session_id": session_id,
    })
    check("first query succeeds", r1.status_code == 200)

    # Second query: follow-up (should use memory)
    r2 = requests.post(f"{BASE_URL}/api/query", json={
        "query": "What are its foreign keys?",
        "session_id": session_id,
    })
    check("follow-up query succeeds", r2.status_code == 200)

    data2 = r2.json()
    if data2.get("routing"):
        is_followup = data2["routing"].get("is_follow_up", False)
        print(f"        Follow-up detected: {is_followup}")
        # Note: whether this is classified as follow-up depends on your
        # query router logic. We just check the endpoint works.


def test_validation_errors():
    """Test that invalid requests return 422 errors."""
    print("\n--- Test: Validation Errors ---")

    # Missing required field
    r1 = requests.post(f"{BASE_URL}/api/query", json={})
    check("missing query -> 422", r1.status_code == 422,
          f"got {r1.status_code}")

    # Query too long (>2000 chars)
    r2 = requests.post(f"{BASE_URL}/api/query", json={
        "query": "x" * 2001,
    })
    check("query too long -> 422", r2.status_code == 422,
          f"got {r2.status_code}")

    # Invalid top_k
    r3 = requests.post(f"{BASE_URL}/api/query", json={
        "query": "valid question",
        "top_k": 0,
    })
    check("top_k=0 -> 422", r3.status_code == 422,
          f"got {r3.status_code}")

    # Wrong type for top_k
    r4 = requests.post(f"{BASE_URL}/api/query", json={
        "query": "valid question",
        "top_k": "not_a_number",
    })
    check("top_k='string' -> 422", r4.status_code == 422,
          f"got {r4.status_code}")


def test_stats_endpoint():
    """Test GET /api/stats returns analytics."""
    print("\n--- Test: Stats Endpoint ---")

    r = requests.get(f"{BASE_URL}/api/stats")

    check("status code is 200", r.status_code == 200, f"got {r.status_code}")

    data = r.json()
    check("total_queries is an int",
          isinstance(data.get("total_queries"), int))
    check("avg_latency_ms is a number",
          isinstance(data.get("avg_latency_ms"), (int, float)))
    check("total_cost_usd is a number",
          isinstance(data.get("total_cost_usd"), (int, float)))


def test_feedback():
    """Test POST /api/feedback accepts ratings."""
    print("\n--- Test: Feedback Endpoint ---")

    r = requests.post(f"{BASE_URL}/api/feedback", json={
        "query": "What is DIM_STORE?",
        "answer": "DIM_STORE is a dimension table.",
        "rating": 5,
        "comment": "Integration test feedback",
    })

    check("status code is 200", r.status_code == 200, f"got {r.status_code}")

    data = r.json()
    check("status is 'ok'", data.get("status") == "ok")


def test_docs_page():
    """Test that the auto-generated docs page loads."""
    print("\n--- Test: Swagger Docs ---")

    r = requests.get(f"{BASE_URL}/docs")
    check("/docs returns 200", r.status_code == 200, f"got {r.status_code}")
    check("/docs returns HTML", "text/html" in r.headers.get("content-type", ""))

    r2 = requests.get(f"{BASE_URL}/openapi.json")
    check("/openapi.json returns 200", r2.status_code == 200)
    check("OpenAPI spec has paths", "paths" in r2.json())


# =====================================================================
# SECTION 4: MAIN RUNNER
# =====================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("STTM RAG API -- INTEGRATION TESTS")
    print("=" * 60)
    print(f"Target: {BASE_URL}")

    # Check server is running
    if not is_server_running():
        print(f"\nERROR: Cannot connect to {BASE_URL}")
        print("Start the server first:")
        print("  uv run uvicorn api_server:app --port 8000")
        print("  -- or --")
        print("  docker compose up")
        sys.exit(1)

    print("Server is reachable. Running tests...\n")

    # Run all tests
    test_health_check()
    test_list_tables()
    test_simple_query()
    test_query_without_sources()
    test_query_with_session()
    test_validation_errors()
    test_stats_endpoint()
    test_feedback()
    test_docs_page()

    # Summary
    total = passed + failed
    print(f"\n{'=' * 60}")
    print(f"RESULTS: {passed}/{total} passed, {failed}/{total} failed")
    print(f"{'=' * 60}")

    if failed > 0:
        print("\nSome tests failed. Check the FAIL messages above.")
        sys.exit(1)
    else:
        print("\nAll tests passed. Your API is working correctly.")
        sys.exit(0)