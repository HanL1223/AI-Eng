"""
Evals script for RAG

HOW TO RUN
python eval.py                          # Basic run, keyword scoring
python eval.py --llm-judge              # Add smart LLM grading (costs API $)
python eval.py --category simple_lookup  # Run only one category
python eval.py --tag "baseline"         # Label this run for comparison
python eval.py --compare fileA.csv fileB.csv  # Compare two runs

"""
import os
import csv
import json
import time
import argparse

try:
    import anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False

from rag import (
    load_documents,      # Loads .txt, .md, .xlsx from docs/ folder
    chunk_text,          # Splits documents into overlapping chunks
    build_vector_store,  # Stores chunks in ChromaDB with embeddings
    retrieve,            # Queries ChromaDB for relevant chunks
    ask_claude,          # Sends query + context to Claude API
    extract_table_name,  # Detects table names mentioned in questions
    DOCS_DIR,            # "docs" — where your STTM files live
    TOP_K,               # 3 — how many chunks to retrieve (default)
)

#Loading Eval question

def load_eval_questions(filepath:str,category_filter:str = None) -> list[dict]:
    """
    Load test questions from a CSV file into a list of dictionaries.
    Optionally filter to only one category (e.g., "simple_lookup").
    """
    questions = []

    with open(filepath,"r",encoding='utf-8') as f:
        reader = csv.DictReader(f)

        for row in reader:
            if category_filter and row['category'] != category_filter:
                continue
            row['expected_keywords'] = [
                kw.strip().upper() for kw in row["expected_answer_keywords"].split(",")
                if kw.strip()
            ]
            questions.append(row)
    print(f"Loaded {len(questions)} eval questions from {filepath}")
    return questions

#Scoring method 1 - Keyword matching
"""
For each question, we define a set of keywords the answer SHOULD contain.
Then we check: how many of those keywords actually appear

This has many limitation should be use as baseline only
"""

def score_keyword(answer:str,expected_keywords:list[str]) -> dict:
    """
    Score an answer by checking how many expected keywords are present
    """

    answer_upper = answer.upper()
    matched = []
    missed = []

    for keyword in expected_keywords:
        if keyword in answer_upper:
            matched.append(keyword)
        else:
            missed.append(keyword)

    total = len(expected_keywords)
    #if condition to mitigate subtract by zero issue
    score = len(matched) / total if total > 0 else 0.0

    return {
        "score" : round(score,2),
        "matched": matched,
        "missed" :missed,
    }


#Scoring Method 2 - Edge Case
"""
THE PROBLEM WITH KEYWORD SCORING FOR "I DON'T KNOW" QUESTIONS

Edge case questions ask about things NOT in the documents.

The correct answer is "I don't have that information."
"""

def score_edge_case(answer:str,category:str) -> dict|None:
    """
    Score edge case questions by detecting decline vs hallucination.

    This function ONLY applies to "edge_case" category questions.
    For other categories, it returns None (not applicable).
    """
    if category != "edge_case":
        return None
    answer_upper = answer.upper()
    #Define decline phrases
    #phrases that indicate the model correctly recognised the information isn't in the provided context.
    decline_phrases = [
        "I DON'T HAVE",
        "DON'T HAVE THAT INFORMATION",
        "NOT FOUND IN",
        "NOT IN THE LOADED",
        "CANNOT FIND",
        "NO INFORMATION",
        "NOT AVAILABLE IN",
        "NOT CONTAINED IN",
        "COULDN'T FIND",
        "UNABLE TO FIND",
        "DON'T SEE",
        "NOT PRESENT",
        "NOT MENTIONED",
        "NO MENTION OF",
    ]
    #Define confident phrases
    #These are phrases that suggest the model is presenting facts.
    confident_phrases = [
        "THE GRAIN IS",
        "THE GRAIN OF",
        "THE TABLE CONTAINS",
        "THE COLUMNS ARE",
        "THE SOURCE SYSTEM IS",
        "IS REFRESHED",
        "THE PRIMARY KEY IS",
        "HERE ARE THE",
        "THIS TABLE",
        "THE TABLE IS",
        "COLUMNS INCLUDE",
    ]
    correctly_declined = any(phrase in answer_upper for phrase in decline_phrases)
    sounds_confident = any(phrase in answer_upper for phrase in confident_phrases)

    #Apply to scoring matrix
    if correctly_declined and not sounds_confident:
        return {"score": 1.0, "label": "CORRECT_DECLINE"}

    elif correctly_declined and sounds_confident:
        return {"score": 0.5, "label": "MIXED_RESPONSE"}

    else:
        return {"score": 0.0, "label": "HALLUCINATED"}


#Scoring method 3 - LLM As Judge
# For each eval question, there are TWO API calls:
#   Call 1 (normal RAG): question → retrieve → ask_claude → answer
#   Call 2 (judge):      question + answer → judge_prompt → score + explanation
def score_with_llm_judge(
        question:str,
        answer:str,
        expected_keywords:list[str],
        category:str,
) -> dict:
    """
    Use a separate Claude API call to grade the answer on a 1-5 scale.
    question: The original eval question ("What is the grain of...")
    answer: The model's full text response to evaluate
    expected_keywords: What the answer should contain (gives the judge context)
    category: "simple_lookup" | "cross_entity" | "edge_case"
              (the judge uses different grading criteria per category)

    RETURNS
    On success:
    {
        "score": 0.75,        # Normalized to 0.0-1.0 (consistent with keyword scorer)
        "raw_score": 4,       # Original 1-5 scale (easier for humans to read)
        "explanation": "Correct grain identified, missing refresh detail"
    }

    On error (API failure, parsing failure, etc.):
    {
        "score": None,
        "raw_score": None,
        "explanation": "Judge error: <error message>"
    }
    """
    #Guard to check if SDK is available
    if not HAS_ANTHROPIC:
        return {"score":None,"raw_score":None,
                "explanation":"anthropic not available"}
    
    client = anthropic.Anthropic()
    judge_prompt = f"""You are evaluating a RAG chatbot's answer about data warehouse documentation.

QUESTION:{question}
CATEGORY:{category}
EXPECTED KEYWORDS/CONCEPTS: {', '.join(expected_keywords)}
ANSWER TO EVALUATE:{answer}

SCORING CRITERIA BY CATEGORY:

For "simple_lookup": 
  Does the answer correctly state the requested fact from the STTM documents?
  5 = correct, well-sourced, includes relevant detail
  3 = partially correct, missing some key info
  1 = wrong, hallucinated, or pulled from wrong table

For "cross_entity": 
  Does the answer correctly synthesize information across multiple tables or sources?
  5 = comprehensive, accurately connects multiple tables, cites sources
  3 = gets some relationships right but misses others
  1 = fails to connect tables, or makes up relationships

For "edge_case": 
  Does the answer correctly say "I don't know" or "not in the documents"?
  5 = clearly and confidently declines to answer, explains info is not available
  3 = hedges but partially declines
  1 = confidently provides made-up information (hallucination)

You MUST provide your response in EXACTLY this format:
SCORE: [1-5]
EXPLANATION: [one sentence explaining your score]

"""

    #Calling API
    # max_tokens=200 is intentionally low:
    #   - We only need "SCORE: 4\nEXPLANATION: one sentence"
    #   - Low token limit saves money and prevents rambling
    #   - If the judge tried to write a long analysis, it would
    #     get cut off — but our parsing only needs the first few lines
    #
    # We wrap everything in try/except because API calls can fail:
    #   - Network timeout
    #   - Rate limiting (429 error — too many requests)
    #   - Server error (500 error)
    #   - Invalid API key
    # On failure, we return None scores rather than crashing the whole eval.

    try:
        response = client.messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens =200,
            messages = [{"role":'user',"content":judge_prompt}]
        )
        judge_text = response.content[0].text.strip()

        score = None
        explanation = ""

        for line in judge_text.split("\n"):
            line = line.strip()
            if line.startswith("SCORE:"):
                try:
                    score = int(line.replace("SCORE:","").strip())
                    score = max(1, min(5, score))
                except ValueError:
                    score = None
            elif line.startswith("EXPLANATION:"):
                explanation = line.replace("EXPLANATION:","").strip()
        normalized = (score - 1) / 4.0 if score else None
        return {
        "score": round(normalized, 2) if normalized is not None else None,
        "raw_score": score,
        "explanation": explanation,
    }
    except Exception as e:
        return {"score": None, "raw_score": None, "explanation": f"Judge error: {e}"}


#Main Eval Runner


def run_evaluation(
        questions:list[dict],
        collection,
        known_tables:list[str],
        use_llm_judge: bool = False,
        top_k: int = TOP_K,
) -> list[dict]:
     """
    Run every eval question through the full RAG pipeline and score it.

    THE EVALUATION LOOP (what happens for each question)
    ────────────────────────────────────────────────────
    For EACH of the 20 questions:
    
      Step 1: extract_table_name()   → Detect which table the question asks about
      Step 2: retrieve()             → Find relevant chunks in ChromaDB  
      Step 3: ask_claude()           → Send question + chunks to Claude, get answer
      Step 4: score_keyword()        → Check for expected keywords in answer
      Step 5: score_edge_case()      → Special scoring for "I don't know" questions
      Step 6: score_with_llm_judge() → Optional: ask Claude to grade the answer
      Step 7: Record EVERYTHING      → Save all data for later analysis

    CRITICAL POINT ABOUT TESTING THE REAL CODE PATH
    ────────────────────────────────────────────────
    Steps 1-3 call the EXACT SAME functions in current chatbox
    imported them from rag.py. If eval.py had its own retrieval
    code

    PARAMETERS
    ──────────
    questions:      List of eval question dicts from load_eval_questions()
    collection:     Your ChromaDB collection (from build_vector_store())
    known_tables:   List of table names for extract_table_name()
    use_llm_judge:  If True, also run LLM-as-judge scoring (costs API $)
    top_k:          How many chunks to retrieve (default from rag.py, overridable)

    RETURNS
    ───────
    A list of result dicts, one per question. Each contains:
    the question, answer, all scores, timing data, and retrieved sources.
    This entire list gets saved to CSV by save_results().
    """
     results = []
     total = len(questions)

     for i,q in enumerate(questions):
         question_id = q['question_id']
         category = q['category']
         question = q['question']
         expected_keywords = q['expected_keywords']
         expected_table = q.get("expected_table","")

         print(f"\n[{i+1}/{total}] {question_id}: {question[:60]}...")

         #Step 1 Table name detection
         detected_table = extract_table_name(question,known_tables)

         #Step 2 Retrieval - Finding relevant chunks
         start_time = time.time()
         chunks = retrieve(collection,question,top_k=top_k,table_name=detected_table)
         retrieval_time = time.time() - start_time

         retrieved_sources = [
            {
                "table_name": c.get("table_name", ""),
                "doc_type": c.get("doc_type", ""),
                "text_preview": c["text"][:150],
            }
            for c in chunks
        ]
         
         #Step3 Generation
         generation_start = time.time()
         try:
             answer = ask_claude(question,chunks)
         except Exception as e:
             answer = f"Error {e}"
         generation_time = time.time() - generation_start
         total_time = retrieval_time + generation_time

         print(f"  Answer ({total_time:.1f}s): {answer[:100]}...")

         #Step 4 Keyword scoring
         #This is a free method so always run as a baseline
         keyword_result = score_keyword(answer, expected_keywords)
         print(f"  Keyword score: {keyword_result['score']} "
              f"(matched: {keyword_result['matched']}, "
              f"missed: {keyword_result['missed']})")
         #Step 5 EDGE CASE SCORING (only for edge_case questions)
         edge_result = score_edge_case(answer, category)
         if edge_result:
            print(f"  Edge case: {edge_result['score']} ({edge_result['label']})")

         #Step 6 LLM Scoring
         judge_result = None

         if use_llm_judge:
             judge_result = score_with_llm_judge(
                 question,answer,expected_keywords,category
             )
             if judge_result["score"] is not None:
                print(f"  Judge: {judge_result['raw_score']}/5 "
                      f"— {judge_result['explanation']}")
         results.append({
            "question_id": question_id,
            "category": category,
            "question": question,
            "expected_table": expected_table,
            "detected_table": detected_table or "",
            "answer": answer,
            # ─── Keyword scores ───
            "keyword_score": keyword_result["score"],
            "keyword_matched": ", ".join(keyword_result["matched"]),
            "keyword_missed": ", ".join(keyword_result["missed"]),
            # ─── Edge case scores (empty string if not applicable) ───
            "edge_case_score": edge_result["score"] if edge_result else "",
            "edge_case_label": edge_result["label"] if edge_result else "",
            # ─── Judge scores (empty string if judge not used) ───
            "judge_score": judge_result["score"] if judge_result else "",
            "judge_raw": judge_result["raw_score"] if judge_result else "",
            "judge_explanation": judge_result["explanation"] if judge_result else "",
            # ─── Timing data (seconds) ───
            "retrieval_time_s": round(retrieval_time, 2),
            "generation_time_s": round(generation_time, 2),
            "total_time_s": round(total_time, 2),
            # ─── Retrieval debug data ───
            "num_chunks_retrieved": len(chunks),
            "retrieved_sources": json.dumps(retrieved_sources),
        })
     return results

#SECTION 6: SAVE RESULTS TO CSV
def save_results(results: list[dict], tag: str = "") -> str:
    """Format: eval_results/eval_results_YYYYMMDD_HHMMSS_tag.csv"""
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag_suffix = f"_{tag}" if tag else ""
    filename = f"eval_results_{timestamp}{tag_suffix}.csv"

    os.makedirs("eval_results",exist_ok = True)
    filepath = os.path.join("eval_results",filename)

    #Write to CSV
    if results:
        fieldnames = results[0].keys()
        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)

    print(f"\nResults saved to: {filepath}")
    return filepath

#SECTION 7: PRINT SUMMARY REPORT 
def print_summary(results: list[dict]):
    """
    Print a human-readable dashboard of eval results.
    
    Shows five sections:
    1. Overall keyword score (single number system health check)
    2. Per-category breakdown (WHERE is the system weak?)
    3. Table detection accuracy (is extract_table_name working?)
    4. Worst performing questions (WHAT to fix next?)
    5. Timing statistics (HOW FAST is the system?)
    """
    if not results:
        print("No results to summarize.")
        return

    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)

    #1. Overall keyword score
    keyword_scores = [r["keyword_score"] for r in results]
    avg_keyword = sum(keyword_scores) / len(keyword_scores)
    print(f"\nOverall Keyword Score: {avg_keyword:.2f} / 1.00")

    #Per category breakdown
    categories = sorted(set(r['category'] for r in results))
    print(f"  {'Category':<20} {'Avg Score':>10} {'Count':>8}")
    print(f"  {'-'*20} {'-'*10} {'-'*8}")

    for cat in categories:
        cat_results = [r for r in results if r["category"] == cat]
        if cat == "edge_case":
            cat_scores = [
                r["edge_case_score"] for r in cat_results
                if r["edge_case_score"] != ""
            ]
        else:
            cat_scores = [r["keyword_score"] for r in cat_results]
        avg = sum(cat_scores) / len(cat_scores) if cat_scores else 0
        print(f"  {cat:<20} {avg:>10.2f} {len(cat_results):>8}")

    #LLM Judge score
    judge_scores = [
        r["judge_score"] for r in results
        if r["judge_score"] != "" and r["judge_score"] is not None
    ]
    if judge_scores:
        avg_judge = sum(judge_scores) / len(judge_scores)
        print(f"\nLLM Judge Score: {avg_judge:.2f} / 1.00")

    #Table detection
    detection_results = [
        r for r in results
        if r["expected_table"] and r["expected_table"] not in ("NONE", "MULTIPLE")
    ]
    if detection_results:
        correct_detections = sum(
            1 for r in detection_results
            if r["detected_table"] == r["expected_table"]
        )
        detection_rate = correct_detections / len(detection_results)
        print(f"\nTable Detection Accuracy: {detection_rate:.0%} "
              f"({correct_detections}/{len(detection_results)})")
    # Worst performing questions
    print(f"\nWorst Performing Questions (fix these first):")

    sorted_results = sorted(results, key=lambda r: r["keyword_score"])

    for r in sorted_results[:5]:
        score_display = r["keyword_score"]
        if r["category"] == "edge_case" and r["edge_case_label"]:
            score_display = f"{r['edge_case_score']} ({r['edge_case_label']})"

        print(f"  {r['question_id']}: score={score_display}")
        print(f"    Q: {r['question'][:70]}...")
        if r["keyword_missed"]:
            print(f"    Missing keywords: {r['keyword_missed']}")
    #timing
    times = [r["total_time_s"] for r in results]
    avg_time = sum(times) / len(times)
    max_time = max(times)
    print(f"\nTiming: avg={avg_time:.1f}s, max={max_time:.1f}s per question")

    print("\n" + "=" * 60)

#SECTION 8: COMPARISON TOOL


     
def compare_runs(file_a:str,file_b:str):
    def load_results(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            return {row["question_id"]: row for row in reader}
    results_a = load_results(file_a)
    results_b = load_results(file_b)

    common_ids = sorted(set(results_a.keys())&set(results_b.keys()))
    print(f"\nComparing:")
    print(f"  A: {os.path.basename(file_a)}")
    print(f"  B: {os.path.basename(file_b)}")
    print(f"\n{'ID':<6} {'Score A':>8} {'Score B':>8} {'Delta':>8} {'Status'}")
    print("-" * 50)

    improved = 0
    regressed= 0
    same = 0

    for qid in common_ids:
        score_a = float(results_a[qid]["keyword_score"])
        score_b = float(results_b[qid]["keyword_score"])
        delta = score_b - score_a

        if delta > 0.01:
            status = "IMPROVED"
            improved += 1
        elif delta < -0.01:
            status = "REGRESSED"
            regressed += 1
        else:
            status = "SAME"
            same += 1

        print(f"{qid:<6} {score_a:>8.2f} {score_b:>8.2f} {delta:>+8.2f} {status}")
    print(f"\nSummary: {improved} improved, {regressed} regressed, {same} unchanged")
    print(f"Total compared: {len(common_ids)} questions")




def main():
    parser = argparse.ArgumentParser(description="RAG Evaluation Framework")
    parser.add_argument(
        "--questions",default = "eval_questions.csv",
        help = "Path to eval questions CSV file"
    )
    parser.add_argument(
        "--llm-judge", action="store_true",
        help="Enable LLM-as-judge scoring (costs extra API calls)"
    )
    parser.add_argument(
        "--category", default=None,
        help="Run only one category: simple_lookup, cross_entity, or edge_case"
    )
    parser.add_argument(
        "--tag", default="",
        help="Label for this run, used in filename (e.g. 'baseline', 'chunk500')"
    )
    parser.add_argument(
        "--top-k", type=int, default=TOP_K,
        help=f"Override TOP_K retrieval count (default: {TOP_K})"
    )
    parser.add_argument(
        "--compare", nargs=2, metavar=("FILE_A", "FILE_B"),
        help="Compare two result CSV files instead of running eval"
    )

    args = parser.parse_args()

     # ─── Comparison mode: compare two files and exit ───
    # If --compare was provided, we don't run an eval at all.
    # Just load two result files, compare them, and exit.

    if args.compare:
        compare_runs(args.compare[0], args.compare[1])
        return
    if not os.environ.get("ANTHROPIC_API_KEY"):
        env_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), ".env"
        )
        if os.path.exists(env_path):
            with open(env_path) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        key, _, value = line.partition("=")
                        os.environ[key.strip()] = value.strip()
    print("RAG EVALUATION FRAMEWORK")
    print("=" * 60)

    questions = load_eval_questions(args.questions, category_filter=args.category)
    if not questions:
        print("No questions found! Check your CSV file path.")
        return
    
    print("\nBuilding RAG pipeline (same as chatbot startup)...")

    # Step 2a: Load documents
    documents = load_documents(DOCS_DIR)
    if not documents:
        print(f"No documents found in {DOCS_DIR}/")
        return
    all_chunks = []
    for doc in documents:
        chunks = chunk_text(doc["content"], doc["source"])
        all_chunks.extend(chunks)
    collection = build_vector_store(all_chunks)

    all_meta = collection.get()
    known_tables = sorted(list(set(
        m.get("table_name", "")
        for m in all_meta["metadatas"]
        if m.get("table_name") and m["table_name"].strip()
    )))

    print(f"Pipeline ready: {len(all_chunks)} chunks from "
          f"{len(documents)} documents, {len(known_tables)} tables indexed")
    # ─── Step 3: Run the evaluation loop ───

    print(f"\nRunning evaluation ({len(questions)} questions)...")
    if args.llm_judge:
        print("  LLM-as-judge: ENABLED (extra API call per question)")
    else:
        print("  LLM-as-judge: DISABLED (keyword scoring only)")
        print("  Tip: use --llm-judge for smarter scoring (costs API $)")

    results = run_evaluation(
        questions=questions,
        collection=collection,
        known_tables=known_tables,
        use_llm_judge=args.llm_judge,
        top_k=args.top_k,
    )

    # ─── Step 4: Save results and print summary dashboard ───

    save_results(results, tag=args.tag)
    print_summary(results)

    # ─── Remind user about the experiment workflow ───
    print("\nNext steps:")
    print("  1. Review the worst-performing questions above")
    print("  2. Change ONE setting in rag.py (chunk_size, TOP_K, or prompt)")
    print("  3. Run eval again: python eval.py --tag 'your_change_name'")
    print("  4. Compare: python eval.py --compare eval_results/old.csv eval_results/new.csv")

if __name__ == "__main__":
    main()







    

         
         

         
    



