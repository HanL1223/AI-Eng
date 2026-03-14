"""eval.py — RAG Evaluation Framework for STTM Chatbot

PURPOSE OF THIS FILE
════════════════════
Your rag.py builds and runs the chatbot.
This file MEASURES how good the chatbot is.

Think of it like a driving test for your RAG system:
    - eval_questions.csv = the test paper (20 questions)
    - eval.py = the examiner (runs each question, grades the answer)
    - eval_results/ = the report card (saved scores for comparison)

WHY EVALUATION MATTERS (The Core Insight)
═════════════════════════════════════════
Without eval, every change to your RAG system is guesswork:
    "I changed chunk_size from 800 to 500... does it seem better?"

With eval, every change is measurable:
    "chunk_size 500 improved 4 questions, broke 1, same on 15."

This is the difference between vibe-based and data-driven engineering.
Every serious ML team follows this loop:

    measure → change ONE thing → measure again → compare → decide

This file gives you the "measure" and "compare" steps.

HOW TO RUN
══════════
  python eval.py                          # Basic run, keyword scoring
  python eval.py --llm-judge              # Add smart LLM grading (costs API $)
  python eval.py --category simple_lookup  # Run only one category
  python eval.py --tag "baseline"         # Label this run for comparison
  python eval.py --compare fileA.csv fileB.csv  # Compare two runs

DEPENDENCIES
════════════
  - Your existing rag.py (we import its functions directly)
  - eval_questions.csv (your test questions)
  - anthropic SDK (you already have this from rag.py)
"""

# ═══════════════════════════════════════════════════════════
# IMPORTS
# ═══════════════════════════════════════════════════════════
#
# PYTHON REFRESHER: What "import" actually does
# ──────────────────────────────────────────────
# When Python sees `import csv`, it:
#   1. Finds the csv.py file in the standard library
#   2. Executes all the code in that file
#   3. Creates a "module object" you can reference as csv.xxx
#
# When Python sees `from rag import load_documents`, it:
#   1. Finds rag.py in the same folder
#   2. Executes ALL the code in rag.py (not just load_documents)
#   3. But only makes load_documents available in THIS file
#
# GOTCHA: "Executes ALL the code" means if rag.py had any
# code outside of functions (like print statements at the
# top level), that code would RUN during import. That's why
# rag.py protects its main() with `if __name__ == "__main__"`.
# Without that guard, importing rag.py would start the chatbot!
# ──────────────────────────────────────────────────────────

import os       # File paths, environment variables
import csv      # Read/write CSV files (our eval data format)
import json     # Serialize retrieved_sources to a string for CSV storage
import time     # time.time() for measuring how long each question takes
import argparse # Parse command-line flags like --llm-judge, --tag

# ─── Import from YOUR existing rag.py ───
# 
# KEY DESIGN PRINCIPLE: We import the SAME functions your chatbot uses.
# eval.py does NOT re-implement retrieval or generation.
# If eval.py had its own retrieval code, you'd be testing
# different code than what actually runs — your scores would be lies.
#
# This is called "testing the real code path" and it's critical.
#
# We also import your CONFIG constants (DOCS_DIR, TOP_K) so that
# eval runs with the same settings as your chatbot by default.
# You can override TOP_K with --top-k flag for experiments.

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

# ─── Import anthropic for the optional LLM-as-judge feature ───
#
# PYTHON REFRESHER: try/except around imports
# ────────────────────────────────────────────
# Sometimes a library might not be installed. Wrapping the import
# in try/except lets the script still run — it just disables
# the features that need that library.
#
# This is a "graceful degradation" pattern:
#   - anthropic installed → keyword scoring + LLM judge both work
#   - anthropic missing   → keyword scoring still works, judge disabled
#
# In practice, you WILL have anthropic installed (rag.py needs it),
# but this pattern is good defensive coding to learn.

try:
    import anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False


# ═══════════════════════════════════════════════════════════
# SECTION 1: LOADING EVAL QUESTIONS
# ═══════════════════════════════════════════════════════════
# 
# Your eval questions live in a CSV file. Each row is one test case
# with a question, expected keywords, and metadata about what
# category it belongs to.
#
# WHY CSV AND NOT XLSX OR JSON?
# ─────────────────────────────
# CSV is the simplest tabular format:
#   - No library needed to read it (csv is built into Python)
#   - You can open and edit it in Excel, Google Sheets, or any text editor
#   - Easy to version control with git (it's just text)
#   - Easy to diff ("what questions did I add since last week?")
#
# JSON would work too, but CSV is more natural for tabular data
# and easier to eyeball in a spreadsheet program.
# ═══════════════════════════════════════════════════════════

def load_eval_questions(filepath: str, category_filter: str = None) -> list[dict]:
    """
    Load test questions from a CSV file into a list of dictionaries.
    Optionally filter to only one category (e.g., "simple_lookup").

    PYTHON REFRESHER: csv.DictReader vs csv.reader
    ───────────────────────────────────────────────
    csv.reader gives you raw lists:
        row = ["Q01", "simple_lookup", "What is the grain of..."]
        # You access by position: row[0], row[1], row[2]
        # Fragile! If you add a column, all positions shift.

    csv.DictReader gives you dictionaries:
        row = {"question_id": "Q01", "category": "simple_lookup", ...}
        # You access by name: row["question_id"]
        # Safe! Adding a column doesn't break existing code.

    DictReader uses the FIRST ROW of the CSV as dictionary keys.
    So your CSV must have a header row.

    PYTHON REFRESHER: Default parameter values
    ───────────────────────────────────────────
    `category_filter: str = None` means:
        - If you call load_eval_questions("file.csv") → category_filter is None
        - If you call load_eval_questions("file.csv", "edge_case") → category_filter is "edge_case"
    
    None is Python's way of saying "no value" — we use it to mean
    "don't filter, load all questions."

    RETURNS
    ───────
    A list of dicts, each representing one eval question:
    [
        {
            "question_id": "Q01",
            "category": "simple_lookup",
            "question": "What is the grain of FACT_STORE_INVENTORY_INTRA?",
            "expected_keywords": ["GRAIN", "STORE", "PRODUCT", "DAILY"],
            ... other fields from CSV ...
        },
        ...
    ]
    """
    questions = []

    # ─── Open and read the CSV file ───
    #
    # PYTHON REFRESHER: encoding="utf-8"
    # ───────────────────────────────────
    # Text files can be stored in different character encodings.
    # UTF-8 is the modern universal standard that handles all languages.
    # Always specify it explicitly — the default varies by OS:
    #   - Linux/Mac: usually utf-8 (fine)
    #   - Windows: often cp1252 (can break on special characters)
    # Being explicit avoids "it works on my machine" bugs.

    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:

            # ─── Apply category filter if specified ───
            #
            # If the user ran: python eval.py --category simple_lookup
            # we skip any row that's NOT simple_lookup.
            # This lets you test just one category quickly during debugging.

            if category_filter and row["category"] != category_filter:
                continue
            
            # ─── Parse expected keywords from CSV string into a Python list ───
            #
            # In the CSV, keywords are stored as a comma-separated string:
            #     "grain,store,product,day,daily"
            #
            # We need to convert this to a Python list for scoring later:
            #     ["GRAIN", "STORE", "PRODUCT", "DAY", "DAILY"]
            #
            # PYTHON REFRESHER: Method chaining and list comprehensions
            # ─────────────────────────────────────────────────────────
            # Let's break down this one-liner step by step:
            #
            #   row["expected_answer_keywords"]          → "grain,store,product"
            #       .split(",")                          → ["grain", "store", "product"]
            #
            # Then the list comprehension:
            #   [kw.strip().upper() for kw in [...] if kw.strip()]
            #
            # This does THREE things to each keyword:
            #   kw.strip()   → removes whitespace: " grain " → "grain"
            #   .upper()     → converts to uppercase: "grain" → "GRAIN"
            #   if kw.strip() → skips empty strings (from trailing commas)
            #
            # We uppercase everything so scoring is case-insensitive:
            # the answer "Daily refresh" will match keyword "DAILY"
            # because we also uppercase the answer during comparison.

            row["expected_keywords"] = [
                kw.strip().upper()
                for kw in row["expected_answer_keywords"].split(",")
                if kw.strip()
            ]

            questions.append(row)

    print(f"Loaded {len(questions)} eval questions from {filepath}")
    return questions


# ═══════════════════════════════════════════════════════════
# SECTION 2: SCORING METHOD 1 — KEYWORD MATCHING
# ═══════════════════════════════════════════════════════════
#
# THE IDEA
# ────────
# For each question, we define a set of keywords the answer SHOULD contain.
# Then we check: how many of those keywords actually appear?
#
# EXAMPLE
# ───────
# Question: "What is the grain of FACT_STORE_INVENTORY_INTRA?"
# Expected keywords: ["GRAIN", "STORE", "PRODUCT", "DAILY"]
# Model answer: "The grain is at the store-product level, refreshed daily."
#
# Checking:  GRAIN ✓  STORE ✓  PRODUCT ✓  DAILY ✓ → Score: 4/4 = 1.0
#
# STRENGTHS
# ─────────
# - FREE: No API calls needed
# - FAST: Instant string matching
# - DETERMINISTIC: Same input always gives same score
#   (LLM scoring is non-deterministic — slight variations each run)
#
# WEAKNESSES (important to understand!)
# ──────────────────────────────────────
# - FALSE POSITIVES: "I don't know about DIM_STORE" matches "DIM_STORE"
#   even though the answer is wrong. The keyword is present but the
#   answer is a DECLINE, not a correct lookup.
#
# - SYNONYM BLINDNESS: If the model says "everyday" but we expected
#   "DAILY", keyword scoring gives 0. A human would give full marks.
#
# - NO QUALITY ASSESSMENT: "The grain is store-product-day, from SAP,
#   refreshed daily via ADF" scores the same as "store product daily"
#   even though the first answer is much more useful.
#
# Despite these weaknesses, keyword scoring is your WORKHORSE scorer.
# Use it for every run. It catches most regressions for free.
# ═══════════════════════════════════════════════════════════

def score_keyword(answer: str, expected_keywords: list[str]) -> dict:
    """
    Score an answer by checking how many expected keywords are present.

    PYTHON REFRESHER: The `in` operator for strings
    ────────────────────────────────────────────────
    "GRAIN" in "THE GRAIN IS DAILY" → True
    "GRAIN" in "the grain is daily" → False (case sensitive!)
    
    That's why we uppercase BOTH the answer and the keywords:
        answer_upper = answer.upper()        → "THE GRAIN IS DAILY"
        keyword = "GRAIN"                    → "GRAIN" 
        "GRAIN" in "THE GRAIN IS DAILY"      → True ✓

    PARAMETERS
    ──────────
    answer: str
        The full text response from Claude (or any model).
    expected_keywords: list[str]
        Already uppercased list like ["GRAIN", "STORE", "DAILY"]

    RETURNS
    ───────
    A dictionary with three fields:
    {
        "score": 0.75,                  # Float 0.0 to 1.0 (percentage matched)
        "matched": ["GRAIN", "STORE", "DAILY"],   # Keywords that WERE found
        "missed": ["REFRESH"],                      # Keywords that were NOT found
    }

    The matched/missed lists are critical for debugging:
    if you see "missed: DAILY" across multiple questions,
    maybe your chunks are cutting off the refresh cadence info.
    """
    # Uppercase the answer once, reuse for all keyword checks.
    #
    # WHY NOT uppercase inside the loop?
    # ──────────────────────────────────
    # answer.upper() creates a NEW string every time it's called.
    # If you have 5 keywords, calling answer.upper() 5 times creates
    # 5 identical copies. Doing it once and storing the result is
    # both cleaner and slightly faster. This is called "hoisting"
    # a computation out of a loop.

    answer_upper = answer.upper()

    matched = []
    missed = []

    for keyword in expected_keywords:
        if keyword in answer_upper:
            matched.append(keyword)
        else:
            missed.append(keyword)

    # ─── Calculate the score as a ratio ───
    #
    # PYTHON REFRESHER: Guarding against division by zero
    # ───────────────────────────────────────────────────
    # If expected_keywords is empty (someone left the field blank in CSV),
    # len(expected_keywords) is 0, and dividing by 0 crashes Python:
    #     ZeroDivisionError: division by zero
    #
    # The pattern `x / total if total > 0 else 0.0` is a conditional
    # expression (ternary operator):
    #     value_if_true IF condition ELSE value_if_false
    #
    # It's equivalent to:
    #     if total > 0:
    #         score = len(matched) / total
    #     else:
    #         score = 0.0
    # 
    # round(score, 2) limits to 2 decimal places: 0.666666 → 0.67

    total = len(expected_keywords)
    score = len(matched) / total if total > 0 else 0.0

    return {
        "score": round(score, 2),
        "matched": matched,
        "missed": missed,
    }


# ═══════════════════════════════════════════════════════════
# SECTION 3: SCORING METHOD 2 — EDGE CASE DETECTION
# ═══════════════════════════════════════════════════════════
#
# THE PROBLEM WITH KEYWORD SCORING FOR "I DON'T KNOW" QUESTIONS
# ──────────────────────────────────────────────────────────────
# Edge case questions ask about things NOT in your documents.
# The correct answer is "I don't have that information."
#
# But keyword scoring CANNOT evaluate this properly:
#
#   Question: "What is the grain of FACT_CUSTOMER_RETURNS?"
#   Expected: "I don't have that information"
#   
#   Bad answer: "FACT_CUSTOMER_RETURNS has a grain of daily by store"
#               → This is WRONG (the table doesn't exist, model hallucinated)
#               → But keyword match on "grain" and "store" would score HIGH
#
#   Good answer: "I don't have that information in the loaded documents"
#               → This is CORRECT
#               → But keyword match has nothing to match on
#
# So edge cases need their OWN scoring logic that checks for the
# OPPOSITE of normal questions: we WANT decline phrases, and we
# PENALIZE confident statements.
#
# WHAT IS HALLUCINATION?
# ──────────────────────
# When an LLM generates confident-sounding information that is
# completely made up. In RAG systems, this happens when the model
# ignores the "only use provided context" instruction and draws
# from its general training data instead.
#
# Hallucination is the #1 failure mode in production RAG systems.
# That's why 20-25% of your eval questions should test for it.
# ═══════════════════════════════════════════════════════════

def score_edge_case(answer: str, category: str) -> dict | None:
    """
    Score edge case questions by detecting decline vs hallucination.

    This function ONLY applies to "edge_case" category questions.
    For other categories, it returns None (not applicable).

    PYTHON REFRESHER: Union return types with | (pipe)
    ──────────────────────────────────────────────────
    `-> dict | None` means this function returns EITHER a dict OR None.
    
    The | syntax for types was introduced in Python 3.10.
    In older Python, you'd write: -> Optional[dict] (from typing module).
    
    Returning None to mean "not applicable" is cleaner than returning
    a fake score of 0.0, because 0.0 is a REAL score meaning "terrible."
    None means "this scoring method doesn't apply to this question."

    HOW THE SCORING MATRIX WORKS
    ────────────────────────────
    We check for two types of phrases in the answer:

    1. DECLINE PHRASES — model correctly said "I don't know"
       "I don't have that information", "not found in the loaded documents"
    
    2. CONFIDENT PHRASES — model stated facts (likely hallucinated)
       "The grain is", "The table contains", "The columns are"

    ┌─────────────────┬────────────────┬────────────────────────────┐
    │                 │ No Confident   │ Yes Confident              │
    ├─────────────────┼────────────────┼────────────────────────────┤
    │ Yes Declined    │ 1.0 CORRECT    │ 0.5 MIXED (hedged)         │
    │ No Declined     │ 0.0 MISSED     │ 0.0 HALLUCINATED           │
    └─────────────────┴────────────────┴────────────────────────────┘
    """
    # ─── Only apply to edge_case questions ───
    # Other categories (simple_lookup, cross_entity) are scored
    # by keyword matching, not by decline detection.

    if category != "edge_case":
        return None

    answer_upper = answer.upper()

    # ─── Define decline phrases ───
    #
    # These are phrases that indicate the model correctly recognized
    # the information isn't in the provided context.
    # We check for several variations because Claude might phrase it
    # differently each time — "I don't have" vs "I cannot find" etc.
    #
    # DESIGN NOTE: These phrases match YOUR system prompt instruction:
    # "If the context doesn't contain the answer, say
    #  'I don't have that information in the loaded documents'"
    #
    # GOTCHA: If you change that instruction in rag.py, you MUST
    # update these phrases too. Otherwise the model might use new
    # phrasing that these patterns don't catch, and every edge case
    # would score 0.0 even when the model behaves correctly.

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

    # ─── Define confident phrases ───
    #
    # These are phrases that suggest the model is presenting facts.
    # If the question is about a table that DOESN'T EXIST, and the
    # model says "The grain is daily by store" — that's hallucination.
    #
    # DOMAIN-SPECIFIC: These patterns are specific to YOUR domain (STTM docs).
    # A RAG system about recipes would check for different patterns
    # like "The ingredients are" or "Cook for 20 minutes."
    # When you adapt this framework to a new domain, update these lists.

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

    # ─── Check for matches ───
    #
    # PYTHON REFRESHER: any() with a generator expression
    # ────────────────────────────────────────────────────
    # any(phrase in answer_upper for phrase in decline_phrases)
    #
    # Let's unpack this step by step:
    #
    # Step 1: The generator expression
    #   (phrase in answer_upper for phrase in decline_phrases)
    #   This produces a sequence of True/False values:
    #     True, False, False, True, False, ...
    #   Each True/False corresponds to one phrase from the list.
    #
    # Step 2: any() consumes the sequence
    #   any() returns True if ANY value is True.
    #   It also "short-circuits": stops checking once it finds a True.
    #   If the first phrase matches, it doesn't check the other 13.
    #
    # The long-form equivalent would be:
    #     correctly_declined = False
    #     for phrase in decline_phrases:
    #         if phrase in answer_upper:
    #             correctly_declined = True
    #             break  # Stop early — we found one match
    #
    # The counterpart is all() — returns True only if EVERY item is True.
    # We don't need all() here because ONE decline phrase is enough.

    correctly_declined = any(phrase in answer_upper for phrase in decline_phrases)
    sounds_confident = any(phrase in answer_upper for phrase in confident_phrases)

    # ─── Apply the scoring matrix ───

    if correctly_declined and not sounds_confident:
        # BEST CASE: Model clearly said "I don't know" and didn't
        # also try to answer. Clean decline. Full marks.
        return {"score": 1.0, "label": "CORRECT_DECLINE"}

    elif correctly_declined and sounds_confident:
        # MIXED: Model said something like "I don't have complete info,
        # but the table might contain store data..." — it hedged.
        # Partial credit because it showed SOME uncertainty awareness,
        # but it still leaked potentially hallucinated info.
        return {"score": 0.5, "label": "MIXED_RESPONSE"}

    else:
        # WORST CASE: Model didn't decline at all.
        # Either it confidently hallucinated ("The grain is daily by store")
        # or gave a vague non-answer without declining.
        # Either way, it failed the edge case test.
        return {"score": 0.0, "label": "HALLUCINATED"}


# ═══════════════════════════════════════════════════════════
# SECTION 4: SCORING METHOD 3 — LLM-AS-JUDGE
# ═══════════════════════════════════════════════════════════
#
# THE IDEA
# ────────
# Instead of dumb keyword matching, ask a SEPARATE Claude call
# to grade the answer like a teacher grading an exam.
#
# This is a well-known technique called "LLM-as-a-Judge"
# (research paper: https://arxiv.org/abs/2306.05685).
#
# HOW IT WORKS (the flow for each question)
# ──────────────────────────────────────────
# For each eval question, there are TWO API calls:
#   Call 1 (normal RAG): question → retrieve → ask_claude → answer
#   Call 2 (judge):      question + answer → judge_prompt → score + explanation
#
# The judge sees: the original question, the expected keywords,
# the category type, and the model's actual answer. It then scores
# 1-5 and gives a brief explanation.
#
# WHY A SEPARATE CALL (not the same call)?
# ────────────────────────────────────────
# You can't ask a model to evaluate its own output in the SAME call.
# It's biased — like grading your own homework. A separate call with
# a "judge persona" prompt gives more honest evaluations.
#
# Even with a separate call, there's still some bias (Claude judging
# Claude tends to be generous). In production, you'd use a STRONGER
# model as judge (e.g., Claude Opus judging Claude Sonnet). But for
# learning, same-model judging is good enough and cheaper.
#
# COST MATH
# ─────────
# 20 eval questions × 2 API calls each = 40 API calls per eval run
# At roughly $0.003-0.01 per call, that's ~$0.10-0.30 per eval run.
# Not expensive, but adds up if you run eval 20 times in a session.
# Use --llm-judge only for final assessments, not every iteration.
#
# WHEN TO USE EACH SCORING METHOD
# ────────────────────────────────
# Keyword scoring:  Every run. Free, fast, good for catching regressions.
# LLM judge:        Final quality check, or when keyword scores are ambiguous.
# Edge case scorer: Automatically applied to all edge_case questions.
# ═══════════════════════════════════════════════════════════

def score_with_llm_judge(
    question: str,
    answer: str,
    expected_keywords: list[str],
    category: str,
) -> dict:
    """
    Use a separate Claude API call to grade the answer on a 1-5 scale.

    PYTHON REFRESHER: Functions with many parameters
    ────────────────────────────────────────────────
    When a function has 4+ parameters, Python lets you call it with
    named arguments for clarity:

        score_with_llm_judge(
            question="What is the grain of...",
            answer="The grain is...",
            expected_keywords=["GRAIN", "STORE"],
            category="simple_lookup",
        )

    The trailing comma after the last argument is optional but
    conventional in Python — it makes git diffs cleaner when you
    add/remove parameters later. Without it, adding a parameter
    changes TWO lines in git (the old last line + new last line).
    With the trailing comma, only ONE line changes (the new line).

    PARAMETERS
    ──────────
    question: The original eval question ("What is the grain of...")
    answer: The model's full text response to evaluate
    expected_keywords: What the answer should contain (gives the judge context)
    category: "simple_lookup" | "cross_entity" | "edge_case"
              (the judge uses different grading criteria per category)

    RETURNS
    ───────
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
    # ─── Guard: Check if anthropic SDK is available ───
    # If the import at the top failed, we can't make API calls.
    # Return None scores so the eval continues without judge scoring.

    if not HAS_ANTHROPIC:
        return {"score": None, "raw_score": None,
                "explanation": "anthropic package not available"}

    client = anthropic.Anthropic()

    # ─── Build the judge prompt ───
    #
    # PROMPT ENGINEERING LESSON: Structured output forcing
    # ────────────────────────────────────────────────────
    # We need the judge to return a score AND an explanation.
    # If we just say "evaluate this answer", Claude might write
    # three paragraphs of analysis but never give a clear score.
    #
    # The fix: specify the EXACT format we want:
    #   "You MUST provide your response in EXACTLY this format:
    #    SCORE: [1-5]
    #    EXPLANATION: [one sentence]"
    #
    # The words "EXACTLY" and "MUST" are intentional — they reduce
    # the chance of the model getting creative with the format.
    # This is a common prompt engineering technique for when you need
    # to parse the output programmatically.
    #
    # PROMPT ENGINEERING LESSON: Category-specific rubrics
    # ────────────────────────────────────────────────────
    # Different question types need different grading criteria.
    # A "5" for simple_lookup means "correct fact, well-sourced."
    # A "5" for edge_case means "correctly said I don't know."
    #
    # Without telling the judge the category, it might score
    # an edge case answer of "I don't know" as 1/5 because
    # it "didn't answer the question." The category context
    # tells the judge WHAT GOOD LOOKS LIKE for each type.
    #
    # PYTHON REFRESHER: Multi-line f-strings
    # ──────────────────────────────────────
    # f"""...""" is a multi-line formatted string.
    # {question}, {category}, etc. get replaced with variable values.
    # {', '.join(expected_keywords)} converts the list to a string:
    #   ["GRAIN", "STORE", "DAILY"] → "GRAIN, STORE, DAILY"
    #
    # ', '.join(list) puts ", " BETWEEN each item (not after the last one).

    judge_prompt = f"""You are evaluating a RAG chatbot's answer about data warehouse documentation.

QUESTION: {question}
CATEGORY: {category}
EXPECTED KEYWORDS/CONCEPTS: {', '.join(expected_keywords)}

ANSWER TO EVALUATE:
{answer}

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
EXPLANATION: [one sentence explaining your score]"""

    # ─── Call Claude as the judge ───
    #
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
            max_tokens=200,
            messages=[{"role": "user", "content": judge_prompt}],
        )

        judge_text = response.content[0].text.strip()

        # ─── Parse the judge's structured response ───
        #
        # We expect something like:
        #   "SCORE: 4
        #    EXPLANATION: Correct grain identified but missing refresh info"
        #
        # PYTHON REFRESHER: String parsing with split() and startswith()
        # ──────────────────────────────────────────────────────────────
        # judge_text.split("\n") breaks the response into lines:
        #     ["SCORE: 4", "EXPLANATION: Correct grain identified..."]
        #
        # For each line, we check what it starts with:
        #   line.startswith("SCORE:")  → True for "SCORE: 4"
        #
        # Then extract the value:
        #   "SCORE: 4".replace("SCORE:", "")  → " 4"
        #   " 4".strip()                      → "4"
        #   int("4")                           → 4
        #
        # PYTHON REFRESHER: Why try/except inside a loop?
        # ────────────────────────────────────────────────
        # The model MIGHT not follow our format perfectly.
        # It could output "SCORE: Four" or "SCORE: 4/5" or "SCORE: 4.0"
        # int() would crash on any of these. The try/except catches
        # the ValueError and sets score to None (unparseable).
        # This is defensive coding — handle what CAN go wrong.

        score = None
        explanation = ""

        for line in judge_text.split("\n"):
            line = line.strip()

            if line.startswith("SCORE:"):
                try:
                    score = int(line.replace("SCORE:", "").strip())

                    # ─── Clamp to valid 1-5 range ───
                    #
                    # PYTHON REFRESHER: Clamping with nested min/max
                    # ─────────────────────────────────────────────
                    # max(1, min(5, score)) ensures score is between 1 and 5.
                    #
                    # How it works with examples:
                    #   score = 7:  min(5, 7) → 5,  max(1, 5) → 5  (clamped down)
                    #   score = 0:  min(5, 0) → 0,  max(1, 0) → 1  (clamped up)
                    #   score = 3:  min(5, 3) → 3,  max(1, 3) → 3  (unchanged)
                    #
                    # This pattern works for any range:
                    #   max(lower_bound, min(upper_bound, value))
                    #
                    # WHY CLAMP? The model might output "SCORE: 10" or "SCORE: 0"
                    # if it ignores our 1-5 instruction. Clamping keeps scores
                    # in the expected range so our normalization math works.

                    score = max(1, min(5, score))

                except ValueError:
                    score = None

            elif line.startswith("EXPLANATION:"):
                explanation = line.replace("EXPLANATION:", "").strip()

        # ─── Normalize 1-5 score to 0.0-1.0 range ───
        #
        # WHY NORMALIZE?
        # ──────────────
        # Keyword scoring returns 0.0-1.0 (e.g., 0.75).
        # If judge returns 1-5 (e.g., 4), comparing them is confusing:
        #   "keyword=0.75, judge=4" — which is better? Apples to oranges.
        #
        # By normalizing: (4 - 1) / (5 - 1) = 3/4 = 0.75
        # Now both are on the same 0.0-1.0 scale:
        #   "keyword=0.75, judge=0.75" — they agree! Apples to apples.
        #
        # The formula (score - 1) / 4.0 maps the 1-5 range:
        #   1 → (1-1)/4 = 0.00  (worst)
        #   2 → (2-1)/4 = 0.25
        #   3 → (3-1)/4 = 0.50
        #   4 → (4-1)/4 = 0.75
        #   5 → (5-1)/4 = 1.00  (best)
        #
        # We also keep raw_score (the original 1-5) because it's easier
        # for humans to read: "4/5" is more intuitive than "0.75/1.0"

        normalized = (score - 1) / 4.0 if score else None

        return {
            "score": round(normalized, 2) if normalized is not None else None,
            "raw_score": score,
            "explanation": explanation,
        }

    except Exception as e:
        return {"score": None, "raw_score": None, "explanation": f"Judge error: {e}"}


# ═══════════════════════════════════════════════════════════
# SECTION 5: THE EVAL RUNNER (Main Engine)
# ═══════════════════════════════════════════════════════════
#
# This is the CORE of the evaluation framework.
# It loops through all questions, runs each through the
# full RAG pipeline, scores the result, and records everything.
#
# Think of it as an automated version of you manually testing:
#   1. Type a question into the chatbot
#   2. Read the answer
#   3. Think "was that correct?"
#   4. Write down your assessment
#   5. Repeat 20 times
#
# Except this script does it in 2-3 minutes instead of 40,
# and it scores consistently every time (no "I'm tired, good enough").
# ═══════════════════════════════════════════════════════════

def run_evaluation(
    questions: list[dict],
    collection,          # ChromaDB collection (no type hint — chromadb types are complex)
    known_tables: list[str],
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
    Steps 1-3 call the EXACT SAME functions your chatbot uses.
    We imported them from rag.py. If eval.py had its own retrieval
    code, you'd be testing different code than what runs in production.
    Your eval scores would be lies.

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

    # ─── Main evaluation loop ───
    #
    # PYTHON REFRESHER: enumerate() for index + value
    # ────────────────────────────────────────────────
    # for i, q in enumerate(questions):
    #
    # enumerate() gives you BOTH the index and the item:
    #   i=0, q=first_question
    #   i=1, q=second_question
    #   etc.
    #
    # Without enumerate, you'd need a separate counter:
    #   i = 0
    #   for q in questions:
    #       ... use i ...
    #       i += 1
    #
    # We use i for the progress display: "[3/20] Q03: What is..."

    for i, q in enumerate(questions):

        # ─── Unpack question data from the dict ───
        #
        # PYTHON REFRESHER: dict.get() vs dict[]
        # ──────────────────────────────────────
        # q["question_id"]            → Crashes with KeyError if key missing
        # q.get("expected_table", "") → Returns "" if key missing (safe)
        #
        # Use [] when the key MUST exist (it's a bug if it doesn't).
        # Use .get(key, default) when the key MIGHT not exist.
        #
        # Here, question_id, category, and question are REQUIRED columns
        # in the CSV — they should always exist. expected_table might be
        # missing for some questions, so we use .get() with a default.

        question_id = q["question_id"]
        category = q["category"]
        question = q["question"]
        expected_keywords = q["expected_keywords"]
        expected_table = q.get("expected_table", "")

        # ─── Progress display ───
        # [:60] truncates long questions so the terminal output stays readable.
        # Without truncation, a 200-character question would wrap messily.

        print(f"\n[{i+1}/{total}] {question_id}: {question[:60]}...")

        # ══════════════════════════════════════════════════
        # Step 1: TABLE NAME DETECTION
        # ══════════════════════════════════════════════════
        # Same extract_table_name() from your rag.py.
        # Scans the question text for any known table names.
        #
        # "What is the grain of FACT_STORE_INVENTORY_INTRA?"
        #   → detected_table = "FACT_STORE_INVENTORY_INTRA"
        #
        # "Tell me about customer returns"
        #   → detected_table = None (no known table found)
        #
        # WHY WE RECORD detected_table IN RESULTS
        # ────────────────────────────────────────
        # If a question scores 0 and expected_table is FACT_SALES_ORDER
        # but detected_table is None — we immediately know the problem
        # is in TABLE DETECTION, not in retrieval or generation.
        # Each recorded field helps isolate WHERE failures happen.
        # This is called "observability" — being able to see inside
        # your system to diagnose problems.

        detected_table = extract_table_name(question, known_tables)

        # ══════════════════════════════════════════════════
        # Step 2: RETRIEVAL (Find relevant chunks)
        # ══════════════════════════════════════════════════
        # Same retrieve() from your rag.py.
        # Queries ChromaDB for the most semantically similar chunks.
        #
        # TIMING MEASUREMENT
        # ──────────────────
        # We measure retrieval time separately from generation time.
        #
        # PYTHON REFRESHER: time.time() for measuring duration
        # ──────────────────────────────────────────────────
        # time.time() returns current time as seconds since
        # January 1, 1970 (the "Unix epoch"), as a float:
        #     1741500000.123456
        #
        # To measure how long something takes:
        #     start = time.time()       # → 1741500000.123
        #     ... do some work ...
        #     elapsed = time.time() - start  # → 0.456 seconds
        #
        # WHY MEASURE RETRIEVAL AND GENERATION SEPARATELY?
        # ────────────────────────────────────────────────
        # If total time is 8 seconds, you need to know:
        #   - Retrieval took 7s → ChromaDB is slow (indexing problem)
        #   - Generation took 7s → Claude is slow (context too large)
        # Without separate timing, you can't tell where the bottleneck is.

        start_time = time.time()
        chunks = retrieve(collection, question, top_k=top_k, table_name=detected_table)
        retrieval_time = time.time() - start_time

        # ─── Record what was retrieved (YOUR #1 DEBUGGING TOOL) ───
        #
        # When a question scores badly, the FIRST thing you check
        # in the results CSV is: what chunks did retrieval find?
        #
        # Example: Question about FACT_SALES_ORDER scores 0.
        # You check retrieved_sources in the CSV:
        #   [{"table_name": "DIM_PRODUCT", "doc_type": "columns"}]
        #
        # AHA! Retrieval found the WRONG TABLE entirely. Problem is
        # in retrieval (embedding quality, chunk boundaries, or
        # metadata filtering) — NOT in Claude's generation.
        #
        # Without this field, you'd have to re-run the question with
        # debug mode in rag.py to figure out what went wrong. Recording
        # retrieval sources NOW saves you from re-running later.
        #
        # PYTHON REFRESHER: List comprehension with dict construction
        # ────────────────────────────────────────────────────────────
        # This creates a NEW list of dicts, each containing only the
        # fields we want to save (not the full chunk text, which could
        # be hundreds of characters and make the CSV unreadable).
        #
        # c["text"][:150] takes just the first 150 characters as a preview.
        # The [:150] is a string slice — same syntax as list slicing.

        retrieved_sources = [
            {
                "table_name": c.get("table_name", ""),
                "doc_type": c.get("doc_type", ""),
                "text_preview": c["text"][:150],
            }
            for c in chunks
        ]

        # ══════════════════════════════════════════════════
        # Step 3: GENERATION (Ask Claude for an answer)
        # ══════════════════════════════════════════════════
        # Same ask_claude() from your rag.py.
        # Sends the question + retrieved chunks to Claude API.
        #
        # ERROR HANDLING: API calls can fail for many reasons:
        #   - Network timeout (your internet or Anthropic's servers)
        #   - Rate limiting (429 error — too many requests per minute)
        #   - Invalid API key (401 error)
        #   - Model overloaded (529 error)
        #
        # Instead of crashing the whole eval, we catch the error
        # and record "ERROR: <message>" as the answer.
        # This way you still get results for the other 19 questions.
        # The failed question will score 0 on keywords, and you'll
        # see the error message in the results CSV.

        generation_start = time.time()
        try:
            answer = ask_claude(question, chunks)
        except Exception as e:
            answer = f"ERROR: {e}"
        generation_time = time.time() - generation_start
        total_time = retrieval_time + generation_time

        # ─── Print answer preview ───
        # .1f means 1 decimal place for time: 2.3s, not 2.345678s
        print(f"  Answer ({total_time:.1f}s): {answer[:100]}...")

        # ══════════════════════════════════════════════════
        # Step 4: KEYWORD SCORING (always runs, free)
        # ══════════════════════════════════════════════════

        keyword_result = score_keyword(answer, expected_keywords)
        print(f"  Keyword score: {keyword_result['score']} "
              f"(matched: {keyword_result['matched']}, "
              f"missed: {keyword_result['missed']})")

        # ══════════════════════════════════════════════════
        # Step 5: EDGE CASE SCORING (only for edge_case questions)
        # ══════════════════════════════════════════════════
        # Returns None for non-edge-case questions.

        edge_result = score_edge_case(answer, category)
        if edge_result:
            print(f"  Edge case: {edge_result['score']} ({edge_result['label']})")

        # ══════════════════════════════════════════════════
        # Step 6: LLM JUDGE SCORING (optional, costs API $)
        # ══════════════════════════════════════════════════
        # Only runs if --llm-judge flag was passed on command line.

        judge_result = None
        if use_llm_judge:
            judge_result = score_with_llm_judge(
                question, answer, expected_keywords, category
            )
            if judge_result["score"] is not None:
                print(f"  Judge: {judge_result['raw_score']}/5 "
                      f"— {judge_result['explanation']}")

        # ══════════════════════════════════════════════════
        # Step 7: RECORD EVERYTHING
        # ══════════════════════════════════════════════════
        #
        # ENGINEERING PRINCIPLE: Record MORE than you think you need.
        # ──────────────────────────────────────────────────────────
        # When you're comparing Run A vs Run B a week from now,
        # you'll be glad you saved every field. Here's what each
        # field is useful for:
        #
        #   question_id        → Identifies the question for comparison
        #   category           → Groups results for per-category analysis
        #   detected_table     → Did table detection work? (vs expected_table)
        #   answer             → The full text — read it when scores are ambiguous
        #   keyword_score      → Primary metric for most questions
        #   keyword_missed     → Which keywords are consistently missing?
        #   edge_case_label    → HALLUCINATED vs CORRECT_DECLINE
        #   retrieval_time_s   → Is ChromaDB slow? (indexing problem?)
        #   generation_time_s  → Is Claude slow? (context too large?)
        #   retrieved_sources  → What chunks were actually found?
        #
        # Storage is cheap (this CSV is a few kilobytes).
        # Re-running experiments is expensive (time + API money).
        # Record it all now so you never have to re-run to diagnose.
        #
        # PYTHON REFRESHER: json.dumps() for nested data in CSV
        # ─────────────────────────────────────────────────────
        # CSV cells must be flat strings. But retrieved_sources is a
        # list of dicts — a nested data structure.
        #
        # json.dumps() converts any Python object to a JSON string:
        #   [{"table_name": "DIM_STORE"}]
        #   → '[{"table_name": "DIM_STORE"}]'
        #
        # The JSON string fits in one CSV cell. To read it back later:
        #   json.loads('[{"table_name": "DIM_STORE"}]')
        #   → [{"table_name": "DIM_STORE"}]  (Python object again)

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


# ═══════════════════════════════════════════════════════════
# SECTION 6: SAVE RESULTS TO CSV
# ═══════════════════════════════════════════════════════════
#
# Each eval run produces a CSV file in the eval_results/ folder.
# File names include timestamps so you never overwrite old results.
# The tag from --tag flag helps you remember what config was tested.
# ═══════════════════════════════════════════════════════════

def save_results(results: list[dict], tag: str = "") -> str:
    """
    Save evaluation results to a timestamped CSV file.

    FILE NAMING CONVENTION
    ──────────────────────
    Format: eval_results/eval_results_YYYYMMDD_HHMMSS_tag.csv
    
    Examples:
        eval_results_20250309_143022_baseline.csv
        eval_results_20250309_151045_chunk500.csv
        eval_results_20250309_160012_chunk500_topk5.csv
    
    The timestamp ensures uniqueness — you can run eval 50 times
    and never accidentally overwrite a previous run's data.
    The tag (from --tag flag) reminds you what config was being tested.

    PYTHON REFRESHER: datetime.now().strftime() formatting
    ──────────────────────────────────────────────────────
    datetime.now() returns the current date and time as an object.
    .strftime() formats it as a string using format codes:
        %Y = 4-digit year  (2025)
        %m = 2-digit month (03)
        %d = 2-digit day   (09)
        %H = 24-hour hour  (14)
        %M = minute        (30)
        %S = second        (22)
    
    So strftime("%Y%m%d_%H%M%S") → "20250309_143022"

    RETURNS
    ───────
    The filepath of the saved CSV (so you can reference it later
    in the comparison command).
    """
    from datetime import datetime

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ─── Build the filename ───
    #
    # PYTHON REFRESHER: f-string with conditional expression
    # ──────────────────────────────────────────────────────
    # f"_{tag}" if tag else ""
    #
    # This is a conditional (ternary) expression:
    #   If tag is truthy (non-empty string) → produce "_baseline"
    #   If tag is falsy (empty string "")   → produce "" (nothing)
    #
    # Empty strings are "falsy" in Python:
    #   bool("")      → False
    #   bool("hello") → True
    # So `if tag` is equivalent to `if tag != ""`

    tag_suffix = f"_{tag}" if tag else ""
    filename = f"eval_results_{timestamp}{tag_suffix}.csv"

    # ─── Create the results directory ───
    #
    # PYTHON REFRESHER: os.makedirs(path, exist_ok=True)
    # ─────────────────────────────────────────────────
    # Creates a directory (and any parent directories needed).
    # exist_ok=True means "don't crash if it already exists."
    # Without exist_ok=True, the SECOND eval run would crash with
    # FileExistsError because eval_results/ was created on the first run.

    os.makedirs("eval_results", exist_ok=True)
    filepath = os.path.join("eval_results", filename)

    # ─── Write results to CSV ───
    #
    # PYTHON REFRESHER: csv.DictWriter
    # ────────────────────────────────
    # The counterpart to csv.DictReader we used for loading.
    # DictWriter takes a list of dicts and writes them as CSV rows.
    # 
    # fieldnames = results[0].keys() gets column names from the
    # first result dict. This assumes all dicts have the same keys
    # (they do — we build them all identically in run_evaluation).
    #
    # writeheader() writes the column names as the first row.
    # writerows(results) writes ALL data rows at once.
    #
    # PYTHON GOTCHA: newline="" with csv.writer
    # ──────────────────────────────────────────
    # Without newline="", Python on Windows adds extra blank lines
    # between CSV rows. This happens because:
    #   - csv module writes \r\n (Windows line ending)
    #   - Python's text mode ALSO converts \n to \r\n on Windows
    #   - Result: \r\r\n (double carriage return = visible blank line)
    #
    # newline="" tells Python: "don't do any line ending conversion,
    # let the csv module handle it." Always use this with csv writers.
    # On Mac/Linux it doesn't matter, but the habit prevents bugs
    # when someone runs your code on Windows.

    if results:
        fieldnames = results[0].keys()
        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)

    print(f"\nResults saved to: {filepath}")
    return filepath


# ═══════════════════════════════════════════════════════════
# SECTION 7: PRINT SUMMARY REPORT
# ═══════════════════════════════════════════════════════════
#
# After each eval run, this prints a human-readable dashboard.
# This is what you look at FIRST after running eval.py.
#
# THE KEY DIAGNOSTIC INSIGHT: Per-Category Breakdown
# ──────────────────────────────────────────────────
# The three categories tell you DIFFERENT things about your system:
#
#   simple_lookup score LOW → RETRIEVAL is broken
#     The right chunks aren't being found by ChromaDB.
#     Fix: adjust chunk_size, improve embeddings, check metadata.
#
#   cross_entity score LOW → CONTEXT/PROMPT is weak
#     Right chunks found, but model can't synthesize them.
#     Fix: increase TOP_K, improve system prompt, better chunk formatting.
#
#   edge_case score LOW → HALLUCINATION GUARDRAILS are weak
#     Model makes up answers instead of saying "I don't know."
#     Fix: strengthen system prompt, add explicit "only use context" rules.
#
# This diagnosis tells you WHERE to focus your engineering effort
# rather than randomly trying things and hoping something helps.
# ═══════════════════════════════════════════════════════════

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

    # ─── 1. Overall keyword score ───
    #
    # PYTHON REFRESHER: List comprehension to extract one field
    # ──────────────────────────────────────────────────────────
    # [r["keyword_score"] for r in results]
    # 
    # This builds a new list by pulling keyword_score from each dict.
    # If results has 20 dicts, this produces a list of 20 floats.
    #
    # It's equivalent to:
    #     keyword_scores = []
    #     for r in results:
    #         keyword_scores.append(r["keyword_score"])
    #
    # sum(list) / len(list) gives the arithmetic mean (average).
    # This is your single-number health check.
    # Higher = better. Track this number across eval runs.

    keyword_scores = [r["keyword_score"] for r in results]
    avg_keyword = sum(keyword_scores) / len(keyword_scores)
    print(f"\nOverall Keyword Score: {avg_keyword:.2f} / 1.00")

    # ─── 2. Per-category breakdown ───
    #
    # PYTHON REFRESHER: set() for unique values
    # ──────────────────────────────────────────
    # set() removes duplicates from a collection:
    #   set(["a", "b", "a", "c", "b"]) → {"a", "b", "c"}
    #
    # set(r["category"] for r in results) gets all unique categories.
    # sorted() converts the set back to an alphabetically sorted list.
    #
    # This is equivalent to SQL: SELECT DISTINCT category FROM results ORDER BY category
    #
    # PYTHON REFRESHER: f-string alignment and padding
    # ─────────────────────────────────────────────────
    # f"{'Category':<20}" left-aligns "Category" in a 20-character field.
    #   < means left-align
    #   > means right-align
    #   20 is the field width (padded with spaces)
    #
    # This creates neat columns in the terminal output:
    #   Category              Avg Score    Count
    #   -------------------- ---------- --------
    #   cross_entity               0.50        6
    #   edge_case                  0.70        5
    #   simple_lookup              0.80        7

    categories = sorted(set(r["category"] for r in results))

    print(f"\nPer-Category Breakdown:")
    print(f"  {'Category':<20} {'Avg Score':>10} {'Count':>8}")
    print(f"  {'-'*20} {'-'*10} {'-'*8}")

    for cat in categories:
        # ─── Filter results to only this category ───
        # This is equivalent to SQL: WHERE category = 'simple_lookup'
        cat_results = [r for r in results if r["category"] == cat]

        # ─── Choose the right score for each category ───
        # For edge cases, use the EDGE CASE score (0.0 or 1.0)
        # not the keyword score (which is meaningless for "I don't know" answers).
        # For other categories, use keyword score as the primary metric.

        if cat == "edge_case":
            cat_scores = [
                r["edge_case_score"] for r in cat_results
                if r["edge_case_score"] != ""
            ]
        else:
            cat_scores = [r["keyword_score"] for r in cat_results]

        avg = sum(cat_scores) / len(cat_scores) if cat_scores else 0
        print(f"  {cat:<20} {avg:>10.2f} {len(cat_results):>8}")

    # ─── 3. LLM Judge scores (only if --llm-judge was used) ───

    judge_scores = [
        r["judge_score"] for r in results
        if r["judge_score"] != "" and r["judge_score"] is not None
    ]
    if judge_scores:
        avg_judge = sum(judge_scores) / len(judge_scores)
        print(f"\nLLM Judge Score: {avg_judge:.2f} / 1.00")

    # ─── 4. Table detection accuracy ───
    #
    # For questions where we KNOW which table the answer should come from,
    # check if extract_table_name() found the right table.
    #
    # We exclude:
    #   - expected_table = "NONE" (edge cases — no real table to detect)
    #   - expected_table = "MULTIPLE" (cross-entity — multiple tables involved)
    #
    # If detection accuracy is low, your extract_table_name() needs work.
    # Common fixes: add fuzzy matching, handle aliases, handle spaces vs underscores.

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

    # ─── 5. Worst performing questions (fix these first) ───
    #
    # PYTHON REFRESHER: sorted() with key= parameter and lambda
    # ──────────────────────────────────────────────────────────
    # sorted(results, key=lambda r: r["keyword_score"])
    #
    # sorted() returns a NEW list in order (doesn't modify the original).
    # The key= parameter tells sorted() HOW to determine order.
    #
    # lambda r: r["keyword_score"] is an anonymous (inline) function:
    #   - Takes one argument: r (a result dict)
    #   - Returns: r["keyword_score"] (the value to sort by)
    #
    # It's equivalent to:
    #   def get_score(r):
    #       return r["keyword_score"]
    #   sorted(results, key=get_score)
    #
    # lambda is useful when the function is too simple to deserve
    # a full def statement. One expression, no name needed.
    #
    # Default sort is ascending (lowest first), so worst scores come first.
    # We show the bottom 5 — these are your highest-impact fix targets.
    #
    # ENGINEERING PRINCIPLE: Fix the worst problems first.
    # A question scoring 0.0 has more room for improvement than one at 0.8.
    # This is the "fix the lowest-hanging fruit" principle.

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

    # ─── 6. Timing statistics ───
    #
    # Quick health check: is the system fast enough?
    # avg > 10s → something might be wrong (too much context?)
    # max > 20s → one question is very slow (investigate why)

    times = [r["total_time_s"] for r in results]
    avg_time = sum(times) / len(times)
    max_time = max(times)
    print(f"\nTiming: avg={avg_time:.1f}s, max={max_time:.1f}s per question")

    print("\n" + "=" * 60)


# ═══════════════════════════════════════════════════════════
# SECTION 8: COMPARISON TOOL
# ═══════════════════════════════════════════════════════════
#
# After running eval twice with different configs, this function
# shows you exactly which questions got BETTER, WORSE, or STAYED SAME.
#
# THIS IS WHERE DATA-DRIVEN DECISIONS HAPPEN
# ───────────────────────────────────────────
# Example: You changed CHUNK_SIZE from 800 to 500.
#
# Comparison output:
#   Q01:  0.75  → 1.00  (+0.25)  IMPROVED
#   Q06:  0.80  → 0.40  (-0.40)  REGRESSED
#   Q09:  0.50  → 0.50  ( 0.00)  SAME
#   Summary: 5 improved, 2 regressed, 13 unchanged
#
# Now you can make an INFORMED decision:
#   "chunk500 improved 5 questions but broke 2 cross_entity ones.
#    Maybe smaller chunks split relationship info. Try 600 next."
#
# USAGE
# ─────
# python eval.py --compare eval_results/run_a.csv eval_results/run_b.csv
# ═══════════════════════════════════════════════════════════

def compare_runs(file_a: str, file_b: str):
    """
    Compare two eval result CSV files side by side.

    PYTHON REFRESHER: Nested (inner) function definitions
    ─────────────────────────────────────────────────────
    Below, load_results() is defined INSIDE compare_runs().
    This is called a "nested function" or "inner function."

    It's only visible inside compare_runs() — you can't call
    load_results() from outside. This is a way to organize
    helper logic that only ONE function needs. Think of it as
    a private method — it keeps the global namespace clean.

    PARAMETERS
    ──────────
    file_a: Path to the FIRST (older/baseline) results CSV
    file_b: Path to the SECOND (newer/experimental) results CSV
    """

    def load_results(filepath):
        """
        Load a CSV file into a dict keyed by question_id.

        PYTHON REFRESHER: Dictionary comprehension
        ───────────────────────────────────────────
        {row["question_id"]: row for row in reader}

        This builds a dictionary where:
            Key   = the question_id ("Q01", "Q02", ...)
            Value = the entire row dict (all columns for that question)

        Equivalent to:
            result = {}
            for row in reader:
                result[row["question_id"]] = row
            return result
        
        Using question_id as the key lets us look up any question
        instantly by ID: results["Q06"] → full data for question 6.
        This is O(1) lookup — much faster than searching through a list.
        """
        with open(filepath, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            return {row["question_id"]: row for row in reader}

    results_a = load_results(file_a)
    results_b = load_results(file_b)

    # ─── Find questions that exist in BOTH runs ───
    #
    # PYTHON REFRESHER: Set intersection with & operator
    # ──────────────────────────────────────────────────
    # set_a & set_b returns elements that appear in BOTH sets.
    #
    # set_a = {"Q01", "Q02", "Q03"}
    # set_b = {"Q02", "Q03", "Q04"}
    # set_a & set_b → {"Q02", "Q03"}
    #
    # We need this because Run A and Run B might have different questions
    # (e.g., you added new test questions between runs). We can only
    # meaningfully compare questions that appear in BOTH files.
    #
    # .keys() returns the dict's keys as a view object.
    # set() converts it to a proper set for & intersection.
    # sorted() gives consistent alphabetical order.

    common_ids = sorted(set(results_a.keys()) & set(results_b.keys()))

    # ─── Print comparison header ───
    #
    # PYTHON REFRESHER: os.path.basename()
    # ────────────────────────────────────
    # os.path.basename("eval_results/eval_results_20250309_baseline.csv")
    #   → "eval_results_20250309_baseline.csv"
    #
    # Strips the directory path, leaving just the filename.
    # Makes the output header more readable.

    print(f"\nComparing:")
    print(f"  A: {os.path.basename(file_a)}")
    print(f"  B: {os.path.basename(file_b)}")
    print(f"\n{'ID':<6} {'Score A':>8} {'Score B':>8} {'Delta':>8} {'Status'}")
    print("-" * 50)

    improved = 0
    regressed = 0
    same = 0

    for qid in common_ids:
        # ─── Convert string scores to floats for comparison ───
        #
        # CSV stores ALL values as strings. The keyword_score "0.75"
        # is a string, not a number. float("0.75") → 0.75 (number).

        score_a = float(results_a[qid]["keyword_score"])
        score_b = float(results_b[qid]["keyword_score"])
        delta = score_b - score_a

        # ─── Classify the change ───
        #
        # We use a 0.01 threshold instead of exactly 0 because
        # floating point arithmetic has tiny rounding errors:
        #   0.1 + 0.2 = 0.30000000000000004 (not exactly 0.3)
        #
        # A threshold of 0.01 means anything within 1% is "same."
        # This avoids false "improved" signals from rounding noise.

        if delta > 0.01:
            status = "IMPROVED"
            improved += 1
        elif delta < -0.01:
            status = "REGRESSED"
            regressed += 1
        else:
            status = "SAME"
            same += 1

        # ─── Print the comparison row ───
        #
        # PYTHON REFRESHER: f-string format specifier {delta:>+8.2f}
        # ──────────────────────────────────────────────────────────
        # This is a compact way to format numbers. Breaking it down:
        #   >    right-align within the field
        #   +    ALWAYS show the sign (+ for positive, - for negative)
        #   8    total field width of 8 characters (padded with spaces)
        #   .2   2 decimal places
        #   f    float format
        #
        # Examples of what this produces:
        #   delta = 0.25  → "   +0.25"
        #   delta = -0.40 → "   -0.40"
        #   delta = 0.00  → "   +0.00"
        #
        # The + sign makes it immediately visible whether a change
        # helped (+) or hurt (-) without having to read the number.

        print(f"{qid:<6} {score_a:>8.2f} {score_b:>8.2f} {delta:>+8.2f} {status}")

    print(f"\nSummary: {improved} improved, {regressed} regressed, {same} unchanged")
    print(f"Total compared: {len(common_ids)} questions")


# ═══════════════════════════════════════════════════════════
# SECTION 9: MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════
#
# This is where the script starts when you run:
#     python eval.py
#
# It handles:
#   1. Parsing command-line arguments (--llm-judge, --tag, etc.)
#   2. Loading the API key (same as rag.py)
#   3. Building the RAG pipeline (same as rag.py startup)
#   4. Running the evaluation loop
#   5. Saving results and printing the summary dashboard
# ═══════════════════════════════════════════════════════════

def main():
    """
    PYTHON REFRESHER: argparse — Command-line argument parsing
    ──────────────────────────────────────────────────────────
    argparse turns your script into a flexible command-line tool.
    
    Without argparse, you'd hardcode everything:
        USE_LLM_JUDGE = False   # Have to edit the code to change this
        TAG = "baseline"        # Have to edit the code to change this
    
    With argparse, you pass options at runtime without editing code:
        python eval.py --llm-judge --tag "baseline"
    
    HOW ARGPARSE WORKS (step by step):
    
    1. Create a parser object:
       parser = argparse.ArgumentParser(description="...")
    
    2. Tell it what arguments your script accepts:
       parser.add_argument("--tag", default="", help="...")
         --tag      = the flag name (user types --tag "baseline")
         default="" = value used if the flag is NOT provided
         help="..." = text shown when user runs: python eval.py --help
    
    3. Parse the actual command line the user typed:
       args = parser.parse_args()
       # This reads sys.argv (the command line arguments) and
       # matches them against the arguments you defined.
    
    4. Access the parsed values:
       args.tag → "baseline" (or "" if --tag wasn't provided)
    
    SPECIAL ARGUMENT TYPES:
    
    action="store_true" — Boolean flag (no value needed):
        parser.add_argument("--llm-judge", action="store_true")
        python eval.py --llm-judge     → args.llm_judge = True
        python eval.py                 → args.llm_judge = False
        
        GOTCHA: --llm-judge (hyphen) becomes args.llm_judge (underscore)
        Python variable names can't contain hyphens, so argparse converts.
    
    type=int — Numeric argument (argparse converts the string to int):
        parser.add_argument("--top-k", type=int, default=3)
        python eval.py --top-k 5       → args.top_k = 5 (integer, not string)
    
    nargs=2 — Argument that takes exactly 2 values:
        parser.add_argument("--compare", nargs=2)
        python eval.py --compare a.csv b.csv 
        → args.compare = ["a.csv", "b.csv"]  (a list of two strings)
    
    metavar — Placeholder names shown in --help output:
        metavar=("FILE_A", "FILE_B") makes --help show:
        "--compare FILE_A FILE_B" instead of "--compare COMPARE COMPARE"
    """
    parser = argparse.ArgumentParser(description="RAG Evaluation Framework")

    parser.add_argument(
        "--questions", default="eval_questions.csv",
        help="Path to eval questions CSV file"
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

    # ─── Load API key (same pattern as your rag.py main) ───
    #
    # First checks if ANTHROPIC_API_KEY is already in the environment.
    # If not, tries to load it from a .env file in the same directory.
    #
    # PYTHON REFRESHER: os.environ.get() vs os.environ[]
    # ───────────────────────────────────────────────────
    # os.environ["ANTHROPIC_API_KEY"]      → KeyError if not set
    # os.environ.get("ANTHROPIC_API_KEY")  → None if not set (safe)
    #
    # We use .get() because the key MIGHT not be in the environment yet
    # (it's in .env file instead). This is the "look before you leap" pattern.
    #
    # PYTHON REFRESHER: str.partition("=") from your rag.py
    # ─────────────────────────────────────────────────────
    # "API_KEY=sk-abc123".partition("=") → ("API_KEY", "=", "sk-abc123")
    # Returns a 3-tuple: (before, separator, after)
    # We use key, _, value to capture before and after, ignoring the "=".
    # The _ is a Python convention for "I don't need this value."

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

    # ─── Print header ───

    print("=" * 60)
    print("RAG EVALUATION FRAMEWORK")
    print("=" * 60)

    # ─── Step 1: Load eval questions from CSV ───

    questions = load_eval_questions(args.questions, category_filter=args.category)
    if not questions:
        print("No questions found! Check your CSV file path.")
        return

    # ─── Step 2: Build the RAG pipeline ───
    #
    # THIS MIRRORS YOUR rag.py main() STARTUP SEQUENCE:
    #   1. Load raw documents from docs/ folder
    #   2. Chunk each document into overlapping pieces
    #   3. Store all chunks in ChromaDB with embeddings
    #   4. Get list of known table names for table detection
    #
    # WHY REBUILD THE VECTOR STORE EVERY TIME?
    # ────────────────────────────────────────
    # If you changed chunk_size or overlap in rag.py between runs,
    # the old ChromaDB database still has chunks from the OLD settings.
    # Rebuilding from scratch ensures eval tests your CURRENT config.
    #
    # This adds ~10-30 seconds of startup time, but guarantees
    # you're always testing what you think you're testing.
    # Correctness first, optimization later.
    #
    # FUTURE OPTIMIZATION: You could hash the docs + config and only
    # rebuild when something changes. But that's premature optimization
    # at this learning stage.

    print("\nBuilding RAG pipeline (same as chatbot startup)...")

    # Step 2a: Load documents
    documents = load_documents(DOCS_DIR)
    if not documents:
        print(f"No documents found in {DOCS_DIR}/")
        return

    # Step 2b: Chunk all documents
    # Same pattern as rag.py: for each doc → chunk_text() → extend into all_chunks
    # (extend adds ALL chunks from one doc, not just the first — see rag.py refresher)

    all_chunks = []
    for doc in documents:
        chunks = chunk_text(doc["content"], doc["source"])
        all_chunks.extend(chunks)

    # Step 2c: Build vector store in ChromaDB
    collection = build_vector_store(all_chunks)

    # Step 2d: Get known table names for extract_table_name()
    # Same logic as rag.py: query ChromaDB metadata for all unique table_name values.
    #
    # PYTHON REFRESHER: This set comprehension (from your rag.py)
    # ──────────────────────────────────────────────────────────
    # set(m.get("table_name", "") for m in all_meta["metadatas"] if ...)
    #
    # Creates a set of unique table names by:
    #   1. Iterating all metadata dicts from ChromaDB
    #   2. Extracting the "table_name" field from each
    #   3. Filtering out empty/whitespace-only names
    #   4. set() removes duplicates automatically

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


# ─── Script entry point guard ───
#
# PYTHON REFRESHER: if __name__ == "__main__"
# ────────────────────────────────────────────
# Every Python file has a special variable called __name__.
# 
# When you RUN the file directly:
#     python eval.py
#     → Python sets __name__ = "__main__"
#     → The if-block EXECUTES → main() runs
#
# When another file IMPORTS from this file:
#     from eval import score_keyword
#     → Python sets __name__ = "eval" (the module name)
#     → The if-block is SKIPPED → main() does NOT run
#
# WITHOUT this guard, `from eval import score_keyword` would
# immediately run the ENTIRE evaluation — rebuild vector store,
# run 20 questions, hit the API 20+ times. That's definitely
# not what you want when you just need one scoring function.
#
# This is the SAME pattern used in your rag.py to protect
# the chatbot from launching when you import rag.py functions.

if __name__ == "__main__":
    main()