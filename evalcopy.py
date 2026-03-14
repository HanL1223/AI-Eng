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

def load_eval_question(filepath:str,category_filter:str = None) -> list[dict]:
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
            row['excepted_keywords'] = [
                kw.strip().upper() for kw in["expected_answer_keywords"].split(",")
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

    answer_uppder = answer.upper()
    matched = []
    missed = []

    for keyword in expected_keywords:
        if keyword in answer:
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
        response = client.message.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens =200,
            messages = [{"role":'user',"content":judge_prompt}]
        )
        judge_text = response.content[0].text.strip()

        score = None
        explanation = ""

        for line in judge_text.split("\n"):
            line = line.strip()
            if line.startwith("SCORE:"):
                try:
                    score = int(line.replace("SCORE:","").strip())
                    score = max(1, min(5, score))
                except ValueError:
                    score = None
            elif line.startwith("EXPLANATION:"):
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
) -> list[str]:
    



