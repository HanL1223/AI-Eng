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


#Loading Eval question

def laod_eval_question(filepath:str,category_filter:str = None) -> list[dict]:
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
    




