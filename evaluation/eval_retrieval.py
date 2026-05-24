"""
Evaluate retrieval pipeline (Problem 2): Semantic RAG vs Keyword search.
Metrics: Precision@K, MRR (Mean Reciprocal Rank).

Usage:
    python evaluation/eval_retrieval.py
"""
import json
from typing import List, Dict
from ner.schemas import PatientProfile
from ner.ner_llm import extract_profile_llm


def is_relevant(trial: Dict, conditions: List[str]) -> bool:
    """A trial is relevant if any condition phrase (or a meaningful root) appears in its text.
    Uses full-phrase matching first, then falls back to the longest word (>=6 chars) in the phrase
    to avoid false positives from short words like 'type', '2', 'of'.
    """
    text = (trial.get("title", "") + " " + trial.get("eligibility_raw", "")).lower()
    for cond in conditions:
        cond_lower = cond.lower()
        # First try: exact full phrase
        if cond_lower in text:
            return True
        # Second try: longest meaningful word in the phrase (avoids "type", "2", "of")
        words = [w for w in cond_lower.split() if len(w) >= 6]
        if words and any(w in text for w in words):
            return True
    return False


def precision_at_k(results: List[Dict], conditions: List[str], k: int = 5) -> float:
    top_k = results[:k]
    if not top_k:
        return 0.0
    relevant = sum(1 for t in top_k if is_relevant(t, conditions))
    return relevant / len(top_k)


def mean_reciprocal_rank(results: List[Dict], conditions: List[str]) -> float:
    for rank, trial in enumerate(results, 1):
        if is_relevant(trial, conditions):
            return 1.0 / rank
    return 0.0


def recall_at_k(results: List[Dict], conditions: List[str], k: int = 10) -> float:
    top_k = results[:k]
    # Count unique condition keywords found across top-k results
    found = set()
    for trial in top_k:
        text = (trial.get("title", "") + " " + trial.get("eligibility_raw", "")).lower()
        for cond in conditions:
            if any(kw in text for kw in cond.lower().split()):
                found.add(cond.lower())
    if not conditions:
        return 0.0
    return len(found) / len(conditions)


def evaluate_retrieval(
    test_path: str = "evaluation/test_patients.json",
    method: str = "semantic",
    top_k: int = 10,
) -> dict:
    with open(test_path) as f:
        test_cases = json.load(f)

    print(f"\nRunning {method.upper()} retrieval on {len(test_cases)} patients...")

    try:
        if method == "semantic":
            from retrieval.retriever_semantic import SemanticRetriever
            retriever = SemanticRetriever()
        else:
            from retrieval.retriever_keyword import KeywordRetriever
            retriever = KeywordRetriever()
    except Exception as e:
        print(f"  ERROR loading retriever: {e}")
        return {"method": method, "error": str(e), "precision_at_5": 0, "mrr": 0}

    p_at_k_scores = []
    recall_scores = []
    mrr_scores = []

    for case in test_cases:
        conditions = case["ground_truth"].get("conditions", [])
        if not conditions:
            continue

        try:
            profile = extract_profile_llm(case["description"])
            results = retriever.retrieve(profile, top_k=top_k)

            p = precision_at_k(results, conditions, k=5)
            r = recall_at_k(results, conditions, k=10)
            m = mean_reciprocal_rank(results, conditions)

            p_at_k_scores.append(p)
            recall_scores.append(r)
            mrr_scores.append(m)
            print(f"  {case['id']}: P@5={p:.3f}  R@10={r:.3f}  MRR={m:.3f}")

        except Exception as e:
            print(f"  ERROR on {case['id']}: {e}")

    n = len(p_at_k_scores)
    summary = {
        "method": method,
        "total_cases": n,
        "precision_at_5": sum(p_at_k_scores) / n if n else 0.0,
        "recall_at_10": sum(recall_scores) / n if n else 0.0,
        "mrr": sum(mrr_scores) / n if n else 0.0,
    }

    return summary


def print_retrieval_comparison(semantic_results: dict, keyword_results: dict):
    print("\n" + "=" * 55)
    print(f"{'Metric':<25} {'Semantic RAG':>14} {'Keyword':>10}")
    print("=" * 55)
    for metric in ["precision_at_5", "recall_at_10", "mrr"]:
        s = semantic_results.get(metric, 0)
        k = keyword_results.get(metric, 0)
        if s > k:
            winner = " <- RAG"
        elif k > s:
            winner = " <- KW"
        else:
            winner = " TIE"
        print(f"{metric:<25} {s:>13.3f} {k:>9.3f}{winner}")
    print("=" * 55)
    print("\nNote: Semantic RAG uses FAISS vector similarity.")
    print("Keyword uses ClinicalTrials.gov native search — the patient status quo.")


if __name__ == "__main__":
    print("Evaluating semantic RAG retrieval...")
    semantic = evaluate_retrieval(method="semantic")

    print("\nEvaluating keyword retrieval (baseline)...")
    keyword = evaluate_retrieval(method="keyword")

    print_retrieval_comparison(semantic, keyword)
