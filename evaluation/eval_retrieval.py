"""
Evaluate retrieval pipeline (Problem 2): Semantic RAG vs Keyword search.
Metrics: Precision@K, MRR (Mean Reciprocal Rank).

Relevance is condition-based: a trial is "relevant" if its title or
eligibility text contains keywords from the patient's conditions.

Usage:
    python evaluation/eval_retrieval.py
"""
import json
from typing import List, Dict
from ner.schemas import PatientProfile
from ner.ner_llm import extract_profile_llm


def is_relevant(trial: Dict, conditions: List[str]) -> bool:
    """A trial is relevant if any condition keyword appears in its text."""
    text = (trial.get("title", "") + " " + trial.get("eligibility_raw", "")).lower()
    for cond in conditions:
        if any(kw in text for kw in cond.lower().split()):
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


def evaluate_retrieval(
    test_path: str = "evaluation/test_patients.json",
    method: str = "semantic",
    top_k: int = 5,
) -> dict:
    with open(test_path) as f:
        test_cases = json.load(f)

    if method == "semantic":
        from retrieval.retriever_semantic import SemanticRetriever
        retriever = SemanticRetriever()
    else:
        from retrieval.retriever_keyword import KeywordRetriever
        retriever = KeywordRetriever()

    p_at_k_scores = []
    mrr_scores = []

    for case in test_cases:
        conditions = case["ground_truth"].get("conditions", [])
        if not conditions:
            continue  # Skip healthy volunteers for retrieval eval

        profile = extract_profile_llm(case["description"])
        results = retriever.retrieve(profile, top_k=top_k)

        p = precision_at_k(results, conditions, k=top_k)
        m = mean_reciprocal_rank(results, conditions)

        p_at_k_scores.append(p)
        mrr_scores.append(m)
        print(f"  {case['id']}: P@{top_k}={p:.3f}  MRR={m:.3f}")

    n = len(p_at_k_scores)
    summary = {
        "method": method,
        "total_cases": n,
        f"precision_at_{top_k}": sum(p_at_k_scores) / n if n else 0.0,
        "mrr": sum(mrr_scores) / n if n else 0.0,
    }

    print(f"\n=== Retrieval Evaluation: {method.upper()} ===")
    for k, v in summary.items():
        print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")

    return summary


if __name__ == "__main__":
    print("Evaluating semantic RAG retrieval...")
    semantic = evaluate_retrieval(method="semantic")

    print("\nEvaluating keyword retrieval (baseline)...")
    keyword = evaluate_retrieval(method="keyword")

    print("\n=== Comparison ===")
    for metric in [f"precision_at_5", "mrr"]:
        s_val = semantic.get(metric, 0)
        k_val = keyword.get(metric, 0)
        winner = "Semantic" if s_val >= k_val else "Keyword"
        print(f"  {metric}: Semantic={s_val:.3f}  Keyword={k_val:.3f}  → {winner} wins")
