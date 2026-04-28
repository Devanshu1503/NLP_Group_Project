"""
Evaluate NER extraction (Problem 1): LLM vs BioBERT.
Runs both approaches on sample_patients.json and compares metrics.

Usage:
    python evaluation/eval_ner.py
"""
import json
from ner.ner_llm import extract_profile_llm
from ner.schemas import PatientProfile


def compute_f1(pred: set, true: set) -> float:
    if not true and not pred:
        return 1.0
    if not true or not pred:
        return 0.0
    tp = len(pred & true)
    precision = tp / len(pred)
    recall = tp / len(true)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def evaluate_extraction(
    test_path: str = "data/sample_patients.json",
    method: str = "llm",
) -> dict:
    """
    Evaluate NER against ground-truth labels.

    Metrics:
        - age_accuracy: exact match
        - gender_accuracy: exact match (case-insensitive)
        - condition_f1: entity-level F1 across all patients
        - medication_f1: entity-level F1 across all patients
    """
    with open(test_path) as f:
        patients = json.load(f)

    age_correct = 0
    gender_correct = 0
    condition_f1s = []
    medication_f1s = []

    for patient in patients:
        desc = patient["description"]
        truth = patient["ground_truth"]

        if method == "llm":
            pred = extract_profile_llm(desc)
        elif method == "bioBERT":
            from ner.ner_bioBERT import extract_profile_bioBERT
            pred = extract_profile_bioBERT(desc)
        else:
            raise ValueError(f"Unknown method: {method}")

        if pred.age == truth.get("age"):
            age_correct += 1

        if pred.gender and truth.get("gender"):
            if pred.gender.lower() == truth["gender"].lower():
                gender_correct += 1

        pred_conds = {c.lower() for c in pred.conditions}
        true_conds = {c.lower() for c in truth.get("conditions", [])}
        condition_f1s.append(compute_f1(pred_conds, true_conds))

        pred_meds = {m.lower() for m in pred.medications}
        true_meds = {m.lower() for m in truth.get("medications", [])}
        medication_f1s.append(compute_f1(pred_meds, true_meds))

    n = len(patients)
    summary = {
        "method": method,
        "total_patients": n,
        "age_accuracy": age_correct / n,
        "gender_accuracy": gender_correct / n,
        "condition_f1": sum(condition_f1s) / n,
        "medication_f1": sum(medication_f1s) / n,
    }

    print(f"\n=== NER Evaluation: {method.upper()} ===")
    for k, v in summary.items():
        print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")

    return summary


if __name__ == "__main__":
    llm_results = evaluate_extraction(method="llm")
    bioBERT_results = evaluate_extraction(method="bioBERT")

    print("\n=== Comparison ===")
    for metric in ["age_accuracy", "gender_accuracy", "condition_f1", "medication_f1"]:
        llm_val = llm_results[metric]
        bio_val = bioBERT_results[metric]
        winner = "LLM" if llm_val >= bio_val else "BioBERT"
        print(f"  {metric}: LLM={llm_val:.3f}  BioBERT={bio_val:.3f}  → {winner} wins")
