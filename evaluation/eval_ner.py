"""
Evaluate NER extraction (Problem 1): LLM vs BioBERT.
Runs both approaches on sample_patients.json and compares metrics.

Usage:
    python evaluation/eval_ner.py
"""
import json
import time
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
        - avg_time_ms: average extraction time in milliseconds
    """
    with open(test_path) as f:
        patients = json.load(f)

    age_correct = 0
    gender_correct = 0
    condition_f1s = []
    medication_f1s = []
    times_ms = []

    print(f"\nRunning {method.upper()} extraction on {len(patients)} patients...")

    # Load BioBERT once outside the loop — avoids reloading model per patient
    bioBERT_instance = None
    if method == "bioBERT":
        from ner.ner_bioBERT import BioBERTExtractor
        bioBERT_instance = BioBERTExtractor()

    for i, patient in enumerate(patients):
        desc = patient["description"]
        truth = patient["ground_truth"]

        try:
            t0 = time.time()
            if method == "llm":
                pred = extract_profile_llm(desc)
            elif method == "bioBERT":
                pred = bioBERT_instance.extract_profile(desc)
            else:
                raise ValueError(f"Unknown method: {method}")
            elapsed_ms = (time.time() - t0) * 1000
            times_ms.append(elapsed_ms)

        except Exception as e:
            print(f"  [{i+1}] ERROR on {patient['id']}: {e}")
            condition_f1s.append(0.0)
            medication_f1s.append(0.0)
            continue

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

        age_ok = "OK" if pred.age == truth.get("age") else "XX"
        print(f"  [{i+1}] {patient['id']} age={age_ok} "
              f"cond_f1={condition_f1s[-1]:.2f} "
              f"med_f1={medication_f1s[-1]:.2f} "
              f"({elapsed_ms:.0f}ms)")

    n = len(patients)
    summary = {
        "method": method,
        "total_patients": n,
        "age_accuracy": age_correct / n,
        "gender_accuracy": gender_correct / n,
        "condition_f1": sum(condition_f1s) / n,
        "medication_f1": sum(medication_f1s) / n,
        "avg_time_ms": sum(times_ms) / len(times_ms) if times_ms else 0,
    }

    return summary


def print_comparison_table(llm_results: dict, bert_results: dict):
    metrics = ["age_accuracy", "gender_accuracy", "condition_f1", "medication_f1"]
    print("\n" + "=" * 55)
    print(f"{'Metric':<25} {'LLM':>12} {'BioBERT':>12}")
    print("=" * 55)
    for m in metrics:
        llm_val = llm_results.get(m, 0)
        bert_val = bert_results.get(m, 0)
        if llm_val > bert_val:
            winner = " <- LLM"
        elif bert_val > llm_val:
            winner = " <- BERT"
        else:
            winner = " TIE"
        print(f"{m:<25} {llm_val:>11.3f} {bert_val:>11.3f}{winner}")
    print("=" * 55)
    print(f"\nLLM avg speed:     {llm_results.get('avg_time_ms', 0):.0f}ms")
    print(f"BioBERT avg speed: {bert_results.get('avg_time_ms', 0):.0f}ms")
    print("\nNote: LLM requires API calls. BioBERT runs fully locally.")


if __name__ == "__main__":
    llm_results = evaluate_extraction(method="llm")
    bert_results = evaluate_extraction(method="bioBERT")
    print_comparison_table(llm_results, bert_results)
