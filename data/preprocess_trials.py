"""
Preprocess raw ClinicalTrials.gov JSON into clean documents for embedding.
Saves to data/processed_trials.json.

Usage:
    python data/preprocess_trials.py
"""
import json
import re
from pathlib import Path
from typing import List, Dict


def clean_text(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def parse_eligibility_criteria(criteria_text: str) -> Dict[str, List[str]]:
    if not criteria_text:
        return {"inclusion": [], "exclusion": []}

    inc_pattern = re.compile(r"inclusion criteria[:\s]*", re.IGNORECASE)
    exc_pattern = re.compile(r"exclusion criteria[:\s]*", re.IGNORECASE)

    parts = re.split(exc_pattern, criteria_text, maxsplit=1)
    inc_section = re.split(inc_pattern, parts[0], maxsplit=1)[-1]
    exc_section = parts[1] if len(parts) > 1 else ""

    inclusion, exclusion = [], []
    for section, target in [(inc_section, inclusion), (exc_section, exclusion)]:
        for item in re.split(r"[\n\-\•\d+\.]", section):
            item = clean_text(item)
            if len(item) > 20:
                target.append(item)

    return {"inclusion": inclusion, "exclusion": exclusion}


def process_trial(raw_trial: dict) -> dict:
    proto = raw_trial.get("protocolSection", {})
    id_mod = proto.get("identificationModule", {})
    desc_mod = proto.get("descriptionModule", {})
    elig_mod = proto.get("eligibilityModule", {})
    contacts_mod = proto.get("contactsLocationsModule", {})
    status_mod = proto.get("statusModule", {})
    design_mod = proto.get("designModule", {})
    cond_mod = proto.get("conditionsModule", {})
    interventions_mod = proto.get("armsInterventionsModule", {})

    locations = contacts_mod.get("locations", [])
    location_strings = [
        f"{loc.get('city', '')}, {loc.get('state', '')}, {loc.get('country', '')}".strip(", ")
        for loc in locations[:5]
        if loc.get("city") or loc.get("state")
    ]

    interventions = interventions_mod.get("interventions", [])
    intervention_names = [i.get("name", "") for i in interventions[:5]]

    raw_criteria = elig_mod.get("eligibilityCriteria", "")
    parsed_criteria = parse_eligibility_criteria(raw_criteria)

    title = id_mod.get("briefTitle", "")
    conditions = cond_mod.get("conditions", [])
    summary = clean_text(desc_mod.get("briefSummary", ""))
    clean_criteria = clean_text(raw_criteria)

    return {
        "nct_id": id_mod.get("nctId", ""),
        "title": title,
        "official_title": id_mod.get("officialTitle", ""),
        "summary": summary,
        "conditions": conditions,
        "interventions": intervention_names,
        "phase": design_mod.get("phases", []),
        "status": status_mod.get("overallStatus", ""),
        "min_age": elig_mod.get("minimumAge", ""),
        "max_age": elig_mod.get("maximumAge", ""),
        "gender": elig_mod.get("sex", "ALL"),
        "healthy_volunteers": elig_mod.get("healthyVolunteers", False),
        "locations": location_strings,
        "eligibility_raw": clean_criteria,
        "inclusion_criteria": parsed_criteria["inclusion"],
        "exclusion_criteria": parsed_criteria["exclusion"],
        "embedding_text": (
            f"Trial: {title}\n"
            f"Conditions: {', '.join(conditions)}\n"
            f"Summary: {summary}\n"
            f"Eligibility: {clean_criteria}\n"
            f"Locations: {', '.join(location_strings)}"
        ).strip(),
    }


def preprocess_trials(
    input_path: str = "data/raw_trials.json",
    output_path: str = "data/processed_trials.json",
) -> List[dict]:
    with open(input_path) as f:
        raw_trials = json.load(f)

    print(f"Processing {len(raw_trials)} trials...")
    processed = []

    for trial in raw_trials:
        try:
            p = process_trial(trial)
            if p["eligibility_raw"] and len(p["eligibility_raw"]) > 100:
                processed.append(p)
        except Exception as e:
            print(f"  Skipping trial: {e}")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(processed, f, indent=2)

    print(f"Saved {len(processed)} processed trials to {output_path}")
    return processed


if __name__ == "__main__":
    preprocess_trials("data/raw_trials.json", "data/processed_trials.json")
