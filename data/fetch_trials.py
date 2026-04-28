"""
Fetch recruiting clinical trials from ClinicalTrials.gov API v2.
No API key required. Saves raw JSON to data/raw_trials.json.

Usage:
    python data/fetch_trials.py
"""
import requests
import json
import time
from pathlib import Path

CTGOV_API = "https://clinicaltrials.gov/api/v2/studies"


def fetch_trials(
    condition: str = None,
    max_trials: int = 500,
    status: str = "RECRUITING",
    output_path: str = "data/raw_trials.json",
):
    all_trials = []
    next_page_token = None
    page_size = 100  # API max per page

    # NOTE: fields param omitted intentionally — v2 field names differ from v1.
    # preprocess_trials.py handles the full nested response structure.
    params = {
        "filter.overallStatus": status,
        "pageSize": page_size,
    }
    if condition:
        params["query.cond"] = condition

    print(f"Fetching up to {max_trials} {status} trials" + (f" for '{condition}'" if condition else "") + "...")

    while len(all_trials) < max_trials:
        if next_page_token:
            params["pageToken"] = next_page_token

        response = requests.get(CTGOV_API, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()

        studies = data.get("studies", [])
        if not studies:
            break

        all_trials.extend(studies)
        print(f"  Fetched {len(all_trials)} trials so far...")

        next_page_token = data.get("nextPageToken")
        if not next_page_token:
            break

        time.sleep(0.5)  # Be polite to the API

    all_trials = all_trials[:max_trials]

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_trials, f, indent=2)

    print(f"Saved {len(all_trials)} trials to {output_path}")
    return all_trials


if __name__ == "__main__":
    # Focused set for fast development
    fetch_trials(condition="diabetes", max_trials=200, output_path="data/raw_trials_diabetes.json")
    # General sample for full pipeline
    fetch_trials(max_trials=500, output_path="data/raw_trials.json")
