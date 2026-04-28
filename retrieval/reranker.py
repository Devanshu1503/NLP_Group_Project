"""
LLM-based reranker for retrieved clinical trials.
Takes top-K candidates from semantic or keyword retrieval and reranks
them using an LLM relevance score. Uses the faster Haiku model to keep
latency and cost low.
"""
import json
import os
from typing import List, Dict
from anthropic import Anthropic
from ner.schemas import PatientProfile

client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

_SYSTEM = """You are a clinical trial eligibility expert. Given a patient profile and a list of trials, score each trial's relevance for this specific patient from 0 to 10.

Scoring criteria:
- 9-10: Condition matches well, patient likely meets age/gender requirements
- 6-8: Partial condition match or minor eligibility uncertainty
- 3-5: Tangentially related or significant eligibility concerns
- 0-2: Unlikely to match or clear exclusion criteria

Return ONLY a JSON array sorted by score descending:
[{"nct_id": "NCT...", "score": 9}, ...]"""


def rerank_trials(
    profile: PatientProfile,
    trials: List[Dict],
    top_k: int = 5,
) -> List[Dict]:
    """
    Rerank a list of retrieved trials using LLM relevance scoring.

    Args:
        profile: Structured patient profile
        trials: Candidates from semantic or keyword retrieval
        top_k: Number of top trials to return after reranking

    Returns:
        Reranked trial list (top_k items)
    """
    if not trials:
        return []

    trial_summaries = [
        {
            "nct_id": t.get("nct_id", ""),
            "title": t.get("title", ""),
            "conditions": t.get("conditions", []),
            "eligibility_summary": t.get("eligibility_raw", "")[:300],
            "min_age": t.get("min_age", ""),
            "max_age": t.get("max_age", ""),
            "gender": t.get("gender", "ALL"),
        }
        for t in trials
    ]

    prompt = (
        f"Patient: {profile.to_query_string()}\n\n"
        f"Trials:\n{json.dumps(trial_summaries, indent=2)}\n\n"
        "Score each trial 0-10 for this patient. Return JSON array."
    )

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",  # Cheaper model fine for scoring
        max_tokens=600,
        system=_SYSTEM,
        messages=[{"role": "user", "content": prompt}],
    )

    raw = response.content[0].text.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]

    scores = json.loads(raw.strip())
    score_map = {s["nct_id"]: s["score"] for s in scores}

    for trial in trials:
        trial["rerank_score"] = score_map.get(trial.get("nct_id", ""), 0)

    return sorted(trials, key=lambda x: x.get("rerank_score", 0), reverse=True)[:top_k]
