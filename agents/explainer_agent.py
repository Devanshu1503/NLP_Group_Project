"""
Agent 4: Plain-English trial explainer with safety guardrails.
Strictly grounded — only uses data present in the retrieved trial records.
"""
import os
from dotenv import load_dotenv
from typing import List, Dict
from anthropic import Anthropic

load_dotenv()
from ner.schemas import PatientProfile

client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

_SYSTEM = """You are a compassionate patient navigator explaining clinical trials in plain language.

For each trial explain: what it studies (simply), key eligibility requirements, location, and how to contact the study team.

CRITICAL GUARDRAILS — never break these:
- Only use information from the provided trial data — never invent details
- Never say a patient IS eligible — say they MAY qualify or MIGHT be eligible
- Always include the NCT ID so patients can verify at ClinicalTrials.gov
- Always end with: "Verify your eligibility directly with the study team and discuss with your doctor before enrolling in any clinical trial."
- If unsure about anything, say so and direct to ClinicalTrials.gov"""


def explain_trials(
    profile: PatientProfile,
    trials: List[Dict],
    max_trials: int = 5,
) -> str:
    """
    Generate a plain-English explanation of matched trials.

    Args:
        profile: Patient's structured profile
        trials: Retrieved trial dicts
        max_trials: Max trials to explain

    Returns:
        Formatted explanation string
    """
    if not trials:
        return (
            "I wasn't able to find any currently recruiting trials that closely match your profile. "
            "You can search directly at ClinicalTrials.gov or ask your doctor about available options."
        )

    trial_summaries = []
    for i, trial in enumerate(trials[:max_trials], 1):
        trial_summaries.append(
            f"Trial {i}:\n"
            f"- NCT ID: {trial.get('nct_id', 'Unknown')}\n"
            f"- Title: {trial.get('title', 'Unknown')}\n"
            f"- Conditions: {', '.join(trial.get('conditions', []))}\n"
            f"- Locations: {', '.join(trial.get('locations', ['Not specified'])[:3])}\n"
            f"- Eligibility (excerpt): {trial.get('eligibility_raw', '')[:500]}..."
        )

    prompt = (
        f"Patient profile:\n"
        f"- Age: {profile.age}, Gender: {profile.gender}\n"
        f"- Conditions: {', '.join(profile.conditions)}\n"
        f"- Medications: {', '.join(profile.medications)}\n"
        f"- Location: {profile.location_city}, {profile.location_state}\n\n"
        f"Potentially matching trials:\n\n" + "\n\n".join(trial_summaries) + "\n\n"
        "Please explain these trials to the patient in plain, friendly language."
    )

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=2000,
        system=[{"type": "text", "text": _SYSTEM, "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": prompt}],
    )

    return response.content[0].text
