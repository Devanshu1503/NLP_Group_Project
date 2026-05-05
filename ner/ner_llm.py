"""
Problem 1 Champion: LLM-based patient profile extraction using Claude.
Uses structured JSON output with prompt caching on the system prompt.
"""
import json
import os
from dotenv import load_dotenv
from anthropic import Anthropic

load_dotenv()
from ner.schemas import PatientProfile

client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

SYSTEM_PROMPT = """You are a medical NLP system. Extract structured patient information from free-text descriptions. Return ONLY valid JSON matching the schema below. Do not include any text outside the JSON object.

Schema:
{
  "age": integer or null,
  "gender": "male" | "female" | "other" | null,
  "conditions": ["list of medical conditions"],
  "medications": ["list of medications with dosages if mentioned"],
  "lab_values": {"test_name": "value"},
  "prior_treatments": ["past treatments, surgeries, chemotherapy etc"],
  "devices": ["medical devices like pacemaker, insulin pump"],
  "location_city": "city name or null",
  "location_state": "2-letter state code or null",
  "travel_distance_miles": integer or null,
  "healthy_volunteer": boolean,
  "disease_status": "remission" | "active" | "stable" | "newly diagnosed" | null
}

Rules:
- Normalize condition names to standard medical terminology where clear
- Extract ALL mentioned medications including OTC
- For lab values use standard names: HbA1c, BMI, eGFR, ejection_fraction, etc.
- If patient has no conditions and wants to join research, set healthy_volunteer: true
- Only extract what is explicitly stated — do not infer or hallucinate"""


def extract_profile_llm(patient_text: str) -> PatientProfile:
    """Extract structured patient profile from free-text using Claude."""
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1000,
        system=[
            {
                "type": "text",
                "text": SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},  # Cache system prompt across calls
            }
        ],
        messages=[
            {"role": "user", "content": f"Extract patient profile from: {patient_text}"}
        ],
    )

    raw_json = response.content[0].text.strip()

    # Strip markdown fences if present
    if raw_json.startswith("```"):
        raw_json = raw_json.split("```")[1]
        if raw_json.startswith("json"):
            raw_json = raw_json[4:]

    data = json.loads(raw_json.strip())
    profile = PatientProfile(**data)
    profile.missing_fields = profile.get_missing_critical_fields()
    return profile


def extract_profile_from_conversation(messages: list) -> PatientProfile:
    """Extract profile from full multi-turn conversation history."""
    user_text = " ".join(m["content"] for m in messages if m["role"] == "user")
    return extract_profile_llm(user_text)
