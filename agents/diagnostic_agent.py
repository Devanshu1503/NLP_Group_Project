"""
Agent 5: Diagnostic Reasoning Agent.
Takes extracted biomarkers + patient symptoms and reasons about
likely conditions and trial-relevant clinical findings.
"""
import os
import json
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()

client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

SYSTEM_PROMPT = """You are a clinical reasoning assistant that helps match
patients to clinical trials. You receive biomarker values from a lab report
and symptoms described by a patient.

Your job:
1. Interpret each biomarker against clinical reference ranges — note what
   abnormal values suggest
2. Combine biomarker findings with reported symptoms
3. Identify the 1-3 most likely clinical conditions this profile is
   consistent with, with confidence levels
4. List which specific biomarkers and symptoms support each condition
5. Identify biomarker values most relevant for clinical trial eligibility
   criteria (e.g. HbA1c > 8% for diabetes trials, eGFR 30-60 for CKD trials)
6. Suggest the types of clinical trials this patient would most likely
   qualify for

CRITICAL RULES:
- NEVER say "you have [condition]" — always say "consistent with" or "suggests"
- ALWAYS recommend the patient confirm findings with their physician
- ONLY reason from data provided — never invent values
- Return ONLY valid JSON — no markdown, no text outside the JSON

Output this exact JSON structure:
{
  "biomarker_interpretation": [
    {
      "biomarker": "HbA1c",
      "value": "9.1%",
      "finding": "Significantly elevated above 6.5% diagnostic threshold",
      "clinical_significance": "Suggests poorly controlled diabetes",
      "status": "HIGH"
    }
  ],
  "suggested_conditions": [
    {
      "condition": "Type 2 Diabetes — Poorly Controlled",
      "confidence": "high",
      "supporting_biomarkers": ["HbA1c 9.1%", "Fasting Glucose 180 mg/dL"],
      "supporting_symptoms": ["fatigue", "frequent urination"],
      "icd_hint": "E11.65"
    }
  ],
  "trial_relevant_values": [
    "HbA1c 9.1% — meets common diabetes trial threshold of HbA1c > 7.5%",
    "eGFR 58 — meets stage 3 CKD criteria used in diabetic kidney disease trials"
  ],
  "recommended_trial_types": [
    "Type 2 Diabetes — glycemic control",
    "Diabetic kidney disease — renoprotective agents"
  ],
  "reasoning_summary": "Brief plain English paragraph summarizing findings for the patient",
  "physician_note": "Always include: These findings should be confirmed by your physician before enrolling in any clinical trial."
}"""


class DiagnosticAgent:
    def reason(
        self,
        biomarkers: dict,
        symptoms: list,
        age: int = None,
        gender: str = None,
    ) -> dict:
        """
        Reason about likely conditions from biomarkers + symptoms.

        Args:
            biomarkers: Dict from BiomarkerExtractor
            symptoms: List of symptom strings from patient chat
            age: Patient age if known
            gender: Patient gender if known

        Returns:
            Dict with biomarker_interpretation, suggested_conditions,
            trial_relevant_values, recommended_trial_types, reasoning_summary
        """
        if not biomarkers:
            return {
                "error": "No biomarkers provided",
                "reasoning_summary": "No lab values were found to analyze.",
            }

        biomarker_text = json.dumps(biomarkers, indent=2)
        symptom_text = ", ".join(symptoms) if symptoms else "None reported"
        context = f"Age: {age or 'Unknown'}, Gender: {gender or 'Unknown'}"

        prompt = (
            f"Patient context: {context}\n\n"
            f"Biomarker values from lab report:\n{biomarker_text}\n\n"
            f"Symptoms reported by patient:\n{symptom_text}\n\n"
            "Please analyze this clinical profile and return your reasoning as JSON."
        )

        try:
            response = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=8096,
                system=[{"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}],
                messages=[{"role": "user", "content": prompt}],
            )

            raw = response.content[0].text.strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]

            return json.loads(raw.strip())

        except Exception as e:
            return {
                "error": str(e),
                "reasoning_summary": "Unable to complete diagnostic reasoning.",
            }
