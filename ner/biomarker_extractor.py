"""
Extract biomarker values from PDF lab reports using PyMuPDF + Claude.
Called by the /analyze-report endpoint.
"""
import os
import json
import fitz
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()

client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

SYSTEM_PROMPT = """You are a medical lab report parser. Extract ALL biomarker
values from the text of a lab report. Return ONLY valid JSON — no markdown,
no explanation, nothing outside the JSON object.

Output structure:
{
  "biomarkers": {
    "HbA1c": {"value": "9.1", "unit": "%", "reference_range": "4.0-5.6", "status": "HIGH"},
    "Glucose": {"value": "180", "unit": "mg/dL", "reference_range": "70-100", "status": "HIGH"}
  },
  "report_date": "YYYY-MM-DD or null",
  "lab_name": "string or null"
}

Rules:
- status must be HIGH, LOW, NORMAL, or UNKNOWN
- Only extract values explicitly present in the document
- Never invent or estimate values
- Return empty biomarkers dict if no values found"""


def extract_from_pdf(pdf_path: str) -> dict:
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()

        if not text.strip():
            return {"biomarkers": {}, "error": "No text found in PDF"}

        return _call_llm(text)

    except Exception as e:
        return {"biomarkers": {}, "error": str(e)}


def extract_from_text(text: str) -> dict:
    """Same extraction but from raw text instead of PDF."""
    try:
        return _call_llm(text)
    except Exception as e:
        return {"biomarkers": {}, "error": str(e)}


def _call_llm(text: str) -> dict:
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=4096,
        system=[{"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": f"Extract biomarkers from this lab report:\n\n{text[:8000]}"}],
    )

    raw = response.content[0].text.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]

    return json.loads(raw.strip())
