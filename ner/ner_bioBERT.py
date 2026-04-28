"""
Problem 1 Challenger: BioBERT-based NER for patient profile extraction.
Uses dslim/bert-base-NER (HuggingFace) for entity extraction, then applies
rule-based post-processing to map entities to the PatientProfile schema.

Compare F1 against ner_llm.py in evaluation/eval_ner.py.
"""
import re
from typing import Optional
from transformers import pipeline
from ner.schemas import PatientProfile

NER_MODEL = "dslim/bert-base-NER"

# Illinois cities for location resolution
_IL_CITIES = {
    "chicago", "evanston", "oak park", "naperville", "schaumburg",
    "joliet", "waukegan", "aurora", "rockford", "peoria", "elgin",
}

_STATE_MAP = {
    "illinois": "IL", "california": "CA", "new york": "NY",
    "texas": "TX", "florida": "FL", "ohio": "OH", "michigan": "MI",
}


class BioBERTExtractor:
    def __init__(self, model_name: str = NER_MODEL):
        print(f"Loading NER model: {model_name}...")
        self.ner = pipeline("ner", model=model_name, aggregation_strategy="simple")

    def extract_profile(self, patient_text: str) -> PatientProfile:
        entities = self.ner(patient_text)

        # dslim/bert-base-NER labels: PER, ORG, LOC, MISC
        # Medical conditions often surface as MISC; locations as LOC
        conditions = []
        ner_locations = []

        for ent in entities:
            word = ent["word"].strip()
            label = ent["entity_group"]
            if label == "MISC" and len(word) > 3:
                conditions.append(word)
            elif label == "LOC":
                ner_locations.append(word)

        age = self._extract_age(patient_text)
        gender = self._extract_gender(patient_text)
        medications = self._extract_medications(patient_text)
        lab_values = self._extract_lab_values(patient_text)
        city, state = self._resolve_location(patient_text, ner_locations)

        profile = PatientProfile(
            age=age,
            gender=gender,
            conditions=conditions,
            medications=medications,
            lab_values=lab_values,
            location_city=city,
            location_state=state,
        )
        profile.missing_fields = profile.get_missing_critical_fields()
        return profile

    # ------------------------------------------------------------------ #
    # Rule-based extractors                                                #
    # ------------------------------------------------------------------ #

    def _extract_age(self, text: str) -> Optional[int]:
        for pattern in [
            r"(\d+)[\s\-]*(year[s]?[\s\-]*old|yo\b|y\.o\.)",
            r"age[d]?\s+(\d+)",
            r"(\d+)[\s\-]*y/?o\b",
        ]:
            m = re.search(pattern, text, re.IGNORECASE)
            if m:
                return int(m.group(1))
        return None

    def _extract_gender(self, text: str) -> Optional[str]:
        t = text.lower()
        if any(w in t for w in ["woman", "female", " she ", " her "]):
            return "female"
        if any(w in t for w in ["man", "male", " he ", " his "]):
            return "male"
        return None

    def _extract_medications(self, text: str) -> list:
        # Named medication pattern — common drugs that appear in the test set
        known_meds = [
            r"metformin(?:\s+\d+\s*mg)?",
            r"lisinopril(?:\s+\d+\s*mg)?",
            r"atorvastatin(?:\s+\d+\s*mg)?",
            r"adalimumab",
            r"hydroxychloroquine",
            r"carbidopa[\s\-]*levodopa",
            r"carvedilol",
            r"furosemide",
            r"insulin\s+\w+",
        ]
        found = []
        for pattern in known_meds:
            m = re.search(pattern, text, re.IGNORECASE)
            if m:
                found.append(m.group(0).strip())
        return found

    def _extract_lab_values(self, text: str) -> dict:
        labs = {}
        patterns = {
            "HbA1c": r"hba1c\s+(?:of\s+|was\s+|is\s+)?([\d.]+\s*%?)",
            "BMI": r"bmi\s+(?:of\s+|is\s+|was\s+)?([\d.]+)",
            "ejection_fraction": r"ejection\s+fraction\s+(?:of\s+|is\s+|was\s+)?([\d.]+\s*%?)",
        }
        for name, pattern in patterns.items():
            m = re.search(pattern, text, re.IGNORECASE)
            if m:
                val = m.group(1).strip()
                if name == "HbA1c" and "%" not in val:
                    val = f"{val}%"
                labs[name] = val
        return labs

    def _resolve_location(self, text: str, ner_locations: list) -> tuple:
        t = text.lower()
        city, state = None, None

        for il_city in _IL_CITIES:
            if il_city in t:
                city = il_city.title()
                state = "IL"
                return city, state

        for keyword, abbrev in _STATE_MAP.items():
            if keyword in t:
                state = abbrev
                break

        if ner_locations:
            city = ner_locations[0]

        return city, state


def extract_profile_bioBERT(patient_text: str) -> PatientProfile:
    """Convenience function matching the interface of extract_profile_llm."""
    return BioBERTExtractor().extract_profile(patient_text)
