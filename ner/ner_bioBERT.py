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

_IL_CITIES = {
    "chicago", "evanston", "oak park", "naperville", "schaumburg",
    "joliet", "waukegan", "aurora", "rockford", "peoria", "elgin",
}

_STATE_MAP = {
    "illinois": "IL", "california": "CA", "new york": "NY",
    "texas": "TX", "florida": "FL", "ohio": "OH", "michigan": "MI",
}

# Expanded medication list covering common drug names + brand names
COMMON_MEDICATIONS = [
    # Diabetes
    "metformin", "insulin glargine", "insulin lispro", "insulin aspart",
    "empagliflozin", "jardiance", "dapagliflozin", "farxiga",
    "semaglutide", "ozempic", "wegovy", "liraglutide", "victoza",
    "sitagliptin", "januvia", "glipizide", "glimepiride", "pioglitazone",
    # Cardiovascular
    "lisinopril", "atorvastatin", "losartan", "amlodipine",
    "carvedilol", "coreg", "furosemide", "lasix", "spironolactone",
    "hydrochlorothiazide", "metoprolol", "bisoprolol",
    "warfarin", "coumadin", "apixaban", "eliquis", "rivaroxaban", "xarelto",
    # Rheumatology / Immunology
    "adalimumab", "humira", "etanercept", "enbrel",
    "hydroxychloroquine", "plaquenil", "prednisone", "methotrexate",
    "rituximab", "tocilizumab",
    # Oncology
    "pembrolizumab", "keytruda", "trastuzumab", "herceptin",
    "ibrutinib", "lenalidomide", "bortezomib",
    # Neurology
    "levodopa", "carbidopa", "carbidopa-levodopa",
    "donepezil", "aricept", "memantine", "namenda",
    "gabapentin", "pregabalin", "lyrica",
    # Psychiatry
    "sertraline", "zoloft", "fluoxetine", "prozac",
    "escitalopram", "lexapro", "duloxetine", "cymbalta",
    "bupropion", "wellbutrin", "quetiapine", "seroquel",
    # GI / Other
    "omeprazole", "pantoprazole", "levothyroxine", "synthroid",
]


class BioBERTExtractor:
    def __init__(self, model_name: str = NER_MODEL):
        print(f"Loading NER model: {model_name}...")
        self.ner = pipeline("ner", model=model_name, aggregation_strategy="simple")

    def extract_profile(self, patient_text: str) -> PatientProfile:
        entities = self.ner(patient_text)

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

    def _extract_age(self, text: str) -> Optional[int]:
        age_patterns = [
            r"\b(\d{1,3})[- ]?year[s]?[- ]?old\b",   # 47-year-old, 47 year old
            r"\bage[d]?\s*(?:is\s*)?(\d{1,3})\b",      # age 47, aged 47
            r"\bI(?:\'m| am)\s+(\d{1,3})\b",           # I'm 47, I am 47
            r"\b(\d{1,3})\s*yo\b",                      # 47yo
            r"\bmy age is\s+(\d{1,3})\b",              # my age is 47
            r"\b(\d{1,3})\s*years?\s+of\s+age\b",      # 47 years of age
        ]
        for pattern in age_patterns:
            m = re.search(pattern, text, re.IGNORECASE)
            if m:
                age = int(m.group(1))
                if 1 <= age <= 120:  # Sanity check
                    return age
        return None

    def _extract_gender(self, text: str) -> Optional[str]:
        t = text.lower()
        if any(w in t for w in ["woman", "female", " she ", " her "]):
            return "female"
        if any(w in t for w in ["man", "male", " he ", " his "]):
            return "male"
        return None

    def _extract_medications(self, text: str) -> list:
        found = []
        text_lower = text.lower()
        for med in COMMON_MEDICATIONS:
            if med.lower() in text_lower:
                found.append(med)
        return list(dict.fromkeys(found))  # Deduplicate while preserving order

    def _extract_lab_values(self, text: str) -> dict:
        labs = {}
        patterns = {
            "HbA1c": r"hba1c\s+(?:of\s+|was\s+|is\s+)?([\d.]+\s*%?)",
            "BMI": r"bmi\s+(?:of\s+|is\s+|was\s+)?([\d.]+)",
            "ejection_fraction": r"ejection\s+fraction\s+(?:of\s+|is\s+|was\s+)?([\d.]+\s*%?)",
            "eGFR": r"egfr\s+(?:of\s+|is\s+|was\s+)?([\d.]+)",
            "glucose": r"(?:fasting\s+)?glucose\s+(?:of\s+|is\s+|was\s+)?([\d.]+)",
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
