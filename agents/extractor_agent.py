"""
Agent 2: Patient profile extractor.
Thin wrapper around both NER approaches so the rest of the pipeline
doesn't need to know which method is in use.
"""
from ner.schemas import PatientProfile
from ner.ner_llm import extract_profile_llm, extract_profile_from_conversation


class ExtractorAgent:
    def __init__(self, method: str = "llm"):
        """
        Args:
            method: "llm" (Champion) or "bioBERT" (Challenger)
        """
        self.method = method
        if method == "bioBERT":
            from ner.ner_bioBERT import BioBERTExtractor
            self._bioBERT = BioBERTExtractor()

    def extract_from_text(self, text: str) -> PatientProfile:
        if self.method == "bioBERT":
            return self._bioBERT.extract_profile(text)
        return extract_profile_llm(text)

    def extract_from_conversation(self, messages: list) -> PatientProfile:
        if self.method == "bioBERT":
            user_text = " ".join(m["content"] for m in messages if m["role"] == "user")
            return self._bioBERT.extract_profile(user_text)
        return extract_profile_from_conversation(messages)
