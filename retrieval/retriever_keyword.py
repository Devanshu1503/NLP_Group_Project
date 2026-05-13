"""
Problem 2 Challenger/Baseline: ClinicalTrials.gov keyword API search.
Mirrors what a patient gets from the CT.gov website directly — the status quo
we are comparing against with semantic RAG.
"""
import requests
from typing import List, Dict
from ner.schemas import PatientProfile

CTGOV_API = "https://clinicaltrials.gov/api/v2/studies"


class KeywordRetriever:
    def retrieve(self, profile: PatientProfile, top_k: int = 10) -> List[Dict]:
        """Retrieve using patient conditions as query.cond (condition-specific search)."""
        keywords = " ".join(profile.conditions) if profile.conditions else "clinical trial"
        return self._fetch({"query.cond": keywords}, top_k, gender_filter=profile.gender)

    def retrieve_raw(self, query: str, top_k: int = 10) -> List[Dict]:
        """Retrieve using a free-text query against query.term (full-text search across all fields)."""
        return self._fetch({"query.term": query}, top_k, gender_filter=None)

    def _fetch(self, query_params: dict, top_k: int, gender_filter) -> List[Dict]:
        params = {
            **query_params,
            "filter.overallStatus": "RECRUITING",
            "pageSize": top_k * 2,
        }

        response = requests.get(CTGOV_API, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()

        results = []
        for study in data.get("studies", []):
            proto      = study.get("protocolSection", {})
            id_mod     = proto.get("identificationModule", {})
            elig_mod   = proto.get("eligibilityModule", {})
            cond_mod   = proto.get("conditionsModule", {})
            loc_mod    = proto.get("contactsLocationsModule", {})
            status_mod = proto.get("statusModule", {})
            design_mod = proto.get("designModule", {})

            trial_gender = elig_mod.get("sex", "ALL").upper()
            if trial_gender not in ("ALL", "") and gender_filter:
                if trial_gender == "MALE" and gender_filter.lower() != "male":
                    continue
                if trial_gender == "FEMALE" and gender_filter.lower() != "female":
                    continue

            locs = [
                f"{l.get('city', '')}, {l.get('state', '')}"
                for l in loc_mod.get("locations", [])[:3]
            ]
            nct_id = id_mod.get("nctId", "")

            results.append({
                "nct_id":          nct_id,
                "title":           id_mod.get("briefTitle", ""),
                "conditions":      cond_mod.get("conditions", []),
                "eligibility_raw": elig_mod.get("eligibilityCriteria", ""),
                "locations":       locs,
                "phase":           design_mod.get("phases", []),
                "status":          status_mod.get("overallStatus", "RECRUITING"),
                "url":             f"https://clinicaltrials.gov/study/{nct_id}",
                "similarity_score": None,
                "retrieval_method": "keyword_ctgov",
            })

        return results[:top_k]
