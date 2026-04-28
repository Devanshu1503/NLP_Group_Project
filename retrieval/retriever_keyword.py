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
        """
        Retrieve trials using ClinicalTrials.gov keyword search.

        Args:
            profile: Patient profile (conditions used as keywords)
            top_k: Max results

        Returns:
            List of trial dicts matching the keyword API format
        """
        keywords = " ".join(profile.conditions) if profile.conditions else "clinical trial"

        params = {
            "query.cond": keywords,
            "filter.overallStatus": "RECRUITING",
            "pageSize": top_k,
        }
        if profile.gender and profile.gender.lower() in ("male", "female"):
            params["filter.sex"] = profile.gender.upper()

        response = requests.get(CTGOV_API, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()

        results = []
        for study in data.get("studies", []):
            proto = study.get("protocolSection", {})
            id_mod = proto.get("identificationModule", {})
            elig_mod = proto.get("eligibilityModule", {})
            cond_mod = proto.get("conditionsModule", {})
            loc_mod = proto.get("contactsLocationsModule", {})

            locs = [
                f"{l.get('city', '')}, {l.get('state', '')}"
                for l in loc_mod.get("locations", [])[:3]
            ]

            results.append({
                "nct_id": id_mod.get("nctId", ""),
                "title": id_mod.get("briefTitle", ""),
                "conditions": cond_mod.get("conditions", []),
                "eligibility_raw": elig_mod.get("eligibilityCriteria", ""),
                "locations": locs,
                "similarity_score": None,
                "retrieval_method": "keyword_ctgov",
            })

        return results
