"""
Semantic RAG retrieval backed by Qdrant + sentence-transformers.
Replaces the previous FAISS-backed implementation; public interface unchanged.

Interface contract (relied on by RetrieverAgent, api/main.py, eval_retrieval.py):
    SemanticRetriever(index_path=..., model_name=...)
    .retrieve(profile: PatientProfile, top_k: int) -> List[Dict]

New additive method (used by /compare-trials endpoint):
    .retrieve_raw(query: str, top_k: int) -> List[Dict]
"""
import os
from typing import List, Dict
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient

load_dotenv()

from ner.schemas import PatientProfile

COLLECTION_NAME = "clinical_trials"
MODEL_NAME      = "sentence-transformers/all-MiniLM-L6-v2"
QDRANT_URL      = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY  = os.getenv("QDRANT_API_KEY")


class SemanticRetriever:
    def __init__(
        self,
        index_path: str = "data/faiss_index",  # kept for call-site compat, unused
        model_name: str = MODEL_NAME,
    ):
        self._client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        self._model  = SentenceTransformer(model_name)
        info = self._client.get_collection(COLLECTION_NAME)
        self.collection_size = info.points_count or 0
        print(f"Qdrant retriever ready. Collection '{COLLECTION_NAME}': {self.collection_size} vectors")

    # ── Public interface ──────────────────────────────────────────────────

    def retrieve(self, profile: PatientProfile, top_k: int = 10) -> List[Dict]:
        """Retrieve via patient profile (existing contract)."""
        query = profile.to_query_string()
        trials = self._search_and_dedup(query, top_k * 4)
        trials = self._apply_hard_filters(trials, profile)
        return trials[:top_k]

    def retrieve_raw(self, query: str, top_k: int = 10) -> List[Dict]:
        """Retrieve by plain text query without a PatientProfile."""
        return self._search_and_dedup(query, top_k * 4)[:top_k]

    # ── Internal helpers ──────────────────────────────────────────────────

    def _search_and_dedup(self, query: str, fetch_limit: int) -> List[Dict]:
        """Embed query, search Qdrant, deduplicate by NCT ID (highest score wins)."""
        query_vec = self._model.encode(
            [query], normalize_embeddings=True
        )[0].tolist()

        response = self._client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vec,
            limit=fetch_limit,
            with_payload=True,
        )

        seen: dict[str, dict] = {}
        for hit in response.points:
            p      = hit.payload
            nct_id = p.get("nct_id", "")
            if not nct_id:
                continue
            score = hit.score
            if nct_id not in seen or score > seen[nct_id]["similarity_score"]:
                seen[nct_id] = {
                    "nct_id":          nct_id,
                    "title":           p.get("title", ""),
                    "conditions":      p.get("conditions", []),
                    "eligibility_raw": p.get("eligibility_raw", ""),
                    "summary":         p.get("summary", ""),
                    "locations":       p.get("locations", []),
                    "min_age":         p.get("min_age", ""),
                    "max_age":         p.get("max_age", ""),
                    "gender":          p.get("gender", "ALL"),
                    "phase":           p.get("phase", []),
                    "status":          p.get("status", ""),
                    "interventions":   p.get("interventions", []),
                    "url":             p.get("url", ""),
                    "chunk_text":      p.get("chunk_text", ""),
                    "similarity_score": score,
                    "retrieval_method": "semantic_rag",
                }

        return sorted(seen.values(), key=lambda x: x["similarity_score"], reverse=True)

    def _apply_hard_filters(self, trials: List[Dict], profile: PatientProfile) -> List[Dict]:
        filtered = []
        for trial in trials:
            if profile.age:
                out_of_bounds = False
                for bound_key, too_far in [
                    ("min_age", lambda age, b: age < b),
                    ("max_age", lambda age, b: age > b),
                ]:
                    bound_str = (
                        trial.get(bound_key, "")
                        .replace(" Years", "")
                        .replace(" Year", "")
                        .strip()
                    )
                    try:
                        if too_far(profile.age, int(bound_str)):
                            out_of_bounds = True
                            break
                    except (ValueError, TypeError):
                        pass
                if out_of_bounds:
                    continue

            trial_gender = trial.get("gender", "ALL").upper()
            if trial_gender not in ("ALL", "") and profile.gender:
                if trial_gender == "MALE" and profile.gender.lower() != "male":
                    continue
                if trial_gender == "FEMALE" and profile.gender.lower() != "female":
                    continue

            filtered.append(trial)
        return filtered
