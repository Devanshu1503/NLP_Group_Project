"""
Problem 2 Champion: Semantic RAG retrieval using FAISS + sentence-transformers.
"""

import pickle
import numpy as np
from typing import List, Dict
from sentence_transformers import SentenceTransformer
import faiss

from ner.schemas import PatientProfile

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class SemanticRetriever:
    def __init__(
        self,
        index_path: str = "data/faiss_index",
        model_name: str = MODEL_NAME,
    ):
        print("Loading FAISS index...")
        self.index = faiss.read_index(f"{index_path}/trials.index")

        with open(f"{index_path}/trials_metadata.pkl", "rb") as f:
            self.trials = pickle.load(f)

        print(f"Loading embedding model: {model_name}...")
        self.model = SentenceTransformer(model_name)
        print(f"Retriever ready. Index size: {self.index.ntotal}")

    def retrieve(self, profile: PatientProfile, top_k: int = 10) -> List[Dict]:
        """
        Retrieve top-k trials matching the patient profile via semantic similarity.
        Applies hard age/gender filters after vector search.
        """
        query = profile.to_query_string()
        query_embedding = self.model.encode(
            [query], normalize_embeddings=True
        ).astype(np.float32)

        scores, indices = self.index.search(query_embedding, top_k * 2)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            trial = self.trials[idx].copy()
            trial["similarity_score"] = float(score)
            trial["retrieval_method"] = "semantic_rag"
            results.append(trial)

        results = self._apply_hard_filters(results, profile)
        return results[:top_k]

    def _apply_hard_filters(self, trials: List[Dict], profile: PatientProfile) -> List[Dict]:
        filtered = []
        for trial in trials:
            # Age filter
            if profile.age:
                for bound_key, comparator in [("min_age", lambda a, b: a < b), ("max_age", lambda a, b: a > b)]:
                    bound_str = trial.get(bound_key, "")
                    bound_str = bound_str.replace(" Years", "").replace(" Year", "").strip()
                    try:
                        if comparator(profile.age, int(bound_str)):
                            break
                    except (ValueError, TypeError):
                        pass
                else:
                    pass  # Both checks passed — continue below

            # Gender filter
            trial_gender = trial.get("gender", "ALL").upper()
            if trial_gender not in ("ALL", "") and profile.gender:
                if trial_gender == "MALE" and profile.gender.lower() != "male":
                    continue
                if trial_gender == "FEMALE" and profile.gender.lower() != "female":
                    continue

            filtered.append(trial)
        return filtered
