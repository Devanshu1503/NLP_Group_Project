"""
Agent 3: Trial retrieval coordinator.
Wraps semantic and keyword retrievers, optionally applying the LLM reranker.
"""
from typing import List, Dict
from ner.schemas import PatientProfile


class RetrieverAgent:
    def __init__(
        self,
        retrieval_method: str = "semantic",
        use_reranker: bool = True,
        index_path: str = "data/faiss_index",
    ):
        """
        Args:
            retrieval_method: "semantic" (Champion) or "keyword" (Challenger)
            use_reranker: Whether to apply LLM reranking on top of retrieval
            index_path: Path to FAISS index (only used for semantic method)
        """
        self.use_reranker = use_reranker

        if retrieval_method == "semantic":
            from retrieval.retriever_semantic import SemanticRetriever
            self.retriever = SemanticRetriever(index_path=index_path)
        else:
            from retrieval.retriever_keyword import KeywordRetriever
            self.retriever = KeywordRetriever()

        if use_reranker:
            from retrieval.reranker import rerank_trials
            self._rerank = rerank_trials

    def retrieve(self, profile: PatientProfile, top_k: int = 10) -> List[Dict]:
        # Fetch more candidates when reranking so the reranker has room to work
        fetch_k = top_k * 2 if self.use_reranker else top_k
        candidates = self.retriever.retrieve(profile, top_k=fetch_k)

        if self.use_reranker and candidates:
            candidates = self._rerank(profile, candidates, top_k=top_k)

        return candidates[:top_k]
