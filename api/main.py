"""
FastAPI app exposing TrialNav as a REST API.

Run:
    uvicorn api.main:app --reload

Endpoints:
    POST /chat          — main chat turn
    DELETE /session/:id — clear a session
    GET  /health        — liveness check
"""
import uuid
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from agents.conversation_agent import ConversationAgent
from retrieval.retriever_keyword import KeywordRetriever
from agents.explainer_agent import explain_trials

# Loaded lazily at startup — requires FAISS index to exist
semantic_retriever = None
keyword_retriever = KeywordRetriever()

# In-memory session store — fine for dev/demo, use Redis in production
sessions: dict[str, ConversationAgent] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    global semantic_retriever
    try:
        from retrieval.retriever_semantic import SemanticRetriever
        semantic_retriever = SemanticRetriever()
        print("Semantic retriever loaded.")
    except Exception as e:
        print(f"Warning: semantic retriever unavailable ({e}). Run retrieval/embed_trials.py first.")
    yield
    sessions.clear()


app = FastAPI(
    title="TrialNav API",
    description="Patient-facing clinical trial matching via conversational NLP + RAG",
    version="0.1.0",
    lifespan=lifespan,
)


class ChatRequest(BaseModel):
    session_id: Optional[str] = None
    message: str
    retrieval_method: str = "semantic"  # "semantic" or "keyword"
    use_reranker: bool = True


class ChatResponse(BaseModel):
    session_id: str
    response: str
    profile_complete: bool
    trials_found: Optional[int] = None


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    session_id = request.session_id or str(uuid.uuid4())
    if session_id not in sessions:
        sessions[session_id] = ConversationAgent()

    agent = sessions[session_id]
    response_text, profile_complete = agent.chat(request.message)
    trials_found = None

    if profile_complete:
        try:
            profile = agent.extract_profile()

            if request.retrieval_method == "semantic" and semantic_retriever:
                trials = semantic_retriever.retrieve(profile, top_k=10)
            else:
                trials = keyword_retriever.retrieve(profile, top_k=10)

            if request.use_reranker and trials:
                from retrieval.reranker import rerank_trials
                trials = rerank_trials(profile, trials, top_k=5)

            trials_found = len(trials)
            explanation = explain_trials(profile, trials, max_trials=5)
            response_text = f"{response_text}\n\n---\n\n{explanation}"

        except Exception as e:
            response_text += (
                "\n\nI found some potential matches but had trouble processing them. "
                "Please search directly at ClinicalTrials.gov."
            )

    return ChatResponse(
        session_id=session_id,
        response=response_text,
        profile_complete=profile_complete,
        trials_found=trials_found,
    )


@app.delete("/session/{session_id}")
async def clear_session(session_id: str):
    sessions.pop(session_id, None)
    return {"status": "cleared"}


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "semantic_retriever": semantic_retriever is not None,
        "index_size": semantic_retriever.index.ntotal if semantic_retriever else 0,
    }
