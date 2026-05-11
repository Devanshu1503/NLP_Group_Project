"""
FastAPI app exposing TrialNav as a REST API.

Run:
    uvicorn api.main:app --reload

Endpoints:
    POST /chat          — main chat turn
    DELETE /session/:id — clear a session
    GET  /health        — liveness check
"""
import json
import os
import shutil
import tempfile
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from agents.conversation_agent import ConversationAgent
from retrieval.retriever_keyword import KeywordRetriever
from agents.explainer_agent import explain_trials
from agents.diagnostic_agent import DiagnosticAgent
from ner.biomarker_extractor import extract_from_pdf, extract_from_text

# Loaded lazily at startup — requires FAISS index to exist
semantic_retriever = None
bioBERT_extractor = None
keyword_retriever = KeywordRetriever()
diagnostic_agent = DiagnosticAgent()

# In-memory session store — fine for dev/demo, use Redis in production
sessions: dict[str, ConversationAgent] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    global semantic_retriever, bioBERT_extractor
    try:
        from retrieval.retriever_semantic import SemanticRetriever
        semantic_retriever = SemanticRetriever()
        print("Semantic retriever loaded.")
    except Exception as e:
        print(f"Warning: semantic retriever unavailable ({e}). Run retrieval/embed_trials.py first.")
    try:
        from ner.ner_bioBERT import BioBERTExtractor
        bioBERT_extractor = BioBERTExtractor()
        print("BioBERT extractor loaded.")
    except Exception as e:
        print(f"Warning: BioBERT extractor unavailable ({e}).")
    yield
    sessions.clear()


_STATIC = Path(__file__).parent / "static"

app = FastAPI(
    title="TrialNav API",
    description="Patient-facing clinical trial matching via conversational NLP + RAG",
    version="0.1.0",
    lifespan=lifespan,
)

app.mount("/static", StaticFiles(directory=_STATIC), name="static")


@app.get("/", include_in_schema=False)
def root():
    return FileResponse(_STATIC / "index.html")


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


class NERCompareRequest(BaseModel):
    text: str


@app.post("/compare-ner")
async def compare_ner(request: NERCompareRequest):
    import time
    from ner.ner_llm import extract_profile_llm

    results = {}

    t0 = time.time()
    try:
        llm_profile = extract_profile_llm(request.text)
        results["llm"] = {
            "profile": llm_profile.model_dump(exclude={"missing_fields"}),
            "missing_fields": llm_profile.missing_fields,
            "time_seconds": round(time.time() - t0, 2),
            "error": None,
        }
    except Exception as e:
        results["llm"] = {"profile": None, "missing_fields": [], "time_seconds": round(time.time() - t0, 2), "error": str(e)}

    t0 = time.time()
    if bioBERT_extractor:
        try:
            bio_profile = bioBERT_extractor.extract_profile(request.text)
            results["bioBERT"] = {
                "profile": bio_profile.model_dump(exclude={"missing_fields"}),
                "missing_fields": bio_profile.missing_fields,
                "time_seconds": round(time.time() - t0, 2),
                "error": None,
            }
        except Exception as e:
            results["bioBERT"] = {"profile": None, "missing_fields": [], "time_seconds": round(time.time() - t0, 2), "error": str(e)}
    else:
        results["bioBERT"] = {"profile": None, "missing_fields": [], "time_seconds": 0, "error": "BioBERT model not loaded at startup"}

    return results


@app.post("/analyze-report")
async def analyze_report(
    file: UploadFile = File(None),
    lab_text: str = Form(None),
    session_id: str = Form(None),
    symptoms: str = Form(""),
):
    """
    Accept either a PDF upload or raw pasted lab text.
    Extract biomarkers, run diagnostic reasoning, return structured findings.
    """
    session_id = session_id or str(uuid.uuid4())

    biomarker_data = {}
    if file and file.filename and file.filename.endswith(".pdf"):
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name
        biomarker_data = extract_from_pdf(tmp_path)
        os.unlink(tmp_path)
    elif lab_text:
        biomarker_data = extract_from_text(lab_text)
    else:
        return {"error": "Provide either a PDF file or lab_text"}

    biomarkers = biomarker_data.get("biomarkers", {})
    symptom_list = [s.strip() for s in symptoms.split(",") if s.strip()]

    diagnosis = diagnostic_agent.reason(biomarkers=biomarkers, symptoms=symptom_list)

    # Inject biomarker context into existing session if present
    if session_id in sessions and biomarkers and not diagnosis.get("error"):
        summary = diagnosis.get("reasoning_summary", "")
        sessions[session_id].messages.append({
            "role": "user",
            "content": f"[LAB REPORT UPLOADED] Biomarkers found: {json.dumps(biomarkers)}. Reasoning: {summary}",
        })

    return {
        "session_id": session_id,
        "biomarkers_found": len(biomarkers),
        "biomarkers": biomarkers,
        "biomarker_interpretation": diagnosis.get("biomarker_interpretation", []),
        "suggested_conditions": diagnosis.get("suggested_conditions", []),
        "trial_relevant_values": diagnosis.get("trial_relevant_values", []),
        "recommended_trial_types": diagnosis.get("recommended_trial_types", []),
        "reasoning_summary": diagnosis.get("reasoning_summary", ""),
        "physician_note": diagnosis.get("physician_note", "Always confirm findings with your physician."),
        "next_step": "Continue the chat to describe your symptoms and get matched to specific clinical trials.",
    }


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
