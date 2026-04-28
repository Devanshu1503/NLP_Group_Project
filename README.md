# TrialNav — Patient Clinical Trial Matching

Conversational NLP + RAG system that matches patients to recruiting clinical trials from ClinicalTrials.gov.  
Built for ADSP 32018 (NLP and Agentic AI) at the University of Chicago.

---

## Project Structure

```
trialnav/
├── data/               # Data pipeline (fetch + preprocess trials)
├── ner/                # Problem 1: Patient profile extraction (LLM vs BioBERT)
├── retrieval/          # Problem 2: Trial retrieval (Semantic RAG vs Keyword)
├── agents/             # Conversation, extraction, retrieval, and explanation agents
├── evaluation/         # Evaluation scripts and labeled test sets
├── api/                # FastAPI REST endpoint
└── notebooks/          # EDA notebook
```

---

## Setup

```bash
git clone <your-repo>
cd trialnav
pip install -r requirements.txt
cp .env.example .env
# Add your ANTHROPIC_API_KEY to .env
```

---

## Build the Data Pipeline

Run these once in order before starting the API or evaluations:

```bash
# 1. Fetch trials from ClinicalTrials.gov (free, no API key needed)
python data/fetch_trials.py

# 2. Clean and structure the raw data
python data/preprocess_trials.py

# 3. Embed trials into FAISS index (~5 min for 500 trials)
python retrieval/embed_trials.py
```

---

## Run the API

```bash
uvicorn api.main:app --reload
# http://localhost:8000
# Docs: http://localhost:8000/docs
```

Test it:

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "I have Type 2 diabetes and live in Chicago"}'
```

---

## Run Evaluations

```bash
# Problem 1: NER — LLM vs BioBERT
python evaluation/eval_ner.py

# Problem 2: Retrieval — Semantic RAG vs Keyword search
python evaluation/eval_retrieval.py
```

---

## Two NLP Problems

| Problem | Champion | Challenger | Metric |
|---|---|---|---|
| 1. Patient Profile Extraction (NER) | LLM (Claude) — `ner/ner_llm.py` | BioBERT — `ner/ner_bioBERT.py` | F1, Accuracy |
| 2. Trial Retrieval | Semantic RAG (FAISS) — `retrieval/retriever_semantic.py` | Keyword API — `retrieval/retriever_keyword.py` | Precision@5, MRR |

---

## Team

- Member 1 — [owns: NER / Problem 1]
- Member 2 — [owns: Retrieval + Reranker / Problem 2]
- Member 3 — [owns: API + Agents + EDA Notebook]
