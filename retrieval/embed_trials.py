"""
Embed processed trials into a FAISS vector index.
Run once after preprocess_trials.py — the index is loaded at API startup.

Usage:
    python retrieval/embed_trials.py
"""
import json
import pickle
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
import faiss
from tqdm import tqdm

# Swap to a biomedical model for better domain performance:
# "pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def build_faiss_index(
    trials_path: str = "data/processed_trials.json",
    index_path: str = "data/faiss_index",
    model_name: str = MODEL_NAME,
):
    print(f"Loading trials from {trials_path}...")
    with open(trials_path) as f:
        trials = json.load(f)

    print(f"Loading embedding model: {model_name}...")
    model = SentenceTransformer(model_name)

    texts = [t["embedding_text"] for t in trials]

    print(f"Embedding {len(texts)} trials...")
    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,  # Enables cosine similarity via inner product
    )

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)  # Inner Product = cosine when normalized
    index.add(embeddings.astype(np.float32))

    Path(index_path).mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, f"{index_path}/trials.index")

    with open(f"{index_path}/trials_metadata.pkl", "wb") as f:
        pickle.dump(trials, f)

    print(f"Index built: {index.ntotal} vectors (dim={dimension})")
    print(f"Saved to {index_path}/")
    return index, trials


if __name__ == "__main__":
    build_faiss_index()
