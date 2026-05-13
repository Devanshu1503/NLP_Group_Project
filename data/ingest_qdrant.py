"""
Ingest clinical trials from ClinicalTrials.gov into a Qdrant vector collection.

Usage:
    python data/ingest_qdrant.py           # incremental: skip NCT IDs already in Qdrant
    python data/ingest_qdrant.py --wipe    # drop & recreate collection, then ingest

Only RECRUITING and NOT_YET_RECRUITING trials are fetched (server-side filter).
Checkpoint + local JSONL cache allow resuming after a crash without re-fetching
from page zero.
"""
import argparse
import json
import os
import re
import time
import uuid
from pathlib import Path

import requests
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

load_dotenv()

# ── Configurable constants ────────────────────────────────────────────────
FETCH_COUNT       = 50_000  # Max valid (nct_id + eligibility) trials to pull
CHUNK_WORDS       = 512
OVERLAP_WORDS     = 50
BATCH_SIZE        = 100     # Points per Qdrant upsert call
PROGRESS_INTERVAL = 1_000   # Print a fetch-progress line every N valid trials

COLLECTION_NAME = "clinical_trials"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CTGOV_API       = "https://clinicaltrials.gov/api/v2/studies"
PAGE_SIZE       = 100       # ClinicalTrials.gov API max per page

# Checkpoint files live beside this script inside data/
_DATA_DIR       = Path(__file__).parent
CHECKPOINT_FILE = _DATA_DIR / "ingest_checkpoint.json"
CACHE_FILE      = _DATA_DIR / "ingest_cache.jsonl"

QDRANT_URL     = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")


# ── Text utilities ────────────────────────────────────────────────────────

def clean(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def chunk_text(text: str) -> list[str]:
    words = text.split()
    if len(words) <= CHUNK_WORDS:
        return [text] if text.strip() else []
    chunks, start = [], 0
    while start < len(words):
        end = min(start + CHUNK_WORDS, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start += CHUNK_WORDS - OVERLAP_WORDS
    return chunks


def build_embed_text(trial: dict, chunk: str) -> str:
    return (
        f"Trial: {trial['title']}\n"
        f"Conditions: {', '.join(trial['conditions'])}\n"
        f"Phase: {', '.join(trial['phase'])}\n"
        f"Status: {trial['status']}\n"
        f"{chunk}"
    )


# ── Checkpoint / JSONL cache ──────────────────────────────────────────────

def load_checkpoint() -> tuple[str | None, int]:
    """Return (next_page_token, total_valid_counted) from the checkpoint file."""
    if CHECKPOINT_FILE.exists():
        try:
            data = json.loads(CHECKPOINT_FILE.read_text(encoding="utf-8"))
            token = data.get("next_page_token")
            count = int(data.get("total_valid", 0))
            print(f"Checkpoint found: resuming after {count:,} valid trials already counted.")
            return token, count
        except Exception:
            pass
    return None, 0


def save_checkpoint(next_page_token: str | None, total_valid: int) -> None:
    CHECKPOINT_FILE.write_text(
        json.dumps({"next_page_token": next_page_token, "total_valid": total_valid}),
        encoding="utf-8",
    )


def load_cache() -> tuple[list[dict], set[str]]:
    """Load previously fetched+kept trials from the JSONL cache."""
    if not CACHE_FILE.exists():
        return [], set()
    trials: list[dict] = []
    nct_ids: set[str] = set()
    with CACHE_FILE.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                t = json.loads(line)
                trials.append(t)
                nct_ids.add(t["nct_id"])
    print(f"Loaded {len(trials):,} trials from local cache (resuming).")
    return trials, nct_ids


def append_to_cache(trial: dict) -> None:
    with CACHE_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(trial) + "\n")


def clear_checkpoint() -> None:
    for f in (CHECKPOINT_FILE, CACHE_FILE):
        if f.exists():
            f.unlink()


# ── Data fetching ─────────────────────────────────────────────────────────

def parse_trial(study: dict) -> dict | None:
    proto        = study.get("protocolSection", {})
    id_mod       = proto.get("identificationModule", {})
    desc_mod     = proto.get("descriptionModule", {})
    elig_mod     = proto.get("eligibilityModule", {})
    status_mod   = proto.get("statusModule", {})
    design_mod   = proto.get("designModule", {})
    cond_mod     = proto.get("conditionsModule", {})
    contacts_mod = proto.get("contactsLocationsModule", {})
    arms_mod     = proto.get("armsInterventionsModule", {})

    nct_id = id_mod.get("nctId", "")
    if not nct_id:
        return None

    locations = [
        f"{l.get('city', '')}, {l.get('state', '')}".strip(", ")
        for l in contacts_mod.get("locations", [])[:5]
        if l.get("city") or l.get("state")
    ]
    interventions = [i.get("name", "") for i in arms_mod.get("interventions", [])[:5]]

    # startDateStruct.date is the canonical start date field in CT.gov API v2.
    # It can be "YYYY-MM-DD", "YYYY-MM", or absent — parse_start_date handles all.
    start_date_str = status_mod.get("startDateStruct", {}).get("date", "")

    return {
        "nct_id":          nct_id,
        "title":           id_mod.get("briefTitle", ""),
        "summary":         clean(desc_mod.get("briefSummary", "")),
        "conditions":      cond_mod.get("conditions", []),
        "interventions":   interventions,
        "phase":           design_mod.get("phases", []),
        "status":          status_mod.get("overallStatus", ""),
        "min_age":         elig_mod.get("minimumAge", ""),
        "max_age":         elig_mod.get("maximumAge", ""),
        "gender":          elig_mod.get("sex", "ALL"),
        "locations":       locations,
        "eligibility_raw": clean(elig_mod.get("eligibilityCriteria", "")),
        "url":             f"https://clinicaltrials.gov/study/{nct_id}",
        "start_date":      start_date_str,
    }


def fetch_trials() -> list[dict]:
    """
    Page through ClinicalTrials.gov and return kept trials (RECRUITING + NOT_YET_RECRUITING).

    Counting:
      - total_valid: trials with nct_id + eligibility_raw (counts toward FETCH_COUNT)

    Supports resuming via CHECKPOINT_FILE + CACHE_FILE.
    """
    next_page_token, total_valid = load_checkpoint()
    all_trials, seen_in_run = load_cache()

    last_logged = (total_valid // PROGRESS_INTERVAL) * PROGRESS_INTERVAL

    params: dict = {
        "filter.overallStatus": "RECRUITING,NOT_YET_RECRUITING",
        "pageSize": PAGE_SIZE,
    }
    if next_page_token:
        params["pageToken"] = next_page_token

    print(f"Fetching up to {FETCH_COUNT:,} recruiting/upcoming trials from ClinicalTrials.gov...")

    while total_valid < FETCH_COUNT:
        resp = requests.get(CTGOV_API, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        studies = data.get("studies", [])
        if not studies:
            break

        for study in studies:
            if total_valid >= FETCH_COUNT:
                break

            trial = parse_trial(study)
            if not trial or not trial["eligibility_raw"]:
                continue
            if trial["nct_id"] in seen_in_run:
                continue

            total_valid += 1
            seen_in_run.add(trial["nct_id"])
            all_trials.append(trial)
            append_to_cache(trial)

        next_page_token = data.get("nextPageToken")
        save_checkpoint(next_page_token, total_valid)

        # Log progress at each PROGRESS_INTERVAL boundary
        milestone = (total_valid // PROGRESS_INTERVAL) * PROGRESS_INTERVAL
        if milestone > last_logged:
            print(f"  Fetched {total_valid:,} / {FETCH_COUNT:,} — kept {len(all_trials):,}")
            last_logged = milestone

        if not next_page_token:
            break

        params["pageToken"] = next_page_token
        time.sleep(0.5)

    print(f"\nFetch complete. Valid: {total_valid:,} | Kept: {len(all_trials):,}")
    return all_trials


# ── Qdrant helpers ────────────────────────────────────────────────────────

def get_existing_nct_ids(client: QdrantClient) -> set[str]:
    """
    Scroll the collection once and return all NCT IDs already stored.
    O(N_points / 1000) round trips — called once per run.
    """
    existing: set[str] = set()
    offset = None
    print("Scanning Qdrant for existing NCT IDs (deduplication)...")
    while True:
        result, next_offset = client.scroll(
            collection_name=COLLECTION_NAME,
            offset=offset,
            limit=1_000,
            with_payload=["nct_id"],
            with_vectors=False,
        )
        for point in result:
            nct_id = (point.payload or {}).get("nct_id")
            if nct_id:
                existing.add(nct_id)
        if next_offset is None:
            break
        offset = next_offset
    print(f"Found {len(existing):,} existing NCT IDs in Qdrant.")
    return existing


# ── Ingestion ─────────────────────────────────────────────────────────────

def ingest(wipe: bool = False) -> None:
    if wipe:
        # Discard any leftover cache/checkpoint from a previous partial run
        clear_checkpoint()

    trials = fetch_trials()
    print(f"\nLoading embedding model...")

    model  = SentenceTransformer(EMBEDDING_MODEL)
    dim    = model.get_embedding_dimension()
    if dim is None:
        raise RuntimeError(f"Could not determine embedding dimension for {EMBEDDING_MODEL}")
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

    if wipe and client.collection_exists(COLLECTION_NAME):
        print(f"Dropping existing collection '{COLLECTION_NAME}'...")
        client.delete_collection(COLLECTION_NAME)

    if not client.collection_exists(COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )
        print(f"Created collection '{COLLECTION_NAME}' (dim={dim})")

    # Load existing NCT IDs once; O(1) set lookup per trial during embedding
    existing_nct_ids = get_existing_nct_ids(client)

    points: list[PointStruct] = []
    total_chunks   = 0
    skipped_dupes  = 0
    ingested_count = 0

    for trial in tqdm(trials, desc="Chunking & embedding"):
        if trial["nct_id"] in existing_nct_ids:
            skipped_dupes += 1
            continue

        # Mark as seen so the same NCT ID can't appear twice within this run
        existing_nct_ids.add(trial["nct_id"])

        long_text = " ".join(filter(None, [trial["summary"], trial["eligibility_raw"]]))
        chunks = chunk_text(long_text) or [trial["title"]]
        ingested_count += 1

        for i, chunk in enumerate(chunks):
            vector = model.encode(
                build_embed_text(trial, chunk), normalize_embeddings=True
            ).tolist()

            points.append(PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload={
                    "nct_id":          trial["nct_id"],
                    "title":           trial["title"],
                    "conditions":      trial["conditions"],
                    "phase":           trial["phase"],
                    "status":          trial["status"],
                    "min_age":         trial["min_age"],
                    "max_age":         trial["max_age"],
                    "gender":          trial["gender"],
                    "locations":       trial["locations"],
                    "eligibility_raw": trial["eligibility_raw"][:2000],
                    "summary":         trial["summary"][:1000],
                    "interventions":   trial["interventions"],
                    "url":             trial["url"],
                    "start_date":      trial["start_date"],
                    "chunk_index":     i,
                    "chunk_text":      chunk[:500],
                },
            ))
            total_chunks += 1

            if len(points) >= BATCH_SIZE:
                client.upsert(collection_name=COLLECTION_NAME, points=points)
                points = []

    if points:
        client.upsert(collection_name=COLLECTION_NAME, points=points)

    if skipped_dupes:
        print(f"Skipped {skipped_dupes:,} trials whose NCT ID was already in Qdrant.")

    clear_checkpoint()
    print("Checkpoint and cache cleared.")

    info = client.get_collection(COLLECTION_NAME)
    print(
        f"\nIngestion complete! {info.points_count:,} vectors "
        f"from {ingested_count:,} new trials in '{COLLECTION_NAME}'"
    )
    if ingested_count:
        print(f"Average {total_chunks / ingested_count:.1f} chunks per trial")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest clinical trials into Qdrant.")
    parser.add_argument(
        "--wipe",
        action="store_true",
        help="Drop and recreate the collection before ingesting (full rebuild).",
    )
    args = parser.parse_args()
    ingest(wipe=args.wipe)
