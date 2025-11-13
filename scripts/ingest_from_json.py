#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Ingest a page-level JSON/JSONL document into ChromaDB using sliding-window chunks.

Features
- Accepts JSON (list of pages) or JSONL (one page per line)
- Filters pages by page_type (e.g., keep only "chapter_body")
- Concatenates selected pages, creates overlapping windows
- Computes page_start/page_end for each window using character offsets
- Batches embedding with SentenceTransformers (CPU-safe)
- Stores doc_id::{window_id} with rich metadata in Chroma

Usage
  # YAML-driven (recommended)
  python scripts/ingest_from_json.py --config config/ingest_json.yaml

  # Or explicit CLI
  python scripts/ingest_from_json.py \
    --json ./guideline_documents/sa-phc-stg-2024.json \
    --doc-id SA_PHC_STG_2024_Full \
    --db ./chroma_db --collection guidelines \
    --include-page-types chapter_body \
    --window 2200 --overlap 400 \
    --embed-model sentence-transformers/all-MiniLM-L6-v2 \
    --batch-size 32 --device cpu
"""

import os
import re
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple, Iterable, Optional

import yaml
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


# --------------------------- Helpers ---------------------------

def load_pages(json_path: str) -> List[Dict[str, Any]]:
    """
    Load pages from a JSON (array) or JSONL (one object per line) file.
    Returns a list of dicts, each representing one page with at least:
      - "page_number" (int)
      - "page_body" (str)
      - optional: "page_type", "chapter_number", "chapter_page_index", "page_header", "page_footer"
    """
    p = Path(json_path)
    if not p.exists():
        raise FileNotFoundError(f"json_path not found: {json_path}")

    pages: List[Dict[str, Any]] = []
    if p.suffix.lower() == ".jsonl":
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                pages.append(json.loads(line))
    else:
        with p.open("r", encoding="utf-8") as f:
            obj = json.load(f)
            if isinstance(obj, list):
                pages = obj
            elif isinstance(obj, dict) and "pages" in obj and isinstance(obj["pages"], list):
                pages = obj["pages"]
            else:
                raise ValueError("JSON must be a list of page objects or have a top-level 'pages' list.")

    # Basic validation
    for i, pg in enumerate(pages):
        if "page_body" not in pg:
            raise ValueError(f"Page index {i} missing 'page_body'.")
        if "page_number" not in pg:
            # If missing, derive a 1-based index
            pg["page_number"] = i + 1
        if "page_type" not in pg:
            # default page_type if absent
            pg["page_type"] = "unknown"

    return pages


def filter_pages(pages: List[Dict[str, Any]], include_types: Tuple[str, ...]) -> List[Dict[str, Any]]:
    """Keep only pages whose page_type is in include_types."""
    include_set = set(t.strip().lower() for t in include_types)
    out = [pg for pg in pages if str(pg.get("page_type", "")).strip().lower() in include_set]
    return out


def normalize_whitespace(text: str) -> str:
    """Collapse excess whitespace for cleaner windows."""
    return re.sub(r"[ \t\r\f\v]+", " ", text).replace("\u00A0", " ").strip()


def build_concatenated_corpus(pages: List[Dict[str, Any]]) -> Tuple[str, List[Tuple[int, int, int]]]:
    """
    Concatenate page bodies into one big string with single newlines between pages.
    Returns:
      corpus_text: str
      index: List of tuples (page_number, start_char, end_char) in corpus_text
    """
    parts: List[str] = []
    index: List[Tuple[int, int, int]] = []
    pos = 0
    for pg in pages:
        body = normalize_whitespace(pg.get("page_body", ""))
        start = pos
        parts.append(body)
        pos += len(body)
        index.append((int(pg.get("page_number", 0)), start, pos))
        # separator between pages
        parts.append("\n")
        pos += 1
    corpus_text = "".join(parts)
    return corpus_text, index


def map_char_range_to_page_span(index: List[Tuple[int, int, int]], lo: int, hi: int) -> Tuple[int, int]:
    """
    Given a char span [lo, hi) in the concatenated corpus, find the first and last
    page_number touched by this range using the page index.
    """
    page_start = None
    page_end = None
    for (pg_no, pg_lo, pg_hi) in index:
        if pg_hi <= lo:
            continue
        if pg_lo >= hi:
            break
        # overlap exists
        if page_start is None:
            page_start = pg_no
        page_end = pg_no
    # Fallback if somehow not found
    if page_start is None:
        page_start = index[0][0]
    if page_end is None:
        page_end = index[-1][0]
    return page_start, page_end


def sliding_windows(text: str, window_chars: int, overlap_chars: int) -> Iterable[Tuple[int, int, str]]:
    """Yield (start, end, chunk) over text with fixed-size windows and overlap."""
    i, n = 0, len(text)
    while i < n:
        j = min(i + window_chars, n)
        yield i, j, text[i:j]
        if j == n:
            break
        i = j - overlap_chars


def default_doc_id_from_filename(path: str) -> str:
    """
    Derive a stable doc_id from a filename:
      - take stem (no extension)
      - non-alphanumeric -> underscore
      - upper case
    """
    stem = Path(path).stem
    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", stem).strip("_")
    return cleaned.upper()


def batched(iterable: List[Any], batch_size: int) -> Iterable[List[Any]]:
    """Yield successive batches from a list."""
    for i in range(0, len(iterable), batch_size):
        yield iterable[i:i + batch_size]


# --------------------------- Core ingest ---------------------------

def ingest_json_pages(
    json_path: str,
    db_path: str,
    collection: str,
    doc_id: Optional[str],
    include_page_types: Tuple[str, ...],
    window_chars: int,
    overlap_chars: int,
    embed_model_name: str,
    batch_size: int = 32,
    device: str = "cpu",
) -> None:
    """
    Ingest selected pages from a page-level JSON/JSONL into Chroma with sliding windows.
    """
    # Derive doc_id if not provided
    if not doc_id:
        doc_id = default_doc_id_from_filename(json_path)

    # Load + filter pages
    print("Loading pages...")
    pages_all = load_pages(json_path)
    pages = filter_pages(pages_all, include_page_types)
    if not pages:
        raise SystemExit(f"No pages found after filtering types {include_page_types}. "
                         f"Available page_types include (examples): "
                         f"{sorted(set(p.get('page_type', 'unknown') for p in pages_all))[:10]}")

    print(f"Loaded {len(pages_all)} pages total; {len(pages)} kept after filtering.")

    # Build big corpus + index for page span mapping
    print("Building concatenated corpus & index...")
    corpus, page_index = build_concatenated_corpus(pages)

    # Create windows (lightweight; we'll embed in batches)
    print("Creating sliding windows...")
    windows: List[Tuple[int, int, str]] = list(sliding_windows(corpus, window_chars, overlap_chars))
    print(f"Total windows: {len(windows)}")

    # Prepare Chroma and embedder
    os.makedirs(db_path, exist_ok=True)
    client = chromadb.PersistentClient(path=db_path, settings=Settings(anonymized_telemetry=False))
    coll = client.get_or_create_collection(name=collection)
    embedder = SentenceTransformer(embed_model_name, device=device)

    # Build metadata + embed in batches
    ids: List[str] = []
    docs: List[str] = []
    metas: List[Dict[str, Any]] = []

    print("Generating metadata...")
    for w_id, (lo, hi, chunk) in enumerate(windows):
        pg_start, pg_end = map_char_range_to_page_span(page_index, lo, hi)
        ids.append(f"{doc_id}::{w_id}")
        docs.append(chunk)
        metas.append({
            "doc_id": doc_id,
            "window_id": w_id,
            "char_start": lo,
            "char_end": hi,
            "page_start": int(pg_start),
            "page_end": int(pg_end),
        })

    # Embed & insert in batches (progress bar)
    print("Embedding and writing to Chroma...")
    for idx_batch in tqdm(list(batched(list(range(len(docs))), batch_size)), desc="Batches"):
        batch_docs = [docs[i] for i in idx_batch]
        batch_ids = [ids[i] for i in idx_batch]
        batch_meta = [metas[i] for i in idx_batch]
        batch_embs = embedder.encode(batch_docs, convert_to_numpy=True).tolist()

        coll.add(
            ids=batch_ids,
            documents=batch_docs,
            metadatas=batch_meta,
            embeddings=batch_embs,
        )

    print(f"Ingest complete: {len(ids)} windows into {db_path}/{collection} (doc_id={doc_id})")


# --------------------------- CLI ---------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", help="Path to YAML config (section: ingest_json)", default=None)

    ap.add_argument("--json", help="Path to JSON/JSONL with page entries", default=None)
    ap.add_argument("--db", default="./chroma_db")
    ap.add_argument("--collection", default="guidelines")
    ap.add_argument("--doc-id", default=None)
    ap.add_argument("--include-page-types", default="chapter_body")  # comma-separated if multiple
    ap.add_argument("--window", type=int, default=2200)
    ap.add_argument("--overlap", type=int, default=400)
    ap.add_argument("--embed-model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    # Load YAML config if provided
    cfg = {}
    if args.config:
        with open(args.config, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        cfg = raw.get("ingest_json") or {}

    # Resolve params (CLI overrides YAML)
    json_path   = args.json or cfg.get("json_path")
    db_path     = cfg.get("db_path", args.db)
    collection  = cfg.get("collection", args.collection)
    doc_id      = args.doc_id if args.doc_id is not None else cfg.get("doc_id")

    window      = cfg.get("window_chars", args.window)
    overlap     = cfg.get("overlap_chars", args.overlap)
    model_name  = cfg.get("embed_model", args.embed_model)
    batch_size  = cfg.get("batch_size", args.batch_size)
    device      = cfg.get("device", args.device)

    include_types = cfg.get("include_page_types", args.include_page_types)
    if isinstance(include_types, str):
        include_types = tuple(s.strip() for s in include_types.split(",") if s.strip())
    else:
        include_types = tuple(include_types or ["chapter_body"])

    if not json_path:
        raise SystemExit("Missing json_path. Set --json or provide in --config under ingest_json.json_path")

    print(f"Config:\n"
          f"  json_path: {json_path}\n"
          f"  db_path: {db_path}\n"
          f"  collection: {collection}\n"
          f"  doc_id: {doc_id or '(auto from filename)'}\n"
          f"  include_page_types: {include_types}\n"
          f"  window_chars: {window} | overlap_chars: {overlap}\n"
          f"  embed_model: {model_name} | device: {device} | batch_size: {batch_size}\n")

    ingest_json_pages(
        json_path=json_path,
        db_path=db_path,
        collection=collection,
        doc_id=doc_id,
        include_page_types=include_types,
        window_chars=window,
        overlap_chars=overlap,
        embed_model_name=model_name,
        batch_size=batch_size,
        device=device,
    )


if __name__ == "__main__":
    main()
