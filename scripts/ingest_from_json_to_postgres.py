#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ingest selected guideline pages into a Postgres + pgvector table.

This mirrors `ingest_from_json.py` but writes each chunk directly into
`guideline_embeddings` (or a user-specified table) via psycopg.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple, Iterable, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import yaml
import psycopg
from psycopg import sql
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from scripts.ingest_from_json import (
    load_pages,
    filter_pages,
    build_concatenated_corpus,
    sliding_windows,
    map_char_range_to_page_span,
    default_doc_id_from_filename,
    batched,
)


def _vector_literal(vec: Iterable[float]) -> str:
    """Format a python list of floats into pgvector's textual representation."""
    parts = []
    for v in vec:
        s = f"{v:.8f}".rstrip("0").rstrip(".")
        parts.append(s or "0")
    return "[" + ", ".join(parts) + "]"


def _resolve_conninfo(args, cfg: Dict[str, Any]) -> str:
    conn = args.pg_conn or cfg.get("pg_conn") or os.getenv("PG_CONNINFO") or os.getenv("POSTGRES_CONN")
    if conn:
        return conn

    host = args.pg_host or cfg.get("pg_host") or os.getenv("PGHOST")
    dbname = args.pg_db or cfg.get("pg_db") or os.getenv("PGDATABASE")
    user = args.pg_user or cfg.get("pg_user") or os.getenv("PGUSER")
    password = args.pg_password or cfg.get("pg_password") or os.getenv("PGPASSWORD")
    port = args.pg_port or cfg.get("pg_port") or os.getenv("PGPORT", 5432)

    if not all([host, dbname, user, password]):
        raise SystemExit("Missing Postgres connection info. Provide --pg-conn or host/db/user/password inputs.")

    return f"host={host} port={port} dbname={dbname} user={user} password={password}"


def ingest_json_to_postgres(
    json_path: str,
    conninfo: str,
    table: str,
    doc_id: Optional[str],
    include_page_types: Tuple[str, ...],
    window_chars: int,
    overlap_chars: int,
    embed_model_name: str,
    batch_size: int,
    device: str,
    delete_existing: bool,
) -> None:
    if not doc_id:
        doc_id = default_doc_id_from_filename(json_path)

    print("Loading pages...")
    pages_all = load_pages(json_path)
    pages = filter_pages(pages_all, include_page_types)
    if not pages:
        raise SystemExit("No pages remained after filtering. Check include_page_types.")
    print(f"Loaded {len(pages_all)} pages total; {len(pages)} kept after filtering.")

    print("Building corpus and sliding windows...")
    corpus, page_index = build_concatenated_corpus(pages)
    windows = list(sliding_windows(corpus, window_chars, overlap_chars))
    print(f"Total windows: {len(windows)}")

    embedder = SentenceTransformer(embed_model_name, device=device)

    entries: List[Dict[str, Any]] = []
    for w_id, (lo, hi, chunk) in enumerate(windows):
        pg_start, pg_end = map_char_range_to_page_span(page_index, lo, hi)
        entries.append(
            {
                "doc_id": doc_id,
                "window_id": w_id,
                "page_start": pg_start,
                "page_end": pg_end,
                "text_chunk": chunk,
            }
        )

    with psycopg.connect(conninfo) as conn:
        conn.autocommit = False
        with conn.cursor() as cur:
            if delete_existing:
                print(f"Deleting existing rows for doc_id={doc_id} from {table}...")
                cur.execute(
                    sql.SQL("DELETE FROM {} WHERE doc_id = %s").format(sql.Identifier(table)),
                    (doc_id,),
                )
                conn.commit()

            insert_sql = sql.SQL(
                """
                INSERT INTO {} (doc_id, window_id, page_start, page_end, text_chunk, embedding)
                VALUES (%s, %s, %s, %s, %s, %s::vector)
                """
            ).format(sql.Identifier(table))

            print("Embedding and inserting batches...")
            for batch_ids in tqdm(list(batched(list(range(len(entries))), batch_size)), desc="Batches"):
                batch_entries = [entries[i] for i in batch_ids]
                batch_docs = [e["text_chunk"] for e in batch_entries]
                batch_embs = embedder.encode(batch_docs, convert_to_numpy=True).tolist()
                rows = []
                for e, emb in zip(batch_entries, batch_embs):
                    rows.append(
                        (
                            e["doc_id"],
                            e["window_id"],
                            e["page_start"],
                            e["page_end"],
                            e["text_chunk"],
                            _vector_literal(emb),
                        )
                    )
                cur.executemany(insert_sql, rows)
            conn.commit()

    print(f"Ingest complete: {len(entries)} rows inserted into {table} (doc_id={doc_id}).")


def parse_args():
    ap = argparse.ArgumentParser(description="Ingest guideline JSON pages into Postgres + pgvector.")
    ap.add_argument("--config", help="Path to YAML config (section: ingest_postgres)", default=None)
    ap.add_argument("--json", help="Path to JSON/JSONL pages file", default=None)
    ap.add_argument("--doc-id", default=None, help="Doc ID label (defaults to filename-based)")
    ap.add_argument("--include-page-types", default="chapter_body", help="Comma-separated page_type filter.")
    ap.add_argument("--window", type=int, default=2200)
    ap.add_argument("--overlap", type=int, default=400)
    ap.add_argument("--embed-model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--table", default="guideline_embeddings")
    ap.add_argument("--delete-existing", action="store_true", help="Delete rows for doc_id before inserting.")

    # Postgres connection options
    ap.add_argument("--pg-conn", help="Full psycopg conninfo or URL.", default=None)
    ap.add_argument("--pg-host", default=None)
    ap.add_argument("--pg-port", type=int, default=5432)
    ap.add_argument("--pg-db", default=None)
    ap.add_argument("--pg-user", default=None)
    ap.add_argument("--pg-password", default=None)
    return ap.parse_args()


def main():
    args = parse_args()
    cfg: Dict[str, Any] = {}
    if args.config:
        with open(args.config, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
            cfg = raw.get("ingest_postgres") or {}

    json_path = args.json or cfg.get("json_path")
    if not json_path:
        raise SystemExit("Missing --json path (or ingest_postgres.json_path in config).")

    include_types = cfg.get("include_page_types", args.include_page_types)
    if isinstance(include_types, str):
        include_types = tuple(s.strip() for s in include_types.split(",") if s.strip())
    else:
        include_types = tuple(include_types or ["chapter_body"])

    table = cfg.get("table", args.table)
    doc_id = args.doc_id or cfg.get("doc_id")
    window_chars = cfg.get("window_chars", args.window)
    overlap_chars = cfg.get("overlap_chars", args.overlap)
    embed_model = cfg.get("embed_model", args.embed_model)
    batch_size = cfg.get("batch_size", args.batch_size)
    device = cfg.get("device", args.device)
    delete_existing = args.delete_existing or cfg.get("delete_existing", False)

    conninfo = _resolve_conninfo(args, cfg)

    print("Config:")
    print(f"  json_path: {json_path}")
    print(f"  table: {table}")
    print(f"  doc_id: {doc_id or '(auto)'}")
    print(f"  include_page_types: {include_types}")
    print(f"  window_chars: {window_chars} | overlap_chars: {overlap_chars}")
    print(f"  embed_model: {embed_model} | device: {device} | batch_size: {batch_size}")
    print(f"  delete_existing: {delete_existing}")

    ingest_json_to_postgres(
        json_path=json_path,
        conninfo=conninfo,
        table=table,
        doc_id=doc_id,
        include_page_types=include_types,
        window_chars=window_chars,
        overlap_chars=overlap_chars,
        embed_model_name=embed_model,
        batch_size=batch_size,
        device=device,
        delete_existing=delete_existing,
    )


if __name__ == "__main__":
    main()
