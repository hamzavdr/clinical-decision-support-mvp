#!/usr/bin/env python3
# utils/db-test2.py
import os
import argparse
import collections
import random
import re
from typing import Optional

import chromadb
from chromadb.config import Settings


def connect(db_path: str, collection: str):
    client = chromadb.PersistentClient(path=db_path, settings=Settings(anonymized_telemetry=False))
    coll = client.get_collection(collection)
    return coll


def print_overview(coll, limit: int = 5):
    print(f"Total items in collection: {coll.count()}")
    # NOTE: do NOT include "ids" here; Chroma returns ids by default
    res = coll.get(limit=limit, include=["metadatas", "documents"])
    ids = res.get("ids") or []
    metas = res.get("metadatas") or []
    docs = res.get("documents") or []
    print(f"\nFirst {len(ids)} items:")
    for i, (_id, m, d) in enumerate(zip(ids, metas, docs), 1):
        print(f"\n--- Item {i} ---")
        print("ID:", _id)
        print("Metadata:", m)
        prev = (d or "")[:180].replace("\n", " ")
        print("Text preview:", prev + ("..." if len(d or "") > 180 else ""))


def print_metadata_stats(coll):
    res = coll.get(include=["metadatas"])
    all_meta = res.get("metadatas") or []
    key_counts = collections.Counter()
    doc_counts = collections.Counter()

    for m in all_meta:
        if isinstance(m, dict):
            key_counts.update(m.keys())
            if "doc_id" in m:
                doc_counts[m["doc_id"]] += 1

    print("\nDoc IDs & window counts:")
    for k, v in doc_counts.items():
        print(f"- {k}: {v} windows")

    print("\nMetadata keys and their frequencies:")
    for k, c in key_counts.most_common():
        print(f"- {k}: {c}")

    # Section distribution if present
    sec_counts = collections.Counter()
    for m in all_meta:
        if isinstance(m, dict) and "section" in m:
            sec_counts[m.get("section")] += 1
    if sec_counts:
        print("\nValues for 'section':")
        for v, c in sec_counts.most_common():
            print(f"- {v}: {c}")
    else:
        print("\nNo 'section' key found in any metadata.")

    # Page range sanity
    bad = 0
    for m in all_meta:
        if not isinstance(m, dict):
            continue
        ps, pe = m.get("page_start"), m.get("page_end")
        if not isinstance(ps, int) or not isinstance(pe, int) or ps < 0 or pe < ps:
            bad += 1
    print(f"\nBad / suspicious page ranges: {bad} of {len(all_meta)}")


def sample_with_section(coll, limit: int = 5):
    res = coll.get(include=["metadatas", "documents"])
    ids = res.get("ids") or []
    metas = res.get("metadatas") or []
    docs = res.get("documents") or []
    shown = 0
    print("\nExamples that have 'section' (up to {}):".format(limit))
    for _id, m, d in zip(ids, metas, docs):
        if isinstance(m, dict) and m.get("section") is not None:
            print("\nID:", _id)
            print("section:", m.get("section"))
            print("page_start:", m.get("page_start"), "page_end:", m.get("page_end"))
            print("Preview:", (d or "")[:180].replace("\n", " ") + "...")
            shown += 1
            if shown >= limit:
                break
    if shown == 0:
        print("(None found.)")


# Simple heuristic to spot TOC / References / Index noise
NOISE_HEAD = re.compile(r"^(contents|table of contents|references|bibliography|index)\b", re.I)
DOTLEADER = re.compile(r"\.{3,}\s*\d{1,4}$", re.M)
YEAR = re.compile(r"\b(19|20)\d{2}\b")
BRACKETED = re.compile(r"\[\d{1,3}\]")

def looks_like_noise(text: str) -> bool:
    if not text:
        return False
    head = "\n".join(text.strip().splitlines()[:5])
    if NOISE_HEAD.search(head): return True
    if DOTLEADER.search(text):  return True
    yrs = len(YEAR.findall(text))
    brk = len(BRACKETED.findall(text))
    return yrs > 20 or brk > 30


def sample_noise_like(coll, limit: int = 5):
    res = coll.get(include=["metadatas", "documents"])
    ids = res.get("ids") or []
    docs = res.get("documents") or []
    metas = res.get("metadatas") or []

    order = list(range(len(ids)))
    random.shuffle(order)

    print("\nNoise-like samples (up to {}):".format(limit))
    hits = 0
    for i in order:
        d = docs[i]
        if d and looks_like_noise(d):
            print("\nNOISE-LIKE ID:", ids[i], "meta:", metas[i])
            print((d[:300] + "...").replace("\n", " "))
            hits += 1
            if hits >= limit:
                break
    if hits == 0:
        print("(No obvious noise-like chunks found in this sample.)")


def run_query(coll, q: str, n: int = 5, doc_id: Optional[str] = None):
    where = {"doc_id": doc_id} if doc_id else None
    print("\n--- Query ---")
    print("q:", q)
    print("n_results:", n)
    if where:
        print("where:", where)

    res = coll.query(
        query_texts=[q],
        n_results=n,
        where=where,
        include=["documents", "metadatas", "distances"],  # distances allowed in query()
    )

    ids = res.get("ids", [[]])[0]
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    dists = res.get("distances", [[]])[0]

    for i in range(len(ids)):
        print("\nHit", i + 1)
        print("id:", ids[i])
        print("distance:", dists[i], "(smaller is closer)")
        print("meta:", metas[i])
        print("preview:", (docs[i] or "")[:200].replace("\n", " ") + "...")


def main():
    ap = argparse.ArgumentParser(description="Inspect and query a Chroma collection.")
    ap.add_argument("--db", default="./chroma_db", help="Path to Chroma persistent dir")
    ap.add_argument("--collection", default="guidelines", help="Collection name")
    ap.add_argument("--limit", type=int, default=5, help="How many items to print in basic overview")
    ap.add_argument("--query", "-q", type=str, default=None, help="Optional: run a query with distances")
    ap.add_argument("--n-results", type=int, default=5, help="n_results for the query")
    ap.add_argument("--doc-id", type=str, default=None, help="Optional doc_id filter for query")
    ap.add_argument("--skip-noise-scan", action="store_true", help="Skip noise-like sampling")
    args = ap.parse_args()

    coll = connect(args.db, args.collection)

    print_overview(coll, limit=args.limit)
    print_metadata_stats(coll)
    sample_with_section(coll, limit=min(args.limit, 5))

    if not args.skip_noise_scan:
        sample_noise_like(coll, limit=min(args.limit, 5))

    if args.query:
        run_query(coll, args.query, n=args.n_results, doc_id=args.doc_id)


if __name__ == "__main__":
    main()
