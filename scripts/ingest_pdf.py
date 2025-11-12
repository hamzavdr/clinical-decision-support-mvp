import os, argparse, fitz, math, tqdm
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

def extract_pages(pdf_path: str):
    doc = fitz.open(pdf_path)
    pages = []
    for i in range(len(doc)):
        text = doc[i].get_text()
        pages.append((i, text))
    return pages

def sliding_windows(text: str, window_chars=2200, overlap_chars=400):
    i, n = 0, len(text)
    while i < n:
        j = min(i + window_chars, n)
        yield (i, j, text[i:j])
        i = j - overlap_chars

def ingest_pdf(pdf_path: str, doc_id: str, db_path: str, collection: str,
               embed_model="sentence-transformers/all-MiniLM-L6-v2",
               window_chars=2200, overlap_chars=400):
    client = chromadb.PersistentClient(path=db_path, settings=Settings(anonymized_telemetry=False))
    coll = client.get_or_create_collection(name=collection)
    embedder = SentenceTransformer(embed_model, device="cpu")

    pages = extract_pages(pdf_path)
    text = "".join(p[1] for p in pages)

    ids, docs, metas, embs = [], [], [], []
    for w_id, (i, j, chunk) in enumerate(sliding_windows(text, window_chars, overlap_chars)):
        ids.append(f"{doc_id}::{w_id}")
        docs.append(chunk)
        metas.append({
            "doc_id": doc_id,
            "window_id": w_id,
            "char_start": i,
            "char_end": j,
            # rough page range estimate
            "page_start": 1,
            "page_end": max(1, len(pages)),
        })
        embs.append(embedder.encode(chunk).tolist())

    coll.add(ids=ids, documents=docs, metadatas=metas, embeddings=embs)
    print(f"Ingested {len(ids)} windows into {db_path}/{collection}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", required=True)
    ap.add_argument("--doc-id", required=True)
    ap.add_argument("--db", default="./chroma_db")
    ap.add_argument("--collection", default="guidelines")
    ap.add_argument("--window", type=int, default=2200)
    ap.add_argument("--overlap", type=int, default=400)
    args = ap.parse_args()
    os.makedirs(args.db, exist_ok=True)
    ingest_pdf(args.pdf, args.doc_id, args.db, args.collection, window_chars=args.window, overlap_chars=args.overlap)
