import os, re, argparse, time
from typing import List, Tuple, Iterator
import fitz  # PyMuPDF
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import yaml
from tqdm.auto import tqdm
import torch

# ---------- helpers ----------
def safe_doc_id_from_path(pdf_path: str) -> str:
    stem = os.path.splitext(os.path.basename(pdf_path))[0]
    return re.sub(r"[^A-Za-z0-9]+", "_", stem).strip("_").upper()

def extract_pages(pdf_path: str) -> List[Tuple[int, str]]:
    print(f"Opening PDF: {pdf_path}")
    doc = fitz.open(pdf_path)
    print(f"PDF opened successfully. Total pages: {len(doc)}")
    pages = []
    for i in tqdm(range(len(doc)), desc="Extracting pages", unit="page"):
        pages.append((i, doc[i].get_text()))
    return pages

def sliding_windows_chunked(text: str, window_chars=2200, overlap_chars=400, chunk_size=100):
    """Generator that yields windows in chunks to save memory"""
    i, n = 0, len(text)
    chunk = []
    step = window_chars - overlap_chars
    
    while i < n:
        j = min(i + window_chars, n)
        chunk.append((i, j, text[i:j]))
        
        if len(chunk) >= chunk_size:
            yield chunk
            chunk = []
        
        i += step  # FIX: Move forward by step, not backwards
        if i + overlap_chars >= n and j >= n:  # Stop if we've covered everything
            break
    
    if chunk:
        yield chunk

def get_optimal_device():
    """Determine best device for embedding model"""
    # Check for CUDA (NVIDIA GPU)
    if torch.cuda.is_available():
        return "cuda", "NVIDIA GPU (CUDA)"
    
    # Check for MPS (Apple Silicon) - but warn about Docker
    if torch.backends.mps.is_available():
        in_docker = os.path.exists('/.dockerenv')
        if in_docker:
            return "cpu", "CPU (Docker doesn't support MPS)"
        return "mps", "Apple Metal (MPS)"
    
    # Fallback to CPU
    return "cpu", "CPU"

def ingest_pdf(
    pdf_path: str,
    doc_id: str,
    db_path: str,
    collection: str,
    embed_model: str,
    window_chars: int,
    overlap_chars: int,
    batch_size: int = 32,
    num_workers: int = 4,
):
    # Chroma + embedder
    print(f"Initializing Chroma client at: {db_path}")
    client = chromadb.PersistentClient(path=db_path, settings=Settings(anonymized_telemetry=False))
    coll = client.get_or_create_collection(name=collection)
    print(f"Collection '{collection}' ready")
    
    print(f"Loading embedding model: {embed_model}")
    device, device_name = get_optimal_device()
    print(f"Using device: {device_name}")
    
    # Set number of threads for CPU inference
    if device == "cpu":
        torch.set_num_threads(num_workers)
        print(f"CPU threads: {num_workers}")
    
    embedder = SentenceTransformer(embed_model, device=device)
    print(f"Model loaded successfully!")

    # 1) Load & concat pages
    print("\n" + "="*60)
    print("STEP 1: Extracting PDF pages")
    print("="*60)
    pages = extract_pages(pdf_path)
    
    print("\nConcatenating page text...")
    text = "".join(p[1] for p in pages)
    text_len = len(text)
    print(f"Total text length: {text_len:,} characters")
    print(f"Estimated memory usage: ~{text_len / (1024*1024):.1f} MB")

    # 2) Estimate total windows
    print("\n" + "="*60)
    print("STEP 2: Calculating window count")
    print("="*60)
    print(f"Window size: {window_chars} chars, Overlap: {overlap_chars} chars")
    
    step = window_chars - overlap_chars
    estimated_windows = max(1, (text_len - overlap_chars + step - 1) // step)
    print(f"Estimated windows: ~{estimated_windows}")

    # 3) Process windows in streaming fashion
    print("\n" + "="*60)
    print("STEP 3: Processing windows (streaming mode)")
    print("="*60)
    
    window_id = 0
    total_processed = 0
    overall_start = time.time()
    
    with tqdm(total=estimated_windows, desc="Overall progress", unit="win") as pbar:
        for window_chunk in sliding_windows_chunked(text, window_chars, overlap_chars, chunk_size=50):
            # Prepare batch
            ids, docs, metas = [], [], []
            
            for i, j, chunk_text in window_chunk:
                ids.append(f"{doc_id}::{window_id}")
                docs.append(chunk_text)
                metas.append({
                    "doc_id": doc_id,
                    "window_id": window_id,
                    "char_start": i,
                    "char_end": j,
                    "page_start": 1,
                    "page_end": max(1, len(pages)),
                })
                window_id += 1
            
            # Encode this chunk
            embeddings = []
            chunk_start = time.time()
            for b in range(0, len(docs), batch_size):
                batch_texts = docs[b : b + batch_size]
                embs = embedder.encode(
                    batch_texts, 
                    convert_to_numpy=False, 
                    show_progress_bar=False,
                    batch_size=batch_size
                )
                
                try:
                    embs = [e.tolist() for e in embs]
                except AttributeError:
                    pass
                
                embeddings.extend(embs)
            
            chunk_time = time.time() - chunk_start
            rate = len(docs) / chunk_time if chunk_time > 0 else 0
            
            # Write this chunk to Chroma
            for b in range(0, len(ids), batch_size):
                coll.add(
                    ids=ids[b : b + batch_size],
                    documents=docs[b : b + batch_size],
                    metadatas=metas[b : b + batch_size],
                    embeddings=embeddings[b : b + batch_size],
                )
            
            total_processed += len(window_chunk)
            elapsed = time.time() - overall_start
            overall_rate = total_processed / elapsed if elapsed > 0 else 0
            eta = (estimated_windows - total_processed) / overall_rate if overall_rate > 0 else 0
            
            pbar.set_postfix({
                "rate": f"{overall_rate:.1f}/s",
                "eta": f"{eta/60:.1f}m"
            })
            pbar.update(len(window_chunk))
    
    # Clear the large text from memory
    del text
    
    total_time = time.time() - overall_start
    print("\n" + "="*60)
    print("✅ INGESTION COMPLETE")
    print("="*60)
    print(f"Documents: {total_processed} windows")
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Average rate: {total_processed/total_time:.1f} windows/sec")
    print(f"Collection: {collection}")
    print(f"Database: {db_path}")
    print(f"Doc ID: {doc_id}")

# ---------- main ----------
def main():
    # Load YAML defaults (optional)
    cfg_path = "./config/config.yaml"
    cfg = {}
    if os.path.exists(cfg_path):
        with open(cfg_path, "r") as f:
            cfg = yaml.safe_load(f) or {}
    ingest_cfg = (cfg.get("ingest") or {})
    default_pdf = ingest_cfg.get("pdf_path", "./guideline_documents/sa-phc-stg-2024-respiratory.pdf")
    default_embed = ingest_cfg.get("embed_model", "sentence-transformers/all-MiniLM-L6-v2")
    default_window = int(ingest_cfg.get("window_chars", 2200))
    default_overlap = int(ingest_cfg.get("overlap_chars", 400))

    ap = argparse.ArgumentParser(description="Ingest a guideline PDF into Chroma with progress bars")
    ap.add_argument("--pdf", default=default_pdf, help="Path to PDF")
    ap.add_argument("--doc-id", default=None, help="Optional; default derives from filename")
    ap.add_argument("--db", default=os.getenv("CHROMA_DB_DIR", "./chroma_db"), help="Chroma persistent dir")
    ap.add_argument("--collection", default="guidelines")
    ap.add_argument("--embed-model", default=default_embed)
    ap.add_argument("--window", type=int, default=default_window)
    ap.add_argument("--overlap", type=int, default=default_overlap)
    ap.add_argument("--batch-size", type=int, default=32, help="Embedding & write batch size")
    ap.add_argument("--num-workers", type=int, default=4, help="CPU threads for encoding")
    args = ap.parse_args()

    if not os.path.exists(args.pdf):
        raise FileNotFoundError(f"PDF not found: {args.pdf}")

    doc_id = args.doc_id or safe_doc_id_from_path(args.pdf)
    os.makedirs(args.db, exist_ok=True)

    print("="*60)
    print("PDF INGESTION SCRIPT")
    print("="*60)
    print(f"PDF: {args.pdf}")
    print(f"Doc ID: {doc_id}")
    print(f"Database: {args.db}")
    print(f"Collection: {args.collection}")
    print(f"Model: {args.embed_model}")
    print("="*60 + "\n")

    ingest_pdf(
        pdf_path=args.pdf,
        doc_id=doc_id,
        db_path=args.db,
        collection=args.collection,
        embed_model=args.embed_model,
        window_chars=args.window,
        overlap_chars=args.overlap,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

if __name__ == "__main__":
    main()