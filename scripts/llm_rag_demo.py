# scripts/llm_rag_demo.py
# RAG → LLM demo with external, versioned prompts: planner → retrieve(+1/+2) → answer

import os, json, argparse, requests, re
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# ------------------- Constants / defaults -------------------
DOC_ID = "SA_PHC_STG_2024_Respiratory"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
OPENAI_MODEL = "gpt-4o-mini"
OPENAI_URL = "https://api.openai.com/v1/chat/completions"

# Project root (…/project), assuming this file is at project/scripts/...
ROOT = Path(__file__).resolve().parents[1]
load_dotenv(dotenv_path=ROOT / ".env")

# ---- Fallback prompts (used if files are missing) ----
PLANNER_SYSTEM_FALLBACK = """You are a query planner for a medical RAG system.
Given a visit note, output 1-3 short search queries targeting a Respiratory chapter in South African paediatric guidelines.
Prefer concrete clinical terms (condition, drug, route, dose, severity, disposition).
Return ONLY valid JSON of the form: {"queries": ["q1","q2", ...]}"""

ANSWER_SYSTEM_FALLBACK = """You are a clinical decision-support assistant.
You MUST use only the provided Context from the guideline to answer.
Cite the context as: [Respiratory chapter, p. {page_start}-{page_end}].
If key info is missing, say so explicitly.
This is a demo and not medical advice."""

ANSWER_USER_TMPL_FALLBACK = """Visit note:
{note}

Context (from guideline):
{context}

Write a management plan tailored to the note. Include:
- Assessment (severity reasoning)
- Treatment (include steroid choice; do NOT invent doses yet)
- Monitoring / reassessment
- Criteria for escalation / disposition
- Explicit citations for key claims using the page range provided.
"""

# ------------------- Versioned prompt loader -------------------
def _read_text(path: Path) -> Optional[str]:
    try:
        if path.is_file():
            return path.read_text(encoding="utf-8").strip()
    except Exception:
        pass
    return None

def _natural_key(s: str):
    # sort v1, v1.1, v2-2025-01-15 naturally
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]

def _resolve_version_dir(base: Path, requested: Optional[str]) -> Path:
    if requested:
        cand = base / requested
        if cand.is_dir():
            return cand
        raise FileNotFoundError(f"Prompt version '{requested}' not found under {base}")
    # Prefer 'current' if present
    cur = base / "current"
    if cur.exists() and cur.is_dir():
        return cur
    # Else pick the latest by natural sort
    versions = [p for p in base.iterdir() if p.is_dir() and not p.name.startswith(".")]
    if not versions:
        raise FileNotFoundError(f"No prompt versions found under {base}")
    versions.sort(key=lambda p: _natural_key(p.name))
    return versions[-1]

def load_prompts_versioned(base_dir: Path, version: Optional[str]) -> Dict[str, str]:
    vdir = _resolve_version_dir(base_dir, version)
    planner = _read_text(vdir / "planner_system.txt") or PLANNER_SYSTEM_FALLBACK
    answer_sys = _read_text(vdir / "answer_system.txt") or ANSWER_SYSTEM_FALLBACK
    answer_user = _read_text(vdir / "answer_user_tmpl.txt") or ANSWER_USER_TMPL_FALLBACK
    return {
        "PLANNER_SYSTEM": planner,
        "ANSWER_SYSTEM": answer_sys,
        "ANSWER_USER_TMPL": answer_user,
    }

# ------------------ Chroma RAG client ------------------
class RAGClient:
    def __init__(self, db_path="./chroma_db", collection="guidelines", embed_model=EMBED_MODEL):
        self.client = chromadb.PersistentClient(path=db_path, settings=Settings())
        self.coll = self.client.get_or_create_collection(name=collection)
        self.embedder = SentenceTransformer(embed_model)

    def list_doc_ids(self) -> list[str]:
        """Return sorted unique doc_id values from collection metadata."""
        res = self.coll.get(include=["metadatas"])  # fine for MVP sizes (689 items)
        metas = res.get("metadatas") or []
        ids = {m.get("doc_id") for m in metas if m and m.get("doc_id")}
        return sorted(ids)

    def _get(self, id_str):
        res = self.coll.get(ids=[id_str])
        if not res["ids"]:
            return None, None
        return res["documents"][0], res["metadatas"][0]

    def fetch(self, doc_id: str, window_id: int):
        return self._get(f"{doc_id}::{window_id}")

    def query(self, q: str, doc_id: str, k=4):
        q_emb = self.embedder.encode([q]).tolist()
        res = self.coll.query(
            query_embeddings=q_emb,
            n_results=k,
            where={"doc_id": doc_id},
            include=["documents", "metadatas", "distances"],
        )
        hits = []
        for i in range(len(res["ids"][0])):
            hits.append({
                "id": res["ids"][0][i],
                "document": res["documents"][0][i],
                "metadata": res["metadatas"][0][i],
                "distance": res["distances"][0][i],
            })
        return hits

    def stitch_neighbors(self, hit: Dict[str, Any], above=1, below=2, cap_words=1600):
        """Include 1 chunk above and 2 chunks below (same doc_id)."""
        m = hit["metadata"]
        center = int(m["window_id"])
        parts, total = [], 0
        for w in range(center - above, center + below + 1):
            doc, meta = self.fetch(m["doc_id"], w)
            if doc is None:
                continue
            words = doc.split()
            parts.append((doc, meta))
            total += len(words)
            if total >= cap_words:
                break
        if not parts:
            return None
        text = "\n".join(p[0] for p in parts)
        page_start = parts[0][1].get("page_start", m["page_start"])
        page_end = parts[-1][1].get("page_end", m["page_end"])
        return {"text": text, "pages": (int(page_start), int(page_end)), "windows": [p[1]["window_id"] for p in parts]}

# ------------------ LLM helpers ------------------
def openai_chat(messages: List[Dict[str, str]], temperature=0.2):
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Set OPENAI_API_KEY")
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    body = {"model": OPENAI_MODEL, "messages": messages, "temperature": temperature}
    r = requests.post(OPENAI_URL, headers=headers, json=body, timeout=60)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]

# ------------------ Pipeline (reads prompts dict) ------------------
def plan_queries(note: str, prompts: Dict[str, str]) -> List[str]:
    msg = [
        {"role": "system", "content": prompts["PLANNER_SYSTEM"]},
        {"role": "user", "content": note.strip()[:4000]},
    ]
    out = openai_chat(msg, temperature=0.0)
    try:
        data = json.loads(out)
        queries = [q.strip() for q in data.get("queries", []) if q.strip()]
        return queries[:3] or ["croup steroid dose management"]
    except Exception:
        return ["croup steroid dose management"]

def retrieve_context(rag: RAGClient, queries: List[str], doc_id: str, k=4, above=1, below=2) -> Dict[str, Any]:
    windows = []
    seen = set()
    for q in queries:
        hits = rag.query(q, doc_id=doc_id, k=k)
        for h in hits:
            m = h["metadata"]; key = (m["doc_id"], m["window_id"])
            if key in seen: 
                continue
            seen.add(key)
            stitched = rag.stitch_neighbors(h, above=above, below=below)
            if stitched:
                windows.append({"text": stitched["text"], "pages": stitched["pages"], "q": q, "dist": h["distance"]})
    windows.sort(key=lambda x: x["dist"])
    if not windows:
        return {"context": "", "pages": (0,0), "queries": queries}
    combined, words, first_pages = [], 0, windows[0]["pages"]
    for w in windows:
        w_words = len(w["text"].split())
        if words + w_words > 1800:
            break
        combined.append(w["text"])
        words += w_words
    return {"context": "\n\n---\n\n".join(combined), "pages": first_pages, "queries": queries}

def answer_with_context(note: str, context: str, pages: Tuple[int, int], prompts: Dict[str, str]):
    page_start, page_end = pages
    msgs = [
        {"role": "system", "content": prompts["ANSWER_SYSTEM"]},
        {"role": "user", "content": prompts["ANSWER_USER_TMPL"].format(note=note, context=context)},
    ]
    out = openai_chat(msgs, temperature=0.2)
    return out + f"\n\n**References:** [Respiratory chapter, p. {page_start}-{page_end}]"

# ------------------ CLI ------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--note", required=False, help="Visit note text; if omitted, a demo note is used.")
    ap.add_argument("--doc-id", default=DOC_ID)
    ap.add_argument("--db", default=str(ROOT / "chroma_db"))
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--above", type=int, default=1)
    ap.add_argument("--below", type=int, default=2)
    ap.add_argument("--prompts-dir", default=str(ROOT / "prompts"), help="Base directory that contains version folders")
    ap.add_argument("--prompts-version", default=None, help="e.g., v1, v1.1, v2-2025-01-15; default = 'current' or latest by name")
    args = ap.parse_args()

    prompts = load_prompts_versioned(Path(args.prompts_dir), args.prompts_version)

    note = args.note or (
        "3-year-old with barky cough, hoarse voice, inspiratory stridor at rest. "
        "Temp 38°C. Mild chest retractions. No drooling. No allergies known."
    )

    rag = RAGClient(db_path=args.db)
    queries = plan_queries(note, prompts)
    ctx = retrieve_context(rag, queries, doc_id=args.doc_id, k=args.k, above=args.above, below=args.below)
    if not ctx["context"]:
        print("No context found. Queries tried:", queries)
        return
    result = answer_with_context(note, ctx["context"], ctx["pages"], prompts)

    print("\n=== Queries ===")
    for q in ctx["queries"]:
        print("-", q)
    print("\n=== Answer ===\n")
    print(result)

if __name__ == "__main__":
    main()
