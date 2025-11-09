import os, json, argparse, requests
from typing import List, Dict, Any
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

DOC_ID = "SA_PHC_STG_2024_Respiratory"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
OPENAI_MODEL = "gpt-4o-mini"  # swap if you prefer another model
OPENAI_URL = "https://api.openai.com/v1/chat/completions"

# ------------------- Loading API key -------------------

load_dotenv()

# ------------------ Chroma RAG client ------------------
class RAGClient:
    def __init__(self, db_path="./chroma_db", collection="guidelines", embed_model=EMBED_MODEL):
        self.client = chromadb.PersistentClient(path=db_path, settings=Settings())
        self.coll = self.client.get_or_create_collection(name=collection)
        self.embedder = SentenceTransformer(embed_model)

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

# ------------------ Prompts ------------------
PLANNER_SYSTEM = """You are a query planner for a medical RAG system.
Given a visit note, output 1-3 short search queries targeting a Respiratory chapter in South African paediatric guidelines.
Prefer concrete clinical terms (condition, drug, route, dose, severity, disposition).
Return ONLY valid JSON of the form: {"queries": ["q1","q2", ...]}"""

ANSWER_SYSTEM = """You are a clinical decision-support assistant.
You MUST use only the provided Context from the guideline to answer.
Cite the context as: [Respiratory chapter, p. {page_start}-{page_end}].
If key info is missing, say so explicitly.
This is a demo and not medical advice."""

ANSWER_USER_TMPL = """Visit note:
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

# ------------------ Pipeline ------------------
def plan_queries(note: str) -> List[str]:
    msg = [
        {"role": "system", "content": PLANNER_SYSTEM},
        {"role": "user", "content": note.strip()[:4000]},
    ]
    out = openai_chat(msg, temperature=0.0)
    try:
        data = json.loads(out)
        queries = [q.strip() for q in data.get("queries", []) if q.strip()]
        return queries[:3] or ["croup steroid dose management"]
    except Exception:
        # fallback: single broad query
        return ["croup steroid dose management"]

def retrieve_context(rag: RAGClient, queries: List[str], doc_id: str, k=4, above=1, below=2) -> Dict[str, Any]:
    windows = []
    seen = set()
    for q in queries:
        hits = rag.query(q, doc_id=doc_id, k=k)
        for h in hits:
            m = h["metadata"]; key = (m["doc_id"], m["window_id"])
            if key in seen: continue
            seen.add(key)
            stitched = rag.stitch_neighbors(h, above=above, below=below)
            if stitched:
                windows.append({"text": stitched["text"], "pages": stitched["pages"], "q": q, "dist": h["distance"]})
    # Sort by distance (smaller is better for cosine in Chroma)
    windows.sort(key=lambda x: x["dist"])
    if not windows:
        return {"context": "", "pages": (0,0), "queries": queries}
    # Concatenate top few windows until ~1500-2000 words
    combined, words, first_pages = [], 0, windows[0]["pages"]
    for w in windows:
        w_words = len(w["text"].split())
        if words + w_words > 1800: break
        combined.append(w["text"])
        words += w_words
    return {"context": "\n\n---\n\n".join(combined), "pages": first_pages, "queries": queries}

def answer_with_context(note: str, context: str, pages):
    page_start, page_end = pages
    cite = f"[Respiratory chapter, p. {page_start}-{page_end}]"
    msgs = [
        {"role": "system", "content": ANSWER_SYSTEM},
        {"role": "user", "content": ANSWER_USER_TMPL.format(note=note, context=context)},
    ]
    out = openai_chat(msgs, temperature=0.2)
    return out + f"\n\n**References:** {cite}"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--note", required=False, help="Visit note text; if omitted, a demo note is used.")
    ap.add_argument("--doc-id", default=DOC_ID)
    ap.add_argument("--db", default="./chroma_db")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--above", type=int, default=1)
    ap.add_argument("--below", type=int, default=2)
    args = ap.parse_args()

    note = args.note or (
        "3-year-old with barky cough, hoarse voice, inspiratory stridor at rest. "
        "Temp 38°C. Mild chest retractions. No drooling. No allergies known."
    )

    rag = RAGClient(db_path=args.db)
    queries = plan_queries(note)
    ctx = retrieve_context(rag, queries, doc_id=args.doc_id, k=args.k, above=args.above, below=args.below)
    if not ctx["context"]:
        print("No context found. Queries tried:", queries)
        return
    result = answer_with_context(note, ctx["context"], ctx["pages"])
    print("\n=== Queries ===")
    for q in ctx["queries"]:
        print("-", q)
    print("\n=== Answer ===\n")
    print(result)

if __name__ == "__main__":
    main()
