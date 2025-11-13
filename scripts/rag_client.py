import os, json
from typing import Dict, Any, List
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

class RAGClient:
    def __init__(self, db_path: str, collection: str, embed_model: str):
        db_path = os.getenv("CHROMA_DB_DIR", db_path)
        self.client = chromadb.PersistentClient(path=db_path, settings=Settings(anonymized_telemetry=False))
        self.coll = self.client.get_or_create_collection(name=collection)
        # force CPU (safe for containers)
        self.embedder = SentenceTransformer(embed_model, device="cpu")
        
        # Debug: Log collection info
        print(f"[RAGClient] Initialized with db_path={db_path}, collection={collection}")
        print(f"[RAGClient] Collection count: {self.coll.count()}")

    def _get(self, id_str):
        res = self.coll.get(ids=[id_str])
        if not res["ids"]:
            return None, None
        return res["documents"][0], res["metadatas"][0]

    def fetch(self, doc_id: str, window_id: int):
        return self._get(f"{doc_id}::{window_id}")

    def query(self, q: str, doc_id: str, k=4):
        print(f"[RAGClient.query] Query: '{q[:50]}...', doc_id: '{doc_id}', k: {k}")
        
        q_emb = self.embedder.encode([q]).tolist()
        
        # Debug: Try query WITHOUT doc_id filter first
        res_all = self.coll.query(
            query_embeddings=q_emb,
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )
        print(f"[RAGClient.query] Query without filter returned: {len(res_all['ids'][0])} results")
        if res_all['ids'][0]:
            print(f"[RAGClient.query] Sample doc_ids in collection: {[m.get('doc_id') for m in res_all['metadatas'][0][:3]]}")
        
        # Now try with filter
        res = self.coll.query(
            query_embeddings=q_emb,
            n_results=k,
            where={"doc_id": doc_id},
            include=["documents", "metadatas", "distances"],
        )
        
        print(f"[RAGClient.query] Query WITH filter returned: {len(res['ids'][0])} results")
        
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
        m = hit["metadata"]
        center = int(m["window_id"])
        parts, total = [], 0
        for w in range(center - above, center + below + 1):
            doc, meta = self.fetch(m["doc_id"], w)
            if doc is None:
                continue
            parts.append((doc, meta))
            total += len(doc.split())
            if total >= cap_words:
                break
        if not parts:
            return None
        text = "\n".join(p[0] for p in parts)
        page_start = parts[0][1].get("page_start", m.get("page_start", 0))
        page_end = parts[-1][1].get("page_end", m.get("page_end", 0))
        return {"text": text, "pages": (int(page_start), int(page_end)), "windows": [p[1]["window_id"] for p in parts]}

def load_prompt(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def plan_queries(note: str, planner_system: str, chat_fn) -> List[str]:
    msgs = [{"role": "system", "content": planner_system},
            {"role": "user", "content": note.strip()[:4000]}]
    out = chat_fn(msgs, temperature=0.0)
    print(f"[plan_queries] LLM response: {out[:200]}...")
    try:
        data = json.loads(out)
        queries = [q.strip() for q in data.get("queries", []) if q.strip()][:3]
        print(f"[plan_queries] Extracted queries: {queries}")
        return queries or ["croup steroid dose management"]
    except Exception as e:
        print(f"[plan_queries] JSON parse error: {e}")
        return ["croup steroid dose management"]

def answer_with_context(note: str, context: str, pages, answer_system: str, answer_user_tmpl: str, chat_fn) -> str:
    page_start, page_end = pages
    cite = f"[Respiratory chapter, p. {page_start}-{page_end}]"
    msg_user = answer_user_tmpl.format(note=note, context=context)
    msgs = [{"role": "system", "content": answer_system},
            {"role": "user", "content": msg_user}]
    out = chat_fn(msgs, temperature=0.2)
    return out + f"\n\n**References:** {cite}"