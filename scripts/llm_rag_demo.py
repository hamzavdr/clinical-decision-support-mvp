# scripts/llm_rag_demo.py
# RAG → LLM demo with external, versioned prompts + dosing-table + LLM-based dosing-query generation

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

# Dosing-table prompts (fallback)
DOSING_SYS_FALLBACK = """You are a clinical dosing summarizer.
Use ONLY the provided context from the guideline.
If dosing details (mg/kg, maximum dose, route, formulation, frequency, or duration) are unclear, return {"status":"NO_TABLE"}.

Return ONLY JSON with:
{
  "status": "OK" | "NO_TABLE",
  "rows": [
    {
      "drug": "string",
      "route": "string",
      "rule_mg_per_kg": "string",
      "max_mg": "string",
      "rounding": "string",
      "usual_formulations": "string",
      "frequency": "string",
      "duration": "string",
      "citation": "e.g., Respiratory chapter p. 12–13"
    }
  ]
}
"""

DOSING_USER_TMPL_FALLBACK = """Management plan (for context):
{plan}

Dosing context (from guideline):
{context}

Extract dosing for medications relevant to the plan. Include:
- mg/kg rule and maximum single dose if specified
- route and usual formulations
- frequency (e.g., once, q6h, daily) and duration (e.g., 3 days, 5–7 days)

Follow the JSON schema exactly and return ONLY JSON.
"""

# LLM-based dosing-query prompts (optional)
DOSING_QUERIES_SYS_FALLBACK = """You extract dosing-focused search queries from a clinical management plan.
Rules:
- Use ONLY information in the plan.
- Target dosing details: mg/kg rules, maximum dose, route, formulations, frequency.
- Include only concise queries (5–12 words).
- Prefer exact terms from the plan that likely appear in the guideline.
- If no medication appears in the plan, return an empty list.
Return ONLY JSON: {"queries":["q1","q2"]}"""

DOSING_QUERIES_USER_TMPL_FALLBACK = """Management plan:
{plan}

Produce dosing-focused search queries following the schema."""

# ------------------- Versioned prompt loader -------------------
def _read_text(path: Path) -> Optional[str]:
    try:
        if path.is_file():
            return path.read_text(encoding="utf-8").strip()
    except Exception:
        pass
    return None

def _natural_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]

def _resolve_version_dir(base: Path, requested: Optional[str]) -> Path:
    if requested:
        cand = base / requested
        if cand.is_dir():
            return cand
        raise FileNotFoundError(f"Prompt version '{requested}' not found under {base}")
    cur = base / "current"
    if cur.exists() and cur.is_dir():
        return cur
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
    dosing_sys = _read_text(vdir / "dosing_table_system.txt") or DOSING_SYS_FALLBACK
    dosing_user = _read_text(vdir / "dosing_table_user_tmpl.txt") or DOSING_USER_TMPL_FALLBACK
    dosing_q_sys = _read_text(vdir / "dosing_queries_system.txt") or DOSING_QUERIES_SYS_FALLBACK
    dosing_q_user = _read_text(vdir / "dosing_queries_user_tmpl.txt") or DOSING_QUERIES_USER_TMPL_FALLBACK
    return {
        "PLANNER_SYSTEM": planner,
        "ANSWER_SYSTEM": answer_sys,
        "ANSWER_USER_TMPL": answer_user,
        "DOSING_SYS": dosing_sys,
        "DOSING_USER_TMPL": dosing_user,
        "DOSING_QUERIES_SYS": dosing_q_sys,
        "DOSING_QUERIES_USER_TMPL": dosing_q_user,
    }

# ------------------ Chroma RAG client ------------------
class RAGClient:
    def __init__(self, db_path="./chroma_db", collection="guidelines", embed_model=EMBED_MODEL):
        self.client = chromadb.PersistentClient(path=db_path, settings=Settings())
        self.coll = self.client.get_or_create_collection(name=collection)
        self.embedder = SentenceTransformer(embed_model, device="cpu")

    def list_doc_ids(self) -> List[str]:
        res = self.coll.get(include=["metadatas"])
        metas = res.get("metadatas") or []
        ids = {m.get("doc_id") for m in metas if m and m.get("doc_id")}
        return sorted(x for x in ids if x)

    def _get(self, id_str):
        res = self.coll.get(ids=[id_str])
        if not res.get("ids"):
            return None, None
        return res["documents"][0], res["metadatas"][0]

    def fetch(self, doc_id: str, window_id: int):
        return self._get(f"{doc_id}::{window_id}")

    def query(self, q: str, doc_id: Optional[str], k=4):
        q_emb = self.embedder.encode([q]).tolist()
        where = {"doc_id": doc_id} if doc_id else {}
        res = self.coll.query(
            query_embeddings=q_emb,
            n_results=k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        hits = []
        if not res.get("ids") or not res["ids"][0]:
            return hits
        for i in range(len(res["ids"][0])):
            hits.append({
                "id": res["ids"][0][i],
                "document": res["documents"][0][i],
                "metadata": res["metadatas"][0][i],
                "distance": res["distances"][0][i],
            })
        return hits

    def stitch_neighbors(self, hit: Dict[str, Any], above=1, below=2, cap_words=1600):
        """Include neighbors around a center window (same doc_id)."""
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
        return {
            "text": text,
            "pages": (int(page_start), int(page_end)),
            "windows": [p[1]["window_id"] for p in parts],
        }

# ------------------ LLM helper ------------------
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

def retrieve_context(
    rag: RAGClient,
    queries: List[str],
    doc_id: Optional[str],
    k=4,
    above=1,
    below=2
) -> Dict[str, Any]:
    windows = []
    seen = set()
    for q in queries:
        hits = rag.query(q, doc_id=doc_id, k=k)
        for h in hits:
            m = h["metadata"]
            key = (m["doc_id"], m["window_id"])
            if key in seen:
                continue
            seen.add(key)
            stitched = rag.stitch_neighbors(h, above=above, below=below)
            if stitched:
                windows.append({
                    "text": stitched["text"],
                    "pages": stitched["pages"],
                    "q": q,
                    "dist": h["distance"]
                })
    windows.sort(key=lambda x: x["dist"])
    if not windows:
        return {"context": "", "pages": (0, 0), "queries": queries}
    # Concatenate top few windows until ~1800 words
    combined, words, first_pages = [], 0, windows[0]["pages"]
    for w in windows:
        w_words = len(w["text"].split())
        if words + w_words > 1800:
            break
        combined.append(w["text"])
        words += w_words
    return {
        "context": "\n\n---\n\n".join(combined),
        "pages": first_pages,
        "queries": queries,
    }

def answer_with_context(
    note: str,
    context: str,
    pages: Tuple[int, int],
    prompts: Dict[str, str],
    dose_block: str = "(none)"
):
    page_start, page_end = pages
    user_content = prompts["ANSWER_USER_TMPL"].format(
        note=note, context=context, dose_block=dose_block
    )
    msgs = [
        {"role": "system", "content": prompts["ANSWER_SYSTEM"]},
        {"role": "user", "content": user_content},
    ]
    out = openai_chat(msgs, temperature=0.2)
    return out + f"\n\n**References:** [Respiratory chapter, p. {page_start}-{page_end}]"

# ------------------ Dosing-table & dosing-query helpers ------------------
def _fallback_dosing_queries(plan_text: str) -> List[str]:
    """
    Very light, plan-driven fallback (regex over the plan) with NO hard-coded global list.
    """
    meds = sorted(set(m.lower() for m in re.findall(r"\b([A-Z][a-zA-Z]{3,})\b", plan_text)))
    meds = [m for m in meds if any(x in m for x in [
        "sone","lone","zone","dexa","predni","rol","line","rine",
        "adrenaline","epinephrine","salbutamol","hydrocortisone"
    ])]
    meds = meds[:4]
    queries = []
    for m in meds:
        queries += [
            f"{m} dose mg/kg",
            f"{m} maximum dose",
            f"{m} route and formulations",
        ]
    if not queries:
        conds = re.findall(r"\b(croup|stridor|asthma|pneumonia|bronchiolitis)\b", plan_text, flags=re.I)
        for c in sorted(set(c.lower() for c in conds))[:2]:
            queries += [f"{c} steroid dose mg/kg", f"{c} maximum dose"]
    return queries[:8]

def generate_dosing_queries_from_plan(plan_text: str, prompts: Dict[str, str]) -> List[str]:
    """
    Primary method: ask LLM to extract dosing-focused queries from the management plan.
    Safe fallback: plan-driven regex if JSON can't be parsed or list is empty.
    """
    sys = prompts.get("DOSING_QUERIES_SYS", DOSING_QUERIES_SYS_FALLBACK)
    usr_tmpl = prompts.get("DOSING_QUERIES_USER_TMPL", DOSING_QUERIES_USER_TMPL_FALLBACK)
    msgs = [
        {"role": "system", "content": sys},
        {"role": "user", "content": usr_tmpl.format(plan=plan_text)},
    ]
    raw = openai_chat(msgs, temperature=0)
    try:
        data = json.loads(raw)
        qs = [q.strip() for q in data.get("queries", []) if isinstance(q, str) and q.strip()]
        return qs or _fallback_dosing_queries(plan_text)
    except Exception:
        return _fallback_dosing_queries(plan_text)

# Backwards compatibility for app imports:
def derive_dosing_queries_from_plan(plan_text: str) -> List[str]:
    """
    Wrapper preserved for compatibility; you can switch your app to
    generate_dosing_queries_from_plan(plan_text, prompts) for versioned behavior.
    Here we fall back to the regex-only variant so this function doesn't need prompts.
    """
    return _fallback_dosing_queries(plan_text)

def retrieve_dosing_context(
    rag: RAGClient,
    queries: List[str],
    doc_id: Optional[str],
    k=6,
    above=1,
    below=2,
    cap_words=1600
) -> Dict[str, Any]:
    windows = []
    seen = set()
    for q in queries:
        for h in rag.query(q, doc_id=doc_id, k=k):
            m = h["metadata"]
            key = (m["doc_id"], m["window_id"])
            if key in seen:
                continue
            seen.add(key)
            stitched = rag.stitch_neighbors(h, above=above, below=below)
            if stitched:
                windows.append({
                    "text": stitched["text"],
                    "pages": stitched["pages"],
                    "dist": h["distance"]
                })
    windows.sort(key=lambda x: x["dist"])
    if not windows:
        return {"context": "", "pages": (0, 0)}
    combined, words, first_pages = [], 0, windows[0]["pages"]
    for w in windows:
        w_words = len(w["text"].split())
        if words + w_words > cap_words:
            break
        combined.append(w["text"])
        words += w_words
    return {"context": "\n\n---\n\n".join(combined), "pages": first_pages}

def build_dosing_table(plan_text: str, dosing_ctx: str, pages, prompts: Dict[str, str]) -> Optional[str]:
    """Ask the LLM for strict JSON; validate; render markdown table with frequency & duration."""
    if not dosing_ctx.strip():
        return None
    msgs = [
        {"role": "system", "content": prompts.get("DOSING_SYS", DOSING_SYS_FALLBACK)},
        {"role": "user", "content": prompts.get("DOSING_USER_TMPL", DOSING_USER_TMPL_FALLBACK).format(
            plan=plan_text, context=dosing_ctx
        )},
    ]
    raw = openai_chat(msgs, temperature=0)
    try:
        data = json.loads(raw)
    except Exception:
        return None
    if data.get("status") != "OK":
        return None
    rows = data.get("rows") or []
    if not rows:
        return None

    header = (
        "| Drug | Route | Rule (mg/kg) | Max (mg) | Rounding | Usual formulations | Frequency | Duration | Citation |\n"
        "|---|---|---:|---:|---|---|---|---|---|\n"
    )
    body = ""
    for r in rows:
        body += (
            f"| {r.get('drug','')} "
            f"| {r.get('route','')} "
            f"| {r.get('rule_mg_per_kg','')} "
            f"| {r.get('max_mg','')} "
            f"| {r.get('rounding','')} "
            f"| {r.get('usual_formulations','')} "
            f"| {r.get('frequency','')} "
            f"| {r.get('duration','')} "
            f"| {r.get('citation','')} |\n"
        )
    return "**Dosing table (from guideline)**\n\n" + header + body


# ------------------ CLI for quick testing ------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--note", required=False, help="Visit note text; if omitted, a demo note is used.")
    ap.add_argument("--doc-id", default=DOC_ID)
    ap.add_argument("--db", default=str(ROOT / "chroma_db"))
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--above", type=int, default=1)
    ap.add_argument("--below", type=int, default=2)
    ap.add_argument("--prompts-dir", default=str(ROOT / "prompts"), help="Base directory that contains version folders")
    ap.add_argument("--prompts-version", default=None, help="e.g., v1, v1.1, v1-full-stg; default = 'current' or latest by name")
    ap.add_argument("--with-dosing-table", action="store_true", help="Also build a dosing table from the plan/context")
    ap.add_argument("--with-dosing-queries-from-plan", action="store_true", help="Use LLM to derive dosing queries from the plan")
    args = ap.parse_args()

    prompts = load_prompts_versioned(Path(args.prompts_dir), args.prompts_version)

    note = args.note or (
        "3-year-old with barky cough, hoarse voice, inspiratory stridor at rest. "
        "Temp 38°C. Mild chest retractions. No drooling. No allergies known."
    )

    rag = RAGClient(db_path=args.db)

    # 1) Plan queries
    queries = plan_queries(note, prompts)

    # 2) Retrieve stitched context
    ctx = retrieve_context(rag, queries, doc_id=args.doc_id, k=args.k, above=args.above, below=args.below)
    if not ctx["context"]:
        print("No context found. Queries tried:", queries)
        return

    # 3) Answer with context
    result = answer_with_context(note, ctx["context"], ctx["pages"], prompts)
    print("\n=== Queries ===")
    for q in ctx["queries"]:
        print("-", q)
    print("\n=== Answer ===\n")
    print(result)

    # 4) (Optional) Dosing queries derived from plan text
    if args.with_dosing_queries_from_plan:
        dq = generate_dosing_queries_from_plan(result, prompts)
        print("\n=== Dosing queries (from plan) ===")
        for q in dq:
            print("-", q)

    # 5) (Optional) Build dosing table derived from the plan
    if args.with_dosing_table:
        print("\n=== Dosing table ===\n")
        # Use LLM-based dosing queries by default if the flag is on; else fallback to wrapper
        dosing_queries = generate_dosing_queries_from_plan(result, prompts) if args.with_dosing_queries_from_plan else derive_dosing_queries_from_plan(result)
        dctx = retrieve_dosing_context(rag, dosing_queries, doc_id=args.doc_id, k=args.k, above=args.above, below=args.below)
        if dctx["context"]:
            table_md = build_dosing_table(result, dctx["context"], dctx["pages"], prompts)
            if table_md:
                print(table_md)
            else:
                print("No dosing table produced from the provided context.")
        else:
            print("No dosing context retrieved for table generation.")

if __name__ == "__main__":
    main()
