import os
import json
import pathlib
import yaml
import streamlit as st
from chromadb.config import Settings
import chromadb
from dotenv import load_dotenv

from scripts.ingest_pdf import ingest_pdf
from scripts.llm_helper import configure as configure_llm, openai_chat
from scripts.rag_client import (
    RAGClient,
    load_prompt,
    plan_queries,
    answer_with_context,
    dosing_queries_from_plan,
    dosing_table_from_context,
)
from scripts.vector_store import ensure_local_vector_store, VectorStoreBootstrapError

# --- Load config
load_dotenv()
CFG = yaml.safe_load(open("./config/config.yaml", "r"))
DB_CFG = CFG.get("database", {})
CHROMA_CFG = DB_CFG.get("chroma", {})

# Check for OpenAI API key
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    st.error("Error: OPENAI_API_KEY not found in environment variables. "
             "Please ensure it's set correctly in your .env file or as a system environment variable.")
    st.stop() # Stop the app if the key is missing

DOC_ID = CFG["app"]["default_doc_id"]
CHROMA_DEFAULT_PATH = CHROMA_CFG.get("db_path", CFG["retrieval"]["db_path"])
DB_PATH = pathlib.Path(os.getenv("CHROMA_DB_DIR", CHROMA_DEFAULT_PATH)).expanduser().resolve()
COLLECTION = CHROMA_CFG.get("collection", CFG["retrieval"]["collection"])
VECTOR_SENTINEL = os.getenv("VECTOR_STORE_SENTINEL", "chroma.sqlite3")

# --- Configure LLM helper
# Assuming configure_llm uses os.getenv("OPENAI_API_KEY") internally or the openai library picks it up.
configure_llm(
    model=CFG["openai"]["model"],
    api_base=CFG["openai"]["api_base"],
)

def _chroma_count(db_path: str, collection: str = "guidelines") -> int:
    try:
        client = chromadb.PersistentClient(path=db_path, settings=Settings(anonymized_telemetry=False))
        coll = client.get_or_create_collection(collection)
        return coll.count()
    except Exception:
        return -1

def _init_state():
    st.session_state.setdefault("plan_state", None)
    st.session_state.setdefault("dosing_state", None)

def _gather_windows(rag, queries, doc_id, k, above, below, cap_words):
    windows, seen = [], set()
    query_details = []
    for q in queries:
        hits = rag.query(q, doc_id=doc_id, k=k)
        dupes = 0
        for h in hits:
            key = (h["metadata"]["doc_id"], h["metadata"]["window_id"])
            if key in seen:
                dupes += 1
                continue
            seen.add(key)
            stitched = rag.stitch_neighbors(h, above=above, below=below, cap_words=cap_words)
            if stitched:
                windows.append({"text": stitched["text"], "pages": stitched["pages"], "q": q, "dist": h["distance"]})
        query_details.append({"query": q, "hit_count": len(hits), "duplicates": dupes})
    windows.sort(key=lambda x: x["dist"])
    return windows, query_details

def _combine_context(windows, limit=3):
    if not windows:
        return "", (0, 0)
    trimmed = windows[:limit]
    context = "\n\n---\n\n".join([w["text"] for w in trimmed])
    start = min(w["pages"][0] for w in trimmed)
    end = max(w["pages"][1] for w in trimmed)
    return context, (start, end)

def _format_citation(doc_label: str, pages):
    start, end = pages
    if start == end:
        page_text = f"p. {start}"
    else:
        page_text = f"p. {start}-{end}"
    return f"{doc_label} {page_text}"

def _vector_store_label():
    return f"Chroma path `{DB_PATH}`"


@st.cache_resource(show_spinner="Loading vector store...")
def _load_rag_client(db_path: str, collection: str):
    local_path = ensure_local_vector_store(db_path)
    return RAGClient(
        db_path=str(local_path),
        collection=collection,
        embed_model=CFG["retrieval"]["embed_model"],
    )


def _make_rag_client():
    try:
        return _load_rag_client(str(DB_PATH), COLLECTION)
    except VectorStoreBootstrapError as exc:
        st.error(f"Unable to prepare vector store: {exc}")
        st.stop()

# --- UI
st.set_page_config(page_title=CFG["app"]["title"], layout="wide")
st.title(CFG["app"]["title"])
st.caption("Develop branch demo")
_init_state()

with st.sidebar:
    st.header("Vector Store")
    st.write(f"Vector DB: {_vector_store_label()}")
    sentinel_path = DB_PATH / VECTOR_SENTINEL
    if sentinel_path.exists():
        st.success("Local cache ready.")
    else:
        st.info("Local cache missing; will download from GCS on first load.")
    doc_id = st.text_input("Guideline Document", value=DOC_ID)

    st.header("Retrieval Settings")
    k = st.number_input("Top-k", 1, 10, CFG["retrieval"]["k"])
    above = st.number_input("Neighbors above", 0, 5, CFG["retrieval"]["above"])
    below = st.number_input("Neighbors below", 0, 5, CFG["retrieval"]["below"])

note = st.text_area("Visit note", height=160, value="3-year-old with barky cough, hoarse voice, inspiratory stridor. Temp 38°C. No drooling.")

# --- Load prompts
planner_system = load_prompt(CFG["prompts"]["planner_system"])
answer_system = load_prompt(CFG["prompts"]["answer_system"])
answer_user_tmpl = load_prompt(CFG["prompts"]["answer_user_template"])
dosing_queries_system = load_prompt(CFG["prompts"]["dosing_queries_system"])
dosing_queries_user_tmpl = load_prompt(CFG["prompts"]["dosing_queries_user_template"])
dosing_table_system = load_prompt(CFG["prompts"]["dosing_table_system"])
dosing_table_user_tmpl = load_prompt(CFG["prompts"]["dosing_table_user_template"])

# --- Vectorise controls ---
# st.subheader("📚 Vectorise guideline PDF into Chroma")

# # Defaults (tweak or wire from your config.yaml if you have one)
# default_pdf = "./guideline_documents/respiratory-stg-2024.pdf"
# default_doc_id = "SA_PHC_STG_2024_Respiratory"
# db_path = os.getenv("CHROMA_DB_DIR", "./chroma_db")

# with st.container(border=True):
#     col1, col2 = st.columns([2, 1])
#     with col1:
#         pdf_path = st.text_input("PDF path", value=default_pdf, help="Path to the guideline PDF on this machine.")
#         doc_id = st.text_input("Document ID", value=default_doc_id, help="Used as the key in the vector store.")
#         window_chars = st.number_input("Window chars", min_value=500, max_value=8000, value=2200, step=100)
#         overlap_chars = st.number_input("Overlap chars", min_value=0, max_value=4000, value=400, step=50)
#     with col2:
#         st.write(" ")
#         st.write(" ")
#         st.caption(f"Vector DB dir: `{db_path}`")
#         current = _chroma_count(db_path)
#         if current >= 0:
#             st.info(f"Current vectors in `{db_path}`: **{current}**")
#         else:
#             st.warning("Vector DB not readable yet.")

generate_plan = st.button("Generate plan", type="primary", use_container_width=True)

if generate_plan:
    st.session_state["dosing_state"] = None
    rag = _make_rag_client()
    queries = plan_queries(note, planner_system, openai_chat)
    windows, query_details = _gather_windows(
        rag,
        queries,
        doc_id=doc_id,
        k=k,
        above=above,
        below=below,
        cap_words=CFG["retrieval"]["cap_words"],
    )
    if not windows:
        st.session_state["plan_state"] = None
        st.error("❌ No context retrieved.")
        st.info("Check the terminal/console for debug logs to see what's happening.")
    else:
        context, pages = _combine_context(windows)
        with st.spinner("Generating plan..."):
            plan = answer_with_context(note, context, pages, answer_system, answer_user_tmpl, openai_chat)
        st.session_state["plan_state"] = {
            "doc_id": doc_id,
            "vector_info": _vector_store_label(),
            "collection": DB_CFG.get("chroma", {}).get("collection", CFG["retrieval"]["collection"]),
            "queries": queries,
            "query_details": query_details,
            "windows": windows,
            "context": context,
            "pages": pages,
            "plan": plan,
            "note": note,
            "k": k,
            "above": above,
            "below": below,
        }

plan_state = st.session_state.get("plan_state")
if plan_state:
    st.write("---")
    st.subheader("🔍 Debug Info")
    st.write(f"**Doc ID being queried:** `{plan_state['doc_id']}`")
    st.write(f"**Vector Store:** {plan_state.get('vector_info', _vector_store_label())}")
    st.write(f"**Collection:** `{plan_state.get('collection', CFG['retrieval']['collection'])}`")
    st.markdown("### Queries generated")
    st.code(json.dumps({"queries": plan_state["queries"]}, indent=2))

    for detail in plan_state["query_details"]:
        st.write(f"Querying: `{detail['query']}`")
        st.write(f"  → Got {detail['hit_count']} hits")
        if detail.get("duplicates"):
            st.write(f"  → Skipped {detail['duplicates']} duplicate windows")

    st.write(f"**Total unique windows retrieved:** {len(plan_state['windows'])}")

    st.markdown("---")
    st.markdown("### 📄 Retrieved Context")
    pages = plan_state["pages"]
    st.caption(f"Showing top 3 windows | Pages: {pages[0]}-{pages[1]}")

    for i, w in enumerate(plan_state["windows"][:3], 1):
        with st.expander(f"Window {i} - Pages {w['pages'][0]}-{w['pages'][1]} (Distance: {w['dist']:.4f})", expanded=False):
            st.caption(f"**Matched query:** `{w['q']}`")
            st.text_area(
                f"Window {i} content",
                value=w["text"],
                height=300,
                disabled=True,
                label_visibility="collapsed",
            )

    with st.expander("View combined context (sent to LLM)", expanded=False):
        st.text_area(
            "Combined context",
            value=plan_state["context"],
            height=400,
            disabled=True,
            label_visibility="collapsed",
        )

    st.markdown("---")
    st.markdown("### 💊 Management Plan")
    st.write(plan_state["plan"])

    st.markdown("---")
    st.markdown("### 💉 Dosing Table")
    generate_dosing = st.button("Generate Dosing Table", type="secondary", use_container_width=True)

    if generate_dosing:
        rag = _make_rag_client()
        dosing_queries = dosing_queries_from_plan(
            plan_state["plan"],
            dosing_queries_system,
            dosing_queries_user_tmpl,
            openai_chat,
        )
        if not dosing_queries:
            st.session_state["dosing_state"] = None
            st.warning("No dosing-focused queries could be generated from the management plan.")
        else:
            windows, query_details = _gather_windows(
                rag,
                dosing_queries,
                doc_id=plan_state["doc_id"],
                k=plan_state["k"],
                above=plan_state["above"],
                below=plan_state["below"],
                cap_words=CFG["retrieval"]["cap_words"],
            )
            if not windows:
                st.session_state["dosing_state"] = None
                st.error("No dosing context retrieved from the guideline.")
            else:
                context, pages = _combine_context(windows)
                with st.spinner("Generating dosing table..."):
                    table = dosing_table_from_context(
                        plan_state["plan"],
                        context,
                        dosing_table_system,
                        dosing_table_user_tmpl,
                        openai_chat,
                    )
                if table.get("status") != "OK" or not table.get("rows"):
                    st.session_state["dosing_state"] = None
                    st.warning("LLM could not build a dosing table from the retrieved context.")
                else:
                    st.session_state["dosing_state"] = {
                        "queries": dosing_queries,
                        "query_details": query_details,
                        "windows": windows,
                        "context": context,
                        "pages": pages,
                        "table": table,
                        "doc_id": plan_state["doc_id"],
                    }

    dosing_state = st.session_state.get("dosing_state")
    if dosing_state:
        citation_label = _format_citation(dosing_state["doc_id"], dosing_state["pages"])
        st.caption(f"Dosing context sourced from `{dosing_state['doc_id']}` ({citation_label})")
        st.markdown("#### Queries used for dosing search")
        st.code(json.dumps({"queries": dosing_state["queries"]}, indent=2))
        for detail in dosing_state["query_details"]:
            st.write(f"Querying: `{detail['query']}`")
            st.write(f"  → Got {detail['hit_count']} hits")
            if detail.get("duplicates"):
                st.write(f"  → Skipped {detail['duplicates']} duplicate windows")

        st.caption(f"Showing top 3 dosing windows | Pages: {dosing_state['pages'][0]}-{dosing_state['pages'][1]}")
        for i, w in enumerate(dosing_state["windows"][:3], 1):
            with st.expander(f"Dosing window {i} - Pages {w['pages'][0]}-{w['pages'][1]} (Distance: {w['dist']:.4f})", expanded=False):
                st.caption(f"**Matched query:** `{w['q']}`")
                st.text_area(
                    f"Dosing window {i} content",
                    value=w["text"],
                    height=250,
                    disabled=True,
                    label_visibility="collapsed",
                )

        with st.expander("View combined dosing context", expanded=False):
            st.text_area(
                "Dosing context",
                value=dosing_state["context"],
                height=350,
                disabled=True,
                label_visibility="collapsed",
            )

        st.markdown("#### Generated dosing table")
        table_rows = []
        desired_cols = [
            "drug",
            "route",
            "rule_mg_per_kg",
            "max_mg",
            "rounding",
            "usual_formulations",
            "frequency",
            "duration",
            "citation",
        ]
        for row in dosing_state["table"]["rows"]:
            filled = {col: row.get(col, "") for col in desired_cols}
            filled["citation"] = citation_label
            table_rows.append(filled)
        st.table(table_rows)
