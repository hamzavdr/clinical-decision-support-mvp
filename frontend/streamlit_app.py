import os, json, yaml, pathlib, streamlit as st
from scripts import llm_helper, rag_client
from scripts.rag_client import RAGClient, load_prompt, plan_queries, answer_with_context

# --- Load config
CFG = yaml.safe_load(open("./config/config.yaml", "r"))
DOC_ID = CFG["app"]["default_doc_id"]
DB_PATH = os.getenv("CHROMA_DB_DIR", CFG["retrieval"]["db_path"])

# --- Configure LLM helper
llm_helper.configure(
    model=CFG["openai"]["model"],
    api_base=CFG["openai"]["api_base"],
)

# --- UI
st.set_page_config(page_title=CFG["app"]["title"], layout="wide")
st.title(CFG["app"]["title"])
st.caption("Develop branch demo")

with st.sidebar:
    st.header("Settings")
    doc_id = st.text_input("Document ID", value=DOC_ID)
    k = st.number_input("Top-k", 1, 10, CFG["retrieval"]["k"])
    above = st.number_input("Neighbors above", 0, 5, CFG["retrieval"]["above"])
    below = st.number_input("Neighbors below", 0, 5, CFG["retrieval"]["below"])

note = st.text_area("Visit note", height=160, value="3-year-old with barky cough, hoarse voice, inspiratory stridor. Temp 38°C. No drooling.")

# --- Load prompts
planner_system = load_prompt(CFG["prompts"]["planner_system"])
answer_system = load_prompt(CFG["prompts"]["answer_system"])
answer_user_tmpl = load_prompt(CFG["prompts"]["answer_user_template"])

# --- Ensure store exists? (optional: call ingest on first run)
st.write(f"Vector DB: {DB_PATH}")
if not pathlib.Path(DB_PATH).exists():
    st.warning("Vector store path not found. Ingest first via scripts/ingest_pdf.py")

if st.button("Generate plan", type="primary", use_container_width=True):
    rag = RAGClient(
        db_path=DB_PATH,
        collection=CFG["retrieval"]["collection"],
        embed_model=CFG["retrieval"]["embed_model"],
    )
    queries = plan_queries(note, planner_system, llm_helper.openai_chat)
    st.code(json.dumps({"queries": queries}, indent=2))

    windows, seen = [], set()
    for q in queries:
        hits = rag.query(q, doc_id=doc_id, k=k)
        for h in hits:
            key = (h["metadata"]["doc_id"], h["metadata"]["window_id"])
            if key in seen: continue
            seen.add(key)
            stitched = rag.stitch_neighbors(h, above=above, below=below, cap_words=CFG["retrieval"]["cap_words"])
            if stitched:
                windows.append({"text": stitched["text"], "pages": stitched["pages"], "q": q, "dist": h["distance"]})
    windows.sort(key=lambda x: x["dist"])
    if not windows:
        st.error("No context retrieved.")
    else:
        context = "\n\n---\n\n".join([w["text"] for w in windows][:3])
        pages = windows[0]["pages"]
        plan = answer_with_context(note, context, pages, answer_system, answer_user_tmpl, llm_helper.openai_chat)
        st.markdown("### Management Plan")
        st.write(plan)
