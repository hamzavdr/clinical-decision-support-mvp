# app.py
import os
from pathlib import Path
import streamlit as st

# Load env (OPENAI_API_KEY from .env at project root)
try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=Path(".") / ".env")
except Exception:
    pass

# Import your RAG/LLM helpers
from scripts.llm_rag_demo import (
    RAGClient,
    plan_queries,
    retrieve_context,
    answer_with_context,
    load_prompts_versioned,
    DOC_ID,
)

DB_PATH = "./chroma_db"

st.set_page_config(page_title="Heidi CDS (PoC)", layout="centered")
st.title("Heidi • Clinical Decision Support (PoC)")
st.caption("Demo only — not medical advice.")

# Cache doc_id list so we don't hit Chroma on every rerun
@st.cache_data(show_spinner=False)
def get_doc_ids(db_path: str):
    try:
        return RAGClient(db_path=db_path).list_doc_ids()
    except Exception:
        return []

with st.sidebar:
    st.subheader("Settings")

    # Vector store / corpus selection
    all_doc_ids = get_doc_ids(DB_PATH)
    if not all_doc_ids:
        st.warning("No doc_ids found in Chroma. Make sure you've ingested a PDF into ./chroma_db.")
        doc_id = DOC_ID
    else:
        # Preselect default if present
        default_index = all_doc_ids.index(DOC_ID) if DOC_ID in all_doc_ids else 0
        doc_id = st.selectbox("Document ID", all_doc_ids, index=default_index)

    # Retrieval hyperparams
    k = st.slider("Top-k", 1, 10, 4)
    above = st.number_input("Neighbors above", min_value=0, max_value=3, value=1, step=1)
    below = st.number_input("Neighbors below", min_value=0, max_value=4, value=2, step=1)

    # Prompts (versioned)
    st.markdown("### Prompts")
    prompts_base = st.text_input("Prompts base dir", "./prompts")
    prompts_version = st.text_input("Version (blank = latest or 'current')", "")
    show_context = st.checkbox("Show retrieved context", value=True)

    api_ok = bool(os.getenv("OPENAI_API_KEY"))
    st.write("✅ OPENAI_API_KEY loaded" if api_ok else "⚠️ OPENAI_API_KEY missing")

# Demo note input
demo_note = ("""
Patient: Jack T.
DOB: 12/03/2022
Age: 3 years
Weight: 14.2 kg

Presenting complaint:
Jack presented with a 2-day history of barky cough, hoarse voice, and low-grade fever. Symptoms worsened overnight, with increased work of breathing and stridor noted at rest this morning. No history of choking, foreign body aspiration, or recent travel. No known sick contacts outside the household. 

History:
- Onset of URTI symptoms 2 days ago, including rhinorrhoea and dry cough
- Barking cough began yesterday evening, hoarseness and intermittent inspiratory stridor overnight
- Mild fever (up to 38.4°C) controlled with paracetamol
- No cyanosis or apnoea reported
- Fully vaccinated and developmentally appropriate for age
- No history of asthma or other chronic respiratory illness
- No previous episodes of croup
- No drug allergies

Examination:
- Alert, mildly distressed, sitting upright with audible inspiratory stridor at rest
- Barky cough noted during assessment
- Mild suprasternal and intercostal recession
- RR 32, HR 124, SpO2 97% on room air, T 37.9°C
- Chest: clear air entry bilaterally, no wheeze or crackles
- ENT: mild erythema of oropharynx, no tonsillar exudate
- CVS: normal S1/S2, no murmurs
- Neurological: alert, interactive, normal tone and reflexes

Assessment:
Jack presents with classic features of moderate croup (laryngotracheobronchitis), likely viral in origin. No signs of severe respiratory distress or impending airway obstruction. No signs suggestive of bacterial tracheitis or other differentials (e.g. foreign body, epiglottitis).

Plan:
- Administer corticosteroids
- Plan as per local guidelines for croup
"""
)
note = st.text_area("Visit note / transcript", value=demo_note, height=200)

col1, col2 = st.columns([1, 1])
with col1:
    run_btn = st.button("Generate plan", type="primary")
with col2:
    clear_btn = st.button("Clear")

if clear_btn:
    st.experimental_rerun()

if run_btn:
    # Fast sanity checks
    if not os.getenv("OPENAI_API_KEY"):
        st.error("OPENAI_API_KEY not set (put it in .env or export it).")
        st.stop()
    if not note.strip():
        st.error("Please enter a visit note.")
        st.stop()

    # Load versioned prompts
    try:
        prompts = load_prompts_versioned(Path(prompts_base), prompts_version or None)
        st.caption(
            f"Using prompts: {Path(prompts_base).resolve()}/"
            f"{(prompts_version or 'latest/current')}"
        )
    except Exception as e:
        st.error(f"Failed to load prompts: {e}")
        st.stop()

    # Init retriever
    rag = RAGClient(db_path=DB_PATH)

    with st.spinner("Planning queries…"):
        queries = plan_queries(note, prompts)
    st.write("**Queries:** ", ", ".join(queries))

    with st.spinner("Retrieving guideline context…"):
        ctx = retrieve_context(
            rag, queries, doc_id=doc_id, k=k, above=above, below=below
        )

    if not ctx["context"]:
        st.error(f"No context found. Tried queries: {queries}")
        st.stop()

    if show_context:
        st.subheader("Retrieved context (stitched)")
        st.code(ctx["context"][:4000] + ("..." if len(ctx["context"]) > 4000 else ""))
        st.caption(f"Source: {doc_id}, p. {ctx['pages'][0]}–{ctx['pages'][1]}")

    with st.spinner("Drafting management plan…"):
        plan_md = answer_with_context(note, ctx["context"], ctx["pages"], prompts)

    st.subheader("Management plan")
    st.markdown(plan_md)
    st.caption("Citations included at the end of the response.")
