# app.py
import os
from pathlib import Path
import streamlit as st

# ---- Env (.env at project root) ----
try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=Path(".") / ".env")
except Exception:
    pass

# ---- Import helpers from your script ----
from scripts.llm_rag_demo import (
    RAGClient,
    plan_queries,
    retrieve_context,
    answer_with_context,
    load_prompts_versioned,
    DOC_ID,
    # dosing helpers
    retrieve_dosing_context,
    build_dosing_table,
    generate_dosing_queries_from_plan,  # LLM-derived queries from plan
)

# ---- Streamlit page ----
st.set_page_config(page_title="Heidi CDS (PoC)", layout="centered")
st.title("Clinical Decision Support (Tool)")
st.caption("""
This app uses a Retrieval-Augmented Generation (RAG) approach to provide clinical decision support. 
It uses a sliding window approach to maintain context. Control the number of chunks with k-hyperparameter.
The context is stitched together with the 
""")

DB_PATH = str(Path.cwd() / "chroma_db")

# ---- Session state init ----
def _init_state():
    defaults = {
        "plan_ready": False,
        "plan_md": "",
        "ctx": None,
        "queries": [],
        "doc_id_used": "",
        "error": "",
        "dosing_table_md": "",
        "dosing_queries": [],
    }
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)

_init_state()

# ---- Utilities ----
@st.cache_data(show_spinner=False)
def get_doc_ids(db_path: str):
    """Return all doc_ids in the vector store (cached)."""
    try:
        rag_tmp = RAGClient(db_path=db_path)
        return rag_tmp.list_doc_ids()
    except Exception:
        return []

def clear_plan_state():
    st.session_state["plan_ready"] = False
    st.session_state["plan_md"] = ""
    st.session_state["ctx"] = None
    st.session_state["queries"] = []
    st.session_state["doc_id_used"] = ""
    st.session_state["error"] = ""
    st.session_state["dosing_table_md"] = ""
    st.session_state["dosing_queries"] = []

# ---- Quick dose calculator (optional max dose, no upper limits) ----
def render_dose_calculator_form():
    """Quick dose calculator as a FORM (optional max dose, no upper limits). Persists last result."""
    st.divider()
    st.markdown("### Quick dose calculator")

    dkey = "dose_state"
    if dkey not in st.session_state:
        st.session_state[dkey] = {
            "weight": 12.0,
            "mgkg": 0.6,
            "drug": "drug name",     # default label
            "use_max": False,        # optional max-dose switch
            "max_mg": 0.0,           # max dose (mg); <=0 means ignored
            "use_round": True,
            "round_inc": 0.5,
            "exact": None,
            "limited": None,         # after applying max dose (if any)
            "final": None,           # after rounding (if any)
        }

    def _round_to_inc(value: float, inc: float) -> float:
        if inc is None or inc <= 0:
            return value
        return round(value / inc) * inc

    with st.form("dose_calc_form", clear_on_submit=False):
        colA, colB, colC = st.columns([1, 1, 1])
        with colA:
            weight_kg = st.number_input(
                "Patient weight (kg)",
                min_value=0.0, step=0.1,
                value=float(st.session_state[dkey]["weight"]),
                key="form_weight",
            )
        with colB:
            mg_per_kg = st.number_input(
                "Recommended dose (mg/kg)",
                min_value=0.0, step=0.01,
                value=float(st.session_state[dkey]["mgkg"]),
                key="form_mgkg",
            )
        with colC:
            drug_name = st.text_input(
                "Medication (label only)",
                value=st.session_state[dkey]["drug"],
                key="form_drug",
            )

        c1, c2 = st.columns([1, 1])
        with c1:
            use_max = st.checkbox(
                "Apply max dose (mg)?",
                value=st.session_state[dkey]["use_max"],
                key="form_max_on",
            )
            # Always enabled so you can edit it before submit
            max_mg = st.number_input(
                "Max dose (mg)",
                min_value=0.0, step=0.1,
                value=float(st.session_state[dkey]["max_mg"]),
                key="form_max",
                help="Applied only if the checkbox above is ticked.",
            )
        with c2:
            use_round = st.checkbox(
                "Round to increment (mg)?",
                value=st.session_state[dkey]["use_round"],
                key="form_round_on",
            )
            round_inc = st.number_input(
                "Rounding increment (mg)",
                min_value=0.0, step=0.1,
                value=float(st.session_state[dkey]["round_inc"]),
                key="form_round",
                help="If 0, rounding is skipped.",
            )

        submitted = st.form_submit_button("Calculate dose")

    if submitted:
        # Store inputs first
        st.session_state[dkey].update({
            "weight": weight_kg, "mgkg": mg_per_kg, "drug": drug_name,
            "use_max": use_max, "max_mg": max_mg,
            "use_round": use_round, "round_inc": round_inc,
        })

        if weight_kg <= 0 or mg_per_kg <= 0:
            st.info("Enter a positive weight and mg/kg rule to calculate a dose.")
            st.session_state[dkey].update({"exact": None, "limited": None, "final": None})
        else:
            exact = weight_kg * mg_per_kg
            # Apply optional max dose (if set and > 0)
            limited = min(exact, max_mg) if (use_max and max_mg > 0) else exact
            # Apply optional rounding (after max-dose)
            final = _round_to_inc(limited, round_inc) if (use_round and round_inc > 0) else limited

            st.session_state[dkey].update({
                "exact": exact,
                "limited": limited,
                "final": final,
            })

    ds = st.session_state[dkey]
    if ds["exact"] is not None:
        st.write("**Calculation**")
        lines = [
            "Weight × mg/kg = exact dose",
            f"{ds['weight']:.2f} kg × {ds['mgkg']:.4g} mg/kg = {ds['exact']:.2f} mg",
        ]
        if ds["use_max"] and ds["max_mg"] > 0:
            if ds["exact"] > ds["max_mg"]:
                lines.append(f"Apply max dose: min({ds['exact']:.2f}, {ds['max_mg']:.2f}) = {ds['limited']:.2f} mg")
            else:
                lines.append(f"Apply max dose: exact does not exceed {ds['max_mg']:.2f} mg → {ds['limited']:.2f} mg")
        if ds["use_round"]:
            if ds["round_inc"] > 0:
                lines.append(f"Round to {ds['round_inc']:.4g} mg → {ds['final']:.2f} mg")
            else:
                lines.append("Rounding disabled (increment ≤ 0).")
        st.code("\n".join(lines))

        st.success(f"**Administered dose: {ds['final']:.2f} mg** of {ds['drug']}")

        # Copyable block
        detail_steps = []
        if ds["use_max"] and ds["max_mg"] > 0:
            detail_steps.append(f"max {ds['max_mg']:.2f} mg")
        if ds["use_round"] and ds["round_inc"] > 0:
            detail_steps.append(f"round {ds['round_inc']:.4g} mg")
        detail = f" ({' • '.join(detail_steps)})" if detail_steps else ""

        dose_block_md = (
            f"- Drug: **{ds['drug']}**\n"
            f"- Weight: **{ds['weight']:.2f} kg**\n"
            f"- Rule: **{ds['mgkg']:.4g} mg/kg**\n"
            f"- Max dose applied: **{'Yes' if (ds['use_max'] and ds['max_mg'] > 0) else 'No'}**\n"
            f"- Result: **{ds['final']:.2f} mg**{detail}"
        )
        with st.expander("Copy dose block"):
            st.markdown(dose_block_md)

        if ds["final"] > 200:
            st.caption("⚠️ Consider double-checking unusually large doses against the guideline.")
    else:
        st.info("Enter values and click **Calculate dose** to see the result.")

# ---- Sidebar ----
with st.sidebar:
    st.subheader("Settings")

    if st.button("🔄 Refresh doc list"):
        get_doc_ids.clear()

    all_doc_ids = get_doc_ids(DB_PATH)
    if not all_doc_ids:
        st.warning(f"No doc_ids found in {DB_PATH}. Make sure you've ingested a PDF.")
        doc_id = DOC_ID
    else:
        default_index = all_doc_ids.index(DOC_ID) if DOC_ID in all_doc_ids else 0
        doc_id = st.selectbox("Guideline Document", all_doc_ids, index=default_index)

    # Retrieval hyperparams
    k = st.slider("Top-k (number of chunks retrieved)", 1, 10, 4)
    above = st.number_input("Neighbors above", min_value=0, max_value=3, value=1, step=1)
    below = st.number_input("Neighbors below", min_value=0, max_value=4, value=2, step=1)

    # Prompts (versioned)
    st.markdown("### Prompts")
    prompts_base = st.text_input("Prompts base dir", "./prompts")
    prompts_version = st.text_input("Version (blank = latest or 'current')", "")
    show_context = st.checkbox("Show retrieved context", value=True)

    # API env
    api_ok = bool(os.getenv("OPENAI_API_KEY"))
    st.write("✅ OPENAI_API_KEY loaded" if api_ok else "⚠️ OPENAI_API_KEY missing")

# ---- Main inputs ----
demo_note = (
"""Patient: Jack T.
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
- Plan as per local guidelines for croup"""
)
note = st.text_area("Visit note / transcript", value=demo_note, height=200, key="note_input")

col1, col2 = st.columns([1, 1])
with col1:
    run_btn = st.button("Generate plan", type="primary")
with col2:
    clear_btn = st.button("Clear")

if clear_btn:
    clear_plan_state()
    st.rerun()

# ---- Run pipeline (store results in session) ----
if run_btn:
    if not os.getenv("OPENAI_API_KEY"):
        st.session_state.error = "OPENAI_API_KEY not set (put it in .env or export it)."
        st.session_state.plan_ready = False
        st.rerun()
    if not note.strip():
        st.session_state.error = "Please enter a visit note."
        st.session_state.plan_ready = False
        st.rerun()

    try:
        prompts = load_prompts_versioned(Path(prompts_base), prompts_version or None)
    except Exception as e:
        st.session_state.error = f"Failed to load prompts: {e}"
        st.session_state.plan_ready = False
        st.rerun()

    rag = RAGClient(db_path=DB_PATH)

    with st.spinner("Planning queries…"):
        queries = plan_queries(note, prompts)
    with st.spinner("Retrieving guideline context…"):
        ctx = retrieve_context(rag, queries, doc_id=doc_id, k=k, above=above, below=below)
    if not ctx["context"]:
        st.session_state.error = f"No context found. Tried queries: {queries}"
        st.session_state.plan_ready = False
        st.rerun()

    with st.spinner("Drafting management plan…"):
        plan_md = answer_with_context(note, ctx["context"], ctx["pages"], prompts)

    # Persist
    st.session_state.plan_ready = True
    st.session_state.plan_md = plan_md
    st.session_state.ctx = ctx
    st.session_state.queries = queries
    st.session_state.doc_id_used = doc_id
    st.session_state.error = ""
    st.session_state.dosing_table_md = ""
    st.session_state.dosing_queries = []
    st.rerun()

# ---- Render from session (stable across reruns) ----
if st.session_state.error:
    st.error(st.session_state.error)

if st.session_state.plan_ready and st.session_state.ctx:
    ctx = st.session_state.ctx

    st.markdown("**Queries used:**")
    for q in st.session_state.queries:
        st.markdown(f"- {q}")

    if show_context:
        st.subheader("Retrieved context (stitched)")
        preview = ctx["context"][:4000] + ("..." if len(ctx["context"]) > 4000 else "")
        st.code(preview)
        st.caption(f"Source: {st.session_state.doc_id_used}, p. {ctx['pages'][0]}–{ctx['pages'][1]}")

    st.subheader("Management plan")
    st.markdown(st.session_state.plan_md)
    st.caption("Citations included at the end of the response.")

    # --- Generate dosing table (placed under the plan) ---
    st.divider()
    st.caption("Uses the management plan to derive dosing queries, retrieves dosing text from the guideline, then renders a table.")
    gen_table_here = st.button("Generate dosing table (from plan + guideline)", type="primary")

    if gen_table_here:
        try:
            prompts_for_table = load_prompts_versioned(Path(prompts_base), prompts_version or None)

            # Derive dosing-specific queries FROM THE PLAN (LLM-driven; versioned prompts)
            dosing_queries = generate_dosing_queries_from_plan(st.session_state.plan_md, prompts_for_table)
            st.session_state["dosing_queries"] = dosing_queries

            # Retrieve dosing context from the same document (tune k/above/below if needed)
            dctx = retrieve_dosing_context(
                RAGClient(db_path=DB_PATH),
                dosing_queries,
                doc_id=st.session_state.doc_id_used,
                k=6, above=1, below=2
            )

            if dctx["context"]:
                table_md = build_dosing_table(
                    st.session_state.plan_md, dctx["context"], dctx["pages"], prompts_for_table
                )
                st.session_state.dosing_table_md = table_md or ""
            else:
                st.session_state.dosing_table_md = ""
                st.info("No dosing context retrieved from the guideline for the current plan.")
        except Exception as e:
            st.info(f"Dosing table generation failed: {e}")

    if st.session_state.get("dosing_queries"):
        st.markdown("**Dosing queries (from plan):**")
        for q in st.session_state.dosing_queries:
            st.markdown(f"- {q}")

    if st.session_state.dosing_table_md:
        st.markdown(st.session_state.dosing_table_md)

    # ---- Calculator appears BELOW the plan (and below table if present) ----
    render_dose_calculator_form()

with st.expander("Debug: vector store contents on server"):
    try:
        rag_dbg = RAGClient(db_path="./chroma_db")
        doc_ids_dbg = rag_dbg.list_doc_ids()
        st.write(f"Found {len(doc_ids_dbg)} doc_ids:")
        for d in doc_ids_dbg:
            st.code(d)
    except Exception as e:
        st.error(f"Failed to inspect vector store: {e}")

with st.expander("🔎 Retrieval diagnostics"):
    try:
        rag_dbg = RAGClient(db_path=DB_PATH)
        # Count windows per doc_id on the server
        metas = rag_dbg.coll.get(include=["metadatas"]).get("metadatas") or []
        counts = {}
        for m in metas:
            d = m.get("doc_id")
            if d: counts[d] = counts.get(d, 0) + 1
        st.write("Doc IDs & window counts on server:")
        for k,v in sorted(counts.items()):
            st.write(f"- {k}: {v} windows")

        st.write("Selected Document ID:", st.session_state.get("doc_id_used") or "(none)")
        st.write("Queries last used:", st.session_state.get("queries", []))
    except Exception as e:
        st.error(f"Diagnostics failed: {e}")


