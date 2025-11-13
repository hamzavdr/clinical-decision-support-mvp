import os, json, yaml, pathlib, streamlit as st
from scripts.ingest_pdf import ingest_pdf
from scripts.llm_helper import configure as configure_llm, openai_chat
from scripts.rag_client import RAGClient, load_prompt, plan_queries, answer_with_context
import os, pathlib
import streamlit as st
from scripts.ingest_pdf import ingest_pdf
from scripts.rag_client import RAGClient
from chromadb.config import Settings
import chromadb
from dotenv import load_dotenv

# --- Load config
load_dotenv()
CFG = yaml.safe_load(open("./config/config.yaml", "r"))

# Check for OpenAI API key
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    st.error("Error: OPENAI_API_KEY not found in environment variables. "
             "Please ensure it's set correctly in your .env file or as a system environment variable.")
    st.stop() # Stop the app if the key is missing

DOC_ID = CFG["app"]["default_doc_id"]
DB_PATH = os.getenv("CHROMA_DB_DIR", CFG["retrieval"]["db_path"])

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

# --- UI
st.set_page_config(page_title=CFG["app"]["title"], layout="wide")
st.title(CFG["app"]["title"])
st.caption("Develop branch demo")

with st.sidebar:
    st.header("Database Settings")
    # --- Ensure store exists? (optional: call ingest on first run)
    st.write(f"Vector DB: {DB_PATH}")
    if not pathlib.Path(DB_PATH).exists():
        st.warning("Vector store path not found. Ingest first via scripts/ingest_pdf.py")
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

#     run_vec = st.button("Vectorise", type="primary", use_container_width=True)
#     if run_vec:
#         # Validate paths
#         if not pathlib.Path(pdf_path).exists():
#             st.error(f"PDF not found at: {pdf_path}")
#         else:
#             pathlib.Path(db_path).mkdir(parents=True, exist_ok=True)
#             with st.spinner("Ingesting PDF and building embeddings (CPU)…"):
#                 try:
#                     # Uses SentenceTransformers on CPU (as in your ingest script)
#                     ingest_pdf(
#                         pdf_path=pdf_path,
#                         doc_id=doc_id,
#                         db_path=db_path,
#                         collection="guidelines",
#                         window_chars=window_chars,
#                         overlap_chars=overlap_chars,
#                     )
#                     after = _chroma_count(db_path)
#                     if after >= 0:
#                         st.success(f"Vectorisation complete. Collection now has **{after}** items.")
#                     else:
#                         st.success("Vectorisation complete.")
#                 except Exception as e:
#                     st.exception(e)

if st.button("Generate plan", type="primary", use_container_width=True):
    st.write("---")
    st.subheader("🔍 Debug Info")
    
    # Show what we're querying with
    st.write(f"**Doc ID being queried:** `{doc_id}`")
    st.write(f"**DB Path:** `{DB_PATH}`")
    st.write(f"**Collection:** `{CFG['retrieval']['collection']}`")
    
    # Initialize RAG client
    rag = RAGClient(
        db_path=DB_PATH,
        collection=CFG["retrieval"]["collection"],
        embed_model=CFG["retrieval"]["embed_model"],
    )
    
    # Generate queries
    queries = plan_queries(note, planner_system, openai_chat)
    st.markdown("### Queries generated")
    st.code(json.dumps({"queries": queries}, indent=2))

    # Retrieve windows
    windows, seen = [], set()
    for q in queries:
        st.write(f"Querying: `{q}`")
        hits = rag.query(q, doc_id=doc_id, k=k)
        st.write(f"  → Got {len(hits)} hits")
        
        for h in hits:
            key = (h["metadata"]["doc_id"], h["metadata"]["window_id"])
            if key in seen: 
                st.write(f"  → Skipping duplicate window {key}")
                continue
            seen.add(key)
            stitched = rag.stitch_neighbors(h, above=above, below=below, cap_words=CFG["retrieval"]["cap_words"])
            if stitched:
                windows.append({"text": stitched["text"], "pages": stitched["pages"], "q": q, "dist": h["distance"]})
    
    st.write(f"**Total unique windows retrieved:** {len(windows)}")
    
    # Sort and generate answer
    windows.sort(key=lambda x: x["dist"])
    if not windows:
        st.error("❌ No context retrieved.")
        st.info("Check the terminal/console for debug logs to see what's happening.")
    else:
        # Build and display the context that will be sent to LLM
        context = "\n\n---\n\n".join([w["text"] for w in windows][:3])
        pages = windows[0]["pages"]
        
        # Display retrieved context
        st.markdown("---")
        st.markdown("### 📄 Retrieved Context")
        st.caption(f"Showing top 3 windows | Pages: {pages[0]}-{pages[1]}")
        
        # Show each window separately
        for i, w in enumerate(windows[:3], 1):
            with st.expander(f"Window {i} - Pages {w['pages'][0]}-{w['pages'][1]} (Distance: {w['dist']:.4f})", expanded=False):
                st.caption(f"**Matched query:** `{w['q']}`")
                st.text_area(
                    f"Window {i} content",
                    value=w["text"],
                    height=300,
                    disabled=True,
                    label_visibility="collapsed"
                )
        
        # Also show combined context
        with st.expander("View combined context (sent to LLM)", expanded=False):
            st.text_area(
                "Combined context",
                value=context,
                height=400,
                disabled=True,
                label_visibility="collapsed"
            )
        
        # Generate management plan
        st.markdown("---")
        st.markdown("### 💊 Management Plan")
        with st.spinner("Generating plan..."):
            plan = answer_with_context(note, context, pages, answer_system, answer_user_tmpl, openai_chat)
        st.write(plan)