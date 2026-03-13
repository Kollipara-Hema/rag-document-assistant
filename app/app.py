import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

import streamlit as st

from src.rag import answer_question
from src.upload_ingest import build_upload_index

st.set_page_config(page_title="RAG Document Assistant", layout="wide")

st.title("LLM-Powered Document Assistant")
st.markdown(
    "Ask questions over your repository documents, uploaded documents, or both."
)

# Sidebar
st.sidebar.header("Settings")
k = st.sidebar.slider("Top-K retrieved chunks", 3, 10, 5)
st.sidebar.info("Model: llama3.1 (local via Ollama)")

mode_label = st.sidebar.radio(
    "Choose document source",
    ["Repository only", "Uploaded only", "Both"],
    index=0,
)

mode_map = {
    "Repository only": "repository",
    "Uploaded only": "uploaded",
    "Both": "both",
}
mode = mode_map[mode_label]

uploaded_files = st.sidebar.file_uploader(
    "Upload documents",
    type=["pdf", "html", "htm", "txt", "csv", "json", "docx", "rtf"],
    accept_multiple_files=True,
)

if "uploads_ready" not in st.session_state:
    st.session_state["uploads_ready"] = False

if st.sidebar.button("Process uploaded documents"):
    if uploaded_files:
        with st.spinner("Indexing uploaded documents..."):
            chunk_count = build_upload_index(uploaded_files)
            st.session_state["uploads_ready"] = chunk_count > 0

        if chunk_count > 0:
            st.sidebar.success(f"Uploaded documents indexed successfully. Chunks created: {chunk_count}")
        else:
            st.sidebar.warning("No supported content could be extracted from uploaded files.")
    else:
        st.sidebar.warning("Please upload at least one file first.")

query = st.text_input(
    "Ask a question:",
    value="What is Poisson PCA and why is it used for count data?",
)

if st.button("Generate Answer"):
    if mode in {"uploaded", "both"} and not st.session_state.get("uploads_ready", False):
        st.warning("You selected uploaded documents, but no uploaded documents have been processed yet.")
    else:
        with st.spinner("Retrieving documents and generating answer..."):
            result = answer_question(query, k=k, mode=mode)

        st.subheader("Answer")
        st.write(result["answer"])

        st.subheader("Sources")
        if result["sources"]:
            for src in result["sources"]:
                st.write(
                    f"• [{src['scope']}] {src['source']} — Page {src['page']} ({src['file_type']})"
                )
        else:
            st.info("No sources retrieved.")
