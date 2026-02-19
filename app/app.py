import sys
from pathlib import Path

# Add project root to PYTHONPATH so `import src...` works
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

import streamlit as st
from src.rag import answer_question

st.set_page_config(
    page_title="RAG Document Assistant",
    layout="wide"
)

st.title("📄 LLM-Powered Document Assistant")
st.markdown(
    "Ask questions over your document collection using Retrieval-Augmented Generation (RAG)."
)

# Sidebar settings
st.sidebar.header("Settings")
k = st.sidebar.slider("Top-K retrieved chunks", 3, 10, 5)
temperature_note = st.sidebar.info("Model: llama3.1 (local via Ollama)")

# Main input
query = st.text_input(
    "Ask a question:",
    value="What is Poisson PCA and why is it used for count data?"
)

if st.button("Generate Answer"):
    with st.spinner("Retrieving documents and generating answer..."):
        result = answer_question(query, k=k)

    st.subheader("Answer")
    st.write(result["answer"])

    st.subheader("Sources")
    for src in result["sources"]:
        st.write(f"• {src['source']} — Page {src['page']}")