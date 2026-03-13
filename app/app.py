"""
Streamlit interface for the modular RAG learning lab.

This app lets users choose key pipeline components and see both
the generated answer and the selected RAG pipeline.
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

import streamlit as st

from src.config import DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE, DEFAULT_TOP_K
from src.pipeline import run_rag_pipeline
from src.registry import LOADERS, CHUNKERS, EMBEDDERS, RETRIEVERS, GENERATORS

st.set_page_config(page_title="RAG Learning Lab", layout="wide")

st.title("RAG Learning Lab")
st.markdown(
    """
This app demonstrates a modular Retrieval-Augmented Generation pipeline.

You can choose pipeline components from the sidebar and see how the system
loads documents, chunks them, embeds them, retrieves relevant context,
and generates an answer.
"""
)

# Sidebar controls let the user configure the current RAG pipeline.
st.sidebar.header("Pipeline Controls")

loader_name = st.sidebar.selectbox("Document Source", list(LOADERS.keys()))
chunker_name = st.sidebar.selectbox("Chunking Strategy", list(CHUNKERS.keys()))
embedder_name = st.sidebar.selectbox("Embedding Model", list(EMBEDDERS.keys()))
retriever_name = st.sidebar.selectbox("Retriever", list(RETRIEVERS.keys()))
generator_name = st.sidebar.selectbox("LLM Generator", list(GENERATORS.keys()))

top_k = st.sidebar.slider("Top-K Retrieved Chunks", 1, 10, DEFAULT_TOP_K)
chunk_size = st.sidebar.slider("Chunk Size", 300, 2000, DEFAULT_CHUNK_SIZE, step=100)
chunk_overlap = st.sidebar.slider("Chunk Overlap", 0, 500, DEFAULT_CHUNK_OVERLAP, step=25)

question = st.text_input(
    "Ask a question about the indexed documents:",
    value="What is retrieval-augmented generation?",
)

if st.button("Run Pipeline"):
    with st.spinner("Running the selected RAG pipeline..."):
        result = run_rag_pipeline(
            question=question,
            loader_name=loader_name,
            chunker_name=chunker_name,
            embedder_name=embedder_name,
            retriever_name=retriever_name,
            generator_name=generator_name,
            top_k=top_k,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    st.subheader("Answer")
    st.write(result["answer"])

    st.subheader("Pipeline Summary")

    # Text summary of the selected pipeline.
    for key, value in result["pipeline_summary"].items():
        st.write(f"**{key}:** {value}")

    # Visual flow of the selected pipeline.
    pipeline_flow = " → ".join(result["pipeline_summary"].values())
    st.info(f"Pipeline Flow: {pipeline_flow}")

    st.subheader("Pipeline Statistics")
    for key, value in result["stats"].items():
        st.write(f"**{key}:** {value}")

    st.subheader("Retrieved Chunks")
    for index, doc in enumerate(result["retrieved_docs"], start=1):
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "NA")

        with st.expander(f"Chunk {index} | Source: {source} | Page: {page}"):
            st.write(doc.page_content[:1500])
