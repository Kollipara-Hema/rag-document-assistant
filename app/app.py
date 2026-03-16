"""
Streamlit interface for the modular RAG learning lab.

Step 3 introduces a two-stage workflow:

1. Build / refresh the vector index
2. Ask many questions using the saved index

This makes the app faster and closer to a real RAG system.
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

import streamlit as st

from src.config import DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE, DEFAULT_TOP_K
from src.pipeline import build_index, answer_with_index
from src.registry import LOADERS, CHUNKERS, EMBEDDERS, RETRIEVERS, GENERATORS
from src.vectordb.chroma_store import chroma_index_exists
from src.pipeline import get_chroma_dir_for_loader


def build_full_pipeline_flow(
    loader_name: str,
    chunker_name: str,
    embedder_name: str,
    retriever_name: str,
    generator_name: str,
) -> str:
    """
    Create a single readable pipeline flow string for display in the UI.

    This helps users understand the full end-to-end RAG path
    from document source to final answer generation.
    """
    return (
        f"{loader_name} → "
        f"{chunker_name} → "
        f"{embedder_name} → "
        f"Chroma → "
        f"{retriever_name} → "
        f"{generator_name}"
    )


st.set_page_config(page_title="RAG Learning Lab", layout="wide")

st.title("RAG Learning Lab")
st.markdown(
    """
This app demonstrates a modular Retrieval-Augmented Generation pipeline.

The workflow is now split into two phases:

1. **Build / Refresh Index**  
2. **Ask Questions from the Existing Index**

This lets users reuse the same indexed document collection for many questions.
"""
)

# Initialize session state so the UI remembers whether the index is ready.
if "index_pipeline_summary" not in st.session_state:
    st.session_state["index_pipeline_summary"] = {}

if "index_stats" not in st.session_state:
    st.session_state["index_stats"] = {}

if "active_index_path" not in st.session_state:
    st.session_state["active_index_path"] = None


# Sidebar controls
st.sidebar.header("Pipeline Controls")

loader_name = st.sidebar.selectbox("Document Source", list(LOADERS.keys()))
active_index_path = st.session_state.get("active_index_path")
st.session_state["index_ready"] = bool(active_index_path) and chroma_index_exists(Path(active_index_path))


source_input = None

if loader_name == "Uploaded Files":
    source_input = st.sidebar.file_uploader(
        "Upload documents",
        accept_multiple_files=True,
        type=["pdf", "html", "htm", "txt", "csv", "json", "docx", "rtf", "md", "yaml", "yml"],
    )

elif loader_name == "Web Page":
    source_input = st.sidebar.text_input("Enter Web Page URL")

elif loader_name == "GitHub Repository":
    source_input = st.sidebar.text_input("Enter GitHub Repository URL")

chunker_name = st.sidebar.selectbox("Chunking Strategy", list(CHUNKERS.keys()))
embedder_name = st.sidebar.selectbox("Embedding Model", list(EMBEDDERS.keys()))
retriever_name = st.sidebar.selectbox("Retriever", list(RETRIEVERS.keys()))
generator_name = st.sidebar.selectbox("LLM Generator", list(GENERATORS.keys()))

chunk_size = st.sidebar.slider("Chunk Size", 300, 2000, DEFAULT_CHUNK_SIZE, step=100)
chunk_overlap = st.sidebar.slider("Chunk Overlap", 0, 500, DEFAULT_CHUNK_OVERLAP, step=25)
top_k = st.sidebar.slider("Top-K Retrieved Chunks", 1, 10, DEFAULT_TOP_K)

st.sidebar.markdown("---")
st.sidebar.subheader("Selected Pipeline")

pipeline_flow = build_full_pipeline_flow(
    loader_name=loader_name,
    chunker_name=chunker_name,
    embedder_name=embedder_name,
    retriever_name=retriever_name,
    generator_name=generator_name,
)

st.sidebar.info(pipeline_flow)

# Show current full pipeline in main app
st.subheader("Current Pipeline Structure")
st.info(pipeline_flow)

st.markdown(
    """
**Pipeline Stages**

1. **Document Source** → where the documents come from  
2. **Chunking** → how documents are split into smaller pieces  
3. **Embedding Model** → how text is converted into vectors  
4. **Vector Database** → where embeddings are stored  
5. **Retriever** → how relevant chunks are selected  
6. **LLM Generator** → how the final answer is produced
"""
)

# Build / refresh index section
st.header("Step 1: Build / Refresh Index")

st.markdown(
    """
Use the selected loader, chunker, and embedder to build the vector index.
This only needs to be done when the document collection or indexing settings change.
"""
)

if st.button("Build / Refresh Index"):
    if loader_name == "Uploaded Files" and not source_input:
        st.error("Please upload at least one file.")
    elif loader_name == "Web Page" and not source_input:
        st.error("Please enter a web page URL.")
    elif loader_name == "GitHub Repository" and not source_input:
        st.error("Please enter a GitHub repository URL.")
    else:
        with st.spinner("Building document index..."):
            result = build_index(
                loader_name=loader_name,
                chunker_name=chunker_name,
                embedder_name=embedder_name,
                source_input=source_input,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )

        st.session_state["index_ready"] = True
        st.session_state["index_pipeline_summary"] = result["pipeline_summary"]
        st.session_state["index_stats"] = result["stats"]
        st.session_state["active_index_path"] = result["stats"]["index_path"]


        st.success("Index built successfully.")


# Show index status
st.subheader("Index Status")

if st.session_state["index_ready"]:
    st.success("Index is ready. You can now ask multiple questions without rebuilding.")

    if st.session_state["index_pipeline_summary"]:
        st.markdown("### Index Pipeline Summary")
        for key, value in st.session_state["index_pipeline_summary"].items():
            st.write(f"**{key}:** {value}")

        index_flow = " → ".join(st.session_state["index_pipeline_summary"].values())
        st.info(f"Index Flow: {index_flow}")

    if st.session_state["index_stats"]:
        st.markdown("### Index Statistics")
        for key, value in st.session_state["index_stats"].items():
            st.write(f"**{key}:** {value}")
    st.caption("Chunk counts and retrieved chunk structure may vary depending on the selected chunking strategy."
    )
else:
    st.warning("No index found yet. Build the index before asking questions.")

st.markdown("---")

# Question answering section
st.header("Step 2: Ask Questions")

question = st.text_input(
    "Ask a question about the indexed documents:",
    value="What is retrieval-augmented generation?",
)

if st.button("Generate Answer"):
    if not st.session_state["index_ready"]:
        st.error("Please build the index first.")
    else:
        with st.spinner("Retrieving context and generating answer..."):
            result = answer_with_index(
                question=question,
                loader_name=loader_name,
                chunker_name=chunker_name,
                embedder_name=embedder_name,
                retriever_name=retriever_name,
                generator_name=generator_name,
                top_k=top_k,
                index_path=st.session_state.get("active_index_path"),
                source_input=source_input,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )

        st.subheader("Answer")
        st.write(result["answer"])

        st.subheader("Query Pipeline Summary")
        for key, value in result["query_summary"].items():
            st.write(f"**{key}:** {value}")

        st.subheader("Full Pipeline Used for This Answer")
        st.info(pipeline_flow)

        st.subheader("Query Statistics")
        for key, value in result["stats"].items():
            st.write(f"**{key}:** {value}")

       
        st.subheader("Retrieved Chunks")

        st.markdown(
            """
            These are the chunks selected by the retriever and passed to the language model.
            Use this section to understand:
            - which document the system relied on
            - which page or source it came from
            - how chunking affected the retrieved context
            """
        )

        for rank, doc in enumerate(result["retrieved_docs"], start=1):
            source = doc.metadata.get("source", "unknown")
            page = doc.metadata.get("page", "NA")
            chunk_id = doc.metadata.get("chunk_id", "NA")
            chunk_length = doc.metadata.get("chunk_length", len(doc.page_content))
            source_scope = doc.metadata.get("source_scope", "unknown")
            file_type = doc.metadata.get("file_type", "unknown")

            expander_title = (
                f"Rank {rank} | Source: {source} | Page: {page} | "
                f"Chunk ID: {chunk_id} | Length: {chunk_length}"
            )

            with st.expander(expander_title):
                st.write(f"**Source Scope:** {source_scope}")
                st.write(f"**File Type:** {file_type}")
                st.write(f"**Chunk Length:** {chunk_length}")
                st.write(doc.page_content[:2000])

