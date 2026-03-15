"""
Main pipeline orchestration for the modular RAG learning lab.

Step 3 separates the workflow into two parts:

1. Build / refresh the index
2. Ask many questions using the existing index

This is closer to how real RAG systems work in practice.
"""

from typing import Dict, Any, List

from src.config import (
    CHROMA_DB_DIR,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
)
from src.registry import LOADERS, CHUNKERS, EMBEDDERS, RETRIEVERS, GENERATORS
from src.vectordb.chroma_store import (
    build_chroma_store,
    load_chroma_store,
    chroma_index_exists,
)


def format_context(retrieved_docs) -> str:
    """
    Convert retrieved documents into a single context string for the LLM.
    """
    context_parts = []

    for doc in retrieved_docs:
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "NA")
        text = doc.page_content.strip().replace("\n", " ")

        context_parts.append(f"[{source}:{page}] {text}")

    return "\n\n".join(context_parts)


def build_index(
    loader_name: str,
    chunker_name: str,
    embedder_name: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> Dict[str, Any]:
    """
    Build the document index once and persist it to Chroma.

    This function is intentionally separate from query answering so that
    repeated user questions do not trigger repeated ingestion and indexing.
    """
    # Load the selected document source.
    loader_fn = LOADERS[loader_name]
    documents = loader_fn()

    # Split the loaded documents into chunks.
    chunker_fn = CHUNKERS[chunker_name]
    chunks = chunker_fn(
        documents=documents,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    # Create the selected embedding model.
    embedder_fn = EMBEDDERS[embedder_name]
    embedding_model = embedder_fn()

    # Build and persist the Chroma vector store.
    build_chroma_store(
        documents=chunks,
        embedding_function=embedding_model,
        persist_directory=CHROMA_DB_DIR,
    )

    return {
        "pipeline_summary": {
            "Loader": loader_name,
            "Chunker": chunker_name,
            "Embedder": embedder_name,
            "Vector DB": "Chroma",
        },
        "stats": {
            "documents_loaded": len(documents),
            "chunks_created": len(chunks),
            "index_path": str(CHROMA_DB_DIR),
        },
    }


def answer_with_index(
    question: str,
    embedder_name: str,
    retriever_name: str,
    generator_name: str,
    top_k: int,
) -> Dict[str, Any]:
    """
    Answer a question using an already-built Chroma index.
    """
    if not chroma_index_exists(CHROMA_DB_DIR):
        raise FileNotFoundError(
            "No Chroma index found. Please build the index before asking questions."
        )

    # Recreate the embedding model so the query can be embedded consistently.
    embedder_fn = EMBEDDERS[embedder_name]
    embedding_model = embedder_fn()

    # Load the existing vector store from disk.
    vector_store = load_chroma_store(
        embedding_function=embedding_model,
        persist_directory=CHROMA_DB_DIR,
    )

    # Retrieve relevant chunks from the saved index.
    retriever_fn = RETRIEVERS[retriever_name]
    retrieved_docs = retriever_fn(
        query=question,
        vector_store=vector_store,
        top_k=top_k,
        fetch_k=max(top_k * 3, 10),
    )


    # Format retrieved chunks for the generator.
    context = format_context(retrieved_docs)

    # Generate the final answer.
    generator_fn = GENERATORS[generator_name]
    answer = generator_fn(question=question, context=context)

    return {
        "answer": answer,
        "retrieved_docs": retrieved_docs,
        "query_summary": {
            "Embedder": embedder_name,
            "Retriever": retriever_name,
            "Generator": generator_name,
            "Top-K": top_k,
        },
        "stats": {
            "chunks_retrieved": len(retrieved_docs),
        },
    }
