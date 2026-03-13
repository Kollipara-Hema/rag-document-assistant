"""
Main pipeline orchestration for the modular RAG learning lab.

This file connects loading, chunking, embedding, indexing, retrieval,
and answer generation into one callable workflow.
"""

from typing import Dict, Any, List

from src.config import (
    CHROMA_DB_DIR,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
)
from src.registry import LOADERS, CHUNKERS, EMBEDDERS, RETRIEVERS, GENERATORS
from src.vectordb.chroma_store import build_chroma_store


def format_context(retrieved_docs) -> str:
    """
    Format retrieved chunks into a context block for the generator.
    """
    context_parts = []

    for doc in retrieved_docs:
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "NA")
        text = doc.page_content.strip().replace("\n", " ")

        context_parts.append(f"[{source}:{page}] {text}")

    return "\n\n".join(context_parts)


def run_rag_pipeline(
    question: str,
    loader_name: str,
    chunker_name: str,
    embedder_name: str,
    retriever_name: str,
    generator_name: str,
    top_k: int,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> Dict[str, Any]:
    """
    Run the full RAG pipeline using selected modular components.
    """
    # Load raw documents from the selected source.
    loader_fn = LOADERS[loader_name]
    documents = loader_fn()

    # Split documents using the selected chunking strategy.
    chunker_fn = CHUNKERS[chunker_name]
    chunks = chunker_fn(
        documents=documents,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    # Create the selected embedding model.
    embedder_fn = EMBEDDERS[embedder_name]
    embedding_model = embedder_fn()

    # Build the Chroma store for the current chunk set.
    vector_store = build_chroma_store(
        documents=chunks,
        embedding_function=embedding_model,
        persist_directory=CHROMA_DB_DIR,
    )

    # Retrieve the most relevant chunks for the user question.
    retriever_fn = RETRIEVERS[retriever_name]
    retrieved_docs = retriever_fn(
        query=question,
        vector_store=vector_store,
        top_k=top_k,
    )

    # Convert retrieved chunks into a context block for generation.
    context = format_context(retrieved_docs)

    # Generate the final answer using the selected LLM backend.
    generator_fn = GENERATORS[generator_name]
    answer = generator_fn(question=question, context=context)

    # Return both the answer and pipeline metadata for display.
    return {
        "answer": answer,
        "retrieved_docs": retrieved_docs,
        "pipeline_summary": {
            "Loader": loader_name,
            "Chunker": chunker_name,
            "Embedder": embedder_name,
            "Vector DB": "Chroma",
            "Retriever": retriever_name,
            "Generator": generator_name,
        },
        "stats": {
            "documents_loaded": len(documents),
            "chunks_created": len(chunks),
            "chunks_retrieved": len(retrieved_docs),
        },
    }
