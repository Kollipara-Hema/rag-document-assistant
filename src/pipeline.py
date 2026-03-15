"""
Main pipeline orchestration for the modular RAG learning lab.
"""

from typing import Dict, Any

from src.config import (
    CHROMA_DB_LOCAL_DIR,
    CHROMA_DB_UPLOAD_DIR,
    CHROMA_DB_WEB_DIR,
    CHROMA_DB_GITHUB_DIR,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
)
from src.registry import LOADERS, CHUNKERS, EMBEDDERS, RETRIEVERS, GENERATORS
from src.vectordb.chroma_store import (
    build_chroma_store,
    load_chroma_store,
    chroma_index_exists,
)


def get_chroma_dir_for_loader(loader_name: str):
    """
    Return a dedicated Chroma directory for each document source.
    """
    mapping = {
        "Local Repository": CHROMA_DB_LOCAL_DIR,
        "Uploaded Files": CHROMA_DB_UPLOAD_DIR,
        "Web Page": CHROMA_DB_WEB_DIR,
        "GitHub Repository": CHROMA_DB_GITHUB_DIR,
    }
    return mapping[loader_name]


def format_context(retrieved_docs) -> str:
    """
    Convert retrieved documents into a context string for the LLM.
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
    source_input=None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> Dict[str, Any]:
    """
    Build the document index once and persist it to Chroma.
    """
    loader_fn = LOADERS[loader_name]

    if loader_name == "Local Repository":
        documents = loader_fn()
    else:
        documents = loader_fn(source_input)

    chunker_fn = CHUNKERS[chunker_name]
    chunks = chunker_fn(
        documents=documents,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    embedder_fn = EMBEDDERS[embedder_name]
    embedding_model = embedder_fn()

    target_chroma_dir = get_chroma_dir_for_loader(loader_name)

    build_chroma_store(
        documents=chunks,
        embedding_function=embedding_model,
        persist_directory=target_chroma_dir,
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
            "index_path": str(target_chroma_dir),
        },
    }


def answer_with_index(
    question: str,
    loader_name: str,
    chunker_name: str,
    embedder_name: str,
    retriever_name: str,
    generator_name: str,
    top_k: int,
    source_input=None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> Dict[str, Any]:
    """
    Answer a question using dense, sparse, or hybrid retrieval.
    """
    retriever_fn = RETRIEVERS[retriever_name]
    target_chroma_dir = get_chroma_dir_for_loader(loader_name)

    needs_chunk_documents = retriever_name in {"Sparse", "Hybrid"}

    chunks = None
    if needs_chunk_documents:
        loader_fn = LOADERS[loader_name]
        if loader_name == "Local Repository":
            documents = loader_fn()
        else:
            documents = loader_fn(source_input)

        chunker_fn = CHUNKERS[chunker_name]
        chunks = chunker_fn(
            documents=documents,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    needs_vector_store = retriever_name in {"Dense", "Hybrid"}

    vector_store = None
    if needs_vector_store:
        if not chroma_index_exists(target_chroma_dir):
            raise FileNotFoundError(
                "No Chroma index found for the selected source. Please build the index first."
            )

        embedder_fn = EMBEDDERS[embedder_name]
        embedding_model = embedder_fn()

        vector_store = load_chroma_store(
            embedding_function=embedding_model,
            persist_directory=target_chroma_dir,
        )

    if retriever_name == "Dense":
        retrieved_docs = retriever_fn(
            query=question,
            vector_store=vector_store,
            top_k=top_k,
            fetch_k=max(top_k * 3, 10),
        )

    elif retriever_name == "Sparse":
        retrieved_docs = retriever_fn(
            query=question,
            documents=chunks,
            top_k=top_k,
        )

    elif retriever_name == "Hybrid":
        retrieved_docs = retriever_fn(
            query=question,
            vector_store=vector_store,
            documents=chunks,
            top_k=top_k,
            dense_fetch_k=max(top_k * 3, 10),
            sparse_fetch_k=max(top_k * 3, 10),
        )

    else:
        raise ValueError(f"Unsupported retriever: {retriever_name}")

    context = format_context(retrieved_docs)

    generator_fn = GENERATORS[generator_name]
    answer = generator_fn(question=question, context=context)

    return {
        "answer": answer,
        "retrieved_docs": retrieved_docs,
        "query_summary": {
            "Loader": loader_name,
            "Chunker": chunker_name,
            "Embedder": embedder_name,
            "Retriever": retriever_name,
            "Generator": generator_name,
            "Top-K": top_k,
        },
        "stats": {
            "chunks_retrieved": len(retrieved_docs),
        },
    }
