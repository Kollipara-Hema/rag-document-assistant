"""
Chroma vector store utilities.
"""

from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_chroma import Chroma


def build_chroma_store(
    documents: List[Document],
    embedding_function,
    persist_directory: Path,
    collection_name: str = "rag_documents",
):
    """
    Build and persist a Chroma vector store from chunked documents.
    """
    persist_directory.mkdir(parents=True, exist_ok=True)

    vector_store = Chroma.from_documents(
        documents=documents,
        embedding=embedding_function,
        persist_directory=str(persist_directory),
        collection_name=collection_name,
    )

    return vector_store


def load_chroma_store(
    embedding_function,
    persist_directory: Path,
    collection_name: str = "rag_documents",
):
    """
    Load an existing Chroma vector store from disk.
    """
    return Chroma(
        persist_directory=str(persist_directory),
        collection_name=collection_name,
        embedding_function=embedding_function,
    )


def chroma_index_exists(persist_directory: Path) -> bool:
    """
    Check whether a Chroma index directory exists and contains files.
    """
    return persist_directory.exists() and any(persist_directory.iterdir())
