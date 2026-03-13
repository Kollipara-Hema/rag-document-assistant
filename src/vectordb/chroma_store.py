"""
Chroma vector store utilities.

This module creates and loads the Chroma index used in v1.
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

    return Chroma.from_documents(
        documents=documents,
        embedding=embedding_function,
        persist_directory=str(persist_directory),
        collection_name=collection_name,
    )


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
