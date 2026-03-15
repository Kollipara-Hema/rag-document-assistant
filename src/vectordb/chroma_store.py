"""
Chroma vector store utilities.

This module handles:
1. building and saving a Chroma index
2. loading an existing Chroma index
3. checking whether an index already exists

We separate these operations so the app can build the index once
and reuse it for many questions.
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

    Parameters
    ----------
    documents : List[Document]
        Chunked documents to index.
    embedding_function :
        Embedding model wrapper used to convert chunks into vectors.
    persist_directory : Path
        Directory where the Chroma database will be stored.
    collection_name : str
        Logical collection name inside Chroma.

    Returns
    -------
    Chroma
        Persisted Chroma vector store.
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

    Parameters
    ----------
    embedding_function :
        Embedding model wrapper used for query embedding.
    persist_directory : Path
        Directory where the Chroma database is stored.
    collection_name : str
        Logical collection name inside Chroma.

    Returns
    -------
    Chroma
        Loaded Chroma vector store.
    """
    return Chroma(
        persist_directory=str(persist_directory),
        collection_name=collection_name,
        embedding_function=embedding_function,
    )


def chroma_index_exists(persist_directory: Path) -> bool:
    """
    Check whether a Chroma index directory exists and contains files.

    This helps the app decide whether it can answer questions immediately
    or whether the user must build the index first.

    Parameters
    ----------
    persist_directory : Path
        Directory where the Chroma database should exist.

    Returns
    -------
    bool
        True if the directory exists and is not empty, otherwise False.
    """
    return persist_directory.exists() and any(persist_directory.iterdir())
