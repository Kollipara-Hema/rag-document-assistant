from __future__ import annotations

from typing import List

from langchain_core.documents import Document
from langchain_chroma import Chroma

from .config import (
    CHROMA_DIR,
    COLLECTION_NAME,
    TOP_K,
    UPLOAD_CHROMA_DIR,
    UPLOAD_COLLECTION_NAME,
)
from .providers import get_embeddings


def _get_db(persist_directory: str, collection_name: str) -> Chroma:
    embeddings = get_embeddings()
    return Chroma(
        persist_directory=persist_directory,
        collection_name=collection_name,
        embedding_function=embeddings,
    )


def _safe_similarity_search(
    persist_directory: str,
    collection_name: str,
    query: str,
    k: int,
) -> List[Document]:
    try:
        db = _get_db(persist_directory, collection_name)
        return db.similarity_search(query, k=k)
    except Exception:
        return []


def retrieve(query: str, k: int = TOP_K, mode: str = "repository") -> List[Document]:
    mode = mode.lower()

    if mode == "repository":
        return _safe_similarity_search(CHROMA_DIR, COLLECTION_NAME, query, k)

    if mode == "uploaded":
        return _safe_similarity_search(UPLOAD_CHROMA_DIR, UPLOAD_COLLECTION_NAME, query, k)

    if mode == "both":
        repo_docs = _safe_similarity_search(CHROMA_DIR, COLLECTION_NAME, query, k)
        upload_docs = _safe_similarity_search(UPLOAD_CHROMA_DIR, UPLOAD_COLLECTION_NAME, query, k)

        combined = repo_docs + upload_docs

        seen = set()
        unique_docs = []
        for d in combined:
            key = (
                d.metadata.get("source"),
                d.metadata.get("page"),
                d.page_content[:200],
            )
            if key not in seen:
                seen.add(key)
                unique_docs.append(d)

        return unique_docs[:k]

    raise ValueError("mode must be one of: repository, uploaded, both")
