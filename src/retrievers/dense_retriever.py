"""
Dense retrieval over the vector store.

This retriever returns similarity-ranked chunks and attaches a retrieval score
to each chunk so the UI can display why a chunk was selected.
"""

from typing import List
from langchain_core.documents import Document


def _make_doc_key(doc: Document) -> tuple:
    """
    Create a lightweight key for deduplication.
    """
    source = doc.metadata.get("source", "unknown")
    page = doc.metadata.get("page", "NA")
    content_prefix = doc.page_content.strip()[:300]

    return (source, page, content_prefix)


def deduplicate_documents(documents: List[Document], max_per_source_page: int = 2) -> List[Document]:
    """
    Remove repeated chunks and limit how many chunks can come from the same source-page pair.
    """
    unique_docs = []
    seen_keys = set()
    source_page_counts = {}

    for doc in documents:
        doc_key = _make_doc_key(doc)

        if doc_key in seen_keys:
            continue

        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "NA")
        source_page_key = (source, page)

        current_count = source_page_counts.get(source_page_key, 0)
        if current_count >= max_per_source_page:
            continue

        seen_keys.add(doc_key)
        unique_docs.append(doc)
        source_page_counts[source_page_key] = current_count + 1

    return unique_docs


def retrieve_dense(query: str, vector_store, top_k: int, fetch_k: int | None = None) -> List[Document]:
    """
    Retrieve the top-k most similar chunks and attach similarity scores.
    """
    raw_fetch_k = fetch_k or max(top_k * 3, 10)

    # similarity_search_with_score returns (Document, score) pairs
    raw_results = vector_store.similarity_search_with_score(query, k=raw_fetch_k)

    scored_docs = []
    for doc, score in raw_results:
        doc.metadata["retrieval_score"] = float(score)
        doc.metadata["retrieval_method"] = "dense"
        scored_docs.append(doc)

    deduped_docs = deduplicate_documents(scored_docs)

    return deduped_docs[:top_k]
