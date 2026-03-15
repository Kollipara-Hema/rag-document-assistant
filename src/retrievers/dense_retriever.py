"""
Dense retrieval over the vector store.

Step 4A adds simple deduplication so the retriever does not return
multiple near-identical chunks from the same source and page.
"""

from typing import List
from langchain_core.documents import Document


def _make_doc_key(doc: Document) -> tuple:
    """
    Create a lightweight key for deduplication.

    We combine source, page, and a prefix of the content so that
    repeated overlapping chunks can be filtered out.
    """
    source = doc.metadata.get("source", "unknown")
    page = doc.metadata.get("page", "NA")
    content_prefix = doc.page_content.strip()[:300]

    return (source, page, content_prefix)


def deduplicate_documents(documents: List[Document], max_per_source_page: int = 2) -> List[Document]:
    """
    Remove repeated chunks and limit how many chunks can come from the same source-page pair.

    Parameters
    ----------
    documents : List[Document]
        Retrieved documents from similarity search.
    max_per_source_page : int
        Maximum number of chunks allowed from the same source and page.

    Returns
    -------
    List[Document]
        Deduplicated and more diverse retrieved documents.
    """
    unique_docs = []
    seen_keys = set()
    source_page_counts = {}

    for doc in documents:
        doc_key = _make_doc_key(doc)

        # Skip exact or near-exact repeated chunks.
        if doc_key in seen_keys:
            continue

        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "NA")
        source_page_key = (source, page)

        # Limit repeated chunks from the same source-page pair.
        current_count = source_page_counts.get(source_page_key, 0)
        if current_count >= max_per_source_page:
            continue

        seen_keys.add(doc_key)
        unique_docs.append(doc)
        source_page_counts[source_page_key] = current_count + 1

    return unique_docs


def retrieve_dense(query: str, vector_store, top_k: int, fetch_k: int | None = None) -> List[Document]:
    """
    Retrieve the top-k most similar chunks for a query and deduplicate them.

    Parameters
    ----------
    query : str
        User question.
    vector_store
        Loaded vector database.
    top_k : int
        Final number of chunks to return.
    fetch_k : int | None
        Number of raw chunks to retrieve before deduplication.
        If None, we fetch more than top_k automatically.

    Returns
    -------
    List[Document]
        Deduplicated retrieved chunks.
    """
    # We fetch more candidates first so deduplication still leaves enough useful chunks.
    raw_fetch_k = fetch_k or max(top_k * 3, 10)

    raw_docs = vector_store.similarity_search(query, k=raw_fetch_k)
    deduped_docs = deduplicate_documents(raw_docs)

    return deduped_docs[:top_k]
