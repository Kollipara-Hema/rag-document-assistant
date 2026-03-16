"""
Hybrid retrieval combines dense and sparse retrieval.

Dense retrieval is good for semantic meaning.
Sparse retrieval is good for exact keywords and technical terms.

We combine both using Reciprocal Rank Fusion (RRF).
"""

from typing import List, Dict, Tuple
from langchain_core.documents import Document

from src.retrievers.dense_retriever import retrieve_dense
from src.retrievers.sparse_retriever import retrieve_sparse


def _make_doc_key(doc: Document) -> Tuple[str, str, str]:
    """
    Create a lightweight identity key for merging results.
    """
    source = str(doc.metadata.get("source", "unknown"))
    page = str(doc.metadata.get("page", "NA"))
    content_prefix = doc.page_content.strip()[:300]

    return (source, page, content_prefix)


def retrieve_hybrid(
    query: str,
    vector_store,
    documents: List[Document],
    top_k: int,
    dense_fetch_k: int | None = None,
    sparse_fetch_k: int | None = None,
    rrf_k: int = 60,
) -> List[Document]:
    """
    Retrieve documents using both dense and sparse retrieval, then merge them
    using Reciprocal Rank Fusion (RRF).
    """
    dense_docs = retrieve_dense(
        query=query,
        vector_store=vector_store,
        top_k=dense_fetch_k or max(top_k * 3, 10),
        fetch_k=dense_fetch_k or max(top_k * 3, 10),
    )

    sparse_docs = retrieve_sparse(
        query=query,
        documents=documents,
        top_k=sparse_fetch_k or max(top_k * 3, 10),
    )

    fused_scores: Dict[Tuple[str, str, str], float] = {}
    doc_lookup: Dict[Tuple[str, str, str], Document] = {}

    for rank, doc in enumerate(dense_docs, start=1):
        key = _make_doc_key(doc)
        fused_scores[key] = fused_scores.get(key, 0.0) + 1.0 / (rrf_k + rank)
        doc_lookup[key] = doc

    for rank, doc in enumerate(sparse_docs, start=1):
        key = _make_doc_key(doc)
        fused_scores[key] = fused_scores.get(key, 0.0) + 1.0 / (rrf_k + rank)
        doc_lookup[key] = doc

    ranked_keys = sorted(
        fused_scores.keys(),
        key=lambda key: fused_scores[key],
        reverse=True,
    )[:top_k]

    retrieved_docs = []
    for key in ranked_keys:
        doc = doc_lookup[key]
        doc.metadata["retrieval_score"] = float(fused_scores[key])
        doc.metadata["retrieval_method"] = "hybrid"
        retrieved_docs.append(doc)

    return retrieved_docs
