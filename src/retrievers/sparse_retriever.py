"""
Sparse retrieval using BM25.

This retriever scores chunks using keyword overlap rather than embeddings.
"""

from typing import List
import re

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi


def _tokenize(text: str) -> List[str]:
    """
    Simple tokenizer for BM25.
    """
    return re.findall(r"\b\w+\b", text.lower())


def retrieve_sparse(
    query: str,
    documents: List[Document],
    top_k: int,
) -> List[Document]:
    """
    Retrieve the top-k chunks using BM25 sparse retrieval and attach BM25 scores.
    """
    if not documents:
        return []

    tokenized_corpus = [_tokenize(doc.page_content) for doc in documents]
    bm25 = BM25Okapi(tokenized_corpus)

    tokenized_query = _tokenize(query)
    scores = bm25.get_scores(tokenized_query)

    ranked_indices = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True,
    )[:top_k]

    retrieved_docs = []
    for rank_index in ranked_indices:
        doc = documents[rank_index]
        doc.metadata["retrieval_score"] = float(scores[rank_index])
        doc.metadata["retrieval_method"] = "sparse"
        retrieved_docs.append(doc)

    return retrieved_docs
