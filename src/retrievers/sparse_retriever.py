"""
Sparse retrieval using BM25.

This retriever scores chunks using keyword overlap rather than embeddings.
It is useful for questions involving exact phrases, technical terms, policy
language, identifiers, and numbers.
"""

from typing import List
import re

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi


def _tokenize(text: str) -> List[str]:
    """
    Simple tokenizer for BM25.

    We lowercase the text and split it into alphanumeric tokens.
    This keeps the implementation easy to understand for learning purposes.
    """
    return re.findall(r"\b\w+\b", text.lower())


def retrieve_sparse(
    query: str,
    documents: List[Document],
    top_k: int,
) -> List[Document]:
    """
    Retrieve the top-k chunks using BM25 sparse retrieval.

    Parameters
    ----------
    query : str
        User question.
    documents : List[Document]
        Chunked documents to search over.
    top_k : int
        Number of top chunks to return.

    Returns
    -------
    List[Document]
        Top-k documents ranked by BM25 score.
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

    return [documents[i] for i in ranked_indices]
