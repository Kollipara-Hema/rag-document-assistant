"""
Dense retrieval over the vector store.

This is the first retrieval mode we implement because it is the
standard dense vector search path in RAG systems.
"""

from typing import List
from langchain_core.documents import Document


def retrieve_dense(query: str, vector_store, top_k: int) -> List[Document]:
    """
    Retrieve the top-k most similar chunks for a query.
    """
    return vector_store.similarity_search(query, k=top_k)
