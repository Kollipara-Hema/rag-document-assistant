"""
Local embedding backend using sentence-transformers.

This is the default embedding option for the learning lab because
it works locally and does not require API calls.
"""

from langchain_community.embeddings import HuggingFaceEmbeddings
from src.config import LOCAL_EMBEDDING_MODEL


def get_local_embedder():
    """
    Return a local embedding model wrapper.
    """
    return HuggingFaceEmbeddings(model_name=LOCAL_EMBEDDING_MODEL)
