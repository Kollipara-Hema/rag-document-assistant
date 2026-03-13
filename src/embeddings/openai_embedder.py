"""
OpenAI embedding backend.

This is optional and requires an OpenAI API key.
"""

from langchain_openai import OpenAIEmbeddings
from src.config import OPENAI_API_KEY, OPENAI_EMBEDDING_MODEL


def get_openai_embedder():
    """
    Return an OpenAI embedding model wrapper.
    """
    return OpenAIEmbeddings(
        api_key=OPENAI_API_KEY,
        model=OPENAI_EMBEDDING_MODEL,
    )
