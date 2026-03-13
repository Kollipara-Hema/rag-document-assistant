"""
Registry of selectable RAG components.

The Streamlit app uses this registry to map dropdown selections
to real implementation functions.
"""

from src.chunkers.fixed_chunker import chunk_with_fixed_size
from src.chunkers.recursive_chunker import chunk_with_recursive_split
from src.embeddings.sentence_transformer_embedder import get_local_embedder
from src.embeddings.openai_embedder import get_openai_embedder
from src.generators.ollama_generator import generate_with_ollama
from src.generators.openai_generator import generate_with_openai
from src.loaders.local_loader import load_local_documents
from src.retrievers.dense_retriever import retrieve_dense

LOADERS = {
    "Local Repository": load_local_documents,
}

CHUNKERS = {
    "Fixed": chunk_with_fixed_size,
    "Recursive": chunk_with_recursive_split,
}

EMBEDDERS = {
    "Local Sentence Transformer": get_local_embedder,
    "OpenAI Embeddings": get_openai_embedder,
}

RETRIEVERS = {
    "Dense": retrieve_dense,
}

GENERATORS = {
    "Ollama": generate_with_ollama,
    "OpenAI": generate_with_openai,
}
