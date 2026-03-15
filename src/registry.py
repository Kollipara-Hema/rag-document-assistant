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
from src.retrievers.dense_retriever import retrieve_dense
from src.retrievers.sparse_retriever import retrieve_sparse
from src.retrievers.hybrid_retriever import retrieve_hybrid
from src.loaders.local_loader import load_local_documents
from src.loaders.upload_loader import load_uploaded_documents
from src.loaders.web_loader import load_web_documents
from src.loaders.github_loader import load_github_documents
from src.chunkers.sliding_window_chunker import chunk_with_sliding_window
from src.chunkers.semantic_chunker import chunk_with_semantic_style


LOADERS = {
    "Local Repository": load_local_documents,
    "Uploaded Files": load_uploaded_documents,
    "Web Page": load_web_documents,
    "GitHub Repository": load_github_documents,
}


CHUNKERS = {
    "Fixed": chunk_with_fixed_size,
    "Recursive": chunk_with_recursive_split,
    "Sliding Window": chunk_with_sliding_window,
    "Semantic": chunk_with_semantic_style,
}

EMBEDDERS = {
    "Local Sentence Transformer": get_local_embedder,
    "OpenAI Embeddings": get_openai_embedder,
}

RETRIEVERS = {
    "Dense": retrieve_dense,
    "Sparse": retrieve_sparse,
    "Hybrid": retrieve_hybrid,
}

GENERATORS = {
    "Ollama": generate_with_ollama,
    "OpenAI": generate_with_openai,
}
