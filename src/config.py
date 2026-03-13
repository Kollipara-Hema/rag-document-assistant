"""
Central configuration for the modular RAG learning lab.

We keep all user-adjustable settings here so that the rest of the code
can import them without hardcoding paths and model names everywhere.
"""

from pathlib import Path
import os
from dotenv import load_dotenv

# Load environment variables from .env if available.
load_dotenv()

# Project root directory.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Data folders.
RAW_DOCS_DIR = PROJECT_ROOT / "data" / "raw_docs"
CHROMA_DB_DIR = PROJECT_ROOT / "data" / "chroma_db"

# Default chunking settings.
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 150

# Default retrieval settings.
DEFAULT_TOP_K = 5

# Embedding model settings.
LOCAL_EMBEDDING_MODEL = os.getenv("LOCAL_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

# LLM settings.
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1")
OPENAI_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")

# API key for OpenAI-backed options.
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
