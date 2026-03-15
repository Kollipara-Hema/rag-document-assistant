"""
Central configuration for the modular RAG learning lab.
"""

from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DOCS_DIR = PROJECT_ROOT / "data" / "raw_docs"

# Separate Chroma directories for each source type.
CHROMA_DB_LOCAL_DIR = PROJECT_ROOT / "data" / "chroma_db_local"
CHROMA_DB_UPLOAD_DIR = PROJECT_ROOT / "data" / "chroma_db_uploads"
CHROMA_DB_WEB_DIR = PROJECT_ROOT / "data" / "chroma_db_web"
CHROMA_DB_GITHUB_DIR = PROJECT_ROOT / "data" / "chroma_db_github"

DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 150
DEFAULT_TOP_K = 5

LOCAL_EMBEDDING_MODEL = os.getenv("LOCAL_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1")
OPENAI_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
