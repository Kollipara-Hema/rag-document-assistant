import os
from dotenv import load_dotenv

load_dotenv()

EMBEDDINGS_PROVIDER = os.getenv("EMBEDDINGS_PROVIDER", "local")  # local | openai
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")              # ollama | openai

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
OPENAI_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")

LOCAL_EMBED_MODEL = os.getenv("LOCAL_EMBED_MODEL", "all-MiniLM-L6-v2")

CHROMA_DIR = os.getenv("CHROMA_DIR", "data/chroma_db")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "docs")

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))
TOP_K = int(os.getenv("TOP_K", "5"))