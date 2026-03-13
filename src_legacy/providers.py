from typing import Any
from .config import (
    EMBEDDINGS_PROVIDER, LLM_PROVIDER,
    OPENAI_API_KEY, OPENAI_EMBED_MODEL, OPENAI_CHAT_MODEL,
    LOCAL_EMBED_MODEL
)

def get_embeddings() -> Any:
    if EMBEDDINGS_PROVIDER == "local":
        from langchain_huggingface import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(model_name=LOCAL_EMBED_MODEL)

    if EMBEDDINGS_PROVIDER == "openai":
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY missing in .env")
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(model=OPENAI_EMBED_MODEL, api_key=OPENAI_API_KEY)

    raise ValueError(f"Unknown EMBEDDINGS_PROVIDER: {EMBEDDINGS_PROVIDER}")

def get_llm() -> Any:
    if LLM_PROVIDER == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(model="llama3.1", temperature=0)

    if LLM_PROVIDER == "openai":
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY missing in .env")
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=OPENAI_CHAT_MODEL, api_key=OPENAI_API_KEY, temperature=0)

    raise ValueError(f"Unknown LLM_PROVIDER: {LLM_PROVIDER}")