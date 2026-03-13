"""
Local generation backend using Ollama.
"""

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_ollama import ChatOllama

from src.config import OLLAMA_MODEL
from src.utils.prompts import RAG_SYSTEM_PROMPT


def generate_with_ollama(question: str, context: str) -> str:
    """
    Generate an answer using a local Ollama chat model.
    """
    llm = ChatOllama(model=OLLAMA_MODEL)

    messages = [
        SystemMessage(content=RAG_SYSTEM_PROMPT),
        HumanMessage(content=f"Question: {question}\n\nContext:\n{context}"),
    ]

    response = llm.invoke(messages)
    return response.content
