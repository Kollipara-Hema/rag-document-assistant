"""
Generation backend using OpenAI chat models.
"""

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI

from src.config import OPENAI_API_KEY, OPENAI_CHAT_MODEL
from src.utils.prompts import RAG_SYSTEM_PROMPT


def generate_with_openai(question: str, context: str) -> str:
    """
    Generate an answer using an OpenAI chat model.
    """
    llm = ChatOpenAI(
        api_key=OPENAI_API_KEY,
        model=OPENAI_CHAT_MODEL,
    )

    messages = [
        SystemMessage(content=RAG_SYSTEM_PROMPT),
        HumanMessage(content=f"Question: {question}\n\nContext:\n{context}"),
    ]

    response = llm.invoke(messages)
    return response.content
