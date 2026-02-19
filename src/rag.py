from __future__ import annotations

from typing import Dict, Any, List
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

from .providers import get_llm
from .retriever import retrieve


PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system",
         "You are a careful assistant. Answer ONLY using the provided context. "
         "If the context is insufficient, say you don't know. "
         "Cite sources as [source:page]."),
        ("human",
         "Question: {question}\n\n"
         "Context:\n{context}\n\n"
         "Return a short, clear answer with citations.")
    ]
)


def format_context(docs: List[Document]) -> str:
    parts = []
    for d in docs:
        src = d.metadata.get("source", "unknown")
        page = d.metadata.get("page", "NA")
        text = d.page_content.strip().replace("\n", " ")
        parts.append(f"[{src}:{page}] {text}")
    return "\n\n".join(parts)


def answer_question(question: str, k: int = 5) -> Dict[str, Any]:
    docs = retrieve(question, k=k)
    context = format_context(docs)

    llm = get_llm()
    chain = PROMPT | llm
    resp = chain.invoke({"question": question, "context": context})

    return {
        "answer": resp.content,
        "sources": [
            {"source": d.metadata.get("source"), "page": d.metadata.get("page")}
            for d in docs
        ],
    }


if __name__ == "__main__":
    q = "What is Poisson PCA and why is it used for count data?"
    out = answer_question(q, k=5)
    print(out["answer"])
    print(out["sources"])