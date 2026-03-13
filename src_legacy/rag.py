from __future__ import annotations

from typing import Any, Dict, List

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

from .providers import get_llm
from .retriever import retrieve


PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a careful assistant. Answer ONLY using the provided context. "
            "Do not use outside knowledge. "
            "If the answer is not present in the provided context, respond exactly with: "
            "'I could not find relevant information in the selected documents. Please upload or use documents that contain the answer.' "
            "If relevant context exists, provide a short clear answer with citations in the format [source:page].",
        ),
        (
            "human",
            "Question: {question}\n\n"
            "Context:\n{context}\n\n"
            "Return a short, clear answer with citations.",
        ),
    ]
)



def format_context(docs: List[Document]) -> str:
    parts = []
    for d in docs:
        src = d.metadata.get("source", "unknown")
        page = d.metadata.get("page", "NA")
        scope = d.metadata.get("source_scope", "repository")
        text = d.page_content.strip().replace("\n", " ")
        parts.append(f"[{scope}|{src}:{page}] {text}")
    return "\n\n".join(parts)


def answer_question(question: str, k: int = 5, mode: str = "repository") -> Dict[str, Any]:
    docs = retrieve(question, k=k, mode=mode)
    context = format_context(docs)

    llm = get_llm()
    chain = PROMPT | llm
    resp = chain.invoke({"question": question, "context": context})

    return {
        "answer": resp.content,
        "sources": [
            {
                "source": d.metadata.get("source"),
                "page": d.metadata.get("page"),
                "scope": d.metadata.get("source_scope", "repository"),
                "file_type": d.metadata.get("file_type", "unknown"),
            }
            for d in docs
        ],
    }


if __name__ == "__main__":
    q = "What is Poisson PCA and why is it used for count data?"
    out = answer_question(q, k=5, mode="repository")
    print(out["answer"])
    print(out["sources"])
