from __future__ import annotations

from typing import List
from langchain_core.documents import Document
from langchain_chroma import Chroma

from .config import CHROMA_DIR, COLLECTION_NAME, TOP_K
from .providers import get_embeddings


def get_retriever(k: int = TOP_K):
    embeddings = get_embeddings()
    db = Chroma(
        persist_directory=CHROMA_DIR,
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
    )
    return db.as_retriever(search_kwargs={"k": k})


def retrieve(query: str, k: int = TOP_K) -> List[Document]:
    retriever = get_retriever(k=k)
    return retriever.invoke(query)