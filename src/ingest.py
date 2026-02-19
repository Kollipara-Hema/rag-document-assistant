from __future__ import annotations

import os
from pathlib import Path
from typing import List, Dict, Any

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma

from .config import CHROMA_DIR, COLLECTION_NAME, CHUNK_SIZE, CHUNK_OVERLAP
from .providers import get_embeddings


def load_pdfs(folder: str) -> List[Document]:
    docs: List[Document] = []
    folder_path = Path(folder)
    pdf_files = sorted(folder_path.glob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in: {folder}")

    for pdf_path in pdf_files:
        loader = PyPDFLoader(str(pdf_path))
        loaded = loader.load()
        for d in loaded:
            d.metadata["source"] = pdf_path.name
        docs.extend(loaded)

    return docs


def chunk_documents(docs: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    return splitter.split_documents(docs)


def build_chroma_index(chunks: List[Document]) -> None:
    os.makedirs(CHROMA_DIR, exist_ok=True)
    embeddings = get_embeddings()

    db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
        collection_name=COLLECTION_NAME
    )
    


def ingest(folder: str = "data/raw_docs") -> Dict[str, Any]:
    docs = load_pdfs(folder)
    chunks = chunk_documents(docs)
    build_chroma_index(chunks)

    return {
        "pdf_files_loaded": len(set(d.metadata.get("source") for d in docs)),
        "pages_loaded": len(docs),
        "chunks_created": len(chunks),
        "chroma_dir": CHROMA_DIR,
        "collection": COLLECTION_NAME,
    }


if __name__ == "__main__":
    stats = ingest()
    print(stats)