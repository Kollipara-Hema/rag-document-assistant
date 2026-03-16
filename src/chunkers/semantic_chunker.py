"""
Semantic-style chunking strategy.

This is a lightweight semantic approximation for educational use.
"""

from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_with_semantic_style(
    documents: List[Document],
    chunk_size: int,
    chunk_overlap: int,
) -> List[Document]:
    """
    Split documents using structure-aware semantic-style chunking.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            "",
        ],
    )

    chunks = splitter.split_documents(documents)

    for chunk_index, chunk in enumerate(chunks, start=1):
        if chunk.metadata is None:
            chunk.metadata = {}

        chunk.metadata["chunk_id"] = chunk_index
        chunk.metadata["chunk_length"] = len(chunk.page_content)

    return chunks
