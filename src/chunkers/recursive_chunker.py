"""
Recursive chunking strategy.

This is usually better than fixed chunking for natural language text
because it tries to split on more meaningful boundaries first.
"""

from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_with_recursive_split(
    documents: List[Document],
    chunk_size: int,
    chunk_overlap: int,
) -> List[Document]:
    """
    Split documents using recursive chunking.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    chunks = splitter.split_documents(documents)

    for chunk_index, chunk in enumerate(chunks, start=1):
        if chunk.metadata is None:
            chunk.metadata = {}

        chunk.metadata["chunk_id"] = chunk_index
        chunk.metadata["chunk_length"] = len(chunk.page_content)

    return chunks
