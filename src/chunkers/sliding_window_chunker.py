"""
Sliding window chunking strategy.

This approach creates overlapping text windows so that information near
chunk boundaries is less likely to be lost during retrieval.
"""

from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import CharacterTextSplitter


def chunk_with_sliding_window(
    documents: List[Document],
    chunk_size: int,
    chunk_overlap: int,
) -> List[Document]:
    """
    Split documents using a sliding window strategy.
    """
    splitter = CharacterTextSplitter(
        separator=" ",
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
