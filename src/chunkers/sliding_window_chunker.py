"""
Sliding window chunking strategy.

This approach creates overlapping text windows so that information near
chunk boundaries is less likely to be lost during retrieval.

It is useful when context continuity matters.
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

    Parameters
    ----------
    documents : List[Document]
        Input documents to split.
    chunk_size : int
        Maximum size of each chunk.
    chunk_overlap : int
        Number of overlapping characters between chunks.

    Returns
    -------
    List[Document]
        Chunked documents.
    """
    splitter = CharacterTextSplitter(
        separator=" ",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    return splitter.split_documents(documents)
