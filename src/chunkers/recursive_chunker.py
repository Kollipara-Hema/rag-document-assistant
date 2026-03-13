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
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    return splitter.split_documents(documents)
