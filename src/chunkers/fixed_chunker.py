"""
Fixed-size chunking strategy.

This method slices text into chunks of a fixed character length
with optional overlap. It is simple and useful for comparison.
"""

from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import CharacterTextSplitter


def chunk_with_fixed_size(
    documents: List[Document],
    chunk_size: int,
    chunk_overlap: int,
) -> List[Document]:
    """
    Split documents into fixed-size character chunks.

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
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separator="\n",
    )

    return splitter.split_documents(documents)
