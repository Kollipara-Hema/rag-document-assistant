"""
Semantic-style chunking strategy.

This is a lightweight semantic approximation for educational use.
Instead of purely splitting by raw character count, it tries to preserve
natural paragraph boundaries first, then falls back to smaller splits if needed.

This helps learners see how structure-aware chunking differs from purely
mechanical fixed-length splitting.
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
        separators=[
            "\n\n",   # paragraph boundary
            "\n",     # line boundary
            ". ",     # sentence-like split
            " ",      # word boundary
            "",       # final fallback
        ],
    )

    return splitter.split_documents(documents)
