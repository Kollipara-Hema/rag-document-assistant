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
    """
    splitter = CharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separator="\n",
    )

    chunks = splitter.split_documents(documents)

    # Add chunk metadata so retrieval results are easier to inspect in the UI.
    for chunk_index, chunk in enumerate(chunks, start=1):
        if chunk.metadata is None:
            chunk.metadata = {}

        chunk.metadata["chunk_id"] = chunk_index
        chunk.metadata["chunk_length"] = len(chunk.page_content)

    return chunks
