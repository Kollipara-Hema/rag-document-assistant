"""
Loader for user-uploaded documents.

This loader accepts files uploaded through the Streamlit UI,
writes them temporarily to disk, and then reuses the same parsing logic
used by the local loader.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import List

from langchain_core.documents import Document

from src.loaders.local_loader import _load_single_file


def load_uploaded_documents(uploaded_files) -> List[Document]:
    """
    Load uploaded files from the Streamlit file uploader.

    Parameters
    ----------
    uploaded_files
        List of uploaded file objects from Streamlit.

    Returns
    -------
    List[Document]
        Parsed documents ready for chunking.
    """
    if not uploaded_files:
        raise ValueError("No uploaded files were provided.")

    temp_dir = Path("data/temp_uploads")

    # Start fresh each time so old uploaded files do not leak into the next run.
    if temp_dir.exists():
        shutil.rmtree(temp_dir)

    temp_dir.mkdir(parents=True, exist_ok=True)

    all_documents: List[Document] = []

    for uploaded_file in uploaded_files:
        file_path = temp_dir / uploaded_file.name

        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        docs = _load_single_file(file_path)

        # Mark source scope for easier debugging in the UI.
        for doc in docs:
            doc.metadata["source_scope"] = "upload"

        all_documents.extend(docs)

    return all_documents
