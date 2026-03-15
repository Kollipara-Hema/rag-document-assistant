"""
Loader for GitHub repositories.

This loader clones a repository temporarily and extracts supported text files
such as markdown, txt, json, yaml, and yml files.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import List

from langchain_core.documents import Document

SUPPORTED_GITHUB_EXTENSIONS = {".md", ".txt", ".json", ".yaml", ".yml"}


def load_github_documents(repo_url: str) -> List[Document]:
    """
    Clone a GitHub repository and load supported text files.

    Parameters
    ----------
    repo_url : str
        Public GitHub repository URL.

    Returns
    -------
    List[Document]
        Parsed repository files as documents.
    """
    if not repo_url:
        raise ValueError("No GitHub repository URL was provided.")

    temp_repo_dir = Path("data/temp_github_repo")

    # Remove old cloned repo if it exists.
    if temp_repo_dir.exists():
        shutil.rmtree(temp_repo_dir)

    subprocess.run(
        ["git", "clone", "--depth", "1", repo_url, str(temp_repo_dir)],
        check=True,
    )

    all_documents: List[Document] = []

    for file_path in temp_repo_dir.rglob("*"):
        if not file_path.is_file():
            continue

        if file_path.suffix.lower() not in SUPPORTED_GITHUB_EXTENSIONS:
            continue

        text = file_path.read_text(encoding="utf-8", errors="ignore")

        relative_path = file_path.relative_to(temp_repo_dir)

        all_documents.append(
            Document(
                page_content=text,
                metadata={
                    "source": str(relative_path),
                    "repo_url": repo_url,
                    "file_type": file_path.suffix.lower().lstrip("."),
                    "page": 1,
                    "source_scope": "github",
                },
            )
        )

    return all_documents
