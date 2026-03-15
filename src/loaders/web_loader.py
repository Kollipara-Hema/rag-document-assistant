"""
Loader for a single web page URL.

This loader fetches HTML content, removes non-visible elements,
and converts the page into a Document object.
"""

from __future__ import annotations

from typing import List

import requests
from bs4 import BeautifulSoup
from langchain_core.documents import Document


def load_web_documents(url: str) -> List[Document]:
    """
    Load a single web page as a document.

    Parameters
    ----------
    url : str
        Web page URL.

    Returns
    -------
    List[Document]
        Parsed web page content as documents.
    """
    if not url:
        raise ValueError("No web URL was provided.")

    response = requests.get(url, timeout=20)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")

    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    text = soup.get_text(separator="\n")
    cleaned_text = "\n".join(line.strip() for line in text.splitlines() if line.strip())

    return [
        Document(
            page_content=cleaned_text,
            metadata={
                "source": url,
                "file_type": "html",
                "page": 1,
                "source_scope": "web",
            },
        )
    ]
