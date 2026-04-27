from __future__ import annotations

from typing import Dict, List, Optional

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def split_text_to_documents(
    text: str,
    *,
    chunk_size: int = 500,
    chunk_overlap: int = 100,
    metadata: Optional[Dict] = None,
) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    md = dict(metadata) if metadata else {}
    return splitter.create_documents([text], metadatas=[md])
