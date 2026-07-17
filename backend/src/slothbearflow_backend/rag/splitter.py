from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Dict, List, Optional

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


CHUNKER_VERSION = "markdown-structure-v2"
CHUNK_SEPARATORS = ["\n# ", "\n## ", "\n### ", "\n\n", "\n", "。", ". ", " "]


def build_chunking_contract(chunk_size: int, chunk_overlap: int) -> Dict[str, Any]:
    return {
        "algorithm": "recursive-character-markdown-sections",
        "chunker_version": CHUNKER_VERSION,
        "chunk_size": int(chunk_size),
        "chunk_overlap": int(chunk_overlap),
        "separators_sha256": hashlib.sha256(
            json.dumps(CHUNK_SEPARATORS, ensure_ascii=False).encode("utf-8")
        ).hexdigest()[:16],
    }


def split_text_to_documents(
    text: str,
    *,
    chunk_size: int = 500,
    chunk_overlap: int = 100,
    metadata: Optional[Dict] = None,
) -> List[Document]:
    normalized = re.sub(r"\r\n?", "\n", str(text or "")).strip()
    if not normalized:
        return []
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=CHUNK_SEPARATORS,
    )
    md = dict(metadata) if metadata else {}
    source = str(md.get("source") or "unknown")
    document_id = str(md.get("document_id") or _digest(source + "\n" + normalized))
    document_version = str(md.get("document_version") or _digest(normalized))
    chunking_contract = dict(
        md.get("chunking_contract")
        or build_chunking_contract(chunk_size, chunk_overlap)
    )
    sections = _markdown_sections(normalized)
    chunks: List[Document] = []
    for section_index, (heading, body) in enumerate(sections):
        section_metadata: Dict[str, Any] = {
            **md,
            "source": source,
            "document_id": document_id,
            "document_version": document_version,
            "chunker_version": CHUNKER_VERSION,
            "chunking_contract": chunking_contract,
            "section": heading,
            "section_index": section_index,
        }
        for chunk in splitter.create_documents([body], metadatas=[section_metadata]):
            if str(chunk.page_content or "").strip():
                chunks.append(chunk)

    total = len(chunks)
    output: List[Document] = []
    for index, chunk in enumerate(chunks):
        content = str(chunk.page_content or "").strip()
        chunk_id = _digest(f"{document_id}:{document_version}:{index}:{content}")
        output.append(
            Document(
                page_content=content,
                metadata={
                    **dict(chunk.metadata or {}),
                    "chunk_id": chunk_id,
                    "chunk_index": index,
                    "chunk_count": total,
                    "content_hash": _digest(content),
                },
            )
        )
    return output


def _markdown_sections(text: str) -> List[tuple[str, str]]:
    sections: List[tuple[str, str]] = []
    heading = "document"
    body: List[str] = []
    for line in text.splitlines():
        match = re.match(r"^(#{1,6})\s+(.+?)\s*$", line)
        if match:
            if body and "\n".join(body).strip():
                sections.append((heading, "\n".join(body).strip()))
            heading = match.group(2).strip()[:200]
            body = [line]
        else:
            body.append(line)
    if body and "\n".join(body).strip():
        sections.append((heading, "\n".join(body).strip()))
    return sections or [("document", text)]


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:32]
