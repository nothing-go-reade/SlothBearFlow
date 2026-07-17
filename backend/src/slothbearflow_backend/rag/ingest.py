from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

from backend.src.slothbearflow_backend import Settings, get_settings
from backend.src.slothbearflow_backend.rag.security import (
    citation_source_is_safe,
    normalize_knowledge_acl,
)
from backend.src.slothbearflow_backend.rag.splitter import (
    build_chunking_contract,
    split_text_to_documents,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class IngestOutcome:
    chunk_count: int
    stale_chunks_deleted: int
    cleanup_confirmed: bool
    metadata: Dict[str, Any]


def build_ingest_metadata(
    text: str,
    *,
    source: str,
    metadata: Optional[Dict[str, Any]] = None,
    settings: Optional[Settings] = None,
) -> Dict[str, Any]:
    settings = settings or get_settings()
    result = dict(metadata or {})
    normalized_source = str(source or "").strip()
    if not citation_source_is_safe(normalized_source):
        raise ValueError("Knowledge source contains unsafe metadata.")

    acl = normalize_knowledge_acl(result)
    tenant_id = str(acl["tenant_id"])
    owner_id = str(acl.get("owner_id") or "")
    visibility = str(acl["visibility"])
    allowed_roles = list(acl["allowed_roles"])

    identity = f"{tenant_id}:{owner_id or '_tenant'}:{normalized_source}"
    # Identity and version fields are service-owned. Allowing caller metadata to
    # override them can make one tenant clean up another tenant's document.
    result["document_id"] = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:32]
    content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()[:32]
    chunking_contract = build_chunking_contract(
        settings.rag_chunk_size,
        settings.rag_chunk_overlap,
    )
    chunk_signature = json.dumps(
        {
            "content_hash": content_hash,
            "chunking_contract": chunking_contract,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    result["document_version"] = hashlib.sha256(
        chunk_signature.encode("utf-8")
    ).hexdigest()[:32]
    result["content_hash"] = content_hash
    result["chunking_contract"] = chunking_contract
    result.update(
        {
            "source": normalized_source,
            "tenant_id": tenant_id,
            "visibility": visibility,
            "allowed_roles": allowed_roles,
        }
    )
    if owner_id:
        result["owner_id"] = owner_id
    return result


def ingest_plain_text(
    text: str,
    *,
    source: str,
    vector_store: Any,
    settings: Optional[Settings] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> int:
    """同步写入向量库（仅应在后台线程 / worker 中调用）。"""
    return ingest_plain_text_with_outcome(
        text,
        source=source,
        vector_store=vector_store,
        settings=settings,
        metadata=metadata,
    ).chunk_count


def ingest_plain_text_with_outcome(
    text: str,
    *,
    source: str,
    vector_store: Any,
    settings: Optional[Settings] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> IngestOutcome:
    """写入 chunks 并显式报告 stale-version cleanup 是否得到确认。"""
    settings = settings or get_settings()
    meta = build_ingest_metadata(
        text,
        source=source,
        metadata=metadata,
        settings=settings,
    )
    docs = split_text_to_documents(
        text,
        chunk_size=settings.rag_chunk_size,
        chunk_overlap=settings.rag_chunk_overlap,
        metadata=meta,
    )
    if not docs:
        raise ValueError("Knowledge ingestion produced no chunks.")
    written_count = vector_store.add_documents(docs)
    if not isinstance(written_count, int) or isinstance(written_count, bool):
        raise RuntimeError("Vector store did not acknowledge the written chunk count.")
    if written_count != len(docs):
        raise RuntimeError(
            "Vector store acknowledged an incomplete write "
            f"({written_count}/{len(docs)} chunks)."
        )

    deleted_count = 0
    cleanup_confirmed = False
    cleanup = getattr(vector_store, "delete_stale_document_versions", None)
    if callable(cleanup):
        deleted_count = int(
            cleanup(
                document_id=str(meta["document_id"]),
                current_version=str(meta["document_version"]),
                source=str(meta["source"]),
                tenant_id=str(meta["tenant_id"]),
                owner_id=str(meta.get("owner_id") or ""),
            )
            or 0
        )
        cleanup_confirmed = True
    logger.info(
        "ingest 完成: chunks=%s stale_chunks=%s collection=%s",
        len(docs),
        deleted_count,
        settings.milvus_collection,
    )
    return IngestOutcome(
        chunk_count=len(docs),
        stale_chunks_deleted=deleted_count,
        cleanup_confirmed=cleanup_confirmed,
        metadata=meta,
    )
