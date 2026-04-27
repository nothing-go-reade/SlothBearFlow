from __future__ import annotations

import logging
import uuid
from typing import Any, Optional

from app import Settings, get_settings
from app.rag.splitter import split_text_to_documents

logger = logging.getLogger(__name__)

def ingest_plain_text(
    text: str,
    *,
    source: str,
    vector_store: Any,
    settings: Optional[Settings] = None,
) -> int:
    """同步写入向量库（仅应在后台线程 / worker 中调用）。"""
    settings = settings or get_settings()
    meta = {"source": source, "job_id": str(uuid.uuid4())}
    docs = split_text_to_documents(text, metadata=meta)
    if not docs:
        return 0
    vector_store.add_documents(docs)
    logger.info(
        "ingest 完成: chunks=%s collection=%s",
        len(docs),
        settings.milvus_collection,
    )
    return len(docs)
