from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from app import Settings, get_settings
from app.rag.embedding import get_embedding_function

logger = logging.getLogger(__name__)

_vector_store: Optional[Any] = None
_vector_store_error: Optional[str] = None


def reset_vector_store_cache() -> None:
    global _vector_store, _vector_store_error
    _vector_store = None
    _vector_store_error = None


def get_vector_store(settings: Optional[Settings] = None) -> Optional[Any]:
    """懒加载 Milvus LangChain VectorStore；失败返回 None（主链路降级）。"""
    global _vector_store, _vector_store_error
    settings = settings or get_settings()
    if settings.skip_milvus or not settings.use_rag:
        return None
    if _vector_store is not None:
        return _vector_store
    if _vector_store_error is not None:
        return None
    try:
        from langchain_milvus import Milvus

        _vector_store = Milvus(
            embedding_function=get_embedding_function(settings),
            collection_name=settings.milvus_collection,
            connection_args={"uri": settings.milvus_uri},
            drop_old=False,
            timeout=settings.milvus_timeout,
        )
        _vector_store_error = None
        return _vector_store
    except Exception as e:  # noqa: BLE001 — 模板需吞掉第三方连接错误
        _vector_store_error = str(e)
        logger.warning("Milvus 初始化失败，RAG 将降级关闭: %s", e)
        return None


def get_vector_store_status(settings: Optional[Settings] = None) -> Dict[str, Any]:
    settings = settings or get_settings()
    if settings.skip_milvus:
        return {"enabled": False, "reason": "SKIP_MILVUS=true"}
    if not settings.use_rag:
        return {"enabled": False, "reason": "USE_RAG=false"}
    try:
        from pymilvus import connections, utility

        alias = "healthcheck"
        connections.connect(
            alias=alias,
            uri=settings.milvus_uri,
            timeout=settings.milvus_timeout,
        )
        has_collection = utility.has_collection(
            settings.milvus_collection,
            using=alias,
            timeout=settings.milvus_timeout,
        )
        connections.disconnect(alias)
        if has_collection:
            return {"enabled": True, "collection": settings.milvus_collection}
        return {
            "enabled": False,
            "reason": f"collection not found: {settings.milvus_collection}",
        }
    except Exception as e:  # noqa: BLE001
        logger.warning("Milvus 健康检查失败: %s", e)
        return {"enabled": False, "reason": str(e)}
