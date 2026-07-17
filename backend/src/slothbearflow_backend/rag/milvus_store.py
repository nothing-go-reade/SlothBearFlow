from __future__ import annotations

import json
import logging
import math
import re
import threading
import time
import uuid
from collections import Counter
from typing import Any, Dict, List, Optional

from langchain_core.documents import Document

from backend.src.slothbearflow_backend import Settings, get_settings
from backend.src.slothbearflow_backend.rag.embedding import get_embedding_function
from backend.src.slothbearflow_backend.rag.security import (
    RagAccessContext,
    build_milvus_acl_filters,
    document_is_authorized,
    query_requests_secret_value,
)

logger = logging.getLogger(__name__)

_vector_store: Optional[Any] = None
_vector_store_error: Optional[str] = None
_vector_store_error_at: float = 0.0
_lock = threading.Lock()


def _tokenize_for_bm25(text: str) -> List[str]:
    text = str(text or "").lower()
    ascii_terms = re.findall(r"[a-z0-9_./:-]{2,}", text)
    cjk_chars = re.findall(r"[\u4e00-\u9fff]", text)
    cjk_bigrams = ["".join(cjk_chars[idx : idx + 2]) for idx in range(max(0, len(cjk_chars) - 1))]
    return ascii_terms + cjk_chars + cjk_bigrams


def _bm25_rank(query: str, documents: List[Document]) -> List[Document]:
    query_terms = _tokenize_for_bm25(query)
    if not query_terms or not documents:
        return []

    tokenized_docs = [
        _tokenize_for_bm25(
            " ".join(
                (
                    str((doc.metadata or {}).get("source") or ""),
                    str((doc.metadata or {}).get("section") or ""),
                    doc.page_content,
                )
            )
        )
        for doc in documents
    ]
    lengths = [len(tokens) for tokens in tokenized_docs]
    avg_len = sum(lengths) / len(lengths) if lengths else 0.0
    if avg_len <= 0:
        return []

    doc_freq: Counter[str] = Counter()
    for tokens in tokenized_docs:
        doc_freq.update(set(tokens))

    total_docs = len(documents)
    k1 = 1.5
    b = 0.75
    scored: List[tuple[float, int, Document]] = []
    for idx, (doc, tokens, doc_len) in enumerate(zip(documents, tokenized_docs, lengths)):
        term_freq = Counter(tokens)
        score = 0.0
        for term in query_terms:
            freq = term_freq.get(term, 0)
            if freq <= 0:
                continue
            idf = math.log(1 + (total_docs - doc_freq[term] + 0.5) / (doc_freq[term] + 0.5))
            denominator = freq + k1 * (1 - b + b * doc_len / avg_len)
            score += idf * (freq * (k1 + 1)) / denominator
        if score <= 0:
            continue
        metadata = dict(getattr(doc, "metadata", None) or {})
        metadata["bm25_score"] = round(score, 6)
        metadata["retrieval"] = "bm25"
        scored.append((score, -idx, Document(page_content=doc.page_content, metadata=metadata)))

    scored.sort(reverse=True)
    return [doc for _, _, doc in scored]


class SimpleMilvusVectorStore:
    def __init__(
        self,
        *,
        embedding_function: Any,
        collection_name: str,
        uri: str,
        timeout: float,
        token: str = "",
    ) -> None:
        from pymilvus import MilvusClient

        self.embedding_function = embedding_function
        self.collection_name = collection_name
        self.timeout = timeout
        client_kwargs: Dict[str, Any] = {"uri": uri, "timeout": timeout}
        if str(token).strip():
            client_kwargs["token"] = str(token).strip()
        self.client = MilvusClient(**client_kwargs)

    def _ensure_collection(self, dimension: int) -> None:
        if self.client.has_collection(self.collection_name, timeout=self.timeout):
            describe = getattr(self.client, "describe_collection", None)
            if not callable(describe):
                raise RuntimeError("Milvus client cannot verify collection dimensions.")
            description = describe(
                collection_name=self.collection_name,
                timeout=self.timeout,
            )
            fields = description.get("fields") if isinstance(description, dict) else None
            vector_field = next(
                (
                    field
                    for field in (fields or [])
                    if isinstance(field, dict) and field.get("name") == "vector"
                ),
                None,
            )
            params = vector_field.get("params") if isinstance(vector_field, dict) else None
            try:
                existing_dimension = int((params or {}).get("dim"))
            except (TypeError, ValueError) as exc:
                raise RuntimeError(
                    "Milvus collection does not expose its vector dimension."
                ) from exc
            if existing_dimension != dimension:
                raise RuntimeError(
                    "Embedding dimension does not match the existing Milvus collection "
                    f"({dimension} != {existing_dimension})."
                )
            return

        from pymilvus import CollectionSchema, DataType, FieldSchema

        schema = CollectionSchema(
            [
                FieldSchema(
                    name="id",
                    dtype=DataType.VARCHAR,
                    is_primary=True,
                    max_length=64,
                ),
                FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
                FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=2048),
                FieldSchema(name="metadata", dtype=DataType.JSON),
                FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=dimension),
            ],
            description="SlothBearFlow knowledge chunks",
        )
        index_params = self.client.prepare_index_params()
        index_params.add_index(
            field_name="vector",
            index_type="AUTOINDEX",
            metric_type="COSINE",
        )
        self.client.create_collection(
            collection_name=self.collection_name,
            schema=schema,
            index_params=index_params,
            timeout=self.timeout,
        )
        self.client.load_collection(self.collection_name, timeout=self.timeout)

    def add_documents(self, documents: List[Document]) -> int:
        if not documents:
            return 0
        texts = [str(doc.page_content or "") for doc in documents]
        vectors = self.embedding_function.embed_documents(texts)
        if vectors is None:
            raise RuntimeError("Embedding provider returned no vectors.")
        try:
            vector_count = len(vectors)
        except TypeError as exc:
            raise RuntimeError("Embedding provider returned an invalid vector batch.") from exc
        if vector_count != len(documents):
            raise RuntimeError(
                "Embedding provider returned an incomplete vector batch "
                f"({vector_count}/{len(documents)})."
            )

        normalized_vectors: List[List[float]] = []
        dimension = 0
        for index, vector in enumerate(vectors):
            try:
                values = [float(value) for value in vector]
            except (TypeError, ValueError) as exc:
                raise RuntimeError(f"Embedding vector {index} is invalid.") from exc
            if not values or any(not math.isfinite(value) for value in values):
                raise RuntimeError(f"Embedding vector {index} is empty or non-finite.")
            if dimension and len(values) != dimension:
                raise RuntimeError("Embedding vectors have inconsistent dimensions.")
            dimension = len(values)
            normalized_vectors.append(values)

        self._ensure_collection(dimension)
        rows = []
        for doc, vector in zip(documents, normalized_vectors):
            metadata = dict(getattr(doc, "metadata", None) or {})
            rows.append(
                {
                    "id": str(metadata.get("chunk_id") or uuid.uuid4()),
                    "text": str(doc.page_content or ""),
                    "source": str(metadata.get("source", "unknown")),
                    "metadata": metadata,
                    "vector": vector,
                }
            )
        upsert = getattr(self.client, "upsert", None)
        if callable(upsert):
            write_result = upsert(
                collection_name=self.collection_name,
                data=rows,
                timeout=self.timeout,
            )
        else:
            write_result = self.client.insert(
                collection_name=self.collection_name,
                data=rows,
                timeout=self.timeout,
            )
        if isinstance(write_result, dict):
            reported_count = write_result.get("upsert_count")
            if reported_count is None:
                reported_count = write_result.get("insert_count")
            if reported_count is not None and int(reported_count) != len(rows):
                raise RuntimeError(
                    "Milvus acknowledged an incomplete write "
                    f"({reported_count}/{len(rows)} rows)."
                )
        self.client.flush(collection_name=self.collection_name, timeout=self.timeout)
        return len(rows)

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        *,
        access_context: Optional[RagAccessContext] = None,
    ) -> List[Document]:
        if query_requests_secret_value(query):
            logger.warning("拒绝检索真实凭据值请求")
            return []
        if access_context is None:
            logger.warning("拒绝缺少访问上下文的向量检索")
            return []
        if not self.client.has_collection(self.collection_name, timeout=self.timeout):
            return []
        vector = self.embedding_function.embed_query(query)
        acl_filters = build_milvus_acl_filters(access_context)
        hit_rows: Dict[str, Dict[str, Any]] = {}
        for acl_filter in acl_filters:
            hits = self.client.search(
                collection_name=self.collection_name,
                data=[vector],
                filter=acl_filter,
                limit=k,
                output_fields=["id", "text", "source", "metadata"],
                search_params={"metric_type": "COSINE", "params": {}},
                timeout=self.timeout,
            )
            for hit in hits[0] if hits else []:
                entity = hit.get("entity", {})
                key = str(
                    hit.get("id")
                    or entity.get("id")
                    or (entity.get("metadata") or {}).get("chunk_id")
                    or f"{entity.get('source', '')}:{entity.get('text', '')}"
                )
                existing = hit_rows.get(key)
                if existing is None or _hit_score(hit) > _hit_score(existing):
                    hit_rows[key] = hit
        ranked_hits = sorted(hit_rows.values(), key=_hit_score, reverse=True)[:k]
        docs: List[Document] = []
        for hit in ranked_hits:
            entity = hit.get("entity", {})
            metadata = dict(entity.get("metadata") or {})
            metadata["source"] = str(entity.get("source") or metadata.get("source") or "unknown")
            distance = hit.get("distance", hit.get("score"))
            if distance is not None:
                metadata["vector_score"] = round(float(distance), 6)
            metadata["retrieval"] = "vector"
            if document_is_authorized(metadata, access_context):
                docs.append(
                    Document(page_content=str(entity.get("text") or ""), metadata=metadata)
                )
        return docs

    def keyword_search(
        self,
        query: str,
        *,
        k: int = 8,
        candidate_limit: int = 512,
        access_context: Optional[RagAccessContext] = None,
    ) -> List[Document]:
        if query_requests_secret_value(query):
            logger.warning("拒绝关键词检索真实凭据值请求")
            return []
        if access_context is None:
            logger.warning("拒绝缺少访问上下文的关键词检索")
            return []
        if not self.client.has_collection(self.collection_name, timeout=self.timeout):
            return []
        acl_filters = build_milvus_acl_filters(access_context)
        row_map: Dict[str, Dict[str, Any]] = {}
        for acl_filter in acl_filters:
            try:
                rows = self.client.query(
                    collection_name=self.collection_name,
                    filter=acl_filter,
                    output_fields=["id", "text", "source", "metadata"],
                    limit=max(k, candidate_limit),
                    timeout=self.timeout,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("Milvus BM25 ACL 分支读取失败: %s", exc)
                continue
            for row in rows:
                key = str(
                    row.get("id")
                    or (row.get("metadata") or {}).get("chunk_id")
                    or f"{row.get('source', '')}:{row.get('text', '')}"
                )
                row_map.setdefault(key, row)

        docs: List[Document] = []
        for row in row_map.values():
            metadata = dict(row.get("metadata") or {})
            metadata["source"] = str(row.get("source") or metadata.get("source") or "unknown")
            if document_is_authorized(metadata, access_context):
                docs.append(Document(page_content=str(row.get("text") or ""), metadata=metadata))
        return _bm25_rank(query, docs)[:k]

    def delete_stale_document_versions(
        self,
        *,
        document_id: str,
        current_version: str,
        source: str,
        tenant_id: str,
        owner_id: str = "",
    ) -> int:
        if not self.client.has_collection(self.collection_name, timeout=self.timeout):
            return 0

        document_value = _milvus_literal(document_id)
        version_value = _milvus_literal(current_version)
        source_value = _milvus_literal(source)
        tenant_value = _milvus_literal(tenant_id)
        owner_value = _milvus_literal(owner_id)
        document_match = f'metadata["document_id"] == {document_value}'
        tenant_match = f'metadata["tenant_id"] == {tenant_value}'
        scope_match = f"{document_match} and {tenant_match}"
        source_scope = f"source == {source_value} and {tenant_match}"
        if owner_id:
            scope_match += f' and metadata["owner_id"] == {owner_value}'
            source_scope += f' and metadata["owner_id"] == {owner_value}'
        missing_version = '(not exists metadata["document_version"])'
        delete_filters = [
            f'{scope_match} and metadata["document_version"] != {version_value}',
            f"{scope_match} and {missing_version}",
            f"{source_scope} and {missing_version}",
        ]
        if owner_id:
            delete_filters.append(
                f"source == {source_value} and {tenant_match} and "
                f'metadata["owner_id"] == {owner_value} and '
                f'metadata["document_id"] != {document_value}'
            )
        if tenant_id == "local":
            delete_filters.extend(
                [
                    f"source == {source_value} and (not exists "
                    f'metadata["tenant_id"]) and {missing_version}',
                    f"source == {source_value} and "
                    f'metadata["tenant_id"] == "" and {missing_version}',
                ]
            )

        deleted_count = 0
        cleanup_errors = []
        for delete_filter in delete_filters:
            try:
                result = self.client.delete(
                    collection_name=self.collection_name,
                    filter=delete_filter,
                    timeout=self.timeout,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("Milvus stale-version cleanup branch failed: %s", exc)
                cleanup_errors.append(exc)
                continue
            if isinstance(result, dict):
                deleted_count += max(0, int(result.get("delete_count") or 0))
        self.client.flush(collection_name=self.collection_name, timeout=self.timeout)
        if cleanup_errors:
            raise RuntimeError(
                f"Milvus stale-version cleanup failed in {len(cleanup_errors)} branch(es)"
            ) from cleanup_errors[0]
        return deleted_count


def reset_vector_store_cache() -> None:
    global _vector_store, _vector_store_error, _vector_store_error_at
    with _lock:
        _vector_store = None
        _vector_store_error = None
        _vector_store_error_at = 0.0


def get_vector_store(settings: Optional[Settings] = None) -> Optional[Any]:
    global _vector_store, _vector_store_error, _vector_store_error_at

    settings = settings or get_settings()

    if settings.skip_milvus or not settings.use_rag:
        return None

    if _vector_store is not None:
        return _vector_store

    # 如果之前失败过，直接降级（可优化点）
    retry_interval = max(0.1, float(settings.milvus_retry_interval_sec))
    if (
        _vector_store_error is not None
        and time.monotonic() - _vector_store_error_at < retry_interval
    ):
        return None

    with _lock:
        if _vector_store is not None:
            return _vector_store

        if (
            _vector_store_error is not None
            and time.monotonic() - _vector_store_error_at < retry_interval
        ):
            return None

        try:
            _vector_store = SimpleMilvusVectorStore(
                embedding_function=get_embedding_function(settings),
                collection_name=settings.milvus_collection,
                uri=settings.milvus_uri,
                timeout=max(float(settings.milvus_timeout), 10.0),
                token=settings.milvus_token,
            )

            _vector_store_error = None
            _vector_store_error_at = 0.0
            return _vector_store

        except Exception as e:
            _vector_store_error = str(e)
            _vector_store_error_at = time.monotonic()
            logger.warning("Milvus 初始化失败，RAG 将降级关闭: %s", e)
            return None


def get_vector_store_status(settings: Optional[Settings] = None) -> Dict[str, Any]:
    settings = settings or get_settings()
    if settings.skip_milvus:
        return {"enabled": False, "reason": "SKIP_MILVUS=true"}
    if not settings.use_rag:
        return {"enabled": False, "reason": "USE_RAG=false"}
    try:
        from pymilvus import MilvusClient

        client_kwargs: Dict[str, Any] = {
            "uri": settings.milvus_uri,
            "timeout": max(float(settings.milvus_timeout), 0.1),
        }
        if str(settings.milvus_token).strip():
            client_kwargs["token"] = str(settings.milvus_token).strip()
        client = MilvusClient(**client_kwargs)
        has_collection = client.has_collection(
            settings.milvus_collection,
            timeout=max(float(settings.milvus_timeout), 0.1),
        )
        return {
            "enabled": True,
            "collection": settings.milvus_collection,
            "collection_exists": bool(has_collection),
            "state": "ready" if has_collection else "empty",
        }
    except Exception as e:  # noqa: BLE001
        logger.warning("Milvus 健康检查失败: %s", e)
        return {"enabled": False, "reason": str(e)}


def _milvus_literal(value: Any) -> str:
    return json.dumps(str(value or ""), ensure_ascii=False)


def _hit_score(hit: Dict[str, Any]) -> float:
    try:
        return float(hit.get("distance", hit.get("score")) or 0.0)
    except (TypeError, ValueError):
        return 0.0
