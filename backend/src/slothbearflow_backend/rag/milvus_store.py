from __future__ import annotations

import logging
import math
import re
import threading
import uuid
from collections import Counter
from typing import Any, Dict, List, Optional

from langchain_core.documents import Document

from backend.src.slothbearflow_backend import Settings, get_settings
from backend.src.slothbearflow_backend.rag.embedding import get_embedding_function

logger = logging.getLogger(__name__)

_vector_store: Optional[Any] = None
_vector_store_error: Optional[str] = None
_lock = threading.Lock()


def _tokenize_for_bm25(text: str) -> List[str]:
    text = str(text or "").lower()
    ascii_terms = re.findall(r"[a-z0-9_./:-]{2,}", text)
    cjk_chars = re.findall(r"[\u4e00-\u9fff]", text)
    cjk_bigrams = [
        "".join(cjk_chars[idx : idx + 2])
        for idx in range(max(0, len(cjk_chars) - 1))
    ]
    return ascii_terms + cjk_chars + cjk_bigrams


def _bm25_rank(query: str, documents: List[Document]) -> List[Document]:
    query_terms = _tokenize_for_bm25(query)
    if not query_terms or not documents:
        return []

    tokenized_docs = [_tokenize_for_bm25(doc.page_content) for doc in documents]
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
    for idx, (doc, tokens, doc_len) in enumerate(
        zip(documents, tokenized_docs, lengths)
    ):
        term_freq = Counter(tokens)
        score = 0.0
        for term in query_terms:
            freq = term_freq.get(term, 0)
            if freq <= 0:
                continue
            idf = math.log(
                1 + (total_docs - doc_freq[term] + 0.5) / (doc_freq[term] + 0.5)
            )
            denominator = freq + k1 * (1 - b + b * doc_len / avg_len)
            score += idf * (freq * (k1 + 1)) / denominator
        if score <= 0:
            continue
        metadata = dict(getattr(doc, "metadata", None) or {})
        metadata["bm25_score"] = round(score, 6)
        metadata["retrieval"] = "bm25"
        scored.append(
            (score, -idx, Document(page_content=doc.page_content, metadata=metadata))
        )

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
    ) -> None:
        from pymilvus import MilvusClient

        self.embedding_function = embedding_function
        self.collection_name = collection_name
        self.timeout = timeout
        self.client = MilvusClient(uri=uri, timeout=timeout)

    def _ensure_collection(self, dimension: int) -> None:
        if self.client.has_collection(self.collection_name, timeout=self.timeout):
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

    def add_documents(self, documents: List[Document]) -> None:
        if not documents:
            return
        texts = [str(doc.page_content or "") for doc in documents]
        vectors = self.embedding_function.embed_documents(texts)
        if not vectors:
            return
        self._ensure_collection(len(vectors[0]))
        rows = []
        for doc, vector in zip(documents, vectors):
            metadata = dict(getattr(doc, "metadata", None) or {})
            rows.append(
                {
                    "id": str(uuid.uuid4()),
                    "text": str(doc.page_content or ""),
                    "source": str(metadata.get("source", "unknown")),
                    "metadata": metadata,
                    "vector": vector,
                }
            )
        self.client.insert(
            collection_name=self.collection_name,
            data=rows,
            timeout=self.timeout,
        )
        self.client.flush(collection_name=self.collection_name, timeout=self.timeout)

    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        if not self.client.has_collection(self.collection_name, timeout=self.timeout):
            return []
        vector = self.embedding_function.embed_query(query)
        hits = self.client.search(
            collection_name=self.collection_name,
            data=[vector],
            limit=k,
            output_fields=["text", "source", "metadata"],
            search_params={"metric_type": "COSINE", "params": {}},
            timeout=self.timeout,
        )
        docs: List[Document] = []
        for hit in hits[0] if hits else []:
            entity = hit.get("entity", {})
            metadata = dict(entity.get("metadata") or {})
            metadata["source"] = str(
                entity.get("source") or metadata.get("source") or "unknown"
            )
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
    ) -> List[Document]:
        if not self.client.has_collection(self.collection_name, timeout=self.timeout):
            return []
        try:
            rows = self.client.query(
                collection_name=self.collection_name,
                filter='id != ""',
                output_fields=["text", "source", "metadata"],
                limit=max(k, candidate_limit),
                timeout=self.timeout,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Milvus BM25 候选读取失败: %s", exc)
            return []

        docs: List[Document] = []
        for row in rows:
            metadata = dict(row.get("metadata") or {})
            metadata["source"] = str(
                row.get("source") or metadata.get("source") or "unknown"
            )
            docs.append(
                Document(page_content=str(row.get("text") or ""), metadata=metadata)
            )
        return _bm25_rank(query, docs)[:k]


def reset_vector_store_cache() -> None:
    global _vector_store, _vector_store_error
    with _lock:
        _vector_store = None
        _vector_store_error = None


def get_vector_store(settings: Optional[Settings] = None) -> Optional[Any]:
    global _vector_store, _vector_store_error

    settings = settings or get_settings()

    if settings.skip_milvus or not settings.use_rag:
        return None

    if _vector_store is not None:
        return _vector_store

    # 如果之前失败过，直接降级（可优化点）
    if _vector_store_error is not None:
        return None

    with _lock:
        if _vector_store is not None:
            return _vector_store

        if _vector_store_error is not None:
            return None

        try:
            _vector_store = SimpleMilvusVectorStore(
                embedding_function=get_embedding_function(settings),
                collection_name=settings.milvus_collection,
                uri=settings.milvus_uri,
                timeout=max(float(settings.milvus_timeout), 10.0),
            )

            _vector_store_error = None
            return _vector_store

        except Exception as e:
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
        from pymilvus import MilvusClient

        client = MilvusClient(
            uri=settings.milvus_uri,
            timeout=max(float(settings.milvus_timeout), 10.0),
        )
        has_collection = client.has_collection(
            settings.milvus_collection,
            timeout=max(float(settings.milvus_timeout), 10.0),
        )
        if has_collection:
            return {"enabled": True, "collection": settings.milvus_collection}
        return {
            "enabled": False,
            "reason": f"collection not found: {settings.milvus_collection}",
        }
    except Exception as e:  # noqa: BLE001
        logger.warning("Milvus 健康检查失败: %s", e)
        return {"enabled": False, "reason": str(e)}
