from __future__ import annotations

import contextvars
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.tools import tool

_rag_sources_ctx: contextvars.ContextVar[Tuple[str, ...]] = contextvars.ContextVar(
    "rag_sources",
    default=(),
)
_rag_citations_ctx: contextvars.ContextVar[Tuple[Tuple[str, str], ...]] = contextvars.ContextVar(
    "rag_citations",
    default=(),
)


def get_last_rag_sources() -> List[str]:
    return list(_rag_sources_ctx.get())


def get_last_rag_citations() -> List[Dict[str, str]]:
    return [
        {"source": source, "excerpt": excerpt}
        for source, excerpt in _rag_citations_ctx.get()
    ]


def reset_rag_sources() -> None:
    _rag_sources_ctx.set(())
    _rag_citations_ctx.set(())


def _set_rag_sources(sources: List[str]) -> None:
    _rag_sources_ctx.set(tuple(sources))


def _set_rag_citations(citations: List[Tuple[str, str]]) -> None:
    _rag_citations_ctx.set(tuple(citations))


@dataclass(frozen=True)
class RagRetrieval:
    context: str
    sources: List[str]
    citations: List[Dict[str, str]]


def _doc_source(doc: Any) -> str:
    metadata = getattr(doc, "metadata", None) or {}
    return str(metadata.get("source", "unknown"))


def _doc_excerpt(doc: Any, *, max_chars: int = 420) -> str:
    return " ".join(str(getattr(doc, "page_content", "")).strip().split())[:max_chars]


def _query_terms(query: str) -> List[str]:
    lower = query.lower()
    terms = re.findall(r"[a-z0-9_./:-]{3,}", lower)
    for item in ("slothbearflow", "redis", "milvus", "postgres", "postgresql", "ollama", "umi", "rag"):
        if item in lower and item not in terms:
            terms.append(item)
    return terms


def _rank_docs(query: str, docs: List[Any]) -> List[Any]:
    terms = _query_terms(query)

    def score(item: Tuple[int, Any]) -> Tuple[int, int]:
        original_idx, doc = item
        source = _doc_source(doc)
        haystack = f"{source} {_doc_excerpt(doc, max_chars=1200)}".lower()
        lexical = sum(1 for term in terms if term in haystack)
        if source == "docs/SlothBearFlow-项目知识库问答卡片.md":
            lexical += 6
        elif source == "docs/SlothBearFlow-项目知识库种子数据.md":
            lexical += 5
        elif source == "docs/SlothBearFlow-本地运行与三组件集成优化记录.md":
            lexical += 3
        elif source == "README.md":
            lexical += 1
        if "是什么" in query and ("是什么" in haystack or "项目定位" in haystack):
            lexical += 6
        if source.startswith("codex-") or source == "manual-note.md":
            lexical -= 2
        return lexical, -original_idx

    ranked = sorted(enumerate(docs), key=score, reverse=True)
    return [doc for _, doc in ranked]


def _dedupe_docs(docs: List[Any]) -> List[Any]:
    seen: set[Tuple[str, str]] = set()
    deduped: List[Any] = []
    for doc in docs:
        key = (_doc_source(doc), _doc_excerpt(doc, max_chars=1200))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(doc)
    return deduped


def _keyword_search(vector_store: Any, query: str, k: int) -> List[Any]:
    search = getattr(vector_store, "keyword_search", None)
    if not callable(search):
        return []
    try:
        return list(search(query, k=k))
    except TypeError:
        return list(search(query))


def retrieve_knowledge_context(
    vector_store: Optional[Any],
    query: str,
    *,
    k: int = 24,
    max_context: int = 4,
) -> RagRetrieval:
    if vector_store is None:
        return RagRetrieval(context="", sources=[], citations=[])

    vector_docs = vector_store.similarity_search(query, k=k)
    keyword_docs = _keyword_search(vector_store, query, k=max_context * 2)
    docs = _dedupe_docs(keyword_docs + vector_docs)
    docs = _rank_docs(query, docs)
    sources: List[str] = []
    citations: List[Dict[str, str]] = []
    parts: List[str] = []
    for idx, doc in enumerate(docs[:max_context], start=1):
        source = _doc_source(doc)
        excerpt = _doc_excerpt(doc)
        if not excerpt:
            continue
        sources.append(source)
        citations.append({"source": source, "excerpt": excerpt[:220]})
        parts.append(f"[{idx}] source={source}\n{excerpt}")

    if not parts:
        return RagRetrieval(context="", sources=[], citations=[])
    return RagRetrieval(
        context="【检索片段】\n" + "\n\n---\n\n".join(parts),
        sources=sources,
        citations=citations,
    )


def build_search_knowledge_tool(vector_store: Optional[Any]):
    # TODO 向量召回 ANN模糊匹配 + BM25 精确召回 + Rerank 重排序
    @tool
    def search_knowledge(query: str) -> str:
        """Search the configured knowledge base for information related to the query."""
        if vector_store is None:
            _set_rag_sources([])
            _set_rag_citations([])
            return "（知识库未启用或暂不可用，请勿编造内部资料。）"
        retrieval = retrieve_knowledge_context(vector_store, query, k=24)
        _set_rag_sources(retrieval.sources)
        _set_rag_citations(
            [(item["source"], item["excerpt"]) for item in retrieval.citations]
        )
        return retrieval.context if retrieval.context else "（未检索到相关内容）"
    return search_knowledge
