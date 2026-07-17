from __future__ import annotations

import contextvars
import inspect
import math
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

from langchain_core.documents import Document
from langchain_core.tools import tool

from backend.src.slothbearflow_backend import Settings, get_settings
from backend.src.slothbearflow_backend.rag.reranker import get_reranker
from backend.src.slothbearflow_backend.rag.security import (
    RagAccessContext,
    citation_source_is_safe,
    contains_prompt_injection,
    document_is_authorized,
    metadata_contains_prompt_injection,
)
from backend.src.slothbearflow_backend.memory.short_memory import estimate_tokens

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


def get_last_rag_citations() -> List[Dict[str, Any]]:
    return [{"source": source, "excerpt": excerpt} for source, excerpt in _rag_citations_ctx.get()]


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
    citations: List[Dict[str, Any]]
    queries: List[str] = None  # type: ignore[assignment]
    blocked_count: int = 0
    no_answer_reason: str = ""

    def __post_init__(self) -> None:
        if self.queries is None:
            object.__setattr__(self, "queries", [])


@dataclass(frozen=True)
class RagToolResponse:
    content: str
    sources: List[str]
    citations: List[Dict[str, Any]]

    @property
    def provenance(self) -> Dict[str, Any]:
        return {"type": "rag", "sources": list(self.sources)}

    def __str__(self) -> str:
        return self.content


def _doc_source(doc: Any) -> str:
    metadata = getattr(doc, "metadata", None) or {}
    return str(metadata.get("source", "unknown"))


def _doc_excerpt(doc: Any, *, max_chars: int = 420) -> str:
    return " ".join(str(getattr(doc, "page_content", "")).strip().split())[:max_chars]


def _query_terms(query: str) -> List[str]:
    lower = query.lower()
    terms = re.findall(r"[a-z0-9_./:-]{2,}", lower)
    for sequence in re.findall(r"[\u4e00-\u9fff]{2,}", lower):
        for size in (2, 3):
            terms.extend(
                sequence[index : index + size] for index in range(max(0, len(sequence) - size + 1))
            )
    return list(dict.fromkeys(terms))


def _rank_docs(query: str, docs: List[Any]) -> List[Any]:
    terms = _query_terms(query)

    def score(item: Tuple[int, Any]) -> Tuple[int, int]:
        original_idx, doc = item
        haystack = f"{_doc_source(doc)} {_doc_excerpt(doc, max_chars=1200)}".lower()
        lexical = sum(1 for term in terms if term in haystack)
        if "是什么" in query and ("是什么" in haystack or "项目定位" in haystack):
            lexical += 2
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


def _query_variants(query: str, *, enabled: bool) -> List[str]:
    original = " ".join(str(query or "").split()).strip()
    if not original:
        return []
    if not enabled:
        return [original]
    simplified = re.sub(
        r"(请问|麻烦|帮我|一下|是什么|有哪些|如何|怎么|为什么|？|\?)",
        " ",
        original,
    )
    simplified = " ".join(simplified.split()).strip()
    terms = _query_terms(original)
    keyword_query = " ".join(terms[:16])
    return list(dict.fromkeys(value for value in (original, simplified, keyword_query) if value))[
        :3
    ]


def _rrf_fuse(rankings: Iterable[List[Any]], *, rrf_k: int) -> List[Any]:
    rows: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for ranking_index, ranking in enumerate(rankings):
        for rank, doc in enumerate(ranking, start=1):
            key = (_doc_source(doc), _doc_excerpt(doc, max_chars=1200))
            row = rows.setdefault(
                key,
                {
                    "document": doc,
                    "score": 0.0,
                    "score_metadata": {},
                    "methods": set(),
                    "first": (ranking_index, rank),
                },
            )
            row["score"] += 1.0 / (max(1, rrf_k) + rank)
            metadata = dict(getattr(doc, "metadata", None) or {})
            row["methods"].add(str(metadata.get("retrieval") or "vector"))
            for score_name in (
                "vector_score",
                "bm25_score",
                "lexical_score",
                "rerank_score",
            ):
                try:
                    score_value = float(metadata.get(score_name) or 0.0)
                except (TypeError, ValueError):
                    continue
                if not math.isfinite(score_value):
                    continue
                if score_value > float(row["score_metadata"].get(score_name) or 0.0):
                    row["score_metadata"][score_name] = score_value
            if _numeric_score(metadata.get("vector_score")) > _numeric_score(
                (getattr(row["document"], "metadata", None) or {}).get("vector_score")
            ):
                row["document"] = doc
    fused = []
    for row in rows.values():
        document = row["document"]
        metadata = dict(getattr(document, "metadata", None) or {})
        metadata.update(row["score_metadata"])
        metadata["rrf_score"] = round(float(row["score"]), 8)
        metadata["retrieval"] = "+".join(sorted(row["methods"]))
        fused.append(
            (
                float(row["score"]),
                tuple(-value for value in row["first"]),
                Document(page_content=document.page_content, metadata=metadata),
            )
        )
    fused.sort(reverse=True)
    return [document for _, _, document in fused]


def _search_accepts_keyword(search: Any, keyword: str) -> bool:
    try:
        parameters = inspect.signature(search).parameters.values()
    except (TypeError, ValueError):
        return False
    return any(
        parameter.name == keyword or parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in parameters
    )


def _similarity_search(
    vector_store: Any,
    query: str,
    k: int,
    access_context: RagAccessContext,
) -> List[Any]:
    search = getattr(vector_store, "similarity_search", None)
    if not callable(search):
        return []
    kwargs: Dict[str, Any] = {}
    if _search_accepts_keyword(search, "k"):
        kwargs["k"] = k
    if _search_accepts_keyword(search, "access_context"):
        kwargs["access_context"] = access_context
    return list(search(query, **kwargs))


def _keyword_search(
    vector_store: Any,
    query: str,
    k: int,
    access_context: RagAccessContext,
) -> List[Any]:
    search = getattr(vector_store, "keyword_search", None)
    if not callable(search):
        return []
    kwargs: Dict[str, Any] = {}
    if _search_accepts_keyword(search, "k"):
        kwargs["k"] = k
    if _search_accepts_keyword(search, "access_context"):
        kwargs["access_context"] = access_context
    return list(search(query, **kwargs))


def _relevance_score(
    metadata: Dict[str, Any],
    *,
    rrf_k: int,
    include_retrieval_scores: bool,
) -> float:
    score_names = ["rerank_score", "vector_score", "lexical_score"]
    scores = [_unit_score(metadata.get(name)) for name in score_names]
    if include_retrieval_scores:
        scores.append(_unit_score(metadata.get("bm25_score")))
        rrf_score = max(0.0, _numeric_score(metadata.get("rrf_score")))
        scores.append(min(1.0, rrf_score * (max(1, int(rrf_k)) + 1)))
    return max(scores, default=0.0)


def _unit_score(value: Any) -> float:
    return max(0.0, min(1.0, _numeric_score(value)))


def _numeric_score(value: Any) -> float:
    try:
        score = float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0
    return score if math.isfinite(score) else 0.0


def retrieve_knowledge_context(
    vector_store: Optional[Any],
    query: str,
    *,
    k: Optional[int] = None,
    max_context: Optional[int] = None,
    settings: Optional[Settings] = None,
    access_context: Optional[RagAccessContext] = None,
) -> RagRetrieval:
    settings = settings or get_settings()
    k = max(1, int(k or settings.rag_retrieval_top_k))
    max_context = max(1, int(max_context or settings.rag_max_context_chunks))
    if vector_store is None:
        return RagRetrieval(
            context="",
            sources=[],
            citations=[],
            no_answer_reason="vector_store_unavailable",
        )

    access = access_context or RagAccessContext(
        allow_legacy=bool(settings.rag_allow_legacy_documents)
    )
    queries = _query_variants(query, enabled=bool(settings.rag_multi_query))
    rankings: List[List[Any]] = []
    for variant in queries:
        rankings.append(_similarity_search(vector_store, variant, k, access))
        rankings.append(_keyword_search(vector_store, variant, max_context * 3, access))
    docs = _rrf_fuse(rankings, rrf_k=settings.rag_rrf_k)
    authorized: List[Any] = []
    blocked_count = 0
    for doc in docs:
        metadata = dict(getattr(doc, "metadata", None) or {})
        if not document_is_authorized(metadata, access):
            blocked_count += 1
            continue
        if not citation_source_is_safe(metadata.get("source")):
            blocked_count += 1
            continue
        if settings.rag_block_prompt_injection and (
            contains_prompt_injection(str(getattr(doc, "page_content", "")))
            or metadata_contains_prompt_injection(metadata)
        ):
            blocked_count += 1
            continue
        authorized.append(doc)
    reranker = get_reranker(settings)
    docs = reranker.rerank(query, authorized) if reranker else _rank_docs(query, authorized)
    include_retrieval_scores = reranker is None
    sources: List[str] = []
    citations: List[Dict[str, Any]] = []
    parts: List[str] = []
    selected: List[Any] = []
    selected_tokens = 0
    for doc in docs:
        metadata = dict(getattr(doc, "metadata", None) or {})
        relevance_score = _relevance_score(
            metadata,
            rrf_k=settings.rag_rrf_k,
            include_retrieval_scores=include_retrieval_scores,
        )
        if relevance_score < settings.rag_relevance_threshold:
            continue
        token_cost = estimate_tokens(str(getattr(doc, "page_content", "")))
        if selected and selected_tokens + token_cost > settings.rag_context_max_tokens:
            continue
        selected.append(doc)
        selected_tokens += token_cost
        if len(selected) >= max_context:
            break

    for idx, doc in enumerate(selected, start=1):
        source = _doc_source(doc)
        excerpt = _doc_excerpt(doc)
        metadata = dict(getattr(doc, "metadata", None) or {})
        if not excerpt:
            continue
        sources.append(source)
        citations.append(
            {
                "source": source,
                "excerpt": excerpt[:220],
                "score": round(
                    _relevance_score(
                        metadata,
                        rrf_k=settings.rag_rrf_k,
                        include_retrieval_scores=include_retrieval_scores,
                    ),
                    6,
                ),
                "chunk_id": str(metadata.get("chunk_id") or ""),
                "retrieval": str(metadata.get("retrieval") or "hybrid"),
            }
        )
        parts.append(f"[{idx}] source={source} chunk_id={metadata.get('chunk_id', '')}\n{excerpt}")

    if not parts:
        return RagRetrieval(
            context="",
            sources=[],
            citations=[],
            queries=queries,
            blocked_count=blocked_count,
            no_answer_reason=(
                "all_candidates_blocked"
                if blocked_count and not authorized
                else "below_relevance_threshold"
            ),
        )
    return RagRetrieval(
        context=(
            "【检索片段】\n【UNTRUSTED_KNOWLEDGE_BEGIN】\n"
            + "\n\n---\n\n".join(parts)
            + "\n【UNTRUSTED_KNOWLEDGE_END】"
        ),
        sources=sources,
        citations=citations,
        queries=queries,
        blocked_count=blocked_count,
    )


def build_search_knowledge_tool(
    vector_store: Optional[Any],
    *,
    settings: Optional[Settings] = None,
    access_context: Optional[RagAccessContext] = None,
):
    settings = settings or get_settings()

    @tool
    def search_knowledge(query: str) -> Any:
        """Search the configured knowledge base for information related to the query."""
        if vector_store is None:
            _set_rag_sources([])
            _set_rag_citations([])
            return RagToolResponse(
                content="（知识库未启用或暂不可用，请勿编造内部资料。）",
                sources=[],
                citations=[],
            )
        retrieval = retrieve_knowledge_context(
            vector_store,
            query,
            settings=settings,
            access_context=access_context,
        )
        _set_rag_sources(retrieval.sources)
        _set_rag_citations([(item["source"], item["excerpt"]) for item in retrieval.citations])
        return RagToolResponse(
            content=retrieval.context if retrieval.context else "（未检索到相关内容）",
            sources=retrieval.sources,
            citations=retrieval.citations,
        )

    return search_knowledge
