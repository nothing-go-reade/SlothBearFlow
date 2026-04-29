from __future__ import annotations

import contextvars
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


def build_search_knowledge_tool(vector_store: Optional[Any]):
    """闭包注入向量库；无库时返回降级提示。"""

    @tool
    def search_knowledge(query: str) -> str:
        """从企业内部知识库检索与问题相关的片段（只读）。"""
        if vector_store is None:
            _set_rag_sources([])
            _set_rag_citations([])
            return "（知识库未启用或暂不可用，请勿编造内部资料。）"
        docs = vector_store.similarity_search(query, k=4)
        sources: List[str] = []
        citations: List[Tuple[str, str]] = []
        parts: List[str] = []
        for idx, d in enumerate(docs, start=1):
            src = "unknown"
            if getattr(d, "metadata", None):
                src = str(d.metadata.get("source", "unknown"))
            sources.append(src)
            excerpt = " ".join(str(d.page_content).strip().split())[:220]
            citations.append((src, excerpt))
            parts.append(f"[{idx}] source={src}\n{excerpt}")
        _set_rag_sources(sources)
        _set_rag_citations(citations)
        header = "【检索片段】\n"
        return header + "\n\n---\n\n".join(parts) if parts else "（未检索到相关内容）"

    return search_knowledge
