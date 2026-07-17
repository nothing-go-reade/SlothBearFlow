from __future__ import annotations

import logging
import math
import re
from functools import lru_cache
from typing import Any, List, Sequence

from langchain_core.documents import Document


logger = logging.getLogger(__name__)


def _terms(text: str) -> set[str]:
    lowered = str(text or "").lower()
    values = set(re.findall(r"[a-z0-9_./:-]{2,}", lowered))
    for sequence in re.findall(r"[\u4e00-\u9fff]{2,}", lowered):
        values.update(
            sequence[index : index + size]
            for size in (2, 3)
            for index in range(max(0, len(sequence) - size + 1))
        )
    return values


def _identifiers(text: str) -> set[str]:
    return {
        value.lower()
        for value in re.findall(
            r"(?<![A-Za-z0-9])(?:[A-Za-z][A-Za-z0-9]*[-_:][A-Za-z0-9._:-]+|[A-Z]{2,}\d{2,})(?![A-Za-z0-9])",
            str(text or ""),
        )
    }


class LexicalReranker:
    def rerank(self, query: str, documents: Sequence[Document]) -> List[Document]:
        query_terms = _terms(query)
        query_identifiers = _identifiers(query)
        scored = []
        for index, document in enumerate(documents):
            metadata = dict(document.metadata or {})
            searchable = " ".join(
                (
                    str(metadata.get("source") or ""),
                    str(metadata.get("section") or ""),
                    document.page_content,
                )
            )
            doc_terms = _terms(searchable)
            overlap = len(query_terms.intersection(doc_terms))
            lexical_score = overlap / max(1, len(query_terms))
            identifier_score = len(query_identifiers.intersection(_identifiers(searchable))) / max(
                1, len(query_identifiers)
            )
            rrf_score = float(metadata.get("rrf_score") or 0.0)
            vector_score = float(metadata.get("vector_score") or 0.0)
            combined = (
                lexical_score * 0.3
                + identifier_score * 0.4
                + min(1.0, rrf_score * 20) * 0.2
                + max(0.0, min(1.0, vector_score)) * 0.1
            )
            metadata.update(
                {
                    "lexical_score": round(lexical_score, 6),
                    "identifier_score": round(identifier_score, 6),
                    "rerank_score": round(combined, 6),
                    "reranker": "lexical",
                }
            )
            scored.append(
                (
                    combined,
                    identifier_score,
                    lexical_score,
                    rrf_score,
                    -index,
                    Document(page_content=document.page_content, metadata=metadata),
                )
            )
        scored.sort(reverse=True)
        return [item[-1] for item in scored]


class CrossEncoderReranker:
    def __init__(self, model_name: str) -> None:
        from sentence_transformers import CrossEncoder  # type: ignore

        self._model = CrossEncoder(model_name)

    def rerank(self, query: str, documents: Sequence[Document]) -> List[Document]:
        if not documents:
            return []
        scores = self._model.predict([(query, doc.page_content) for doc in documents])
        raw_scores = [float(score) for score in scores]
        if len(raw_scores) != len(documents):
            raise RuntimeError(
                "Cross-encoder returned an incomplete score batch "
                f"({len(raw_scores)}/{len(documents)})."
            )
        already_probabilities = bool(raw_scores) and all(
            math.isfinite(score) and 0.0 <= score <= 1.0 for score in raw_scores
        )
        ranked = []
        for index, (document, raw_score) in enumerate(zip(documents, raw_scores)):
            if not math.isfinite(raw_score):
                score = 0.0
            elif already_probabilities:
                score = raw_score
            else:
                score = 1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, raw_score))))
            metadata = dict(document.metadata or {})
            metadata.update(
                {
                    "rerank_score": round(score, 6),
                    "rerank_raw_score": round(raw_score, 6),
                    "reranker": "cross_encoder",
                }
            )
            ranked.append(
                (
                    score,
                    -index,
                    Document(page_content=document.page_content, metadata=metadata),
                )
            )
        ranked.sort(reverse=True)
        return [document for _, _, document in ranked]


@lru_cache(maxsize=4)
def _cross_encoder(model_name: str) -> Any:
    return CrossEncoderReranker(model_name)


def get_reranker(settings: Any) -> Any:
    provider = str(getattr(settings, "rag_reranker_provider", "lexical") or "lexical")
    if provider == "none":
        return None
    if provider == "cross_encoder":
        try:
            return _cross_encoder(str(settings.rag_cross_encoder_model))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Cross-encoder reranker unavailable; using lexical: %s", exc)
    return LexicalReranker()
