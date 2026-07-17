from __future__ import annotations

import re
from collections import Counter
from typing import Any, Iterable, List, Mapping

from backend.src.slothbearflow_backend.rag.security import (
    citation_is_from_current_recall,
    citation_source_is_safe,
    contains_prompt_injection,
)


def filter_safe_citations(citations: Iterable[Any]) -> List[Any]:
    safe = []
    for citation in citations:
        source = str(_citation_field(citation, "source") or "").strip()
        excerpt = str(_citation_field(citation, "excerpt") or "").strip()
        chunk_id = str(_citation_field(citation, "chunk_id") or "").strip()
        if not citation_source_is_safe(source) or not excerpt:
            continue
        recalled = citation_is_from_current_recall(source, chunk_id)
        if recalled is not True:
            continue
        if len(excerpt) > 4000 or contains_prompt_injection(excerpt):
            continue
        if any(ord(character) == 0 or 127 <= ord(character) <= 159 for character in excerpt):
            continue
        safe.append(citation)
    return safe


def verify_citation_support(answer: str, citations: Iterable[Any]) -> List[Any]:
    answer_terms = _terms(answer)
    rows = filter_safe_citations(citations)
    excerpt_terms = [_terms(str(_citation_field(citation, "excerpt") or "")) for citation in rows]
    document_frequency: Counter[str] = Counter()
    for terms in excerpt_terms:
        document_frequency.update(answer_terms.intersection(terms))

    verified = []
    for citation, terms in zip(rows, excerpt_terms):
        overlap = answer_terms.intersection(terms)
        weighted_overlap = sum(1.0 / max(1, document_frequency[term]) for term in overlap)
        support_score = weighted_overlap / max(1, len(answer_terms))
        if hasattr(citation, "model_copy"):
            verified.append(
                citation.model_copy(
                    update={
                        "support_score": round(support_score, 6),
                        "supported": support_score >= 0.08,
                    }
                )
            )
        elif isinstance(citation, Mapping):
            verified.append(
                {
                    **dict(citation),
                    "support_score": round(support_score, 6),
                    "supported": support_score >= 0.08,
                }
            )
        else:
            verified.append(citation)
    return verified


def _terms(text: str) -> set[str]:
    lowered = str(text or "").lower()
    terms = set(re.findall(r"[a-z0-9_./:-]{2,}", lowered))
    chars = re.findall(r"[\u4e00-\u9fff]", lowered)
    terms.update("".join(chars[index : index + 2]) for index in range(max(0, len(chars) - 1)))
    return terms


def _citation_field(citation: Any, field_name: str) -> Any:
    if isinstance(citation, Mapping):
        return citation.get(field_name)
    return getattr(citation, field_name, "")
