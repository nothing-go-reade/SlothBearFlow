from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Iterable, List, Mapping, Optional

from langchain_core.documents import Document

from backend.src.slothbearflow_backend.evaluation.schema import (
    EvaluationReport,
    RagAnswerResult,
    RagCaseScore,
    RagEvaluationCase,
)


DEFAULT_DATASET = Path(__file__).parent / "datasets" / "rag_regression_v1.jsonl"
MUST_PASS_TAGS = frozenset({"acl", "security", "no-answer"})


class ProductionAclEvaluationStore:
    """Deterministic candidates passed through the production RAG ACL pipeline."""

    def __init__(self) -> None:
        self._documents = [
            _acl_document(
                "ACL_PRIVATE_ALICE_MARKER",
                "eval/private-alice.md",
                tenant_id="local",
                owner_id="alice",
                visibility="private",
            ),
            _acl_document(
                "ACL_ROLE_OPS_MARKER",
                "eval/role-ops.md",
                tenant_id="local",
                visibility="tenant",
                allowed_roles=["ops"],
            ),
            _acl_document(
                "ACL_CROSS_TENANT_MARKER",
                "eval/cross-tenant.md",
                tenant_id="other-tenant",
                visibility="tenant",
            ),
        ]

    def similarity_search(self, query: str, k: int, **_kwargs: Any) -> List[Document]:
        marker = str(query or "").strip().upper()
        return [
            document
            for document in self._documents
            if marker and marker in document.page_content.upper()
        ][:k]

    def keyword_search(self, query: str, *, k: int, **_kwargs: Any) -> List[Document]:
        del query, k
        return []


def _acl_document(
    marker: str,
    source: str,
    *,
    tenant_id: str,
    visibility: str,
    owner_id: str = "",
    allowed_roles: Optional[List[str]] = None,
) -> Document:
    return Document(
        page_content=f"{marker} production-equivalent ACL evaluation fact.",
        metadata={
            "source": source,
            "chunk_id": f"chunk-{marker.lower()}",
            "tenant_id": tenant_id,
            "owner_id": owner_id,
            "visibility": visibility,
            "allowed_roles": list(allowed_roles or []),
            "vector_score": 0.999,
            "retrieval": "vector",
        },
    )


def load_dataset(path: Any = DEFAULT_DATASET) -> List[RagEvaluationCase]:
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped:
            rows.append(RagEvaluationCase(**json.loads(stripped)))
    return rows


def evaluate_rag_dataset(
    cases: Iterable[RagEvaluationCase],
    retrieve: Callable[[str], Any],
    *,
    dataset_version: str = "rag-regression-v1",
    retrieve_case: Optional[Callable[[RagEvaluationCase], Any]] = None,
    answer_case: Optional[Callable[[RagEvaluationCase, Any], Any]] = None,
) -> EvaluationReport:
    case_rows = list(cases)
    scores: List[RagCaseScore] = []
    for case in case_rows:
        retrieval = retrieve_case(case) if retrieve_case is not None else retrieve(case.question)
        citations = list(getattr(retrieval, "citations", None) or [])
        sources = [str(item.get("source") or "") for item in citations]
        context = str(getattr(retrieval, "context", "") or "").lower()
        ranks = [
            index + 1
            for index, source in enumerate(sources)
            if source in set(case.expected_sources)
        ]
        source_hit = not case.expected_sources or bool(ranks)
        acl_safe = not bool(set(sources).intersection(case.forbidden_sources))
        reciprocal_rank = 1.0 / min(ranks) if ranks else 0.0
        matched_terms = sum(1 for term in case.expected_terms if term.lower() in context)
        term_recall = matched_terms / max(1, len(case.expected_terms))
        has_answer = bool(context.strip())
        no_answer_correct = has_answer == bool(case.should_answer)
        retrieval_passed = (
            source_hit
            and acl_safe
            and no_answer_correct
            and (not case.expected_terms or term_recall >= 0.5)
        )
        answer_evaluated = False
        answer_source_hit: Optional[bool] = None
        answer_term_recall: Optional[float] = None
        answer_no_answer_correct: Optional[bool] = None
        answer_acl_safe: Optional[bool] = None
        answer_passed: Optional[bool] = None
        if answer_case is not None:
            raw_answer = answer_case(case, retrieval)
            if raw_answer is not None:
                answer = _coerce_answer_result(raw_answer)
                answer_evaluated = True
                answer_text = answer.text.lower()
                answer_source_hit = not case.expected_sources or bool(
                    set(answer.sources).intersection(case.expected_sources)
                )
                answer_matches = sum(
                    1 for term in case.expected_terms if term.lower() in answer_text
                )
                answer_term_recall = answer_matches / max(1, len(case.expected_terms))
                answer_no_answer_correct = answer.answered == bool(case.should_answer)
                answer_acl_safe = not bool(
                    set(answer.sources).intersection(case.forbidden_sources)
                )
                answer_passed = (
                    answer_source_hit
                    and answer_no_answer_correct
                    and answer_acl_safe
                    and (not answer.answered or bool(answer.text.strip()))
                    and (not case.expected_terms or answer_term_recall >= 0.5)
                )
        must_pass = _case_is_must_pass(case)
        passed = retrieval_passed and answer_passed is not False
        scores.append(
            RagCaseScore(
                id=case.id,
                source_hit=source_hit,
                reciprocal_rank=round(reciprocal_rank, 6),
                term_recall=round(term_recall, 6),
                no_answer_correct=no_answer_correct,
                acl_safe=acl_safe,
                retrieval_passed=retrieval_passed,
                answer_evaluated=answer_evaluated,
                answer_source_hit=answer_source_hit,
                answer_term_recall=_round_optional(answer_term_recall),
                answer_no_answer_correct=answer_no_answer_correct,
                answer_acl_safe=answer_acl_safe,
                answer_passed=answer_passed,
                must_pass=must_pass,
                passed=passed,
            )
        )
    count = len(scores)
    must_pass_failures = [score.id for score in scores if score.must_pass and not score.passed]
    answer_scores = [score for score in scores if score.answer_evaluated]
    retrieval_no_answer_accuracy = _mean(
        score.no_answer_correct
        for case, score in zip(case_rows, scores)
        if not case.should_answer
    )
    return EvaluationReport(
        dataset_version=dataset_version,
        case_count=count,
        pass_rate=_mean(item.passed for item in scores),
        source_hit_rate=_mean(
            score.source_hit for case, score in zip(case_rows, scores) if case.expected_sources
        ),
        mean_reciprocal_rank=_mean(
            score.reciprocal_rank for case, score in zip(case_rows, scores) if case.expected_sources
        ),
        mean_term_recall=_mean(
            score.term_recall for case, score in zip(case_rows, scores) if case.expected_terms
        ),
        no_answer_accuracy=retrieval_no_answer_accuracy,
        acl_safety_rate=_mean(
            score.acl_safe
            for case, score in zip(case_rows, scores)
            if case.forbidden_sources
        ),
        retrieval_pass_rate=_mean(item.retrieval_passed for item in scores),
        retrieval_no_answer_accuracy=retrieval_no_answer_accuracy,
        answer_case_count=len(answer_scores),
        answer_pass_rate=_mean_optional(
            score.answer_passed for score in answer_scores if score.answer_passed is not None
        ),
        answer_source_hit_rate=_mean_optional(
            score.answer_source_hit
            for case, score in zip(case_rows, scores)
            if score.answer_evaluated
            and case.expected_sources
            and score.answer_source_hit is not None
        ),
        answer_term_recall=_mean_optional(
            score.answer_term_recall
            for case, score in zip(case_rows, scores)
            if score.answer_evaluated
            and case.expected_terms
            and score.answer_term_recall is not None
        ),
        answer_no_answer_accuracy=_mean_optional(
            score.answer_no_answer_correct
            for case, score in zip(case_rows, scores)
            if score.answer_evaluated
            and not case.should_answer
            and score.answer_no_answer_correct is not None
        ),
        answer_acl_safety_rate=_mean_optional(
            score.answer_acl_safe
            for case, score in zip(case_rows, scores)
            if score.answer_evaluated
            and case.forbidden_sources
            and score.answer_acl_safe is not None
        ),
        must_pass_case_count=sum(1 for score in scores if score.must_pass),
        must_pass_failure_count=len(must_pass_failures),
        must_pass_passed=not must_pass_failures,
        must_pass_failures=must_pass_failures,
        cases=scores,
    )


def evaluation_gate_passed(
    report: EvaluationReport,
    *,
    minimum_pass_rate: float,
    require_answer_evaluation: bool = False,
) -> bool:
    if not 0.0 <= minimum_pass_rate <= 1.0:
        raise ValueError("minimum_pass_rate must be between 0 and 1")
    answers_complete = (
        not require_answer_evaluation or report.answer_case_count == report.case_count
    )
    return (
        report.pass_rate >= minimum_pass_rate
        and report.must_pass_passed
        and answers_complete
    )


def _case_is_must_pass(case: RagEvaluationCase) -> bool:
    tags = {tag.strip().lower() for tag in case.tags}
    return bool(
        case.must_pass
        or case.forbidden_sources
        or not case.should_answer
        or tags.intersection(MUST_PASS_TAGS)
    )


def _coerce_answer_result(value: Any) -> RagAnswerResult:
    if isinstance(value, RagAnswerResult):
        return value
    if isinstance(value, Mapping):
        payload = dict(value)
        if "text" not in payload and "answer" in payload:
            payload["text"] = payload.pop("answer")
        return RagAnswerResult.model_validate(payload)
    raise TypeError(
        "answer_case must return RagAnswerResult, a compatible mapping, or None"
    )


def _mean(values: Iterable[Any]) -> float:
    rows = [float(value) for value in values]
    return round(sum(rows) / len(rows), 6) if rows else 0.0


def _mean_optional(values: Iterable[Any]) -> Optional[float]:
    rows = [float(value) for value in values]
    return round(sum(rows) / len(rows), 6) if rows else None


def _round_optional(value: Optional[float]) -> Optional[float]:
    return round(value, 6) if value is not None else None
