from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class RagEvaluationCase(BaseModel):
    id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    expected_sources: List[str] = Field(default_factory=list)
    forbidden_sources: List[str] = Field(default_factory=list)
    expected_terms: List[str] = Field(default_factory=list)
    should_answer: bool = True
    tags: List[str] = Field(default_factory=list)
    tenant_id: str = "local"
    user_id: str = "local-user"
    roles: List[str] = Field(default_factory=lambda: ["viewer"])
    allow_legacy: bool = False
    must_pass: bool = False


class RagAnswerResult(BaseModel):
    """Explicit answer semantics supplied by an optional end-to-end evaluator."""

    text: str = ""
    answered: bool
    sources: List[str] = Field(default_factory=list)


class RagCaseScore(BaseModel):
    id: str
    source_hit: bool
    reciprocal_rank: float
    term_recall: float
    no_answer_correct: bool
    acl_safe: bool
    retrieval_passed: bool
    answer_evaluated: bool
    answer_source_hit: Optional[bool]
    answer_term_recall: Optional[float]
    answer_no_answer_correct: Optional[bool]
    answer_acl_safe: Optional[bool]
    answer_passed: Optional[bool]
    must_pass: bool
    passed: bool


class EvaluationReport(BaseModel):
    dataset_version: str
    case_count: int
    pass_rate: float
    source_hit_rate: float
    mean_reciprocal_rank: float
    mean_term_recall: float
    no_answer_accuracy: float
    acl_safety_rate: float
    retrieval_pass_rate: float
    retrieval_no_answer_accuracy: float
    answer_case_count: int
    answer_pass_rate: Optional[float]
    answer_source_hit_rate: Optional[float]
    answer_term_recall: Optional[float]
    answer_no_answer_accuracy: Optional[float]
    answer_acl_safety_rate: Optional[float]
    must_pass_case_count: int
    must_pass_failure_count: int
    must_pass_passed: bool
    must_pass_failures: List[str] = Field(default_factory=list)
    cases: List[RagCaseScore] = Field(default_factory=list)
