from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class Citation(BaseModel):
    """RAG / tool 产出的引用信息。"""

    source: str = Field(default="", description="来源标识，如文档名、工具名等")
    excerpt: str = Field(default="", description="可回显给用户的关键片段")
    score: Optional[float] = Field(default=None, description="可选相关度分数")
    chunk_id: str = Field(default="", description="稳定知识块 ID")
    retrieval: str = Field(default="", description="召回/融合方式")
    supported: Optional[bool] = Field(default=None, description="回答是否得到该引用支持")
    support_score: Optional[float] = Field(default=None, description="引用支持度")


class ChatOutput(BaseModel):
    """面向 API / 下游系统的结构化输出。"""

    answer: str = Field(description="给用户的最终自然语言回答")
    source: str = Field(
        default="",
        description="信息来源说明：工具名、知识库 source 字段或「无」",
    )
    citations: List[Citation] = Field(
        default_factory=list,
        description="引用片段列表，适合下游直接渲染出处",
    )
    tools_used: List[str] = Field(
        default_factory=list,
        description="本次回答中实际使用过的工具名",
    )


class ToolTraceOutput(BaseModel):
    call_id: str = ""
    name: str = ""
    args: Dict[str, Any] = Field(default_factory=dict)
    ok: bool = False
    status: str = ""
    duration_ms: float = 0.0
    observation: str = ""
    error_code: str = ""
    policy_decision: str = ""
    provenance: Dict[str, Any] = Field(default_factory=dict)
