from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field

# 与项目自动记忆惯例对齐的 memory 类型枚举。
MEMORY_TYPES = ("user", "feedback", "project", "reference")


class MemoryItem(BaseModel):
    """一条长期记忆（用户是谁 / 偏好 / 希望 agent 怎么工作）。"""

    name: str = Field(description="kebab-case 短 slug，作为文件名与去重键")
    description: str = Field(default="", description="一行摘要，用于索引与读回")
    type: str = Field(default="user", description="user | feedback | project | reference")
    body: str = Field(default="", description="记忆正文")
    confidence: float = Field(default=0.8, ge=0, le=1)
    source_tenant_id: str = ""
    source_user_id: str = ""
    source_session_id: str = ""
    source_turn_id: str = ""
    source_generation: int = Field(default=0, ge=0)


class SkillItem(BaseModel):
    """一条可复用技巧（这类任务以后应该怎么做）。"""

    name: str = Field(description="kebab-case 短 slug，作为文件名与去重键")
    trigger: str = Field(default="", description="何时套用该技巧")
    body: str = Field(default="", description="技巧正文 / 步骤")
    confidence: float = Field(default=0.8, ge=0, le=1)
    source_tenant_id: str = ""
    source_user_id: str = ""
    source_session_id: str = ""
    source_turn_id: str = ""
    source_generation: int = Field(default=0, ge=0)


class ReviewResult(BaseModel):
    """review agent 的结构化产出（路径 B 直接产出；路径 A 由工具聚合而成）。"""

    should_save: bool = Field(default=False, description="本轮是否有值得沉淀的内容")
    rationale: str = Field(default="", description="判断理由，便于排查")
    memories: List[MemoryItem] = Field(default_factory=list)
    skills: List[SkillItem] = Field(default_factory=list)


def normalize_memory_type(value: str) -> str:
    v = str(value or "").strip().lower()
    return v if v in MEMORY_TYPES else "user"
