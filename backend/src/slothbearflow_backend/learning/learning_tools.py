from __future__ import annotations

from typing import Callable, Optional

from langchain_core.tools import tool

from backend.src.slothbearflow_backend.learning.schema import (
    MemoryItem,
    SkillItem,
    normalize_memory_type,
)
from backend.src.slothbearflow_backend.learning.store import LearningStore

# 工具路径（A）允许执行的工具名白名单（对标 Hermes memory/skills 工具集）。
REVIEW_TOOL_WHITELIST = {"save_memory", "save_skill"}


def build_review_tools(
    store: LearningStore,
    *,
    source_tenant_id: str = "",
    source_user_id: str = "",
    source_session_id: str = "",
    source_turn_id: str = "",
    source_generation: int = 0,
    write_guard: Optional[Callable[[], bool]] = None,
):
    """构建后台复盘写工具：仅 save_memory / save_skill，内部委托 LearningStore。

    供支持工具调用的模型在 ExplicitReActRuntime 内调用；非白名单工具会被
    review_guard 的线程级白名单拦截。
    """

    @tool
    def save_memory(name: str, body: str, description: str = "", type: str = "user") -> str:
        """Persist a long-term MEMORY about the user (identity, preference, how they want the agent to work).

        name: short kebab-case slug; body: the memory content; type: user|feedback|project|reference.
        """
        path = store.upsert_memory(
            MemoryItem(
                name=name,
                description=description,
                type=normalize_memory_type(type),
                body=body,
                source_tenant_id=source_tenant_id,
                source_user_id=source_user_id,
                source_session_id=source_session_id,
                source_turn_id=source_turn_id,
                source_generation=source_generation,
            ),
            write_guard=write_guard,
        )
        return f"saved memory: {path.name}" if path else "memory rejected"

    @tool
    def save_skill(name: str, body: str, trigger: str = "") -> str:
        """Persist a reusable SKILL for this kind of task (how to do it next time).

        name: short kebab-case slug; trigger: when to apply it; body: the technique/steps.
        """
        path = store.upsert_skill(
            SkillItem(
                name=name,
                trigger=trigger,
                body=body,
                source_tenant_id=source_tenant_id,
                source_user_id=source_user_id,
                source_session_id=source_session_id,
                source_turn_id=source_turn_id,
                source_generation=source_generation,
            ),
            write_guard=write_guard,
        )
        return f"saved skill: {path.name}" if path else "skill rejected"

    return [save_memory, save_skill]
