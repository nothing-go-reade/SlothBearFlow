from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from langchain_core.prompts import ChatPromptTemplate

from backend.src.slothbearflow_backend import (
    Settings,
    get_chat_llm,
    get_settings,
    llm_supports_tools,
)
from backend.src.slothbearflow_backend.agent.react_runtime import ExplicitReActRuntime
from backend.src.slothbearflow_backend.learning.learning_tools import build_review_tools
from backend.src.slothbearflow_backend.learning.review_guard import (
    clear_thread_tool_whitelist,
    set_thread_tool_whitelist,
)
from backend.src.slothbearflow_backend.learning.schema import ReviewResult
from backend.src.slothbearflow_backend.learning.snapshot import TurnSnapshot
from backend.src.slothbearflow_backend.learning.store import LearningStore

logger = logging.getLogger(__name__)

_BASE = (
    "你是一个「后台复盘」助手。下面是主助手与用户的一轮对话快照。"
    "你的任务是判断本轮是否有值得**长期保存**的内容，并保守地提炼。"
    "只保存稳定、可复用、对未来有帮助的信息；忽略一次性的、与具体任务强绑定的细节。"
    "命名用简短的 kebab-case slug；同一主题复用同名以便覆盖更新，不要制造重复条目。\n\n"
)

_MEMORY_REVIEW_PROMPT = _BASE + (
    "【本次只复盘 Memory（用户长期记忆）】关注：\n"
    "- 用户是谁、身份/角色/背景；\n"
    "- 用户的稳定偏好（语言、风格、详略、格式）；\n"
    "- 用户希望 agent 如何工作的长期约定。\n"
)

_SKILL_REVIEW_PROMPT = _BASE + (
    "【本次只复盘 Skills（可复用做法）】关注：\n"
    "- 这类任务以后应该怎么做、稳定的流程/步骤；\n"
    "- 用户纠正过的格式/语气/工作流（这属于一等的 skill 信号）；\n"
    "- 可复用的技巧、修复路径、工具使用模式。\n"
)

_COMBINED_REVIEW_PROMPT = _BASE + (
    "【本次同时复盘 Memory 与 Skills】\n"
    "- Memory：用户是谁、偏好、希望 agent 怎么工作；\n"
    "- Skills：这类任务以后怎么做、用户纠正过的格式/流程、可复用技巧。\n"
    "把「用户长期关系」放入 memory，把「某类任务的方法」放入 skill，不要混淆。\n"
)

_STRUCTURED_SUFFIX = (
    "\n请输出结构化结果。若本轮没有值得长期保存的内容，"
    "should_save=false 且 memories/skills 留空。"
)

_TOOL_SUFFIX = (
    "\n若有值得保存的内容，调用 save_memory / save_skill 工具逐条保存；"
    "保存完成后直接结束。若没有值得保存的内容，直接简短说明无需保存，不要调用工具。"
)


def _select_prompt(review_memory: bool, review_skills: bool) -> str:
    if review_memory and review_skills:
        return _COMBINED_REVIEW_PROMPT
    if review_memory:
        return _MEMORY_REVIEW_PROMPT
    return _SKILL_REVIEW_PROMPT


def _review_settings(settings: Settings) -> Settings:
    """可选 REVIEW_MODEL 覆盖：复盘用单独模型，否则复用主 LLM。"""
    if str(settings.review_model or "").strip():
        try:
            return settings.model_copy(update={"llm_model": settings.review_model})
        except Exception:  # pragma: no cover
            return settings
    return settings


def _run_structured_path(
    review_settings: Settings,
    store: LearningStore,
    system_prompt: str,
    turn_text: str,
    snap: TurnSnapshot,
    max_items: int,
) -> None:
    llm = get_chat_llm(review_settings, temperature=0.0).with_structured_output(
        ReviewResult
    )
    prompt = ChatPromptTemplate.from_messages(
        [("system", system_prompt + _STRUCTURED_SUFFIX), ("human", "{turn}")]
    )
    result = (prompt | llm).invoke({"turn": turn_text})
    if not isinstance(result, ReviewResult):
        return
    if not (result.should_save or result.memories or result.skills):
        logger.info("后台复盘(结构化)：本轮无需保存")
        return
    written = store.save_many(
        memories=result.memories if snap.review_memory else [],
        skills=result.skills if snap.review_skills else [],
        max_items=max_items,
    )
    logger.info("后台复盘(结构化)落盘: %s", written)


def _run_tool_path(
    review_settings: Settings,
    store: LearningStore,
    system_prompt: str,
    turn_text: str,
    snap: TurnSnapshot,
    max_items: int,
) -> None:
    llm = get_chat_llm(review_settings, temperature=0.0)
    tools = build_review_tools(store)
    allowed = set()
    if snap.review_memory:
        allowed.add("save_memory")
    if snap.review_skills:
        allowed.add("save_skill")
    tools = [t for t in tools if t.name in allowed]
    if not tools:
        return
    runtime = ExplicitReActRuntime(llm=llm, tools=tools, max_steps=max_items + 2)
    # 执行层白名单：只放行 memory/skills 写工具（对标 Hermes set_thread_tool_whitelist）。
    set_thread_tool_whitelist(allowed)
    try:
        result = runtime.invoke(
            {
                "input": turn_text,
                "system_prompt": system_prompt + _TOOL_SUFFIX,
                "chat_history": [],
            }
        )
        logger.info(
            "后台复盘(工具)完成: tools_used=%s steps=%s",
            result.get("tools_used"),
            result.get("steps"),
        )
    finally:
        clear_thread_tool_whitelist()


def run_review_job(snapshot: Any, settings: Optional[Settings] = None) -> None:
    """后台复盘 job 入口（在 worker 线程中通过 asyncio.to_thread 调用）。

    全程 best-effort：任何异常都吞掉，绝不影响主链路。
    """
    settings = settings or get_settings()
    if not settings.enable_background_review:
        return
    try:
        data = snapshot if isinstance(snapshot, dict) else snapshot.to_dict()
        snap = TurnSnapshot.from_dict(data)
    except Exception:
        logger.exception("后台复盘：快照解析失败")
        return
    if not (snap.review_memory or snap.review_skills):
        return

    try:
        store = LearningStore(settings.review_base_dir)
        review_settings = _review_settings(settings)
        system_prompt = _select_prompt(snap.review_memory, snap.review_skills)
        turn_text = snap.render()
        max_items = max(1, int(settings.review_max_items))
        use_tools = llm_supports_tools(settings) and not settings.review_force_structured
        if use_tools:
            _run_tool_path(
                review_settings, store, system_prompt, turn_text, snap, max_items
            )
        else:
            _run_structured_path(
                review_settings, store, system_prompt, turn_text, snap, max_items
            )
    except Exception:
        logger.exception(
            "后台复盘失败（已忽略，不影响主链路）: session_id=%s", snap.session_id
        )
