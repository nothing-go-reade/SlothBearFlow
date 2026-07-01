from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from fastapi import HTTPException

from backend.src.slothbearflow_backend import Settings
from backend.src.slothbearflow_backend.deps import InMemoryRedis
from backend.src.slothbearflow_backend.learning.snapshot import TurnSnapshot
from backend.src.slothbearflow_backend.learning.store import LearningStore
from backend.src.slothbearflow_backend.memory.redis_memory import (
    append_turn_and_save,
    get_redis_session,
    messages_from_payload,
)
from backend.src.slothbearflow_backend.memory.short_memory import trim_message_window
from backend.src.slothbearflow_backend.memory.summary_memory import enqueue_summary_update
from backend.src.slothbearflow_backend.output_schema import ChatOutput, Citation
from backend.src.slothbearflow_backend.persistence.postgres import postgres_persistence
from backend.src.slothbearflow_backend.tools.rag_tool import (
    RagRetrieval,
    reset_rag_sources,
    retrieve_knowledge_context,
)

logger = logging.getLogger(__name__)

_STREAM_DONE = object()


def _next_stream_chunk(iterator: Any) -> Any:
    return next(iterator, _STREAM_DONE)


def detect_used_tools(raw_output: str, has_citations: bool) -> List[str]:
    tools: List[str] = []
    raw_lower = raw_output.lower()
    if "weather" in raw_lower or "天气查询结果" in raw_output:
        tools.append("get_weather")
    if "最近会话上下文" in raw_output:
        tools.append("get_session_context")
    if has_citations:
        tools.append("search_knowledge")
    return tools


def build_rag_augmented_input(
    user_message: str, retrieval: Optional[RagRetrieval]
) -> str:
    if not retrieval or not retrieval.context:
        return user_message
    return (
        "请优先根据下面的知识库检索片段回答用户问题。"
        "如果片段与问题无关，请明确说明没有找到可靠依据；"
        "如果片段中包含答案，请引用来源文件并给出直接结论。"
        "回答控制在 6 句话以内，避免复述无关片段。\n\n"
        f"{retrieval.context}\n\n"
        f"【用户问题】\n{user_message}"
    )


def merge_citations(*groups: List[Citation]) -> List[Citation]:
    seen: set = set()
    merged: List[Citation] = []
    for group in groups:
        for item in group:
            key = (item.source, item.excerpt)
            if key in seen:
                continue
            seen.add(key)
            merged.append(item)
    return merged


def citation_sources(citations: List[Citation]) -> List[str]:
    return [item.source for item in citations if item.source]


def should_stream_response(settings: Any, executor: Any) -> tuple:
    if not settings.stream_output:
        return False, "stream_output_disabled"
    if settings.structured_output:
        return False, "structured_output_enabled"
    if not hasattr(executor, "stream"):
        return False, "executor_not_streamable"
    return True, "enabled"


def normalize_stream_output_format(value: str) -> str:
    normalized = str(value or "plain").strip().lower()
    return normalized if normalized in {"plain", "sse"} else "plain"


@dataclass
class TurnInput:
    session_id: str
    message: str


@dataclass
class PreparedTurn:
    turn: TurnInput
    payload: Dict[str, Any]
    client: Any
    windowed: List[Any]
    rag_retrieval: Optional[RagRetrieval]
    prefetched_citations: List[Citation]
    llm_input: str
    executor: Any
    should_stream: bool
    stream_reason: str
    stream_format: str
    rolling_summary: str
    tool_trace: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class TurnResult:
    answer: str
    source: str
    citations: List[Citation]
    tools_used: List[str]
    session_id: str
    raw_output: str


class ChatTurnRunner:
    """统一的对话回合编排层（对标 Hermes conversation_loop.run_conversation）。

    封装一轮 /chat 的全部编排：载入会话 → RAG 预检索 → 构建 executor →
    流式/非流式产出 → 落库 → 入队摘要 →（后续）入队后台复盘。
    `_finalize` 是唯一的「回合收尾」钩子，后台复盘触发器挂在此处。

    可被 monkeypatch 的协作者（build_agent_executor / get_vector_store /
    structured_chat_output_from_text / get_last_rag_sources / get_last_rag_citations）
    由 web 层（main）在请求期注入，从而保留模块级 patch 语义。
    """

    def __init__(
        self,
        settings: Settings,
        queue: Optional[asyncio.Queue],
        *,
        build_agent_executor: Callable[..., Any],
        get_vector_store: Callable[..., Any],
        structured_chat_output_from_text: Callable[..., Any],
        get_last_rag_sources: Callable[[], List[str]],
        get_last_rag_citations: Callable[[], List[Dict[str, str]]],
    ) -> None:
        self.settings = settings
        self.queue = queue
        self._build_agent_executor = build_agent_executor
        self._get_vector_store = get_vector_store
        self._structured_chat_output_from_text = structured_chat_output_from_text
        self._get_last_rag_sources = get_last_rag_sources
        self._get_last_rag_citations = get_last_rag_citations

    async def prepare(self, turn: TurnInput) -> PreparedTurn:
        started_at = time.perf_counter()
        settings = self.settings
        logger.info(
            "chat start session_id=%s message_length=%s",
            turn.session_id,
            len(turn.message),
        )

        reset_rag_sources()

        payload, client = get_redis_session(turn.session_id, settings=settings)
        logger.info(
            "chat session load backend=%s messages=%s",
            "memory" if isinstance(client, InMemoryRedis) else "redis",
            len(payload.get("messages") or []),
        )

        history = messages_from_payload(list(payload.get("messages") or []))
        windowed = trim_message_window(history, settings.memory_window_pairs)

        vs = self._get_vector_store(settings)

        rag_retrieval: Optional[RagRetrieval] = None
        if vs is not None:
            try:
                rag_retrieval = await asyncio.to_thread(
                    retrieve_knowledge_context,
                    vs,
                    turn.message,
                    k=24,
                )
            except Exception:
                logger.exception("chat rag prefetch failed")
                rag_retrieval = None
        prefetched_citations = [
            Citation(**item)
            for item in (rag_retrieval.citations if rag_retrieval else [])
        ]
        llm_input = build_rag_augmented_input(turn.message, rag_retrieval)

        rolling_summary = str(payload.get("summary") or "")
        learning_context = self._build_learning_context(turn.message)
        executor = self._build_agent_executor(
            vector_store=vs,
            chat_history=windowed,
            rolling_summary=rolling_summary or None,
            settings=settings,
            learning_context=learning_context,
        )

        should_stream, stream_reason = should_stream_response(settings, executor)
        stream_format = normalize_stream_output_format(settings.stream_output_format)
        logger.info(
            "chat prepared in %.3fs stream=%s reason=%s",
            time.perf_counter() - started_at,
            should_stream,
            stream_reason,
        )

        return PreparedTurn(
            turn=turn,
            payload=payload,
            client=client,
            windowed=windowed,
            rag_retrieval=rag_retrieval,
            prefetched_citations=prefetched_citations,
            llm_input=llm_input,
            executor=executor,
            should_stream=should_stream,
            stream_reason=stream_reason,
            stream_format=stream_format,
            rolling_summary=rolling_summary,
        )

    async def run_blocking(self, prepared: PreparedTurn) -> TurnResult:
        settings = self.settings
        turn = prepared.turn
        try:
            result = await asyncio.to_thread(
                prepared.executor.invoke,
                {"input": prepared.llm_input, "chat_history": prepared.windowed},
            )
        except Exception as e:  # noqa: BLE001
            logger.exception("Agent 调用失败")
            raise HTTPException(status_code=502, detail=str(e)) from e

        if settings.review_tool_trace:
            prepared.tool_trace = list(result.get("tool_trace") or [])

        raw = str(result.get("output") or "")
        rag_citations = merge_citations(
            prepared.prefetched_citations,
            [Citation(**item) for item in self._get_last_rag_citations()],
        )
        rag_sources = sorted(
            set(citation_sources(rag_citations) + self._get_last_rag_sources())
        )
        rag_hint = ",".join(sorted(set(rag_sources))) if rag_sources else ""
        tools_used = detect_used_tools(raw, bool(rag_citations))

        if settings.structured_output:
            try:
                structured = await asyncio.to_thread(
                    self._structured_chat_output_from_text,
                    raw,
                    rag_hint=rag_hint,
                    citations=rag_citations,
                    tools_used=tools_used,
                    settings=settings,
                )
            except Exception:
                logger.exception("结构化输出失败，回退为原文本")
                structured = ChatOutput(
                    answer=raw,
                    source=rag_hint or "agent",
                    citations=rag_citations,
                    tools_used=tools_used,
                )
        else:
            structured = ChatOutput(
                answer=raw,
                source=rag_hint or "agent",
                citations=rag_citations,
                tools_used=tools_used,
            )

        await self._finalize(
            prepared,
            answer=structured.answer,
            raw=raw,
            response_source=structured.source or rag_hint or "agent",
            citations=structured.citations,
            tools_used=structured.tools_used,
            response_mode="invoke",
            stream_format="",
        )

        return TurnResult(
            answer=structured.answer,
            source=structured.source or rag_hint or "",
            citations=structured.citations,
            tools_used=structured.tools_used,
            session_id=turn.session_id,
            raw_output=raw,
        )

    async def iter_stream(self, prepared: PreparedTurn):
        settings = self.settings
        turn = prepared.turn
        stream_format = prepared.stream_format
        payload_input = {"input": prepared.llm_input, "chat_history": prepared.windowed}
        full_output_parts: List[str] = []
        persisted_stream_events: List[Dict[str, Any]] = []

        if stream_format == "sse":
            start_event = {"type": "start", "session_id": turn.session_id}
            persisted_stream_events.append(
                {
                    "seq": 0,
                    "event_type": "start",
                    "content": json.dumps(start_event, ensure_ascii=False),
                }
            )
            yield "data: " + json.dumps(start_event, ensure_ascii=False) + "\n\n"

        iterator = prepared.executor.stream(payload_input)
        while True:
            chunk = await asyncio.to_thread(_next_stream_chunk, iterator)
            if chunk is _STREAM_DONE:
                break
            text = str(chunk.get("output") or "")
            if not text:
                continue
            full_output_parts.append(text)
            persisted_stream_events.append(
                {
                    "seq": len(persisted_stream_events),
                    "event_type": "chunk",
                    "content": text,
                }
            )
            if stream_format == "sse":
                yield (
                    "data: "
                    + json.dumps({"type": "chunk", "content": text}, ensure_ascii=False)
                    + "\n\n"
                )
            else:
                yield text

        answer = "".join(full_output_parts)
        tool_rag_citations = [Citation(**item) for item in self._get_last_rag_citations()]
        stream_rag_citations = merge_citations(
            prepared.prefetched_citations,
            tool_rag_citations,
        )
        stream_rag_sources = sorted(
            set(citation_sources(stream_rag_citations) + self._get_last_rag_sources())
        )
        stream_rag_hint = ",".join(stream_rag_sources)
        stream_tools_used = detect_used_tools(answer, bool(stream_rag_citations))

        done_payload: Optional[Dict[str, Any]] = None
        if stream_format == "sse":
            done_payload = {
                "type": "done",
                "session_id": turn.session_id,
                "answer": answer,
                "source": stream_rag_hint or "agent",
                "citations": [item.model_dump() for item in stream_rag_citations],
                "tools_used": stream_tools_used,
            }
            persisted_stream_events.append(
                {
                    "seq": len(persisted_stream_events),
                    "event_type": "done",
                    "content": json.dumps(done_payload, ensure_ascii=False),
                }
            )

        await self._finalize(
            prepared,
            answer=answer,
            raw=answer,
            response_source=stream_rag_hint or "agent",
            citations=stream_rag_citations,
            tools_used=stream_tools_used,
            response_mode="stream",
            stream_format=stream_format,
            stream_events=persisted_stream_events,
        )

        if stream_format == "sse" and done_payload is not None:
            yield "data: " + json.dumps(done_payload, ensure_ascii=False) + "\n\n"

    async def _finalize(
        self,
        prepared: PreparedTurn,
        *,
        answer: str,
        raw: str,
        response_source: str,
        citations: List[Citation],
        tools_used: List[str],
        response_mode: str,
        stream_format: str,
        stream_events: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """唯一的「回合收尾」钩子：落库 + 入队摘要（+ 后续后台复盘）。"""
        settings = self.settings
        turn = prepared.turn

        append_turn_and_save(
            prepared.client,
            turn.session_id,
            prepared.payload,
            turn.message,
            answer,
        )
        postgres_persistence.persist_chat_turn(
            session_id=turn.session_id,
            user_message=turn.message,
            assistant_message=answer,
            raw_output=raw,
            response_source=response_source,
            tools_used=tools_used,
            citations=[item.model_dump() for item in citations],
            response_mode=response_mode,
            stream_format=stream_format,
            settings=settings,
        )
        if stream_events is not None:
            postgres_persistence.persist_stream_events(
                session_id=turn.session_id,
                events=stream_events,
                settings=settings,
            )

        if settings.async_summary_update and self.queue is not None:
            await enqueue_summary_update(self.queue, turn.session_id)

        self._maybe_enqueue_review(
            prepared,
            answer=answer,
            raw=raw,
            tools_used=tools_used,
            citations=citations,
        )

    def _build_learning_context(self, query: str) -> Optional[str]:
        """读回：把历史复盘沉淀的相关 memory/skills 注入 system prompt（有界、可关）。"""
        settings = self.settings
        if not settings.inject_learning_into_prompt:
            return None
        try:
            store = LearningStore(settings.review_base_dir)
            block = store.select_for_injection(
                query, settings.learning_prompt_budget_chars
            )
            return block or None
        except Exception:
            logger.exception("读回学习上下文失败（已忽略）")
            return None

    def _review_dimensions(self, turn_no: int) -> tuple:
        """按 nudge 间隔判断本轮该复盘哪些维度（对标 _memory/_skill_nudge_interval）。"""
        settings = self.settings
        mem_interval = max(1, int(settings.review_memory_interval))
        skill_interval = max(1, int(settings.review_skills_interval))
        review_memory = turn_no % mem_interval == 0
        review_skills = turn_no % skill_interval == 0
        return review_memory, review_skills

    def _maybe_enqueue_review(
        self,
        prepared: PreparedTurn,
        *,
        answer: str,
        raw: str,
        tools_used: List[str],
        citations: List[Citation],
    ) -> None:
        """间隔触发后台复盘。fire-and-forget：队列满/异常一律跳过，绝不影响主回合。"""
        settings = self.settings
        if not settings.enable_background_review or self.queue is None:
            return
        # 回合数 = append_turn_and_save 后的对话轮数（user+assistant 成对）。
        turn_no = len(list(prepared.payload.get("messages") or [])) // 2
        if turn_no <= 0:
            return
        review_memory, review_skills = self._review_dimensions(turn_no)
        if not (review_memory or review_skills):
            return

        retrieval = prepared.rag_retrieval
        snapshot = TurnSnapshot(
            session_id=prepared.turn.session_id,
            user_message=prepared.turn.message,
            final_answer=answer,
            raw_output=raw,
            tools_used=list(tools_used),
            tool_trace=list(prepared.tool_trace or []),
            rag_context=(retrieval.context if retrieval else ""),
            citations=[item.model_dump() for item in citations],
            rolling_summary=prepared.rolling_summary,
            review_memory=review_memory,
            review_skills=review_skills,
        )
        try:
            self.queue.put_nowait(
                {"type": "review", "snapshot": snapshot.to_dict()}
            )
            logger.info(
                "已入队后台复盘 session_id=%s turn=%s memory=%s skills=%s",
                prepared.turn.session_id,
                turn_no,
                review_memory,
                review_skills,
            )
        except asyncio.QueueFull:
            logger.warning("任务队列已满，跳过本轮后台复盘")
        except Exception:
            logger.exception("入队后台复盘失败（已忽略）")
