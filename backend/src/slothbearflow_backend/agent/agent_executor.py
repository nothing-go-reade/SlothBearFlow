from __future__ import annotations

import time
from typing import Any, Dict, Iterable, Optional

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.runnables import RunnableSerializable

from backend.src.slothbearflow_backend import (
    Settings,
    build_tools,
    get_agent_prompt,
    get_chat_llm,
    get_settings,
    llm_supports_tools,
)
from backend.src.slothbearflow_backend.agent.react_runtime import ExplicitReActRuntime
from backend.src.slothbearflow_backend.agent.content import extract_model_text
from backend.src.slothbearflow_backend.agent.run_result import AgentRunResult
from backend.src.slothbearflow_backend.agent.tool_trace import (
    begin_tool_trace,
    end_tool_trace,
    get_tool_trace,
)
from backend.src.slothbearflow_backend.llm import get_llm_model_name
from backend.src.slothbearflow_backend.prompt import build_system_prompt, get_basic_chat_prompt
from backend.src.slothbearflow_backend.rag.security import RagAccessContext
from backend.src.slothbearflow_backend.observability import get_observability
from backend.src.slothbearflow_backend.security.scrub import scrub_observation


class BasicChatExecutor:
    def __init__(
        self,
        runnable: RunnableSerializable[Any, Any],
        *,
        model: str,
        prompt_version: str,
        settings: Settings,
    ) -> None:
        self._runnable = runnable
        self._model = model
        self._prompt_version = prompt_version
        self._settings = settings

    def invoke(self, payload: dict[str, Any]) -> dict[str, Any]:
        started_at = time.perf_counter()
        result = self._runnable.invoke(payload)
        run_result = AgentRunResult(
            output=extract_model_text(result),
            stop_reason="final_answer",
            steps=1,
            latency_ms=(time.perf_counter() - started_at) * 1000,
            model=self._model,
            executor="basic",
            prompt_version=self._prompt_version,
        )
        get_observability(self._settings).record_generation(
            name="llm.generation",
            model=self._model,
            input_chars=len(str(payload.get("input") or "")),
            output_chars=len(run_result.output),
            latency_ms=run_result.latency_ms,
            stop_reason=run_result.stop_reason,
        )
        return run_result.to_dict()

    def stream(self, payload: dict[str, Any]):
        started_at = time.perf_counter()
        parts = []
        for chunk in self._runnable.stream(payload):
            text = extract_model_text(chunk)
            if text:
                parts.append(text)
                yield {"output": text}
        run_result = AgentRunResult(
            output="".join(parts),
            stop_reason="final_answer",
            steps=1,
            latency_ms=(time.perf_counter() - started_at) * 1000,
            model=self._model,
            executor="basic",
            prompt_version=self._prompt_version,
        )
        get_observability(self._settings).record_generation(
            name="llm.generation",
            model=self._model,
            input_chars=len(str(payload.get("input") or "")),
            output_chars=len(run_result.output),
            latency_ms=run_result.latency_ms,
            stop_reason=run_result.stop_reason,
        )
        yield {
            "output": "",
            "_agent_result": run_result.to_dict(),
        }


class ToolCallingExecutorAdapter:
    def __init__(self, executor: AgentExecutor, *, settings: Settings) -> None:
        self._executor = executor
        self._settings = settings

    def invoke(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        started_at = time.perf_counter()
        begin_tool_trace()
        try:
            raw = self._executor.invoke(payload)
            trace = get_tool_trace()
        finally:
            end_tool_trace()
        intermediate_steps = list(raw.get("intermediate_steps") or [])
        if not trace:
            trace = _trace_from_intermediate_steps(
                intermediate_steps, settings=self._settings
            )
        output = extract_model_text(raw.get("output"))
        stop_reason = "final_answer"
        if "iteration limit" in output.lower():
            stop_reason = "max_steps"
        elif "time limit" in output.lower():
            stop_reason = "max_execution_time"
        citations, sources = _rag_provenance_from_trace(trace)
        run_result = AgentRunResult(
            output=output,
            stop_reason=stop_reason,
            steps=max(1, len(intermediate_steps)),
            tools_used=_tools_from_trace(trace),
            tool_trace=trace,
            citations=citations,
            rag_sources=sources,
            latency_ms=(time.perf_counter() - started_at) * 1000,
            model=get_llm_model_name(self._settings),
            executor="tool_calling",
            prompt_version=self._settings.agent_prompt_version,
        )
        get_observability(self._settings).record_generation(
            name="agent.generation",
            model=run_result.model,
            input_chars=len(str(payload.get("input") or "")),
            output_chars=len(run_result.output),
            latency_ms=run_result.latency_ms,
            stop_reason=run_result.stop_reason,
        )
        return run_result.to_dict()

    def stream(self, payload: Dict[str, Any]):
        result = self.invoke(payload)
        if result.get("output"):
            yield {"output": str(result["output"])}
        yield {"output": "", "_agent_result": result}


def _tools_from_trace(trace: Iterable[Dict[str, Any]]) -> list[str]:
    return list(
        dict.fromkeys(
            str(item.get("name") or "")
            for item in trace
            if str(item.get("name") or "").strip()
        )
    )


def _rag_provenance_from_trace(
    trace: Iterable[Dict[str, Any]],
) -> tuple[list[Dict[str, Any]], list[str]]:
    citations: list[Dict[str, Any]] = []
    sources: list[str] = []
    seen = set()
    for item in trace:
        provenance = item.get("provenance") or {}
        for source in provenance.get("sources") or []:
            if source not in sources:
                sources.append(str(source))
        for citation in provenance.get("citations") or []:
            if not isinstance(citation, dict):
                continue
            key = (str(citation.get("source") or ""), str(citation.get("excerpt") or ""))
            if key in seen:
                continue
            seen.add(key)
            citations.append(dict(citation))
    return citations, sources


def _trace_from_intermediate_steps(
    steps: Iterable[Any], *, settings: Optional[Settings] = None
) -> list[Dict[str, Any]]:
    trace: list[Dict[str, Any]] = []
    for index, step in enumerate(steps, start=1):
        if not isinstance(step, (tuple, list)) or len(step) != 2:
            continue
        action, observation = step
        safe_observation = scrub_observation(observation, settings)
        lowered = safe_observation.lower()
        denied = "denied" in lowered or "not allowed" in lowered
        failed = denied or any(
            marker in lowered
            for marker in ("failed safely", "timed out", "cancelled", "unavailable")
        )
        trace.append(
            {
                "call_id": str(getattr(action, "tool_call_id", "") or f"step-{index}"),
                "name": str(getattr(action, "tool", "") or "unknown"),
                "args": dict(getattr(action, "tool_input", {}) or {}),
                "ok": not failed,
                "status": "denied" if denied else "failed" if failed else "completed",
                "duration_ms": 0.0,
                "observation": safe_observation[:800],
                "error_code": "tool_denied" if denied else "tool_error" if failed else "",
                "policy_decision": "deny" if denied else "allow",
                "provenance": {},
            }
        )
    return trace


def build_agent_executor(
    *,
    vector_store: Optional[Any],
    chat_history: Optional[list[Any]] = None,
    rolling_summary: Optional[str] = None,
    settings: Optional[Settings] = None,
    learning_context: Optional[str] = None,
    rag_access_context: Optional[RagAccessContext] = None,
) -> Any:
    settings = settings or get_settings()
    llm = get_chat_llm(settings)

    if not llm_supports_tools(settings):
        #  普通提示词: Prompt = System + History + Input
        prompt = get_basic_chat_prompt(
            rolling_summary=rolling_summary,
            structured_output=settings.structured_output,
            learning_context=learning_context,
            prompt_version=settings.agent_prompt_version,
        )
        return BasicChatExecutor(
            prompt | llm,
            model=get_llm_model_name(settings),
            prompt_version=settings.agent_prompt_version,
            settings=settings,
        )

    # 调试——工具
    tool_kwargs: Dict[str, Any] = {
        "chat_history": chat_history,
        "settings": settings,
    }
    if rag_access_context is not None:
        tool_kwargs["rag_access_context"] = rag_access_context
    tools = build_tools(vector_store, **tool_kwargs)

    # ReAct Agent 追踪
    if settings.enable_explicit_react_runtime:
        system_prompt = build_system_prompt(
            rolling_summary=rolling_summary,
            supports_tools=True,
            structured_output=settings.structured_output,
            learning_context=learning_context,
            prompt_version=settings.agent_prompt_version,
        )

        class ReActExecutorAdapter(ExplicitReActRuntime):
            def invoke(self, payload: dict[str, Any]) -> dict[str, Any]:
                runtime_payload = {
                    **payload,
                    "chat_history": list(chat_history or []),
                    "system_prompt": system_prompt,
                }
                started_at = time.perf_counter()
                begin_tool_trace()
                try:
                    result = super().invoke(runtime_payload)
                    exact_trace = get_tool_trace()
                finally:
                    end_tool_trace()
                if exact_trace:
                    result["tool_trace"] = _merge_tool_traces(
                        result.get("tool_trace") or [], exact_trace
                    )
                    result["tools_used"] = _tools_from_trace(result["tool_trace"])
                citations, sources = _rag_provenance_from_trace(
                    result.get("tool_trace") or []
                )
                result.update(
                    {
                        "rag_citations": citations,
                        "rag_sources": sources,
                        "latency_ms": (time.perf_counter() - started_at) * 1000,
                        "model": get_llm_model_name(settings),
                        "executor": "explicit_react",
                        "prompt_version": settings.agent_prompt_version,
                    }
                )
                get_observability(settings).record_generation(
                    name="agent.generation",
                    model=str(result.get("model") or ""),
                    input_chars=len(str(runtime_payload.get("input") or "")),
                    output_chars=len(str(result.get("output") or "")),
                    latency_ms=float(result.get("latency_ms") or 0.0),
                    stop_reason=str(result.get("stop_reason") or "final_answer"),
                )
                return result

            def stream(self, payload: dict[str, Any]):
                runtime_payload = {
                    **payload,
                    "chat_history": list(chat_history or []),
                    "system_prompt": system_prompt,
                }
                result = self.invoke(runtime_payload)
                if result.get("output"):
                    yield {"output": str(result["output"])}
                yield {"output": "", "_agent_result": result}

        return ReActExecutorAdapter(
            llm=llm,
            tools=tools,
            max_steps=settings.react_max_steps,
            max_tool_calls=settings.max_tool_calls_per_turn,
        )

    # Standard LangChain tool-calling path with bounded iterations and deadline.
    prompt = get_agent_prompt(
        rolling_summary=rolling_summary,
        structured_output=settings.structured_output,
        learning_context=learning_context,
        prompt_version=settings.agent_prompt_version,
    )
    agent = create_tool_calling_agent(llm, tools, prompt)
    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=False,
        handle_parsing_errors=True,
        max_iterations=settings.agent_max_iterations,
        max_execution_time=settings.agent_timeout_sec,
        early_stopping_method="generate",
        return_intermediate_steps=True,
    )
    return ToolCallingExecutorAdapter(executor, settings=settings)


def _merge_tool_traces(
    runtime_trace: Iterable[Dict[str, Any]],
    exact_trace: Iterable[Dict[str, Any]],
) -> list[Dict[str, Any]]:
    merged = [dict(item) for item in exact_trace]
    fingerprints = {
        (
            str(item.get("name") or ""),
            repr(item.get("args") or {}),
            bool(item.get("ok")),
        )
        for item in merged
    }
    for item in runtime_trace:
        row = dict(item)
        fingerprint = (
            str(row.get("name") or ""),
            repr(row.get("args") or {}),
            bool(row.get("ok")),
        )
        if fingerprint not in fingerprints:
            merged.append(row)
            fingerprints.add(fingerprint)
    return merged
