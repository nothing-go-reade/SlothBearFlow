from __future__ import annotations

import hashlib
import json
import logging
import time
import uuid
from typing import Any, List

from langchain_core.tools import BaseTool
from pydantic import ValidationError

from backend.src.slothbearflow_backend.security.engine import evaluate_tool_call
from backend.src.slothbearflow_backend.security.execution import (
    ToolArgumentError,
    ToolCircuitOpen,
    ToolExecutionCancelled,
    ToolExecutionTimeout,
    ToolInvocationError,
    ToolResultUncertain,
    execute_async,
    execute_sync,
)
from backend.src.slothbearflow_backend.security.schema import PolicyBundle
from backend.src.slothbearflow_backend.security.scrub import scrub_observation
from backend.src.slothbearflow_backend.security.identity import current_principal
from backend.src.slothbearflow_backend.security.turn_state import (
    current_turn_cancellation_token,
    current_turn_id,
)

logger = logging.getLogger(__name__)

# BaseTool.run 会把这些运行期参数混进 _run 的 kwargs，重建 arg dict 时要剔除。
_RESERVED_KWARGS = {"run_manager", "callbacks", "config"}

_VALIDATION_ERROR_MSG = "Tool input validation failed; please correct the arguments."


class PolicyGuardedTool(BaseTool):
    """把一个内部工具包裹进安全策略。

    两条执行路径都必经 BaseTool.run → _run（AgentExecutor 走 tool.run，
    ExplicitReActRuntime 走 tool.invoke→run），因此在 _run/_arun 里拦截可覆盖两者。
    复用内部工具的 name/description/args_schema → function-calling schema 不变。
    """

    inner_tool: Any = None
    policy: Any = None
    settings: Any = None

    def __init__(self, *, inner_tool: BaseTool, policy: PolicyBundle, settings: Any) -> None:
        super().__init__(
            name=inner_tool.name,
            description=inner_tool.description,
            args_schema=inner_tool.args_schema,
            # 类型错参数会在 _run 前触发 ValidationError → 返回短观测串而非抛异常，
            # 契合 AgentExecutor(handle_parsing_errors=True) 与 ReAct 的容错。
            handle_validation_error=_VALIDATION_ERROR_MSG,
            inner_tool=inner_tool,
            policy=policy,
            settings=settings,
        )

    def _clean_args(self, kwargs: dict) -> dict:
        return {k: v for k, v in kwargs.items() if k not in _RESERVED_KWARGS}

    def _run(self, *args: Any, **kwargs: Any) -> Any:
        arg_dict = self._clean_args(kwargs)
        call_id = str(uuid.uuid4())
        started_at = time.perf_counter()
        decision = evaluate_tool_call(
            self.name, arg_dict, settings=self.settings, policy=self.policy, quota=True
        )
        if not decision.allowed:
            self._record_trace(
                call_id=call_id,
                args=arg_dict,
                ok=False,
                status="denied",
                observation=decision.reason,
                error_code="tool_denied",
                decision="deny",
                started_at=started_at,
            )
            return decision.reason
        tool_policy = self.policy.tools.get(self.name)
        side_effecting = self._side_effecting(tool_policy)
        try:
            result = execute_sync(
                self._execution_scope_name(),
                lambda: self.inner_tool.invoke(arg_dict),
                timeout_sec=self._setting(tool_policy, "timeout_sec", "tool_timeout_sec", 15.0),
                retries=self._retries(tool_policy),
                failure_threshold=int(
                    self._setting(
                        tool_policy,
                        "circuit_failure_threshold",
                        "tool_circuit_failure_threshold",
                        3,
                    )
                ),
                recovery_sec=self._setting(
                    tool_policy,
                    "circuit_recovery_sec",
                    "tool_circuit_recovery_sec",
                    30.0,
                ),
                retry_safe=self._retry_safe(tool_policy),
                cancellation_token=current_turn_cancellation_token(),
                idempotency_key=self._idempotency_key(
                    arg_dict, fallback=call_id if side_effecting else ""
                ),
                side_effecting=side_effecting,
            )
            observation = scrub_observation(result, self.settings)
            observation = self._truncate_observation(observation)
            self._record_trace(
                call_id=call_id,
                args=arg_dict,
                ok=True,
                status="completed",
                observation=str(observation),
                error_code="",
                decision="allow",
                started_at=started_at,
                result=result,
            )
            return observation
        except ToolResultUncertain as exc:
            return self._failed_observation(
                call_id,
                arg_dict,
                started_at,
                "tool_result_uncertain",
                "Tool deadline exceeded; side effects may have completed. "
                "Result is uncertain and must not be retried automatically. "
                f"Idempotency key: {exc.idempotency_key}.",
                status="uncertain",
            )
        except ToolExecutionTimeout:
            return self._failed_observation(
                call_id, arg_dict, started_at, "tool_timeout", "Tool execution timed out."
            )
        except ToolCircuitOpen:
            return self._failed_observation(
                call_id, arg_dict, started_at, "circuit_open", "Tool is temporarily unavailable."
            )
        except ToolExecutionCancelled:
            return self._failed_observation(
                call_id, arg_dict, started_at, "tool_cancelled", "Tool execution was cancelled."
            )
        except (ToolArgumentError, ValidationError):
            return self._failed_observation(
                call_id,
                arg_dict,
                started_at,
                "tool_invalid_arguments",
                _VALIDATION_ERROR_MSG,
            )
        except ToolInvocationError:
            return self._failed_observation(
                call_id,
                arg_dict,
                started_at,
                "tool_invocation_rejected",
                "Tool invocation was rejected.",
            )
        except Exception:
            logger.exception("tool execution failed safely: %s", self.name)
            return self._failed_observation(
                call_id, arg_dict, started_at, "tool_error", "Tool execution failed safely."
            )

    async def _arun(self, *args: Any, **kwargs: Any) -> Any:
        arg_dict = self._clean_args(kwargs)
        call_id = str(uuid.uuid4())
        started_at = time.perf_counter()
        decision = evaluate_tool_call(
            self.name, arg_dict, settings=self.settings, policy=self.policy, quota=True
        )
        if not decision.allowed:
            self._record_trace(
                call_id=call_id,
                args=arg_dict,
                ok=False,
                status="denied",
                observation=decision.reason,
                error_code="tool_denied",
                decision="deny",
                started_at=started_at,
            )
            return decision.reason
        tool_policy = self.policy.tools.get(self.name)
        side_effecting = self._side_effecting(tool_policy)
        try:
            result = await execute_async(
                self._execution_scope_name(),
                lambda: self.inner_tool.ainvoke(arg_dict),
                timeout_sec=self._setting(tool_policy, "timeout_sec", "tool_timeout_sec", 15.0),
                retries=self._retries(tool_policy),
                failure_threshold=int(
                    self._setting(
                        tool_policy,
                        "circuit_failure_threshold",
                        "tool_circuit_failure_threshold",
                        3,
                    )
                ),
                recovery_sec=self._setting(
                    tool_policy,
                    "circuit_recovery_sec",
                    "tool_circuit_recovery_sec",
                    30.0,
                ),
                retry_safe=self._retry_safe(tool_policy),
                cancellation_token=current_turn_cancellation_token(),
                idempotency_key=self._idempotency_key(
                    arg_dict, fallback=call_id if side_effecting else ""
                ),
                side_effecting=side_effecting,
            )
            observation = scrub_observation(result, self.settings)
            observation = self._truncate_observation(observation)
            self._record_trace(
                call_id=call_id,
                args=arg_dict,
                ok=True,
                status="completed",
                observation=str(observation),
                error_code="",
                decision="allow",
                started_at=started_at,
                result=result,
            )
            return observation
        except ToolResultUncertain as exc:
            return self._failed_observation(
                call_id,
                arg_dict,
                started_at,
                "tool_result_uncertain",
                "Tool deadline exceeded; side effects may have completed. "
                "Result is uncertain and must not be retried automatically. "
                f"Idempotency key: {exc.idempotency_key}.",
                status="uncertain",
            )
        except ToolExecutionTimeout:
            return self._failed_observation(
                call_id, arg_dict, started_at, "tool_timeout", "Tool execution timed out."
            )
        except ToolCircuitOpen:
            return self._failed_observation(
                call_id, arg_dict, started_at, "circuit_open", "Tool is temporarily unavailable."
            )
        except ToolExecutionCancelled:
            return self._failed_observation(
                call_id, arg_dict, started_at, "tool_cancelled", "Tool execution was cancelled."
            )
        except (ToolArgumentError, ValidationError):
            return self._failed_observation(
                call_id,
                arg_dict,
                started_at,
                "tool_invalid_arguments",
                _VALIDATION_ERROR_MSG,
            )
        except ToolInvocationError:
            return self._failed_observation(
                call_id,
                arg_dict,
                started_at,
                "tool_invocation_rejected",
                "Tool invocation was rejected.",
            )
        except Exception:
            logger.exception("async tool execution failed safely: %s", self.name)
            return self._failed_observation(
                call_id, arg_dict, started_at, "tool_error", "Tool execution failed safely."
            )

    def _setting(
        self,
        tool_policy: Any,
        policy_name: str,
        settings_name: str,
        default: float,
    ) -> float:
        policy_value = getattr(tool_policy, policy_name, None) if tool_policy else None
        value = policy_value if policy_value is not None else getattr(
            self.settings, settings_name, default
        )
        return max(0.001, float(value))

    def _retries(self, tool_policy: Any) -> int:
        configured = getattr(tool_policy, "retry_attempts", None) if tool_policy else None
        if configured is None:
            configured = getattr(self.settings, "tool_retry_attempts", 0)
        # Automatic retries are restricted to read-only, side-effect-free tools.
        if tool_policy is not None and getattr(tool_policy, "cls", "read") != "read":
            return 0
        return max(0, min(5, int(configured)))

    @staticmethod
    def _retry_safe(tool_policy: Any) -> bool:
        return bool(
            tool_policy is not None
            and getattr(tool_policy, "cls", "read") == "read"
            and getattr(tool_policy, "retry_safe", False)
        )

    @staticmethod
    def _side_effecting(tool_policy: Any) -> bool:
        return bool(
            tool_policy is not None
            and getattr(tool_policy, "cls", "read") in {"write", "network", "system"}
        )

    def _idempotency_key(self, args: dict, *, fallback: str = "") -> str:
        turn_id = current_turn_id()
        if not turn_id:
            return fallback
        principal = current_principal()
        payload = json.dumps(
            {
                "turn_id": turn_id,
                "tenant_id": principal.tenant_id,
                "user_id": principal.user_id,
                "tool": self.name,
                "args": args,
            },
            sort_keys=True,
            ensure_ascii=False,
            default=str,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _execution_scope_name(self) -> str:
        principal = current_principal()
        provenance = getattr(self.inner_tool, "provenance", None)
        server = ""
        if isinstance(provenance, dict):
            server = str(provenance.get("server") or "")
        payload = json.dumps(
            {
                "tenant_id": principal.tenant_id or "local",
                "server": server or "local",
                "tool": self.name,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return "tool-circuit:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _failed_observation(
        self,
        call_id: str,
        args: dict,
        started_at: float,
        error_code: str,
        message: str,
        *,
        status: str = "failed",
    ) -> str:
        self._record_trace(
            call_id=call_id,
            args=args,
            ok=False,
            status=status,
            observation=message,
            error_code=error_code,
            decision="allow",
            started_at=started_at,
        )
        return message

    def _truncate_observation(self, observation: Any) -> str:
        value = str(observation)
        limit = max(
            256,
            int(getattr(self.settings, "tool_observation_max_chars", 12000)),
        )
        if len(value) <= limit:
            return value
        return value[:limit] + "\n[TOOL_OUTPUT_TRUNCATED]"

    def _record_trace(
        self,
        *,
        call_id: str,
        args: dict,
        ok: bool,
        status: str,
        observation: str,
        error_code: str,
        decision: str,
        started_at: float,
        result: Any = None,
    ) -> None:
        from backend.src.slothbearflow_backend.agent.tool_trace import (
            record_tool_trace,
            safe_tool_args,
        )

        max_chars = max(
            64,
            int(getattr(self.settings, "tool_trace_observation_max_chars", 800)),
        )
        provenance = getattr(result, "provenance", None)
        if not isinstance(provenance, dict):
            provenance = {}
        citations = getattr(result, "citations", None)
        if citations:
            provenance["citations"] = list(citations)
        sources = getattr(result, "sources", None)
        if sources:
            provenance["sources"] = list(sources)
        record_tool_trace(
            {
                "call_id": call_id,
                "name": self.name,
                "args": safe_tool_args(args, self.settings),
                "ok": ok,
                "status": status,
                "duration_ms": round((time.perf_counter() - started_at) * 1000, 3),
                "observation": str(observation)[:max_chars],
                "error_code": error_code,
                "policy_decision": decision,
                "provenance": provenance,
            }
        )
        try:
            from backend.src.slothbearflow_backend.observability import get_observability

            get_observability(self.settings).event(
                "tool.call",
                component="tool",
                metadata={
                    "tool": self.name,
                    "status": status,
                    "error_code": error_code,
                    "duration_ms": round(
                        (time.perf_counter() - started_at) * 1000, 3
                    ),
                    "policy_decision": decision,
                    "provenance": provenance,
                },
            )
        except Exception:
            logger.exception("tool observability event failed")
        try:
            from backend.src.slothbearflow_backend.security.audit import audit_event

            principal = current_principal()
            event_type = (
                "tool.call_completed"
                if ok
                else "tool.call_denied" if status == "denied" else "tool.call_failed"
            )
            audit_event(
                self.settings,
                event_type,
                actor=principal.user_id,
                tenant_id=principal.tenant_id,
                target=self.name,
                outcome="success" if ok else status,
                metadata={
                    "call_id": call_id,
                    "error_code": error_code,
                    "duration_ms": round(
                        (time.perf_counter() - started_at) * 1000, 3
                    ),
                    "policy_decision": decision,
                },
            )
        except Exception:
            logger.exception("tool audit event failed")


def apply_tool_policy(
    tools: List[Any], policy: PolicyBundle, settings: Any
) -> List[Any]:
    """按策略过滤 + 包裹工具列表。

    - enforce：丢弃 allow:false / 未列入(default deny) 的工具（模型看不到），再包裹存活工具；
    - log：不过滤（观察模式），仅包裹（wrapper 内部按 log 模式记录不阻断）；
    - off：理论上 build_tools 不会调用本函数，此处双保险直接原样返回。
    """
    mode = str(getattr(settings, "tool_guard_mode", "enforce") or "enforce").lower()
    if mode == "off":
        return list(tools)

    result: List[Any] = []
    for tool in tools:
        name = getattr(tool, "name", None)
        if name is None:
            logger.warning("[tool-guard] filtered unnamed tool")
            continue
        if mode == "enforce":
            tp = policy.tools.get(name)
            if tp is not None and not tp.allow:
                logger.info("[tool-guard] filtered disabled tool: %s", name)
                continue
            if tp is None and str(policy.default_action or "deny").lower() != "allow":
                logger.info("[tool-guard] filtered non-allowlisted tool: %s", name)
                continue
        result.append(
            PolicyGuardedTool(inner_tool=tool, policy=policy, settings=settings)
        )
    return result
