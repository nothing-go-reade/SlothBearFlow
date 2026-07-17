from __future__ import annotations

import asyncio
import contextvars
import math
import threading
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, Optional

from pydantic import ValidationError


class ToolExecutionTimeout(TimeoutError):
    pass


class ToolCircuitOpen(RuntimeError):
    pass


class ToolExecutionCancelled(RuntimeError):
    pass


class ToolInvocationError(RuntimeError):
    """The service was reachable, but the invocation itself was rejected."""


class ToolArgumentError(ToolInvocationError):
    pass


class ToolResultUncertain(ToolExecutionTimeout):
    def __init__(self, message: str, *, idempotency_key: str) -> None:
        super().__init__(message)
        self.idempotency_key = idempotency_key


class CancellationToken:
    def __init__(self) -> None:
        self._event = threading.Event()
        self._reason = "tool execution cancelled"
        self._lock = threading.Lock()

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()

    @property
    def reason(self) -> str:
        with self._lock:
            return self._reason

    def cancel(self, reason: str = "tool execution cancelled") -> bool:
        with self._lock:
            first = not self._event.is_set()
            if first:
                self._reason = str(reason or "tool execution cancelled")
            self._event.set()
            return first

    def wait(self, timeout: Optional[float] = None) -> bool:
        return self._event.wait(timeout)

    def raise_if_cancelled(self) -> None:
        if self.cancelled:
            raise ToolExecutionCancelled(self.reason)


@dataclass
class CircuitState:
    failures: int = 0
    opened_at: float = 0.0


@dataclass
class _IdempotencyRecord:
    future: Future
    expires_at: float
    ttl_sec: float


_executor = ThreadPoolExecutor(max_workers=16, thread_name_prefix="agent-tool")
_circuits: Dict[str, CircuitState] = {}
_circuit_lock = threading.Lock()
_idempotency_records: Dict[str, _IdempotencyRecord] = {}
_idempotency_lock = threading.Lock()
_current_cancellation_token: contextvars.ContextVar[Optional[CancellationToken]] = (
    contextvars.ContextVar("tool_cancellation_token", default=None)
)
_current_idempotency_key: contextvars.ContextVar[str] = contextvars.ContextVar(
    "tool_idempotency_key", default=""
)
_current_tool_deadline: contextvars.ContextVar[Optional[float]] = contextvars.ContextVar(
    "tool_execution_deadline", default=None
)


def current_cancellation_token() -> Optional[CancellationToken]:
    return _current_cancellation_token.get()


def current_idempotency_key() -> str:
    return _current_idempotency_key.get()


def current_tool_deadline() -> Optional[float]:
    return _current_tool_deadline.get()


def _before_call(name: str, *, failure_threshold: int, recovery_sec: float) -> None:
    with _circuit_lock:
        state = _circuits.setdefault(name, CircuitState())
        if state.failures < failure_threshold:
            return
        if time.monotonic() - state.opened_at >= recovery_sec:
            state.failures = max(0, failure_threshold - 1)
            state.opened_at = 0.0
            return
        raise ToolCircuitOpen("tool circuit is open")


def _record_success(name: str) -> None:
    with _circuit_lock:
        _circuits[name] = CircuitState()


def _record_failure(name: str, *, failure_threshold: int) -> None:
    with _circuit_lock:
        state = _circuits.setdefault(name, CircuitState())
        state.failures += 1
        if state.failures >= failure_threshold:
            state.opened_at = time.monotonic()


def _is_service_failure(error: BaseException) -> bool:
    return not isinstance(
        error,
        (ToolInvocationError, ToolExecutionCancelled, ValidationError),
    )


def _validate_execution_options(
    name: str,
    *,
    timeout_sec: float,
    retries: int,
    failure_threshold: int,
    recovery_sec: float,
    retry_backoff_sec: float,
    idempotency_ttl_sec: float,
) -> None:
    if not str(name or "").strip():
        raise ValueError("tool execution scope name is required")
    positive_values = {
        "timeout_sec": timeout_sec,
        "recovery_sec": recovery_sec,
        "idempotency_ttl_sec": idempotency_ttl_sec,
    }
    for label, value in positive_values.items():
        if not math.isfinite(float(value)) or float(value) <= 0:
            raise ValueError(f"{label} must be finite and greater than zero")
    if not isinstance(retries, int) or isinstance(retries, bool) or not 0 <= retries <= 5:
        raise ValueError("retries must be between 0 and 5")
    if (
        not isinstance(failure_threshold, int)
        or isinstance(failure_threshold, bool)
        or failure_threshold <= 0
    ):
        raise ValueError("failure_threshold must be greater than zero")
    if not math.isfinite(float(retry_backoff_sec)) or float(retry_backoff_sec) < 0:
        raise ValueError("retry_backoff_sec must be finite and non-negative")


def _claim_idempotency(
    name: str,
    key: Optional[str],
    *,
    ttl_sec: float,
) -> tuple[bool, Optional[_IdempotencyRecord]]:
    stable_key = str(key or "").strip()
    if not stable_key:
        return True, None
    scoped_key = "%s\x00%s" % (name, stable_key)
    now = time.monotonic()
    with _idempotency_lock:
        expired = [
            record_key
            for record_key, record in _idempotency_records.items()
            if record.future.done() and record.expires_at <= now
        ]
        for record_key in expired:
            _idempotency_records.pop(record_key, None)
        existing = _idempotency_records.get(scoped_key)
        if existing is not None:
            return False, existing
        ttl = max(1.0, float(ttl_sec))
        record = _IdempotencyRecord(
            future=Future(),
            expires_at=now + ttl,
            ttl_sec=ttl,
        )
        _idempotency_records[scoped_key] = record
        return True, record


def _complete_idempotency(
    record: Optional[_IdempotencyRecord],
    *,
    result: Any = None,
    error: Optional[BaseException] = None,
) -> None:
    if record is None:
        return
    with _idempotency_lock:
        if record.future.done():
            return
        record.expires_at = time.monotonic() + record.ttl_sec
        if error is not None:
            record.future.set_exception(error)
        else:
            record.future.set_result(result)


def _wait_for_idempotent_sync(
    record: _IdempotencyRecord,
    *,
    timeout_sec: float,
    cancellation_token: CancellationToken,
) -> Any:
    deadline = time.monotonic() + max(0.001, timeout_sec)
    while True:
        cancellation_token.raise_if_cancelled()
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise ToolExecutionTimeout("idempotent tool execution is still in progress")
        try:
            return record.future.result(timeout=min(0.05, remaining))
        except FutureTimeoutError:
            if record.future.done():
                return record.future.result()


async def _wait_for_idempotent_async(
    record: _IdempotencyRecord,
    *,
    timeout_sec: float,
    cancellation_token: CancellationToken,
) -> Any:
    deadline = time.monotonic() + max(0.001, timeout_sec)
    wrapped = asyncio.wrap_future(record.future)
    while True:
        cancellation_token.raise_if_cancelled()
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise ToolExecutionTimeout("idempotent tool execution is still in progress")
        try:
            return await asyncio.wait_for(
                asyncio.shield(wrapped),
                timeout=min(0.05, remaining),
            )
        except asyncio.TimeoutError:
            if record.future.done():
                return record.future.result()


def _run_sync_operation(
    operation: Callable[[], Any],
    cancellation_token: CancellationToken,
    idempotency_key: str,
) -> Any:
    token_handle = _current_cancellation_token.set(cancellation_token)
    key_handle = _current_idempotency_key.set(idempotency_key)
    try:
        cancellation_token.raise_if_cancelled()
        return operation()
    finally:
        _current_idempotency_key.reset(key_handle)
        _current_cancellation_token.reset(token_handle)


async def _run_async_operation(
    operation: Callable[[], Awaitable[Any]],
    cancellation_token: CancellationToken,
    idempotency_key: str,
) -> Any:
    token_handle = _current_cancellation_token.set(cancellation_token)
    key_handle = _current_idempotency_key.set(idempotency_key)
    try:
        cancellation_token.raise_if_cancelled()
        return await operation()
    finally:
        _current_idempotency_key.reset(key_handle)
        _current_cancellation_token.reset(token_handle)


def _consume_async_task_result(task: "asyncio.Task[Any]") -> None:
    try:
        task.result()
    except BaseException:
        pass


def execute_sync(
    name: str,
    operation: Callable[[], Any],
    *,
    timeout_sec: float,
    retries: int,
    failure_threshold: int,
    recovery_sec: float,
    retry_backoff_sec: float = 0.1,
    retry_safe: bool = False,
    cancellation_token: Optional[CancellationToken] = None,
    idempotency_key: Optional[str] = None,
    idempotency_ttl_sec: float = 300.0,
    side_effecting: bool = False,
) -> Any:
    _validate_execution_options(
        name,
        timeout_sec=timeout_sec,
        retries=retries,
        failure_threshold=failure_threshold,
        recovery_sec=recovery_sec,
        retry_backoff_sec=retry_backoff_sec,
        idempotency_ttl_sec=idempotency_ttl_sec,
    )
    token = cancellation_token or CancellationToken()
    stable_idempotency_key = str(idempotency_key or "").strip()
    if side_effecting and not stable_idempotency_key:
        stable_idempotency_key = uuid.uuid4().hex
    owner, idempotency_record = _claim_idempotency(
        name,
        stable_idempotency_key,
        ttl_sec=idempotency_ttl_sec,
    )
    if not owner:
        assert idempotency_record is not None
        return _wait_for_idempotent_sync(
            idempotency_record,
            timeout_sec=timeout_sec,
            cancellation_token=token,
        )
    try:
        token.raise_if_cancelled()
        _before_call(
            name,
            failure_threshold=failure_threshold,
            recovery_sec=recovery_sec,
        )
    except BaseException as exc:
        _complete_idempotency(idempotency_record, error=exc)
        raise

    effective_retries = max(0, retries) if retry_safe and not side_effecting else 0
    last_error: Optional[BaseException] = None
    for attempt in range(effective_retries + 1):
        try:
            token.raise_if_cancelled()
        except ToolExecutionCancelled as exc:
            last_error = exc
            break
        deadline_handle = _current_tool_deadline.set(
            time.monotonic() + max(0.001, timeout_sec)
        )
        try:
            context = contextvars.copy_context()
        finally:
            _current_tool_deadline.reset(deadline_handle)
        future = _executor.submit(
            context.run,
            _run_sync_operation,
            operation,
            token,
            stable_idempotency_key,
        )
        try:
            result = future.result(timeout=max(0.001, timeout_sec))
            _record_success(name)
            _complete_idempotency(idempotency_record, result=result)
            return result
        except FutureTimeoutError:
            operation_timed_out = True
            if future.done():
                try:
                    result = future.result()
                except FutureTimeoutError:
                    pass
                except Exception as completed_error:  # noqa: BLE001
                    last_error = completed_error
                    operation_timed_out = False
                except BaseException as completed_error:
                    _complete_idempotency(idempotency_record, error=completed_error)
                    raise
                else:
                    _record_success(name)
                    _complete_idempotency(idempotency_record, result=result)
                    return result
            if operation_timed_out:
                future.cancel()
                timeout_message = "tool execution exceeded %.3fs" % timeout_sec
                last_error = (
                    ToolResultUncertain(
                        timeout_message,
                        idempotency_key=stable_idempotency_key,
                    )
                    if side_effecting
                    else ToolExecutionTimeout(timeout_message)
                )
                token.cancel(str(last_error))
                _record_failure(name, failure_threshold=failure_threshold)
                _complete_idempotency(idempotency_record, error=last_error)
                raise last_error
        except ToolExecutionTimeout as exc:
            token.cancel(str(exc))
            last_error = exc
            break
        except ToolExecutionCancelled as exc:
            last_error = exc
            break
        except Exception as exc:  # noqa: BLE001
            last_error = exc
        except BaseException as exc:
            _complete_idempotency(idempotency_record, error=exc)
            raise
        if last_error is not None and not _is_service_failure(last_error):
            break
        if attempt < effective_retries:
            if token.wait(max(0.0, retry_backoff_sec) * (2**attempt)):
                last_error = ToolExecutionCancelled(token.reason)
                break
    if last_error is None:
        last_error = RuntimeError("tool execution failed")
    if _is_service_failure(last_error):
        _record_failure(name, failure_threshold=failure_threshold)
    _complete_idempotency(idempotency_record, error=last_error)
    raise last_error


async def execute_async(
    name: str,
    operation: Callable[[], Awaitable[Any]],
    *,
    timeout_sec: float,
    retries: int,
    failure_threshold: int,
    recovery_sec: float,
    retry_backoff_sec: float = 0.1,
    retry_safe: bool = False,
    cancellation_token: Optional[CancellationToken] = None,
    idempotency_key: Optional[str] = None,
    idempotency_ttl_sec: float = 300.0,
    side_effecting: bool = False,
) -> Any:
    _validate_execution_options(
        name,
        timeout_sec=timeout_sec,
        retries=retries,
        failure_threshold=failure_threshold,
        recovery_sec=recovery_sec,
        retry_backoff_sec=retry_backoff_sec,
        idempotency_ttl_sec=idempotency_ttl_sec,
    )
    token = cancellation_token or CancellationToken()
    stable_idempotency_key = str(idempotency_key or "").strip()
    if side_effecting and not stable_idempotency_key:
        stable_idempotency_key = uuid.uuid4().hex
    owner, idempotency_record = _claim_idempotency(
        name,
        stable_idempotency_key,
        ttl_sec=idempotency_ttl_sec,
    )
    if not owner:
        assert idempotency_record is not None
        return await _wait_for_idempotent_async(
            idempotency_record,
            timeout_sec=timeout_sec,
            cancellation_token=token,
        )
    try:
        token.raise_if_cancelled()
        _before_call(
            name,
            failure_threshold=failure_threshold,
            recovery_sec=recovery_sec,
        )
    except BaseException as exc:
        _complete_idempotency(idempotency_record, error=exc)
        raise

    effective_retries = max(0, retries) if retry_safe and not side_effecting else 0
    last_error: Optional[BaseException] = None
    for attempt in range(effective_retries + 1):
        try:
            token.raise_if_cancelled()
        except ToolExecutionCancelled as exc:
            last_error = exc
            break
        deadline_handle = _current_tool_deadline.set(
            time.monotonic() + max(0.001, timeout_sec)
        )
        try:
            task = asyncio.create_task(
                _run_async_operation(operation, token, stable_idempotency_key)
            )
        finally:
            _current_tool_deadline.reset(deadline_handle)
        try:
            done, _ = await asyncio.wait({task}, timeout=max(0.001, timeout_sec))
            if not done:
                timeout_message = "tool execution exceeded %.3fs" % timeout_sec
                last_error = (
                    ToolResultUncertain(
                        timeout_message,
                        idempotency_key=stable_idempotency_key,
                    )
                    if side_effecting
                    else ToolExecutionTimeout(timeout_message)
                )
                token.cancel(str(last_error))
                task.cancel()
                task.add_done_callback(_consume_async_task_result)
                raise last_error
            result = task.result()
            _record_success(name)
            _complete_idempotency(idempotency_record, result=result)
            return result
        except ToolExecutionTimeout as exc:
            token.cancel(str(exc))
            _record_failure(name, failure_threshold=failure_threshold)
            _complete_idempotency(idempotency_record, error=exc)
            raise
        except ToolExecutionCancelled as exc:
            last_error = exc
            break
        except asyncio.CancelledError:
            token.cancel("tool execution cancelled")
            task.cancel()
            task.add_done_callback(_consume_async_task_result)
            cancelled = ToolExecutionCancelled(token.reason)
            _complete_idempotency(idempotency_record, error=cancelled)
            raise
        except Exception as exc:  # noqa: BLE001
            last_error = exc
        except BaseException as exc:
            _complete_idempotency(idempotency_record, error=exc)
            raise
        if last_error is not None and not _is_service_failure(last_error):
            break
        if attempt < effective_retries:
            try:
                await asyncio.sleep(max(0.0, retry_backoff_sec) * (2**attempt))
            except asyncio.CancelledError:
                token.cancel("tool execution cancelled")
                cancelled = ToolExecutionCancelled(token.reason)
                _complete_idempotency(idempotency_record, error=cancelled)
                raise
            if token.cancelled:
                last_error = ToolExecutionCancelled(token.reason)
                break
    if last_error is None:
        last_error = RuntimeError("tool execution failed")
    if isinstance(last_error, ToolExecutionTimeout):
        token.cancel(str(last_error))
    if _is_service_failure(last_error):
        _record_failure(name, failure_threshold=failure_threshold)
    _complete_idempotency(idempotency_record, error=last_error)
    raise last_error


def reset_circuits() -> None:
    with _circuit_lock:
        _circuits.clear()
    with _idempotency_lock:
        _idempotency_records.clear()
