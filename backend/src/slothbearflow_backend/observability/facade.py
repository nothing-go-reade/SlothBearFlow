from __future__ import annotations

import contextlib
import copy
import hashlib
import logging
import threading
import time
import uuid
from collections import deque
from dataclasses import asdict
from typing import Any, Deque, Dict, Iterator, List, Optional

from backend.src.slothbearflow_backend.observability.context import (
    TraceContext,
    bind_trace_context,
    current_trace_context,
    current_trace_id,
    new_trace_context,
    reset_trace_context,
)
from backend.src.slothbearflow_backend.observability.redaction import (
    sanitize_observability_data,
)


logger = logging.getLogger(__name__)


class _LangfuseBridge:
    def __init__(self, client: Any) -> None:
        self.client = client
        self.api = (
            "v3"
            if hasattr(client, "start_observation")
            else "legacy"
            if hasattr(client, "trace")
            else "unsupported"
        )

    def start_trace(
        self, trace_id: str, operation: str, metadata: Dict[str, Any]
    ) -> Any:
        if self.api == "v3":
            kwargs = {
                "name": operation,
                "as_type": "span",
                "metadata": metadata,
                "trace_context": {"trace_id": trace_id},
            }
            try:
                return self.client.start_observation(**kwargs)
            except TypeError:
                kwargs.pop("trace_context", None)
                return self.client.start_observation(**kwargs)
        if self.api == "legacy":
            return self.client.trace(id=trace_id, name=operation, metadata=metadata)
        return None

    def finish_trace(self, observation: Any, *, status: str, error: str) -> None:
        if observation is None:
            return
        self._finish_observation(
            observation,
            output={"status": status, "error": error},
            level="ERROR" if status == "error" else "DEFAULT",
        )

    def record_span(self, trace_id: str, span: Dict[str, Any]) -> None:
        if self.api == "v3":
            kwargs = {
                "name": str(span.get("name") or "span"),
                "as_type": "span",
                "metadata": {
                    "component": span.get("component"),
                    "duration_ms": span.get("duration_ms"),
                    **dict(span.get("metadata") or {}),
                },
                "trace_context": {"trace_id": trace_id},
            }
            try:
                observation = self.client.start_observation(**kwargs)
            except TypeError:
                kwargs.pop("trace_context", None)
                observation = self.client.start_observation(**kwargs)
            self._finish_observation(
                observation,
                output={"status": span.get("status")},
                level="ERROR" if span.get("status") == "error" else "DEFAULT",
            )
            return
        if self.api == "legacy":
            trace = self.client.trace(id=trace_id)
            child = trace.span(
                name=str(span.get("name") or "span"),
                metadata=dict(span.get("metadata") or {}),
            )
            child.end(output={"status": span.get("status")})

    def record_generation(
        self,
        *,
        trace_id: str,
        name: str,
        model: str,
        input_summary: Dict[str, Any],
        output_summary: Dict[str, Any],
        metadata: Dict[str, Any],
    ) -> None:
        if self.api != "v3":
            return
        kwargs = {
            "name": name,
            "as_type": "generation",
            "model": model,
            "input": input_summary,
            "metadata": metadata,
            "trace_context": {"trace_id": trace_id},
        }
        try:
            observation = self.client.start_observation(**kwargs)
        except TypeError:
            kwargs.pop("trace_context", None)
            observation = self.client.start_observation(**kwargs)
        self._finish_observation(observation, output=output_summary)

    @staticmethod
    def _finish_observation(
        observation: Any,
        *,
        output: Any,
        level: str = "DEFAULT",
    ) -> None:
        # Langfuse Python 3.15 moved output/level from end(...) to update(...).
        # Prefer the current contract while retaining compatibility with older v3 clients.
        if hasattr(observation, "update"):
            observation.update(output=output, level=level)
            observation.end()
            return
        observation.end(output=output, level=level)

    def flush(self) -> None:
        if hasattr(self.client, "flush"):
            self.client.flush()


class ObservabilityFacade:
    def __init__(self, settings: Any) -> None:
        self.settings = settings
        self.enabled = bool(getattr(settings, "observability_enabled", False))
        self.include_content = bool(getattr(settings, "trace_include_content", False))
        self._traces: Deque[Dict[str, Any]] = deque(
            maxlen=max(1, int(getattr(settings, "trace_store_size", 200)))
        )
        self._active: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.RLock()
        self._langfuse = self._build_langfuse()
        self._metrics = self._build_metrics()

    def _build_langfuse(self) -> Any:
        if not self.enabled or not bool(getattr(self.settings, "langfuse_enabled", False)):
            return None
        public_key = str(getattr(self.settings, "langfuse_public_key", "") or "")
        secret_key = str(getattr(self.settings, "langfuse_secret_key", "") or "")
        if not public_key or not secret_key:
            logger.warning("Langfuse enabled but API keys are missing; using local traces")
            return None
        host = str(getattr(self.settings, "langfuse_host", "") or "")
        try:
            from langfuse import Langfuse  # type: ignore

            return _LangfuseBridge(
                Langfuse(
                    public_key=public_key,
                    secret_key=secret_key,
                    base_url=host,
                )
            )
        except TypeError:
            try:
                from langfuse import Langfuse  # type: ignore

                return _LangfuseBridge(
                    Langfuse(
                        public_key=public_key,
                        secret_key=secret_key,
                        host=host,
                    )
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Langfuse initialization failed; using local traces: %s", exc
                )
                return None
        except Exception as exc:  # noqa: BLE001
            logger.warning("Langfuse initialization failed; using local traces: %s", exc)
            return None

    def _build_metrics(self) -> Dict[str, Any]:
        if not self.enabled or not bool(getattr(self.settings, "prometheus_enabled", False)):
            return {}
        try:
            from prometheus_client import Counter, Gauge, Histogram, REGISTRY

            def existing_or_create(name: str, factory: Any) -> Any:
                collector = getattr(REGISTRY, "_names_to_collectors", {}).get(name)
                return collector or factory()

            return {
                "requests": existing_or_create(
                    "slothbearflow_http_requests_total",
                    lambda: Counter(
                        "slothbearflow_http_requests_total",
                        "HTTP requests",
                        ["method", "path", "status"],
                    ),
                ),
                "duration": existing_or_create(
                    "slothbearflow_http_request_duration_seconds",
                    lambda: Histogram(
                        "slothbearflow_http_request_duration_seconds",
                        "HTTP request duration",
                        ["method", "path"],
                    ),
                ),
                "spans": existing_or_create(
                    "slothbearflow_spans_total",
                    lambda: Counter(
                        "slothbearflow_spans_total",
                        "Agent spans",
                        ["component", "name", "status"],
                    ),
                ),
                "active": existing_or_create(
                    "slothbearflow_active_requests",
                    lambda: Gauge(
                        "slothbearflow_active_requests",
                        "Active HTTP requests",
                    ),
                ),
            }
        except Exception as exc:  # noqa: BLE001
            logger.warning("Prometheus metrics unavailable: %s", exc)
            return {}

    def start_trace(
        self,
        operation: str,
        *,
        request_id: str = "",
        trace_id: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> tuple[TraceContext, Any]:
        safe_metadata = self._sanitize(dict(metadata or {}))
        context = new_trace_context(
            operation,
            request_id=request_id,
            trace_id=trace_id,
            metadata=safe_metadata,
        )
        token = bind_trace_context(context)
        if not self.enabled:
            return context, token
        now = time.time()
        record = {
            **asdict(context),
            "owner_id": context.user_id,
            "started_at": now,
            "status": "running",
            "duration_ms": 0.0,
            "spans": [],
        }
        with self._lock:
            self._active[context.trace_id] = record
        if self._metrics.get("active"):
            self._metrics["active"].inc()
        if self._langfuse is not None:
            try:
                record["_langfuse_trace"] = self._langfuse.start_trace(
                    context.trace_id, operation, safe_metadata
                )
            except Exception:
                logger.exception("Langfuse trace creation failed")
        return context, token

    def finish_trace(self, token: Any, *, status: str = "ok", error: str = "") -> None:
        context = current_trace_context()
        if context is None:
            return
        if not self.enabled:
            reset_trace_context(token)
            return
        safe_error = self._sanitize(error)
        with self._lock:
            record = self._active.pop(context.trace_id, None)
            if record is not None:
                record["status"] = status
                record["error"] = safe_error
                record["user_id"] = context.user_id
                record["owner_id"] = context.user_id
                record["tenant_id"] = context.tenant_id
                record["metadata"] = self._sanitize(context.metadata)
                record["duration_ms"] = round(
                    (time.time() - float(record["started_at"])) * 1000, 3
                )
                langfuse_trace = record.pop("_langfuse_trace", None)
                self._traces.append(record)
            else:
                langfuse_trace = None
        if langfuse_trace is not None:
            try:
                update = getattr(langfuse_trace, "update", None)
                if callable(update):
                    identity_metadata = self._sanitize(
                        {
                            "tenant_id": context.tenant_id,
                            "user_id_hash": hashlib.sha256(
                                context.user_id.encode("utf-8")
                            ).hexdigest()[:24],
                        }
                    )
                    update(metadata=identity_metadata)
                self._langfuse.finish_trace(
                    langfuse_trace, status=status, error=safe_error
                )
            except Exception:
                logger.exception("Langfuse trace finalization failed")
        if self._metrics.get("active"):
            self._metrics["active"].dec()
        reset_trace_context(token)

    @contextlib.contextmanager
    def span(
        self,
        name: str,
        *,
        component: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Iterator[Dict[str, Any]]:
        started = time.perf_counter()
        span = {
            "span_id": uuid.uuid4().hex,
            "name": name,
            "component": component,
            "status": "ok",
            "metadata": dict(metadata or {}),
        }
        try:
            yield span
        except BaseException as exc:  # noqa: BLE001
            span["status"] = "error"
            span["error_type"] = type(exc).__name__
            raise
        finally:
            span["duration_ms"] = round((time.perf_counter() - started) * 1000, 3)
            self.record_span(span)

    def record_span(self, span: Dict[str, Any]) -> None:
        if not self.enabled:
            return
        trace_id = current_trace_id()
        if not trace_id:
            return
        safe_span = self._sanitize(dict(span))
        with self._lock:
            record = self._active.get(trace_id)
            if record is not None:
                record["spans"].append(safe_span)
        if self._langfuse is not None:
            try:
                self._langfuse.record_span(trace_id, safe_span)
            except Exception:
                logger.exception("Langfuse span export failed")
        metric = self._metrics.get("spans")
        if metric:
            metric.labels(
                str(safe_span.get("component") or "unknown"),
                str(safe_span.get("name") or "unknown"),
                str(safe_span.get("status") or "unknown"),
            ).inc()

    def event(
        self,
        name: str,
        *,
        component: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not self.enabled:
            return
        self.record_span(
            {
                "span_id": uuid.uuid4().hex,
                "name": name,
                "component": component,
                "status": "ok",
                "duration_ms": 0.0,
                "metadata": dict(metadata or {}),
            }
        )

    def record_http(self, method: str, path: str, status: int, duration: float) -> None:
        if not self.enabled:
            return
        counter = self._metrics.get("requests")
        histogram = self._metrics.get("duration")
        if counter:
            counter.labels(method, path, str(status)).inc()
        if histogram:
            histogram.labels(method, path).observe(duration)

    def record_generation(
        self,
        *,
        name: str,
        model: str,
        input_chars: int,
        output_chars: int,
        latency_ms: float,
        stop_reason: str,
    ) -> None:
        if not self.enabled:
            return
        trace_id = current_trace_id()
        metadata = {
            "latency_ms": round(latency_ms, 3),
            "stop_reason": stop_reason,
        }
        self.event(name, component="llm", metadata={**metadata, "model": model})
        if self._langfuse is not None and trace_id:
            try:
                self._langfuse.record_generation(
                    trace_id=trace_id,
                    name=name,
                    model=model,
                    input_summary={"chars": input_chars},
                    output_summary={"chars": output_chars},
                    metadata=metadata,
                )
            except Exception:
                logger.exception("Langfuse generation export failed")

    def recent_traces(
        self,
        limit: int = 50,
        *,
        tenant_id: str = "",
        user_id: str = "",
        owner_id: str = "",
    ) -> List[Dict[str, Any]]:
        if not self.enabled:
            return []
        with self._lock:
            rows = list(self._traces)
            rows = [
                row
                for row in rows
                if self._trace_visible(
                    row,
                    tenant_id=tenant_id,
                    user_id=user_id,
                    owner_id=owner_id,
                )
            ]
            selected = rows[-max(1, min(limit, 200)) :][::-1]
            return [self._public_trace(row) for row in selected]

    def get_trace(
        self,
        trace_id: str,
        *,
        tenant_id: str = "",
        user_id: str = "",
        owner_id: str = "",
    ) -> Optional[Dict[str, Any]]:
        if not self.enabled:
            return None
        with self._lock:
            if trace_id in self._active:
                value = self._active[trace_id]
                return (
                    self._public_trace(value)
                    if self._trace_visible(
                        value,
                        tenant_id=tenant_id,
                        user_id=user_id,
                        owner_id=owner_id,
                    )
                    else None
                )
            for trace in reversed(self._traces):
                if trace.get("trace_id") == trace_id:
                    return (
                        self._public_trace(trace)
                        if self._trace_visible(
                            trace,
                            tenant_id=tenant_id,
                            user_id=user_id,
                            owner_id=owner_id,
                        )
                        else None
                    )
        return None

    def _sanitize(self, value: Any) -> Any:
        return sanitize_observability_data(
            value,
            include_content=self.include_content,
        )

    @staticmethod
    def _trace_visible(
        trace: Dict[str, Any],
        *,
        tenant_id: str,
        user_id: str,
        owner_id: str,
    ) -> bool:
        if tenant_id and str(trace.get("tenant_id") or "") != tenant_id:
            return False
        trace_user_id = str(trace.get("owner_id") or trace.get("user_id") or "")
        if user_id and trace_user_id != user_id:
            return False
        if owner_id and trace_user_id != owner_id:
            return False
        return True

    def _public_trace(self, trace: Dict[str, Any]) -> Dict[str, Any]:
        value = {
            key: copy.deepcopy(item)
            for key, item in trace.items()
            if not str(key).startswith("_")
        }
        return self._sanitize(value)

    def status(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "local_trace_store": self.enabled,
            "prometheus": bool(self._metrics),
            "langfuse": bool(
                self._langfuse and self._langfuse.api != "unsupported"
            ),
            "langfuse_api": self._langfuse.api if self._langfuse else "disabled",
            "langfuse_configured": bool(
                self.enabled and getattr(self.settings, "langfuse_enabled", False)
            ),
            "langfuse_host": str(getattr(self.settings, "langfuse_host", "") or ""),
        }

    def flush(self) -> None:
        if self.enabled and self._langfuse is not None:
            try:
                self._langfuse.flush()
            except Exception:
                logger.exception("Langfuse flush failed")


_instance: Optional[ObservabilityFacade] = None
_instance_key = ""
_instance_lock = threading.Lock()


def get_observability(settings: Any) -> ObservabilityFacade:
    global _instance, _instance_key
    key = repr(
        (
            bool(getattr(settings, "observability_enabled", False)),
            bool(getattr(settings, "prometheus_enabled", False)),
            int(getattr(settings, "trace_store_size", 200)),
            bool(getattr(settings, "trace_include_content", False)),
            bool(getattr(settings, "langfuse_enabled", False)),
            str(getattr(settings, "langfuse_host", "") or ""),
            bool(
                getattr(settings, "langfuse_public_key", "")
                and getattr(settings, "langfuse_secret_key", "")
            ),
        )
    )
    if _instance is None or key != _instance_key:
        with _instance_lock:
            if _instance is None or key != _instance_key:
                if _instance is not None:
                    _instance.flush()
                _instance = ObservabilityFacade(settings)
                _instance_key = key
    return _instance


def reset_observability() -> None:
    global _instance, _instance_key
    with _instance_lock:
        if _instance is not None:
            _instance.flush()
        _instance = None
        _instance_key = ""
