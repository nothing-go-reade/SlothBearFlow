from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from typing import Any

import pytest

from backend.src.slothbearflow_backend.observability.context import (
    current_trace_context,
)
from backend.src.slothbearflow_backend.observability.facade import (
    ObservabilityFacade,
)
from backend.src.slothbearflow_backend.observability.middleware import (
    NOT_FOUND_ROUTE,
    RequestTraceMiddleware,
)
from backend.src.slothbearflow_backend.observability.redaction import (
    REDACTED,
    sanitize_observability_data,
)


def _settings(**overrides: Any) -> SimpleNamespace:
    values = {
        "observability_enabled": True,
        "prometheus_enabled": False,
        "trace_store_size": 20,
        "trace_include_content": False,
        "langfuse_enabled": False,
        "langfuse_host": "http://127.0.0.1:3000",
        "langfuse_public_key": "",
        "langfuse_secret_key": "",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class _Exporter:
    api = "v3"

    def __init__(self) -> None:
        self.calls: list[tuple[str, Any]] = []

    def start_trace(self, trace_id: str, operation: str, metadata: dict[str, Any]) -> Any:
        self.calls.append(("trace", metadata))
        return self

    def finish_trace(self, observation: Any, *, status: str, error: str) -> None:
        self.calls.append(("finish", {"status": status, "error": error}))

    def record_span(self, trace_id: str, span: dict[str, Any]) -> None:
        self.calls.append(("span", span))

    def record_generation(self, **kwargs: Any) -> None:
        self.calls.append(("generation", kwargs))

    def flush(self) -> None:
        self.calls.append(("flush", None))


class _Metric:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[Any, ...]]] = []

    def labels(self, *labels: Any) -> "_Metric":
        self.calls.append(("labels", labels))
        return self

    def inc(self) -> None:
        self.calls.append(("inc", ()))

    def dec(self) -> None:
        self.calls.append(("dec", ()))

    def observe(self, value: float) -> None:
        self.calls.append(("observe", (value,)))


def test_observability_disabled_has_no_storage_or_export() -> None:
    facade = ObservabilityFacade(
        _settings(
            observability_enabled=False,
            prometheus_enabled=True,
            langfuse_enabled=True,
            langfuse_public_key="public",
            langfuse_secret_key="secret",
        )
    )
    assert facade.status() == {
        "enabled": False,
        "local_trace_store": False,
        "prometheus": False,
        "langfuse": False,
        "langfuse_api": "disabled",
        "langfuse_configured": False,
        "langfuse_host": "http://127.0.0.1:3000",
    }

    exporter = _Exporter()
    metrics = {name: _Metric() for name in ("active", "requests", "duration", "spans")}
    facade._langfuse = exporter
    facade._metrics = metrics

    context, token = facade.start_trace("disabled", metadata={"input": "private"})
    with facade.span("ignored", component="test", metadata={"output": "private"}):
        pass
    facade.event("ignored.event", component="test", metadata={"citations": ["private"]})
    facade.record_generation(
        name="ignored.generation",
        model="demo",
        input_chars=7,
        output_chars=8,
        latency_ms=1.0,
        stop_reason="done",
    )
    facade.record_http("GET", "/ignored", 200, 0.01)
    facade.finish_trace(token)
    facade.flush()

    assert facade.get_trace(context.trace_id) is None
    assert facade.recent_traces() == []
    assert facade._active == {}
    assert list(facade._traces) == []
    assert exporter.calls == []
    assert all(metric.calls == [] for metric in metrics.values())
    assert current_trace_context() is None


def test_content_is_redacted_once_for_local_store_and_exporter() -> None:
    raw_input = "private user question"
    raw_output = "private model answer"
    raw_excerpt = "private retrieved paragraph"
    facade = ObservabilityFacade(_settings())
    exporter = _Exporter()
    facade._langfuse = exporter

    context, token = facade.start_trace(
        "content-policy",
        metadata={
            "input": {"question": raw_input, "chars": len(raw_input)},
            "output": raw_output,
            "provenance": {"server": "private-server", "kind": "rag"},
            "citations": [
                {"source": "private.md", "excerpt": raw_excerpt, "score": 0.9}
            ],
            "safe_label": "prepare",
        },
    )
    with facade.span(
        "content.span",
        component="test",
        metadata={
            "output": {"answer": raw_output},
            "provenance": {"citations": [{"excerpt": raw_excerpt}]},
            "safe_label": "run",
        },
    ):
        pass
    facade.finish_trace(token)

    trace = facade.get_trace(context.trace_id)
    assert trace is not None
    assert trace["metadata"]["input"] == {
        "question": REDACTED,
        "chars": len(raw_input),
    }
    assert trace["metadata"]["output"] == REDACTED
    assert trace["metadata"]["provenance"] == {
        "server": REDACTED,
        "kind": REDACTED,
    }
    assert trace["metadata"]["citations"][0] == {
        "source": REDACTED,
        "excerpt": REDACTED,
        "score": 0.9,
    }
    assert trace["metadata"]["safe_label"] == "prepare"
    assert trace["spans"][0]["metadata"]["output"]["answer"] == REDACTED
    assert trace["spans"][0]["metadata"]["safe_label"] == "run"

    persisted = json.dumps(trace, ensure_ascii=False)
    exported = json.dumps(exporter.calls, ensure_ascii=False)
    for raw_value in (raw_input, raw_output, raw_excerpt, "private-server", "private.md"):
        assert raw_value not in persisted
        assert raw_value not in exported
    assert REDACTED in persisted
    assert REDACTED in exported


def test_content_can_be_included_without_disabling_secret_redaction() -> None:
    facade = ObservabilityFacade(_settings(trace_include_content=True))
    context, token = facade.start_trace(
        "full-content",
        metadata={
            "input": "benign prompt",
            "citations": [{"excerpt": "benign excerpt"}],
            "api_key": "plain-secret-value",
        },
    )
    facade.finish_trace(token)

    trace = facade.get_trace(context.trace_id)
    assert trace is not None
    assert trace["metadata"]["input"] == "benign prompt"
    assert trace["metadata"]["citations"][0]["excerpt"] == "benign excerpt"
    assert trace["metadata"]["api_key"] == REDACTED


def test_secret_redaction_covers_camel_case_tokens_and_complete_private_keys() -> None:
    private_key = (
        "-----BEGIN PRIVATE KEY-----\n"
        "c3VwZXItc2VjcmV0LWtleS1tYXRlcmlhbA==\n"
        "-----END PRIVATE KEY-----"
    )
    raw_values = [
        "sk-proj-ABCDEFGHIJKLMNOPQRSTUVWX",
        "github_pat_ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890",
        "eyJabcdefgh.ijklmnop.qrstuvwx",
        "postgresql://agent:super-secret@db.internal/app",
        private_key,
    ]
    sanitized = sanitize_observability_data(
        {
            "accessToken": raw_values[0],
            "privateKey": private_key,
            "message": " ".join(raw_values),
        },
        include_content=True,
    )
    encoded = json.dumps(sanitized, ensure_ascii=False)

    assert sanitized["accessToken"] == REDACTED
    assert sanitized["privateKey"] == REDACTED
    assert all(value not in encoded for value in raw_values)


def _record_owned_trace(
    facade: ObservabilityFacade,
    *,
    tenant_id: str,
    user_id: str,
) -> str:
    context, token = facade.start_trace("owned")
    context.tenant_id = tenant_id
    context.user_id = user_id
    facade.finish_trace(token)
    return context.trace_id


def test_trace_queries_support_tenant_and_owner_user_acl() -> None:
    facade = ObservabilityFacade(_settings())
    alice_trace = _record_owned_trace(facade, tenant_id="tenant-a", user_id="alice")
    bob_trace = _record_owned_trace(facade, tenant_id="tenant-a", user_id="bob")
    other_tenant_trace = _record_owned_trace(
        facade,
        tenant_id="tenant-b",
        user_id="alice",
    )

    tenant_rows = facade.recent_traces(tenant_id="tenant-a")
    assert {row["trace_id"] for row in tenant_rows} == {alice_trace, bob_trace}
    assert [
        row["trace_id"]
        for row in facade.recent_traces(tenant_id="tenant-a", user_id="alice")
    ] == [alice_trace]
    assert [
        row["trace_id"]
        for row in facade.recent_traces(tenant_id="tenant-a", owner_id="bob")
    ] == [bob_trace]
    assert facade.get_trace(alice_trace, tenant_id="tenant-a", user_id="bob") is None
    assert facade.get_trace(alice_trace, tenant_id="tenant-b", user_id="alice") is None
    assert facade.get_trace(
        alice_trace,
        tenant_id="tenant-a",
        user_id="alice",
        owner_id="alice",
    ) is not None
    assert facade.get_trace(other_tenant_trace, tenant_id="tenant-a") is None


def test_middleware_records_route_templates_and_bounds_404_labels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class CapturingFacade:
        enabled = True

        def __init__(self) -> None:
            self.http_calls: list[tuple[str, str, int, float]] = []

        def start_trace(self, operation: str, **kwargs: Any) -> tuple[Any, object]:
            return (
                SimpleNamespace(
                    request_id=kwargs.get("request_id") or "request-id",
                    trace_id="trace-id",
                    metadata=dict(kwargs.get("metadata") or {}),
                ),
                object(),
            )

        def record_http(
            self,
            method: str,
            path: str,
            status: int,
            duration: float,
        ) -> None:
            self.http_calls.append((method, path, status, duration))

        def finish_trace(self, token: object, **kwargs: Any) -> None:
            return None

    facade = CapturingFacade()
    monkeypatch.setattr(
        "backend.src.slothbearflow_backend.observability.middleware.get_observability",
        lambda settings: facade,
    )

    async def app(scope: dict[str, Any], receive: Any, send: Any) -> None:
        route_path = scope.get("test_route")
        if route_path:
            scope["route"] = SimpleNamespace(path=route_path, path_format=route_path)
        await send(
            {
                "type": "http.response.start",
                "status": int(scope["test_status"]),
                "headers": [],
            }
        )
        await send({"type": "http.response.body", "body": b""})

    middleware = RequestTraceMiddleware(app, settings=object())

    async def call(path: str, status: int, route: str = "") -> None:
        async def receive() -> dict[str, Any]:
            return {"type": "http.request", "body": b"", "more_body": False}

        async def send(message: dict[str, Any]) -> None:
            return None

        await middleware(
            {
                "type": "http",
                "method": "GET",
                "path": path,
                "headers": [],
                "test_status": status,
                "test_route": route,
            },
            receive,
            send,
        )

    asyncio.run(call("/widgets/123", 200, "/widgets/{widget_id}"))
    asyncio.run(call("/missing/first-id", 404))
    asyncio.run(call("/missing/second-id", 404))

    assert facade.http_calls[0][1] == "/widgets/{widget_id}"
    assert facade.http_calls[1][1] == NOT_FOUND_ROUTE
    assert facade.http_calls[2][1] == NOT_FOUND_ROUTE
    labels = {item[1] for item in facade.http_calls}
    assert "/widgets/123" not in labels
    assert "/missing/first-id" not in labels
    assert "/missing/second-id" not in labels


def test_middleware_blocks_metrics_export_when_observability_is_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    facade = ObservabilityFacade(
        _settings(observability_enabled=False, prometheus_enabled=True)
    )
    monkeypatch.setattr(
        "backend.src.slothbearflow_backend.observability.middleware.get_observability",
        lambda settings: facade,
    )
    app_called = False

    async def app(scope: dict[str, Any], receive: Any, send: Any) -> None:
        nonlocal app_called
        app_called = True

    middleware = RequestTraceMiddleware(app, settings=object())
    messages: list[dict[str, Any]] = []

    async def call() -> None:
        async def receive() -> dict[str, Any]:
            return {"type": "http.request", "body": b"", "more_body": False}

        async def send(message: dict[str, Any]) -> None:
            messages.append(message)

        await middleware(
            {
                "type": "http",
                "method": "GET",
                "path": "/metrics",
                "headers": [],
            },
            receive,
            send,
        )

    asyncio.run(call())

    assert app_called is False
    assert messages[0]["status"] == 404
    assert messages[1]["body"] == b'{"detail":"Observability is disabled."}'


def test_middleware_replaces_non_ascii_request_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class CapturingFacade:
        enabled = True

        def start_trace(self, _operation: str, **kwargs: Any) -> tuple[Any, object]:
            request_id = kwargs.get("request_id") or "generated-request-id"
            return (
                SimpleNamespace(
                    request_id=request_id,
                    trace_id="trace-id",
                    metadata={},
                    error_type="",
                ),
                object(),
            )

        def record_http(self, *_args: Any, **_kwargs: Any) -> None:
            return None

        def finish_trace(self, *_args: Any, **_kwargs: Any) -> None:
            return None

    monkeypatch.setattr(
        "backend.src.slothbearflow_backend.observability.middleware.get_observability",
        lambda _settings: CapturingFacade(),
    )

    async def app(_scope: dict[str, Any], _receive: Any, send: Any) -> None:
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b""})

    messages: list[dict[str, Any]] = []

    async def scenario() -> None:
        async def receive() -> dict[str, Any]:
            return {"type": "http.request", "body": b"", "more_body": False}

        async def send(message: dict[str, Any]) -> None:
            messages.append(message)

        await RequestTraceMiddleware(app, settings=object())(
            {
                "type": "http",
                "method": "GET",
                "path": "/",
                "headers": [(b"x-request-id", b"\xffinvalid")],
            },
            receive,
            send,
        )

    asyncio.run(scenario())
    headers = dict(messages[0]["headers"])
    assert headers[b"x-request-id"] == b"generated-request-id"
