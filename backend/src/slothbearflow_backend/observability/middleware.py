from __future__ import annotations

import re
import time
from typing import Any, Dict

from backend.src.slothbearflow_backend.observability.facade import get_observability
from backend.src.slothbearflow_backend.config import get_settings


NOT_FOUND_ROUTE = "__not_found__"
UNKNOWN_ROUTE = "__unknown__"
_REQUEST_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")


def route_template(scope: Dict[str, Any], status_code: int) -> str:
    """Resolve a bounded Prometheus route label without using the request URL."""

    if status_code == 404:
        return NOT_FOUND_ROUTE
    route = scope.get("route")
    for attribute in ("path_format", "path"):
        value = str(getattr(route, attribute, "") or "")
        if value:
            return value
    return UNKNOWN_ROUTE


class RequestTraceMiddleware:
    def __init__(self, app: Any, settings: Any = None) -> None:
        self.app = app
        self.settings = settings

    async def __call__(self, scope: Dict[str, Any], receive: Any, send: Any) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return
        headers = {
            key.decode("latin-1").lower(): value.decode("latin-1")
            for key, value in scope.get("headers") or []
        }
        request_id = headers.get("x-request-id", "")[:128]
        if not _REQUEST_ID_RE.fullmatch(request_id):
            request_id = ""
        method = str(scope.get("method") or "GET").upper()
        settings = self.settings or get_settings()
        facade = get_observability(settings)
        if not facade.enabled and scope.get("path") == "/metrics":
            body = b'{"detail":"Observability is disabled."}'
            await send(
                {
                    "type": "http.response.start",
                    "status": 404,
                    "headers": [
                        (b"content-type", b"application/json"),
                        (b"content-length", str(len(body)).encode("ascii")),
                    ],
                }
            )
            await send({"type": "http.response.body", "body": body})
            return
        context, token = facade.start_trace(
            "http.request",
            request_id=request_id,
            metadata={"method": method},
        )
        started = time.perf_counter()
        status_code = 500

        async def traced_send(message: Dict[str, Any]) -> None:
            nonlocal status_code
            if message.get("type") == "http.response.start":
                status_code = int(message.get("status") or 500)
                response_headers = list(message.get("headers") or [])
                response_headers.extend(
                    [
                        (b"x-request-id", context.request_id.encode("ascii")),
                        (b"x-trace-id", context.trace_id.encode("ascii")),
                    ]
                )
                message["headers"] = response_headers
            await send(message)

        error = ""
        try:
            await self.app(scope, receive, traced_send)
        except BaseException as exc:  # noqa: BLE001
            error = type(exc).__name__
            raise
        finally:
            duration = time.perf_counter() - started
            path = route_template(scope, status_code)
            context.metadata["path"] = path
            facade.record_http(method, path, status_code, duration)
            traced_error = error or str(getattr(context, "error_type", "") or "")
            facade.finish_trace(
                token,
                status="ok" if status_code < 500 and not traced_error else "error",
                error=traced_error,
            )
