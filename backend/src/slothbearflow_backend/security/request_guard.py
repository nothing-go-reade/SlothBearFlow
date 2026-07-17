from __future__ import annotations

import ipaddress
from typing import Any, Dict, List

from backend.src.slothbearflow_backend.config import get_settings


class RequestSizeLimitMiddleware:
    def __init__(self, app: Any, max_bytes: int) -> None:
        self.app = app
        self.max_bytes = max(1, int(max_bytes))

    async def __call__(self, scope: Dict[str, Any], receive: Any, send: Any) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return
        headers = {
            key.decode("latin-1").lower(): value.decode("latin-1")
            for key, value in scope.get("headers") or []
        }
        try:
            content_length = int(headers.get("content-length") or 0)
        except ValueError:
            content_length = self.max_bytes + 1
        if content_length > self.max_bytes:
            await self._reject(send)
            return

        # Content-Length is optional and untrusted. Buffer the bounded JSON body
        # before invoking the application so chunked requests cannot bypass the
        # limit after a handler has already started its response.
        messages: List[Dict[str, Any]] = []
        received_bytes = 0
        while True:
            message = await receive()
            messages.append(message)
            if message.get("type") != "http.request":
                break
            received_bytes += len(message.get("body") or b"")
            if received_bytes > self.max_bytes:
                await self._reject(send)
                return
            if not message.get("more_body", False):
                break

        index = 0

        async def replay_receive() -> Dict[str, Any]:
            nonlocal index
            if index < len(messages):
                message = messages[index]
                index += 1
                return message
            return await receive()

        await self.app(scope, replay_receive, send)

    @staticmethod
    async def _reject(send: Any) -> None:
        body = b'{"detail":"Request body is too large."}'
        await send(
            {
                "type": "http.response.start",
                "status": 413,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode()),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})


class LocalAuthBoundaryMiddleware:
    """Keep the intentionally unauthenticated local profile on the local machine."""

    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(self, scope: Dict[str, Any], receive: Any, send: Any) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return
        settings = get_settings()
        if (
            settings.auth_required
            or settings.app_env == "test"
            or _is_loopback_client(scope.get("client"))
        ):
            await self.app(scope, receive, send)
            return
        body = b'{"detail":"Unauthenticated local mode only accepts loopback clients."}'
        await send(
            {
                "type": "http.response.start",
                "status": 403,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode("ascii")),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})


def _is_loopback_client(client: Any) -> bool:
    host = str(client[0] if isinstance(client, (tuple, list)) and client else "")
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False
