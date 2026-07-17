from __future__ import annotations

import argparse
import hashlib
import json
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlparse


def _embedding(value: object, dimensions: int = 32) -> list[float]:
    digest = hashlib.sha256(str(value).encode("utf-8")).digest()
    raw = [((digest[index % len(digest)] / 255.0) * 2.0) - 1.0 for index in range(dimensions)]
    norm = sum(item * item for item in raw) ** 0.5 or 1.0
    return [round(item / norm, 8) for item in raw]


def _usage(messages: list[dict[str, Any]], content: str) -> dict[str, int]:
    prompt_chars = sum(len(json.dumps(item, ensure_ascii=False)) for item in messages)
    return {
        "prompt_tokens": max(1, prompt_chars // 4),
        "completion_tokens": max(1, len(content) // 4),
        "total_tokens": max(2, (prompt_chars + len(content)) // 4),
    }


class OpenAIStubHandler(BaseHTTPRequestHandler):
    server_version = "SlothBearFlowOpenAIStub/1.0"
    protocol_version = "HTTP/1.1"

    def log_message(self, format: str, *args: object) -> None:
        print(f"[{self.log_date_time_string()}] {format % args}", flush=True)

    def _json_body(self) -> dict[str, Any]:
        try:
            length = int(self.headers.get("content-length", "0"))
            value = json.loads(self.rfile.read(length) or b"{}")
        except (ValueError, json.JSONDecodeError):
            self._send(400, {"error": {"message": "invalid JSON body"}})
            raise
        if not isinstance(value, dict):
            self._send(400, {"error": {"message": "JSON body must be an object"}})
            raise ValueError("JSON body must be an object")
        return value

    def _send(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path == "/health":
            self._send(200, {"status": "ok"})
            return
        if path.startswith("/v1/models/"):
            model = path.rsplit("/", 1)[-1]
            self._send(200, {"id": model, "object": "model", "owned_by": "local"})
            return
        if path == "/v1/models":
            self._send(
                200,
                {
                    "object": "list",
                    "data": [
                        {"id": "slothbearflow-e2e", "object": "model", "owned_by": "local"}
                    ],
                },
            )
            return
        self._send(404, {"error": {"message": "not found"}})

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        try:
            body = self._json_body()
        except (ValueError, json.JSONDecodeError):
            return
        if path == "/v1/embeddings":
            raw_inputs = body.get("input", [])
            inputs = raw_inputs if isinstance(raw_inputs, list) else [raw_inputs]
            self._send(
                200,
                {
                    "object": "list",
                    "model": str(body.get("model") or "slothbearflow-e2e-embedding"),
                    "data": [
                        {
                            "object": "embedding",
                            "index": index,
                            "embedding": _embedding(value),
                        }
                        for index, value in enumerate(inputs)
                    ],
                    "usage": {"prompt_tokens": len(inputs), "total_tokens": len(inputs)},
                },
            )
            return
        if path == "/v1/chat/completions":
            self._chat_completion(body)
            return
        self._send(404, {"error": {"message": "not found"}})

    def _chat_completion(self, body: dict[str, Any]) -> None:
        messages = [item for item in body.get("messages", []) if isinstance(item, dict)]
        tools = [item for item in body.get("tools", []) if isinstance(item, dict)]
        tool_names = {
            str(item.get("function", {}).get("name") or "")
            for item in tools
            if isinstance(item.get("function"), dict)
        }
        has_tool_result = any(item.get("role") == "tool" for item in messages)
        message: dict[str, Any]
        finish_reason: str
        if "get_current_time" in tool_names and not has_tool_result:
            message = {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_local_time",
                        "type": "function",
                        "function": {"name": "get_current_time", "arguments": "{}"},
                    }
                ],
            }
            finish_reason = "tool_calls"
            content = ""
        else:
            content = json.dumps(
                {
                    "answer": "SlothBearFlow 本地全链路验收完成。",
                    "citations": [],
                    "confidence": 0.99,
                },
                ensure_ascii=False,
            )
            message = {"role": "assistant", "content": content}
            finish_reason = "stop"
        if body.get("stream"):
            self._stream_chat_completion(
                model=str(body.get("model") or "slothbearflow-e2e"),
                message=message,
                finish_reason=finish_reason,
            )
            return
        self._send(
            200,
            {
                "id": "chatcmpl-" + uuid.uuid4().hex,
                "object": "chat.completion",
                "created": int(time.time()),
                "model": str(body.get("model") or "slothbearflow-e2e"),
                "choices": [
                    {"index": 0, "message": message, "finish_reason": finish_reason}
                ],
                "usage": _usage(messages, content),
            },
        )

    def _stream_chat_completion(
        self,
        *,
        model: str,
        message: dict[str, Any],
        finish_reason: str,
    ) -> None:
        completion_id = "chatcmpl-" + uuid.uuid4().hex
        created = int(time.time())
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()

        delta: dict[str, Any] = {"role": "assistant"}
        if message.get("tool_calls"):
            delta["tool_calls"] = [
                {
                    "index": index,
                    **tool_call,
                }
                for index, tool_call in enumerate(message["tool_calls"])
            ]
        else:
            delta["content"] = str(message.get("content") or "")
        chunks = [
            {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
            },
            {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [
                    {"index": 0, "delta": {}, "finish_reason": finish_reason}
                ],
            },
        ]
        try:
            for chunk in chunks:
                self.wfile.write(
                    ("data: " + json.dumps(chunk, ensure_ascii=False) + "\n\n").encode(
                        "utf-8"
                    )
                )
                self.wfile.flush()
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        finally:
            self.close_connection = True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a loopback-only OpenAI-compatible server for deterministic E2E tests."
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18080)
    args = parser.parse_args()
    if args.host not in {"127.0.0.1", "::1", "localhost"}:
        raise SystemExit("The integration stub may only bind to a loopback address.")
    server = ThreadingHTTPServer((args.host, args.port), OpenAIStubHandler)
    print(f"OpenAI-compatible stub listening on http://{args.host}:{args.port}/v1", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
