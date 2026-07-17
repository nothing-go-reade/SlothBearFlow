from __future__ import annotations

import argparse
import base64
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


def _load_env(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip()
    return values


def _request_json(
    url: str,
    *,
    headers: dict[str, str] | None = None,
    timeout: float = 5.0,
) -> Any:
    request = urllib.request.Request(url, headers=headers or {})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _wait_json(
    url: str,
    *,
    headers: dict[str, str] | None = None,
    timeout: float,
) -> Any:
    deadline = time.monotonic() + timeout
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            return _request_json(url, headers=headers)
        except (OSError, ValueError, urllib.error.HTTPError) as exc:
            last_error = exc
            time.sleep(1.0)
    raise RuntimeError(f"service did not become ready: {url}: {last_error}")


def _basic_auth(username: str, password: str) -> dict[str, str]:
    encoded = base64.b64encode(f"{username}:{password}".encode()).decode("ascii")
    return {"Authorization": f"Basic {encoded}"}


def _rows(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if not isinstance(payload, dict):
        return []
    for key in ("data", "items", "traces", "observations"):
        value = payload.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
    return []


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify Langfuse, Prometheus and Grafana with live API readback."
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=Path(__file__).resolve().parents[1] / ".env.observability",
    )
    parser.add_argument("--backend-url", default="http://127.0.0.1:8000")
    parser.add_argument("--langfuse-url", default="http://127.0.0.1:3000")
    parser.add_argument("--prometheus-url", default="http://127.0.0.1:9090")
    parser.add_argument("--grafana-url", default="http://127.0.0.1:3001")
    parser.add_argument("--trace-id", default="")
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args()

    values = _load_env(args.env_file.expanduser().resolve())
    langfuse_headers = _basic_auth(
        values["LANGFUSE_INIT_PROJECT_PUBLIC_KEY"],
        values["LANGFUSE_INIT_PROJECT_SECRET_KEY"],
    )
    grafana_headers = _basic_auth(
        values.get("GRAFANA_ADMIN_USER", "admin"),
        values["GRAFANA_ADMIN_PASSWORD"],
    )

    backend = _wait_json(args.backend_url.rstrip("/") + "/health", timeout=args.timeout)
    langfuse = _wait_json(
        args.langfuse_url.rstrip("/") + "/api/public/health", timeout=args.timeout
    )
    prometheus = _wait_json(
        args.prometheus_url.rstrip("/") + "/api/v1/status/buildinfo", timeout=args.timeout
    )
    grafana = _wait_json(
        args.grafana_url.rstrip("/") + "/api/health", timeout=args.timeout
    )
    datasource = _wait_json(
        args.grafana_url.rstrip("/") + "/api/datasources/uid/prometheus/health",
        headers=grafana_headers,
        timeout=args.timeout,
    )

    query = urllib.parse.urlencode({"query": 'up{job="slothbearflow-backend"}'})
    metrics = _wait_json(
        args.prometheus_url.rstrip("/") + "/api/v1/query?" + query,
        timeout=args.timeout,
    )
    metric_rows = metrics.get("data", {}).get("result", []) if isinstance(metrics, dict) else []
    backend_up = any(
        isinstance(row, dict)
        and isinstance(row.get("value"), list)
        and len(row["value"]) > 1
        and str(row["value"][1]) == "1"
        for row in metric_rows
    )

    trace_found = not args.trace_id
    observation_names: list[str] = []
    if args.trace_id:
        deadline = time.monotonic() + args.timeout
        while time.monotonic() < deadline and not trace_found:
            traces = _request_json(
                args.langfuse_url.rstrip("/") + "/api/public/traces?limit=100",
                headers=langfuse_headers,
            )
            trace_found = any(str(row.get("id") or "") == args.trace_id for row in _rows(traces))
            if not trace_found:
                time.sleep(1.0)
        if trace_found:
            observation_query = urllib.parse.urlencode(
                {"traceId": args.trace_id, "limit": "100"}
            )
            observations = _request_json(
                args.langfuse_url.rstrip("/")
                + "/api/public/observations?"
                + observation_query,
                headers=langfuse_headers,
            )
            observation_names = sorted(
                {
                    str(row.get("name") or "")
                    for row in _rows(observations)
                    if str(row.get("name") or "")
                }
            )

    checks = {
        "backend": bool(isinstance(backend, dict)),
        "langfuse": bool(isinstance(langfuse, dict)),
        "prometheus": bool(isinstance(prometheus, dict)),
        "prometheus_backend_up": backend_up,
        "grafana": bool(isinstance(grafana, dict)),
        "grafana_prometheus_datasource": str(
            datasource.get("status") if isinstance(datasource, dict) else ""
        ).lower()
        in {"ok", "success"},
        "langfuse_trace_readback": trace_found,
    }
    output = {
        "ok": all(checks.values()),
        "checks": checks,
        "trace_id": args.trace_id,
        "observation_names": observation_names,
    }
    print(json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True))
    if not output["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
