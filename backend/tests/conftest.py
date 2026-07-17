from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _isolate_runtime_files(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("APP_ENV", "test")
    monkeypatch.setenv("AUTH_REQUIRED", "false")
    monkeypatch.setenv("AUTH_LOCAL_ROLES_JSON", '["operator"]')
    monkeypatch.setenv("LANGFUSE_ENABLED", "false")
    monkeypatch.setenv("ENABLE_POSTGRES_PERSISTENCE", "false")
    monkeypatch.setenv("POSTGRES_DSN", "")
    monkeypatch.setenv("POSTGRES_RESTORE_ON_REDIS_MISS", "false")
    monkeypatch.setenv("SKIP_MILVUS", "true")
    monkeypatch.setenv("USE_RAG", "false")
    monkeypatch.setenv("LLM_HEALTHCHECK_ENABLED", "false")
    monkeypatch.setenv("MCP_ENABLED", "false")
    monkeypatch.setenv("AUDIT_LOG_FILE", str(tmp_path / "audit.jsonl"))

    from backend.src.slothbearflow_backend.config import get_settings

    get_settings.cache_clear()
    yield
    get_settings.cache_clear()
