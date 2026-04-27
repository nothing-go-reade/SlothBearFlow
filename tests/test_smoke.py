from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(autouse=True)
def _env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SKIP_MILVUS", "true")
    monkeypatch.setenv("USE_RAG", "false")
    monkeypatch.setenv("STRUCTURED_OUTPUT", "false")
    monkeypatch.setenv("ASYNC_SUMMARY_UPDATE", "false")
    from app.config import get_settings

    get_settings.cache_clear()


def test_health_ok() -> None:
    from app.main import app

    with TestClient(app) as client:
        r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body.get("ok") is True
    assert "redis" in body
    assert "milvus" in body
    assert "session_store" in body


def test_root_ok() -> None:
    from app.main import app

    with TestClient(app) as client:
        r = client.get("/")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["docs"] == "/docs"


def test_config_loads() -> None:
    from app.config import get_settings

    s = get_settings()
    assert s.ollama_base_url.startswith("http")
    assert isinstance(s.ollama_model_supports_tools, bool)
    assert s.redis_socket_timeout > 0
    assert s.milvus_timeout > 0
    assert s.app_log_file.endswith(".log")
    assert s.access_log_file.endswith(".log")
    assert s.error_log_file.endswith(".log")


def test_build_agent_executor_falls_back_when_model_has_no_tools(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OLLAMA_MODEL_SUPPORTS_TOOLS", "false")
    from app.agent.agent_executor import BasicChatExecutor, build_agent_executor
    from app.config import get_settings

    get_settings.cache_clear()
    executor = build_agent_executor(vector_store=None, chat_history=[], rolling_summary=None)
    assert isinstance(executor, BasicChatExecutor)


def test_config_defaults_to_plain_text_output() -> None:
    from app.config import Settings

    s = Settings(_env_file=None)
    assert s.stream_output is False
    assert s.structured_output is False
    assert s.enable_postgres_persistence is False


def test_chat_streams_when_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("STREAM_OUTPUT", "true")
    monkeypatch.setenv("STRUCTURED_OUTPUT", "false")
    monkeypatch.setenv("STREAM_OUTPUT_FORMAT", "sse")
    from app.config import get_settings
    from app.main import app

    class FakeStreamExecutor:
        def stream(self, payload):
            assert payload["input"] == "你好"
            yield {"output": "你好，"}
            yield {"output": "这是流式返回。"}

    monkeypatch.setattr("app.main.build_agent_executor", lambda **kwargs: FakeStreamExecutor())
    monkeypatch.setattr("app.main.get_vector_store", lambda settings=None: None)
    get_settings.cache_clear()

    with TestClient(app) as client:
        with client.stream("POST", "/chat", json={"session_id": "s-stream", "message": "你好"}) as r:
            body = "".join(chunk for chunk in r.iter_text())

    assert r.status_code == 200
    assert '"type": "chunk"' in body
    assert "这是流式返回" in body


def test_chat_streams_plain_text_when_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("STREAM_OUTPUT", "true")
    monkeypatch.setenv("STRUCTURED_OUTPUT", "false")
    monkeypatch.setenv("STREAM_OUTPUT_FORMAT", "plain")
    from app.config import get_settings
    from app.main import app

    class FakeStreamExecutor:
        def stream(self, payload):
            yield {"output": "你好，"}
            yield {"output": "纯文本流式返回。"}

    monkeypatch.setattr("app.main.build_agent_executor", lambda **kwargs: FakeStreamExecutor())
    monkeypatch.setattr("app.main.get_vector_store", lambda settings=None: None)
    get_settings.cache_clear()

    with TestClient(app) as client:
        with client.stream("POST", "/chat", json={"session_id": "s-plain", "message": "你好"}) as r:
            body = "".join(chunk for chunk in r.iter_text())

    assert r.status_code == 200
    assert "data:" not in body
    assert body == "你好，纯文本流式返回。"


def test_chat_works_with_in_memory_session_store(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.main import app
    from app.output_schema import ChatOutput

    class FakeExecutor:
        def invoke(self, payload):
            assert payload["input"] == "你好"
            return {"output": "这是一个测试回答"}

    monkeypatch.setattr("app.main.build_agent_executor", lambda **kwargs: FakeExecutor())
    monkeypatch.setattr("app.main.get_vector_store", lambda settings=None: None)
    monkeypatch.setattr(
        "app.main.structured_chat_output_from_text",
        lambda raw, rag_hint="", settings=None: ChatOutput(answer=raw, source="agent"),
    )

    with TestClient(app) as client:
        r = client.post("/chat", json={"session_id": "s1", "message": "你好"})

    assert r.status_code == 200
    body = r.json()
    assert body["answer"] == "这是一个测试回答"
    assert body["session_id"] == "s1"


def test_chat_persists_metadata_when_postgres_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ENABLE_POSTGRES_PERSISTENCE", "true")
    monkeypatch.setenv("POSTGRES_DSN", "postgresql://demo")
    from app.main import app

    persisted: list[dict] = []

    class FakeExecutor:
        def invoke(self, payload):
            return {"output": "这是一个测试回答"}

    monkeypatch.setattr("app.main.build_agent_executor", lambda **kwargs: FakeExecutor())
    monkeypatch.setattr("app.main.get_vector_store", lambda settings=None: None)
    monkeypatch.setattr(
        "app.main.postgres_persistence.ensure_schema",
        lambda settings=None: True,
    )
    monkeypatch.setattr(
        "app.main.postgres_persistence.persist_chat_turn",
        lambda **kwargs: persisted.append(kwargs),
    )

    with TestClient(app) as client:
        r = client.post("/chat", json={"session_id": "s-pg", "message": "你好"})

    assert r.status_code == 200
    assert len(persisted) == 1
    assert persisted[0]["session_id"] == "s-pg"
    assert persisted[0]["user_message"] == "你好"
    assert persisted[0]["assistant_message"] == "这是一个测试回答"


def test_ingest_blocked_when_rag_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SKIP_MILVUS", "true")
    monkeypatch.setenv("USE_RAG", "false")
    from app.config import get_settings
    from app.main import app

    get_settings.cache_clear()
    with TestClient(app) as client:
        r = client.post("/ingest", json={"source": "a.txt", "text": "hello"})

    assert r.status_code == 400
    assert "向量库写入" in r.json()["detail"]


def test_ingest_accepts_job_when_rag_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SKIP_MILVUS", "false")
    monkeypatch.setenv("USE_RAG", "true")
    from app.config import get_settings
    from app.main import app

    async def fake_worker_loop(queue, settings=None):
        while True:
            await queue.get()
            queue.task_done()

    monkeypatch.setattr("app.main.worker_loop", fake_worker_loop)
    get_settings.cache_clear()
    with TestClient(app) as client:
        r = client.post("/ingest", json={"source": "a.txt", "text": "hello"})

    assert r.status_code == 200
    body = r.json()
    assert body["accepted"] is True
    assert body["job_id"]


def test_ingest_persists_job_metadata_when_postgres_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SKIP_MILVUS", "false")
    monkeypatch.setenv("USE_RAG", "true")
    monkeypatch.setenv("ENABLE_POSTGRES_PERSISTENCE", "true")
    monkeypatch.setenv("POSTGRES_DSN", "postgresql://demo")
    from app.config import get_settings
    from app.main import app

    persisted: list[dict] = []

    async def fake_worker_loop(queue, settings=None):
        while True:
            await queue.get()
            queue.task_done()

    monkeypatch.setattr("app.main.worker_loop", fake_worker_loop)
    monkeypatch.setattr(
        "app.main.postgres_persistence.ensure_schema",
        lambda settings=None: True,
    )
    monkeypatch.setattr(
        "app.main.postgres_persistence.persist_ingest_job",
        lambda **kwargs: persisted.append(kwargs),
    )
    get_settings.cache_clear()

    with TestClient(app) as client:
        r = client.post("/ingest", json={"source": "a.txt", "text": "hello"})

    assert r.status_code == 200
    assert len(persisted) == 1
    assert persisted[0]["source"] == "a.txt"
    assert persisted[0]["text_length"] == 5
    assert persisted[0]["status"] == "queued"


def test_chat_returns_rag_citations(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.main import app

    class FakeExecutor:
        def invoke(self, payload):
            return {"output": "根据知识库，退款申请需要财务审核。"}

    monkeypatch.setattr("app.main.build_agent_executor", lambda **kwargs: FakeExecutor())
    monkeypatch.setattr("app.main.get_vector_store", lambda settings=None: object())
    monkeypatch.setattr("app.main.get_last_rag_sources", lambda: ["refund-policy.md"])
    monkeypatch.setattr(
        "app.main.get_last_rag_citations",
        lambda: [{"source": "refund-policy.md", "excerpt": "退款申请需要先提交工单，再经过财务审核。"}],
    )
    monkeypatch.setenv("SKIP_MILVUS", "false")
    monkeypatch.setenv("USE_RAG", "true")
    monkeypatch.setenv("STRUCTURED_OUTPUT", "false")

    from app.config import get_settings

    get_settings.cache_clear()
    with TestClient(app) as client:
        r = client.post("/chat", json={"session_id": "rag-1", "message": "退款流程是什么？"})

    assert r.status_code == 200
    body = r.json()
    assert body["source"] == "refund-policy.md"
    assert body["tools_used"] == ["search_knowledge"]
    assert body["citations"][0]["source"] == "refund-policy.md"
    assert "财务审核" in body["citations"][0]["excerpt"]
