from __future__ import annotations

import asyncio

import pytest


@pytest.fixture(autouse=True)
def _env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SKIP_MILVUS", "true")
    monkeypatch.setenv("USE_RAG", "false")
    monkeypatch.setenv("ASYNC_SUMMARY_UPDATE", "false")
    from backend.src.slothbearflow_backend.config import get_settings

    get_settings.cache_clear()


# --------------------------- store / index ---------------------------


def test_learning_store_writes_and_rejects_traversal(tmp_path) -> None:
    from backend.src.slothbearflow_backend.learning.schema import MemoryItem, SkillItem
    from backend.src.slothbearflow_backend.learning.store import LearningStore

    store = LearningStore(str(tmp_path))
    mem = store.upsert_memory(
        MemoryItem(name="Prefers Short", description="偏好简短", type="feedback", body="简短作答")
    )
    skill = store.upsert_skill(SkillItem(name="cite", trigger="内部文档", body="先检索再答"))
    assert mem is not None and mem.exists()
    assert skill is not None and skill.exists()
    assert mem.name == "prefers-short.md"

    # 路径穿越被净化，落点仍在 memory 目录内。
    bad = store.upsert_memory(MemoryItem(name="../../etc/passwd", body="x"))
    assert bad is not None
    assert str(bad.resolve()).startswith(str(store.memory_dir.resolve()))


def test_learning_store_upsert_dedupes_by_name(tmp_path) -> None:
    from backend.src.slothbearflow_backend.learning.schema import MemoryItem
    from backend.src.slothbearflow_backend.learning.store import LearningStore

    store = LearningStore(str(tmp_path))
    store.upsert_memory(MemoryItem(name="pref", description="旧", body="旧正文"))
    store.upsert_memory(MemoryItem(name="pref", description="新", body="新正文"))
    files = list((tmp_path / "memory").glob("*.md"))
    assert len(files) == 1
    row = store.index.get("memory", "pref")
    assert row is not None and row["description"] == "新"


def test_learning_index_search_and_reindex(tmp_path) -> None:
    from backend.src.slothbearflow_backend.learning.schema import MemoryItem
    from backend.src.slothbearflow_backend.learning.store import LearningStore
    from backend.src.slothbearflow_backend.learning.index_db import LearningIndex

    store = LearningStore(str(tmp_path))
    store.upsert_memory(MemoryItem(name="short", description="用户偏好简短回答", body="简短"))
    store.upsert_memory(MemoryItem(name="lang", description="用户使用中文", body="中文"))
    hits = store.index.search("简短 回答", kind="memory", limit=5)
    assert hits and hits[0]["name"] == "short"

    # 重建索引：删库后从磁盘 .md 全量恢复。
    fresh = LearningIndex(str(tmp_path / "index.sqlite"))
    count = fresh.reindex_from_disk(str(tmp_path))
    assert count == 2


def test_select_for_injection_is_bounded(tmp_path) -> None:
    from backend.src.slothbearflow_backend.learning.schema import MemoryItem
    from backend.src.slothbearflow_backend.learning.store import LearningStore

    store = LearningStore(str(tmp_path))
    store.upsert_memory(
        MemoryItem(name="m1", description="简短 " * 200, body="正文 " * 200)
    )
    block = store.select_for_injection("简短", budget_chars=80)
    assert block
    assert len(block) <= 80 + 2  # 预算 + " …" 截断标记


# --------------------------- review guard / runtime ---------------------------


def test_thread_tool_whitelist() -> None:
    from backend.src.slothbearflow_backend.learning.review_guard import (
        clear_thread_tool_whitelist,
        is_tool_allowed,
        set_thread_tool_whitelist,
    )

    assert is_tool_allowed("anything") is True  # 未设置 → 全放行
    set_thread_tool_whitelist({"save_memory"})
    try:
        assert is_tool_allowed("save_memory") is True
        assert is_tool_allowed("terminal") is False
    finally:
        clear_thread_tool_whitelist()
    assert is_tool_allowed("terminal") is True


def test_react_runtime_denies_non_whitelisted_tool() -> None:
    from langchain_core.messages import AIMessage

    from backend.src.slothbearflow_backend.agent.react_runtime import ExplicitReActRuntime
    from backend.src.slothbearflow_backend.learning.review_guard import (
        clear_thread_tool_whitelist,
        set_thread_tool_whitelist,
    )

    class DummyBoundLLM:
        def __init__(self) -> None:
            self.calls = 0

        def invoke(self, messages):
            self.calls += 1
            if self.calls == 1:
                return AIMessage(
                    content="",
                    tool_calls=[
                        {"name": "terminal", "args": {}, "id": "c1", "type": "tool_call"}
                    ],
                )
            return AIMessage(content="done")

    class DummyLLM:
        def __init__(self) -> None:
            self._bound = DummyBoundLLM()

        def bind_tools(self, tools):
            return self._bound

    class EvilTool:
        name = "terminal"

        def invoke(self, payload):  # pragma: no cover - must never run
            raise AssertionError("denied tool must not execute")

    runtime = ExplicitReActRuntime(llm=DummyLLM(), tools=[EvilTool()], max_steps=2)
    set_thread_tool_whitelist({"save_memory", "save_skill"})
    try:
        result = runtime.invoke({"input": "x"})
    finally:
        clear_thread_tool_whitelist()

    trace = result["tool_trace"]
    assert any(s["name"] == "terminal" and s["ok"] is False for s in trace)
    assert "denied" in trace[0]["observation"].lower()


# --------------------------- review job (structured path) ---------------------------


def test_run_review_job_structured_writes_files(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ENABLE_BACKGROUND_REVIEW", "true")
    monkeypatch.setenv("REVIEW_BASE_DIR", str(tmp_path))
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setenv("OLLAMA_MODEL_SUPPORTS_TOOLS", "false")  # → 结构化路径
    from langchain_core.runnables import RunnableLambda

    import backend.src.slothbearflow_backend.learning.review_agent as ra
    from backend.src.slothbearflow_backend.config import get_settings
    from backend.src.slothbearflow_backend.learning.schema import (
        MemoryItem,
        ReviewResult,
        SkillItem,
    )

    def fake_llm(settings=None, temperature=None):
        class FakeLLM:
            def with_structured_output(self, schema):
                return RunnableLambda(
                    lambda _m: ReviewResult(
                        should_save=True,
                        rationale="ok",
                        memories=[MemoryItem(name="short", description="偏好简短", body="简短")],
                        skills=[SkillItem(name="cite", trigger="内部文档", body="先检索")],
                    )
                )

        return FakeLLM()

    monkeypatch.setattr(ra, "get_chat_llm", fake_llm)
    get_settings.cache_clear()

    ra.run_review_job(
        {
            "session_id": "s1",
            "user_message": "以后简短点",
            "final_answer": "好的",
            "review_memory": True,
            "review_skills": True,
        },
        get_settings(),
    )

    assert (tmp_path / "memory" / "short.md").exists()
    assert (tmp_path / "skills" / "cite.md").exists()


def test_run_review_job_noop_when_disabled(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ENABLE_BACKGROUND_REVIEW", "false")
    monkeypatch.setenv("REVIEW_BASE_DIR", str(tmp_path))
    import backend.src.slothbearflow_backend.learning.review_agent as ra
    from backend.src.slothbearflow_backend.config import get_settings

    def boom(*a, **k):  # pragma: no cover
        raise AssertionError("review must not call LLM when disabled")

    monkeypatch.setattr(ra, "get_chat_llm", boom)
    get_settings.cache_clear()

    ra.run_review_job(
        {"session_id": "s", "user_message": "x", "final_answer": "y", "review_memory": True},
        get_settings(),
    )
    assert not (tmp_path / "memory").exists() or not list((tmp_path / "memory").glob("*.md"))


# --------------------------- trigger / interval gating ---------------------------


def _make_runner(settings, queue):
    from backend.src.slothbearflow_backend.agent.conversation_loop import ChatTurnRunner

    return ChatTurnRunner(
        settings,
        queue,
        build_agent_executor=lambda **k: None,
        get_vector_store=lambda settings=None: None,
        structured_chat_output_from_text=lambda *a, **k: None,
        get_last_rag_sources=lambda: [],
        get_last_rag_citations=lambda: [],
    )


def _make_prepared(payload):
    from backend.src.slothbearflow_backend.agent.conversation_loop import (
        PreparedTurn,
        TurnInput,
    )

    return PreparedTurn(
        turn=TurnInput(session_id="s1", message="hi"),
        payload=payload,
        client=None,
        windowed=[],
        rag_retrieval=None,
        prefetched_citations=[],
        llm_input="hi",
        executor=None,
        should_stream=False,
        stream_reason="",
        stream_format="plain",
        rolling_summary="",
    )


def test_review_dimensions_interval_gating(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("REVIEW_MEMORY_INTERVAL", "2")
    monkeypatch.setenv("REVIEW_SKILLS_INTERVAL", "3")
    from backend.src.slothbearflow_backend.config import get_settings

    get_settings.cache_clear()
    runner = _make_runner(get_settings(), None)
    assert runner._review_dimensions(1) == (False, False)
    assert runner._review_dimensions(2) == (True, False)
    assert runner._review_dimensions(3) == (False, True)
    assert runner._review_dimensions(6) == (True, True)


def test_review_enqueued_when_due(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ENABLE_BACKGROUND_REVIEW", "true")
    monkeypatch.setenv("REVIEW_MEMORY_INTERVAL", "1")
    monkeypatch.setenv("REVIEW_SKILLS_INTERVAL", "5")
    from backend.src.slothbearflow_backend.config import get_settings

    get_settings.cache_clear()
    queue: asyncio.Queue = asyncio.Queue()
    runner = _make_runner(get_settings(), queue)
    prepared = _make_prepared(
        {
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "yo"},
            ]
        }
    )
    runner._maybe_enqueue_review(
        prepared, answer="yo", raw="yo", tools_used=[], citations=[]
    )
    assert queue.qsize() == 1
    job = queue.get_nowait()
    assert job["type"] == "review"
    assert job["snapshot"]["review_memory"] is True
    assert job["snapshot"]["review_skills"] is False
    assert job["snapshot"]["user_message"] == "hi"


def test_review_not_enqueued_when_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ENABLE_BACKGROUND_REVIEW", "false")
    monkeypatch.setenv("REVIEW_MEMORY_INTERVAL", "1")
    from backend.src.slothbearflow_backend.config import get_settings

    get_settings.cache_clear()
    queue: asyncio.Queue = asyncio.Queue()
    runner = _make_runner(get_settings(), queue)
    prepared = _make_prepared(
        {"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}]}
    )
    runner._maybe_enqueue_review(
        prepared, answer="yo", raw="yo", tools_used=[], citations=[]
    )
    assert queue.qsize() == 0


def test_worker_loop_dispatches_review_job(monkeypatch: pytest.MonkeyPatch) -> None:
    import backend.src.slothbearflow_backend.worker.background as bg
    from backend.src.slothbearflow_backend.config import get_settings

    get_settings.cache_clear()
    seen: list = []
    monkeypatch.setattr(bg, "run_review_job", lambda snap, settings=None: seen.append(snap))

    async def run() -> None:
        queue: asyncio.Queue = asyncio.Queue()
        await queue.put(
            {"type": "review", "snapshot": {"session_id": "s1", "review_memory": True}}
        )
        task = asyncio.create_task(bg.worker_loop(queue, get_settings()))
        await queue.join()
        task.cancel()

    asyncio.run(run())
    assert seen and seen[0]["session_id"] == "s1"


# --------------------------- read-back prompt injection ---------------------------


def test_build_system_prompt_injects_learning() -> None:
    from backend.src.slothbearflow_backend.prompt import build_system_prompt

    with_ctx = build_system_prompt(
        supports_tools=False, learning_context="用户偏好简短回答"
    )
    assert "长期记忆与技巧" in with_ctx
    assert "用户偏好简短回答" in with_ctx

    without_ctx = build_system_prompt(supports_tools=False)
    assert "长期记忆与技巧" not in without_ctx
