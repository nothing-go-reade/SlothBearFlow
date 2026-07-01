from __future__ import annotations

import types

import pytest
from langchain_core.tools import tool

from backend.src.slothbearflow_backend.learning.review_guard import (
    clear_thread_tool_whitelist,
    set_thread_tool_whitelist,
)
from backend.src.slothbearflow_backend.security import (
    ArgConstraint,
    PolicyBundle,
    PolicyGuardedTool,
    ToolPolicy,
    apply_tool_policy,
    begin_turn,
    end_turn,
    evaluate_tool_call,
    load_tool_policy,
    scrub_observation,
)


@pytest.fixture(autouse=True)
def _reset() -> None:
    load_tool_policy.cache_clear()
    clear_thread_tool_whitelist()
    end_turn()
    yield
    clear_thread_tool_whitelist()
    end_turn()
    load_tool_policy.cache_clear()


def _settings(mode="enforce", scrub=True, max_calls=8):
    return types.SimpleNamespace(
        tool_guard_mode=mode,
        tool_scrub_output=scrub,
        max_tool_calls_per_turn=max_calls,
        tool_policy_file="",
    )


def _weather_policy(**kwargs):
    tp = ToolPolicy(allow=True, **kwargs)
    return PolicyBundle(default_action="deny", max_tool_calls_per_turn=8, tools={"get_weather": tp})


# ---------------------------------------------------------------- 加载 / 默认策略


def test_builtin_default_allows_readonly_denies_unknown():
    pol = load_tool_policy("", "enforce")
    assert pol.default_action == "deny"
    assert set(pol.allowed_tool_names()) == {
        "get_current_time",
        "get_weather",
        "get_session_context",
        "search_knowledge",
    }
    s = _settings()
    assert evaluate_tool_call("get_weather", {"city": "北京"}, settings=s, policy=pol).allowed
    assert not evaluate_tool_call("run_shell", {}, settings=s, policy=pol).allowed


def test_loader_reads_yaml_file(tmp_path):
    f = tmp_path / "policy.yaml"
    f.write_text(
        "version: 1\n"
        "default_action: deny\n"
        "tools:\n"
        "  get_weather:\n"
        "    allow: true\n"
        "    class: read\n"
        "    args:\n"
        "      city:\n"
        "        type: string\n"
        "        max_len: 8\n",
        encoding="utf-8",
    )
    pol = load_tool_policy(str(f), "enforce")
    assert pol.tools["get_weather"].cls == "read"
    assert pol.tools["get_weather"].args["city"].max_len == 8


def test_loader_missing_file_falls_back(tmp_path):
    pol = load_tool_policy(str(tmp_path / "nope.yaml"), "enforce")
    assert "get_weather" in pol.allowed_tool_names()


# ---------------------------------------------------------------- 白名单准入 / 过滤


def test_apply_tool_policy_filters_unlisted_and_wraps():
    @tool
    def get_weather(city: str) -> str:
        "weather"
        return "sun"

    @tool
    def run_shell(cmd: str) -> str:
        "danger"
        return "ran"

    pol = load_tool_policy("", "enforce")
    out = apply_tool_policy([get_weather, run_shell], pol, _settings())
    assert [t.name for t in out] == ["get_weather"]
    assert isinstance(out[0], PolicyGuardedTool)


def test_wrapped_run_denies_via_langchain_entry():
    # 证明拦截点对 AgentExecutor 路径同样生效（AgentExecutor 调 tool.run）。
    @tool
    def run_shell(cmd: str) -> str:
        "danger"
        return "ran"

    pol = PolicyBundle(default_action="deny", tools={})
    guarded = PolicyGuardedTool(inner_tool=run_shell, policy=pol, settings=_settings())
    observation = guarded.run({"cmd": "rm -rf /"})
    assert "not in allowlist" in observation


# ---------------------------------------------------------------- 参数校验


def test_arg_over_limit_rejected_and_inner_not_called():
    called = {"n": 0}

    @tool
    def get_weather(city: str) -> str:
        "weather"
        called["n"] += 1
        return "sun"

    pol = _weather_policy(args={"city": ArgConstraint(type="string", max_len=3)})
    guarded = PolicyGuardedTool(inner_tool=get_weather, policy=pol, settings=_settings())
    observation = guarded.run({"city": "超长的城市名"})
    assert "max_len" in observation
    assert called["n"] == 0  # 被拒时内部工具绝不执行


# ---------------------------------------------------------------- 配额


def test_quota_exhaustion():
    pol = _weather_policy(max_calls_per_turn=2)
    s = _settings()
    begin_turn()
    try:
        results = [
            evaluate_tool_call("get_weather", {"city": "北"}, settings=s, policy=pol).allowed
            for _ in range(3)
        ]
    finally:
        end_turn()
    assert results == [True, True, False]


def test_no_turn_means_no_quota():
    # 未开回合（如后台复盘 / 直调）→ 不受配额限制。
    pol = _weather_policy(max_calls_per_turn=1)
    s = _settings()
    results = [
        evaluate_tool_call("get_weather", {"city": "北"}, settings=s, policy=pol).allowed
        for _ in range(3)
    ]
    assert results == [True, True, True]


# ---------------------------------------------------------------- 危险动作


def test_requires_approval_auto_denied():
    pol = PolicyBundle(
        default_action="deny",
        tools={"delete_file": ToolPolicy(allow=True, requires_approval=True)},
    )
    d = evaluate_tool_call("delete_file", {"path": "x"}, settings=_settings(), policy=pol)
    assert not d.allowed
    assert "approval" in d.reason


# ---------------------------------------------------------------- 输出脱敏


def test_output_scrub_redacts_secret():
    s = _settings(scrub=True)
    out = scrub_observation("token sk-ABCDEFGHIJKLMNOP1234 tail", s)
    assert "[REDACTED]" in out
    assert "sk-ABCDEFGHIJKLMNOP1234" not in out


def test_output_scrub_disabled_passthrough():
    s = _settings(scrub=False)
    raw = "token sk-ABCDEFGHIJKLMNOP1234 tail"
    assert scrub_observation(raw, s) == raw


# ------------------------------------------------------------ 复盘白名单不受影响


def test_review_thread_whitelist_short_circuits_policy():
    # 策略文件里 get_weather 是放行的，但复盘线程白名单一旦设置就独立裁决、
    # 跳过策略文件；save_memory 不在策略文件却被放行、get_weather 反被拒。
    pol = load_tool_policy("", "enforce")
    s = _settings()
    set_thread_tool_whitelist({"save_memory", "save_skill"})
    try:
        assert evaluate_tool_call("save_memory", {}, settings=s, policy=pol).allowed
        assert not evaluate_tool_call("get_weather", {"city": "北京"}, settings=s, policy=pol).allowed
    finally:
        clear_thread_tool_whitelist()


# ---------------------------------------------------------------- 模式：off / log


def test_mode_off_passthrough_no_wrap():
    @tool
    def run_shell(cmd: str) -> str:
        "danger"
        return "ran"

    pol = load_tool_policy("", "enforce")
    out = apply_tool_policy([run_shell], pol, _settings(mode="off"))
    assert out[0] is run_shell  # off：不过滤、不包裹，原样返回


def test_mode_log_wraps_but_does_not_filter_or_block():
    @tool
    def run_shell(cmd: str) -> str:
        "danger"
        return "ran"

    pol = load_tool_policy("", "enforce")  # run_shell 不在名单
    out = apply_tool_policy([run_shell], pol, _settings(mode="log"))
    assert [t.name for t in out] == ["run_shell"]  # log：不过滤
    assert isinstance(out[0], PolicyGuardedTool)
    # log 模式下不阻断：观察即放行，内部工具照常执行。
    assert out[0].run({"cmd": "ls"}) == "ran"
