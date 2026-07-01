# SlothBearFlow — 工具调用安全加固（Tool Guard）

> 描述已落地的 function calling / tool use 安全层：一个**配置文件驱动、初始化即生效、覆盖全部执行路径**的工具准入与校验机制。对标 **OWASP LLM Top 10 2025 · LLM08 Excessive Agency**。
>
> 整理日期：2026-07-01 ｜ 相关代码：`security/*`、`tools/registry.py`、`agent/conversation_loop.py`、`config.py`、`backend/config/tool_policy.yaml`

---

## 一、为什么做

加固前，工具调用**容错合格但安全拦截几乎空白**：唯一的白名单只作用于后台复盘；主链路 `/chat` 对工具零准入控制；且旧的白名单检查只在 `ExplicitReActRuntime` 生效，默认的 LangChain `AgentExecutor` 路径直接绕过。当前 4 个工具恰好只读、无副作用，是「没有危险工具」的侥幸而非「有防护」。一旦接入写操作/外部 API，将没有任何一层拦截危险参数或越权调用。

## 二、核心架构：单一拦截点覆盖两条路径

两条工具执行路径最终都汇聚到 `BaseTool.run → _run`：

- LangChain `AgentExecutor` → `tool.run(...)`
- `ExplicitReActRuntime._invoke_tool` → `tool.invoke(...)` → `run` → `_run`

因此用一个重写 `_run`/`_arun` 的 `BaseTool` 子类包裹器（`security/wrapper.py` 的 `PolicyGuardedTool`），即可**同时拦住两条路径**，且复用内部工具的 `name/description/args_schema` → function-calling schema 字节不变（已用 `convert_to_openai_tool` 验证相等）。

三层防御：

| 层 | 位置 | 覆盖 |
|---|---|---|
| ① 构建期过滤 | `tools/registry.py::build_tools` → `apply_tool_policy` | 丢弃 `allow:false`/未列入的工具，模型根本看不到 |
| ② 执行期包裹 | `PolicyGuardedTool._run/_arun` | 参数校验、每回合配额、`requires_approval` 拒绝、输出脱敏 |
| ③ 复盘白名单 | `learning/review_guard`（原有，被 engine 优先短路） | 复盘上下文独立裁决，不受策略文件影响 |

**每回合配额**用 `contextvars`（`security/turn_state.py`）：能穿透 `asyncio.to_thread`，每请求独立 context → 不跨请求泄漏。`ChatTurnRunner.prepare` 开回合、`_finalize` 收尾。

## 三、策略文件 `backend/config/tool_policy.yaml`

初始化时由 `security/loader.py` 加载一次（`lru_cache`）。缺失/解析失败/PyYAML 不可用 → 回退代码内置默认策略（放行现有 4 只读工具、`default_action: deny`），保证不 breaks 现有工具。

每个工具可声明：`allow`、`class: read|write`、`requires_approval`、`max_calls_per_turn`、以及逐参数约束 `type / max_len / min_len / enum / regex / min / max / path_within`。全局 `default_action: deny` 拒绝一切未列入工具。

## 四、7 个安全维度 → 落点

| # | 维度 | 落点 |
|---|---|---|
| 1 | 工具白名单 | 策略 `allow` + 构建期过滤（两路径通用） |
| 2 | 主链路工具准入 | `PolicyGuardedTool` 拦住 LangChain 路径（旧 bypass 关闭） |
| 3 | 参数校验/清洗 | `security/validators.py`（类型/长度/枚举/正则/区间/路径） |
| 4 | 危险动作拦截 | `requires_approval` → headless 无 HITL，自动拒绝 |
| 5 | 调用频率/配额 | `contextvars` 每回合全局 + 每工具上限 |
| 6 | 参数注入/越权 | allowlist 式校验（enum/regex known-good、`path_within` 防穿越） |
| 7 | 敏感信息过滤 | `security/scrub.py` 对工具**输出**脱敏后再回灌模型 |

## 五、配置项（默认全部安全默认）

| Env Var | 默认 | 说明 |
|---|---|---|
| `TOOL_GUARD_MODE` | `enforce` | `off`｜`log`（只记录不阻断）｜`enforce` |
| `TOOL_POLICY_FILE` | `backend/config/tool_policy.yaml` | 策略文件路径 |
| `MAX_TOOL_CALLS_PER_TURN` | `8` | 单回合总配额（策略文件可覆盖） |
| `TOOL_SCRUB_OUTPUT` | `true` | 工具输出脱敏（保守高精度） |

## 六、向后兼容

`enforce` 默认 + 提交的 YAML 放行现有 4 只读工具 → 今天所有调用照常；未知/未来工具默认被拒（新增安全收益）。`off` 模式 `build_tools` 不介入、纯透传。后台复盘的线程白名单**优先短路**策略文件，故 `save_memory`/`save_skill` 不受影响。回归：既有 41 测试 + 新增 14 测试全绿（`pytest -q backend/tests`）。

## 七、与 OWASP 对齐

- **LLM08 Excessive Agency**：最小功能（allowlist）、最小权限（read/write class）、人审（`requires_approval` 自动拒）、输入校验、限流（配额）、决策与执行分离（`engine` 决策 vs `inner` 执行）、日志。
- **LLM01 Prompt Injection / LLM02 Sensitive Info Disclosure**：参数 allowlist 缩小注入爆炸半径；输出脱敏应对敏感信息外泄。

## 八、诚实标注的边界

- 当前工具全只读；写操作工具的 `requires_approval` 通道已就绪但无真实写工具端到端验证。
- `path_within`/SSRF 类防护已实现，但因无网络类工具，属「就绪未实战」。
- 输出脱敏为保守正则；若接入返回密钥的工具，应按需扩充 `security/scrub.py` 的模式集。
