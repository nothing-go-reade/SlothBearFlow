# SlothBearFlow — Reflection 反思复盘机制

> 本文描述 SlothBearFlow 已落地的 **Agent Reflection（反思）** 能力：一个 Hermes 风格的「后台复盘学习层」。内容基于 `backend/src/slothbearflow_backend/` 实际源码整理，标注了精确的 `文件:行号`。
>
> 整理日期：2026-07-01
>
> 相关代码：`agent/conversation_loop.py`、`worker/background.py`、`learning/*`、`config.py`、`prompt.py`

---

## 一、这是哪一种 Reflection

学术与工程上，反思分两条方向相反的路子。**本项目实现的是第二种（后台复盘 / 记忆固化型），不是第一种。**

| | 触发时机 | 改什么 | 对当前答案的影响 | 代表 |
|---|---|---|---|---|
| **自我纠错型** | 主循环内、同步 | 改**这一轮**的答案（先自评 → 再改稿 → 才返回给用户） | 直接改写 | Reflexion / Self-Refine |
| **后台复盘型（本项目）** | 回合结束后、**异步** | **不碰当前答案**，只把偏好/经验/技巧蒸馏进 Memory / Skills | 零影响，收益体现在**未来**对话 | **Hermes Background Review** |

一句话定性：

> 主 Agent 完成回答 → 用户**立即**拿到结果 → 后台起一个复盘任务 → 复盘刚才这轮对话 → 若有值得长期记住的偏好/经验/技巧，写入 Memory / Skills → **主会话历史与 prompt 不被污染**。

用户拿到的答案永远不会被复盘改动；复盘只在事后为「以后更懂用户」做沉淀。

---

## 二、完整链路（生产者 → 队列 → 消费者）

反思不是独立进程/线程，而是复用现有的共享 `asyncio.Queue`，新增一个 `"review"` job 类型，与 `ingest` / `summarize` 并列。效果等价 Hermes 的 daemon thread（fire-and-forget、异常吞掉、不阻塞用户响应），但更贴 FastAPI 的异步模型。

```
/chat  ──prepare──▶ iter_stream / run_blocking ──▶ _finalize ──put_nowait──▶
  asyncio.Queue  ──worker_loop(get)──▶  type=="review"  ──to_thread──▶  run_review_job
   (main.py:123)     (background.py:17)     (background.py:60)        (review_agent.py:149)
```

### 生产侧（回合收尾时入队）

| 步骤 | 位置 | 说明 |
|---|---|---|
| HTTP 入口 | `main.py:267` `chat()` | 唯一对话入口，很薄，委托给 `ChatTurnRunner` |
| 编排准备 | `conversation_loop.py:163` `prepare()` | 载会话 → RAG 预检索 → 建 executor |
| 两条输出路径 | `:310` `iter_stream()` / `:238` `run_blocking()` | 流式 / 非流式 |
| **唯一回合收尾钩子** | `:399` `_finalize()` | 落库 + 入队摘要 + **入队复盘**都在此汇聚 |
| 触发门控 | `:477` `_maybe_enqueue_review()` | 间隔 N 轮判断，`turn_no = len(messages)//2`（`:491`） |
| **真正入队** | `:513` `put_nowait({"type":"review", ...})` | `QueueFull`/任何异常一律跳过（`:523-526`），**绝不阻塞用户回合** |

### 队列本体（应用启动时创建）

| 步骤 | 位置 |
|---|---|
| 建队列 | `main.py:123` `asyncio.Queue(maxsize=settings.job_queue_max)` |
| 拉起常驻 worker | `main.py:125` `asyncio.create_task(worker_loop(queue, settings))` |

### 消费侧（后端任务入口）

| 步骤 | 位置 | 说明 |
|---|---|---|
| **队列消费入口** | `worker/background.py:17` `worker_loop()` | `await queue.get()` 常驻协程，按 `type` 分发 |
| review 分支 | `background.py:60-62` | `await asyncio.to_thread(run_review_job, snapshot, settings)` |
| **复盘业务入口** | `learning/review_agent.py:149` `run_review_job()` | review agent 的实际执行体 |

---

## 三、回合快照 TurnSnapshot（复盘的输入）

复盘任务消费的是一份**快照副本**（`learning/snapshot.py`），不持有活会话对象。字段来源都在 `_finalize` 上下文里齐备：

- `session_id` / `user_message` / `final_answer` / `raw_output`
- `tools_used`、`tool_trace`（工具调用轨迹，见下方说明）
- `rag_context`、`citations`（本轮 RAG 检索片段与引用）
- `rolling_summary`（滚动摘要）
- `review_memory` / `review_skills`（本轮该复盘哪些维度，由间隔门控决定）

> **`tool_trace` 的边界**：只有显式 ReAct 路径（`agent/react_runtime.py`）会产出工具调用轨迹；默认 LangChain `AgentExecutor` 路径**未接** `return_intermediate_steps`，该路径下 trace 为空。当前默认执行器是 `BasicChatExecutor`（无工具），trace 也为空——复盘仍可基于「问题 + 回答 + RAG 片段」正常工作。此项受 `REVIEW_TOOL_TRACE` 开关控制，默认关。

---

## 四、Review Agent 的双路径写入（按模型能力自动选）

复盘产出统一为 `MemoryItem` / `SkillItem`（`learning/schema.py`），但**写入机制按模型能力分两条路**，最终都落到同一个 `LearningStore`（唯一写入面）：

```
llm_supports_tools(settings) and not review_force_structured
        │
        ├── True  ── 路径 A：工具调用（Hermes 式）  review_agent.py:111 _run_tool_path
        │             review agent 调 save_memory / save_skill 工具
        │             经线程级白名单只放行这两个工具，其余 deny
        │
        └── False ── 路径 B：结构化输出（弱模型兼容） review_agent.py:83 _run_structured_path
                      with_structured_output(ReviewResult) 产 JSON，代码经 store 落盘
```

- **路径 A（支持工具的模型，如 qwen2.5 / llama3.1 / OpenAI）**：构建 `save_memory` / `save_skill` 写工具（`learning/learning_tools.py`），复用 `ExplicitReActRuntime` 跑有界循环让 review agent 自行决定调用。隔离靠**线程级白名单**（`learning/review_guard.py` 的 `set_thread_tool_whitelist`）：`ExplicitReActRuntime._invoke_tool` 执行前查白名单，白名单外的工具直接返回 deny 文案（对标 Hermes `set_thread_tool_whitelist` + 危险命令 auto-deny）。白名单据本轮维度动态收窄：只复盘 memory 时仅放行 `save_memory`，反之亦然（`review_agent.py:121-131`）。
- **路径 B（不支持工具的模型，当前默认 `deepseek-r1:7b`）**：`get_chat_llm(...).with_structured_output(ReviewResult)` 产 JSON，代码经 store 落盘。review LLM 仅产 JSON，结构上无法执行任何动作。

> **验证边界**：路径 B 已用 `deepseek-r1:7b` 端到端验证；路径 A 目前仅有单元测试（白名单 deny + tool_trace 记录），尚未用真实工具模型端到端跑过。

### 三套复盘 Prompt（按维度选择，`review_agent.py:33-70`）

| 维度 | Prompt | 关注点 |
|---|---|---|
| 仅 Memory | `_MEMORY_REVIEW_PROMPT` | 用户是谁、身份/角色/背景、稳定偏好（语言/风格/详略/格式）、长期约定 |
| 仅 Skills | `_SKILL_REVIEW_PROMPT` | 这类任务以后怎么做、**用户纠正过的格式/语气/流程（一等 skill 信号）**、可复用技巧 |
| 两者皆中 | `_COMBINED_REVIEW_PROMPT` | 合并复盘；「用户长期关系」入 memory，「某类任务的方法」入 skill |

所有 prompt 共享 `_BASE`：保守提炼，只存稳定/可复用/对未来有帮助的内容；命名用 kebab-case slug；同一主题复用同名以覆盖更新，不制造重复。

---

## 五、存储：Markdown 真相源 + SQLite 派生索引

### 唯一写入面 `LearningStore`（`learning/store.py`）

review 的所有写操作只经此类，落在 `base_dir/{memory,skills}/*.md`——这就是 Hermes「memory/skills 白名单」在本项目的映射：**除 store API 外无任何写操作，结构上不可能碰系统其他部分。**

- 文件格式对齐项目自动记忆惯例：frontmatter（memory 用 `name/description/metadata.type`，skill 用 `name/trigger`）+ 正文。
- **路径穿越防护**：`_safe_slug()` 只保留 `[a-z0-9-]`（`store.py:17,41`），`_resolve_within()` 用 `is_relative_to` 校验解析后路径不越界（`store.py:46`），越界拒写。
- **去重**：按 `(kind, name)` slug 覆盖式 upsert，同一主题不反复堆文件。
- **落盘条数**：`save_many(..., max_items)` 截断（`store.py:122`），默认单次最多 5 条。

### 派生索引 `LearningIndex`（`learning/index_db.py`）

**纯索引，Markdown 才是真相源。** 用标准库 `sqlite3`（零新依赖），DB 丢失不丢数据、可由扫描 `.md` 完全重建（`reindex_from_disk`）。

- 检索用 **Python 关键词重叠打分**（含 CJK 分词），而非 FTS5——为可移植性放弃了 FTS5 依赖。
- 索引读写异常一律吞掉降级（复盘/读回是尽力而为，绝不影响主链路）。
- `index.sqlite` 落在 `agent_learning/` 学习命名空间，**与会话存储（Redis / PG）物理隔离**——对标 Hermes `skip_memory`，不污染用户会话记忆命名空间。

---

## 六、闭环的两半：写侧 + 读侧（都默认关）

⚠️ **这是理解本机制最关键的一点。** 反思闭环由两个**独立开关**控制，缺一不成闭环：

| | 开关 | 作用 | 缺失后果 |
|---|---|---|---|
| **写侧** | `ENABLE_BACKGROUND_REVIEW` | 复盘 → 提取 → 落盘 memory/skills | 关：完全不反思 |
| **读侧** | `INJECT_LEARNING_INTO_PROMPT` | 把学到的东西读回、注入未来 system prompt | 关：只写不读，文件在长但行为不变 |

- **开箱默认（两个都 false）**：agent 不反思，行为与未引入本层完全一致（零行为变更、opt-in）。
- **只开写侧**：memory/skills 文件持续增长，但从不回灌进 prompt → 反思**空转**，「越用越懂用户」不成立。
- **写侧 + 读侧都开**：才形成完整闭环。

### 读回通道（读侧唯一入口）

`ChatTurnRunner.prepare` → `_build_learning_context()`（`conversation_loop.py:453`）→ `LearningStore.select_for_injection(user_message, budget)`（`store.py:144`）：按相关度选相关 memory/skills → 拼成有界「长期记忆/技巧」块 → 经 `prompt.py build_system_prompt(learning_context=...)` 注入 system prompt。

> **prefix cache 取舍**：注入会跨轮改变 system prompt（影响 prefix cache）。设计上注入块**剔除易变字段**（mtime/路径），仅用稳定内容，以在学习集不变时保持 system prompt 前缀字节稳定；且受 `LEARNING_PROMPT_BUDGET_CHARS` 限长。本地 Ollama 无 prefix cache 收益，此项主要为接入 OpenAI 等云端模型时的成本考量。

---

## 七、配置项（`config.py:165-191`，默认全关）

| Env Var | 默认 | 说明 |
|---|---|---|
| `ENABLE_BACKGROUND_REVIEW` | `false` | **写侧总开关**（opt-in） |
| `REVIEW_MEMORY_INTERVAL` | `3` | 每 N 轮复盘 memory |
| `REVIEW_SKILLS_INTERVAL` | `5` | 每 N 轮复盘 skills |
| `REVIEW_BASE_DIR` | `agent_learning` | 学习文件根目录 |
| `REVIEW_MAX_ITEMS` | `5` | 单次复盘最多落盘条数 |
| `REVIEW_MODEL` | `` | 可选复盘专用模型；空则复用主 LLM |
| `REVIEW_FORCE_STRUCTURED` | `false` | 强制走结构化路径（即便模型支持工具），便于调试/兼容 |
| `REVIEW_TOOL_TRACE` | `false` | 是否捕获工具调用轨迹进 snapshot |
| `INJECT_LEARNING_INTO_PROMPT` | `false` | **读侧总开关**：是否把学习读回注入 system prompt |
| `LEARNING_PROMPT_BUDGET_CHARS` | `1200` | 读回注入字符上限 |

---

## 八、隔离保证（逐条对账 Hermes）

- review 消费 snapshot **副本**，不持有活会话对象。
- 只经 `LearningStore` 写 `agent_learning/{memory,skills}` 下文件 + `index.sqlite`，带路径穿越防护。
- **绝不**写 Redis 会话 messages、**绝不**写 `agent_chat_turns`、**绝不** append 对话历史、**绝不**改主回合缓存 prompt；学习命名空间与会话命名空间物理隔离。
- 跑在 worker 任务上、**响应送达之后**执行 → 不与用户回合争抢模型。
- 路径 A：线程级白名单只放行 `save_memory`/`save_skill`，其余 deny；路径 B：LLM 仅产 JSON，结构上不能执行动作。
- 读回是 learning→主回合的**唯一**通道，有界 + 开关控制。

---

## 九、与 Hermes 原文的取舍差异（诚实标注）

| 维度 | Hermes 原文 | 本项目 | 原因 |
|---|---|---|---|
| 并发载体 | daemon thread | 复用 `asyncio.Queue` + `"review"` job | 更贴 FastAPI 异步；语义等价（fire-and-forget） |
| 危险命令拦截 | 独立 `auto-deny` 回调 | 线程级白名单（只放行两个写工具） | 本项目无命令审批系统，用白名单覆盖同一诉求 |
| system prompt | review 继承主回合 `_cached_system_prompt`（保 prefix cache） | review 用**自己的**复盘 prompt | 本地 Ollama 无 prefix cache 收益，判定不值得复刻 |
| 索引全文检索 | — | Python 关键词打分（非 FTS5） | 换取零依赖与可移植性 |
| 工具路径验证 | — | 仅单测，未真机验证 | 待接入真实工具模型后补端到端验证 |

---

## 附：一句话总结

SlothBearFlow 现具备 **Hermes 后台复盘这一支的 Reflection**——事后异步蒸馏本轮对话为长期 Memory / Skills、不污染当前回合；双路径写入（工具调用 / 结构化输出）自动适配模型能力；Markdown 为真相源、SQLite 为可重建的派生索引。它是 **opt-in** 的，且需**写侧（`ENABLE_BACKGROUND_REVIEW`）+ 读侧（`INJECT_LEARNING_INTO_PROMPT`）两个开关都开**，反思才真正闭环生效。
