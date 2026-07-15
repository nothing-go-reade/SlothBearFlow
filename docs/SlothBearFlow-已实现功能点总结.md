# SlothBearFlow 已实现功能点总结

> 本文基于对后端全部源码（`backend/src/slothbearflow_backend/`，约 2900 行）、前端控制台、测试、编排与配置文件的实际通读整理，描述的是**当前代码已落地**的能力，而非规划或文档声明。
>
> 整理日期：2026-06-30

---

## 一、整体形态

**SlothBearFlow** 是一个企业级 LangChain Agent 服务，采用 monorepo 结构：

- `backend/` — FastAPI + LangChain，**完整可运行**
- `frontend/` — Umi + React + TypeScript 单页控制台
- `docs/` — 5 篇中文文档（能力梳理、运行优化记录、知识库种子/问答卡片、代码审查）

服务入口：

```bash
uvicorn backend.src.slothbearflow_backend.main:app --host 0.0.0.0 --port 8000
```

---

## 二、API 层（`main.py`）

| 端点 | 功能 |
|------|------|
| `GET /` | 服务自描述 |
| `GET /health` | 聚合健康检查：Redis / Milvus / 会话存储 / Postgres / LLM / Embedding 状态 |
| `POST /chat` | 统一对话入口（核心） |
| `POST /ingest` | 异步入库（投递到后台队列，返回 job_id） |
| `GET /favicon.ico` | 返回 204 |

附带能力：

- **CORS** 放行前端 `127.0.0.1:5173`
- **滚动文件日志**：app / access / error 三套 `RotatingFileHandler`
- **lifespan 生命周期**：启动时建 Postgres 表 + 拉起后台 worker，关闭时优雅取消
- `/chat` 全链路埋了**耗时日志**（每阶段 `perf_counter`）

---

## 三、对话主流程（`/chat` 已实现的编排）

1. 重置 RAG 引用上下文（contextvars）
2. 从 Redis 载入会话（messages + summary），Redis 未命中可**从 Postgres 回填**
3. 短期记忆**窗口裁剪**（最近 N 轮）
4. **RAG 预检索**：无论模型是否支持工具，都先做一次向量 + 关键词混合检索，把片段注入 LLM 输入（解决“无工具模型也能用知识库”）
5. 构建执行器（三条路径，见第四节）
6. **流式 / 非流式**分流输出
7. 可选**二段式结构化输出**
8. 回写 Redis + **持久化到 Postgres**（chat turn / stream events）
9. 异步入队**摘要更新**任务

---

## 四、Agent 执行层（`agent/`）— 三条可切换路径

`build_agent_executor` 按「模型是否支持工具 + 开关」选择：

1. **无工具模型** → `BasicChatExecutor`（`prompt | llm`，支持 invoke / stream）
2. **支持工具 + 显式 ReAct 开关开**（`ENABLE_EXPLICIT_REACT_RUNTIME=true`）→ `ExplicitReActRuntime`：自研**有界 ReAct 循环**，`bind_tools` → 解析 tool_calls → 执行 → ToolMessage 回填 observation，带 `max_steps` 上限与 `stop_reason`（final_answer / max_steps）
3. **支持工具（默认）** → LangChain `create_tool_calling_agent` + `AgentExecutor`（max_iterations=4，early_stopping=generate，容错解析）

> 显式 ReAct 是 feature-flag 灰度能力，默认仍走 LangChain，响应结构保持一致。

---

## 五、LLM 层（`llm.py`）

- **多 provider**：`ollama`（默认 `deepseek-r1:7b`）/ `openai`（默认 `gpt-4o-mini`），懒加载导入
- **工具能力探测**：按 provider 默认值，可被 `LLM_SUPPORTS_TOOLS` 覆盖
- 可配参数：temperature / top_p / max_tokens / **reasoning_effort**（`deep_think=true` → `high` 映射）/ `model_kwargs` / `extra_body`（OpenAI 兼容自定义端点）

---

## 六、工具层（`tools/`）

| 工具 | 状态 |
|------|------|
| `get_current_time` | 已实现（本地时区时间） |
| `get_weather` | **离线样例**数据（6 个城市，预留接真实 API） |
| `get_session_context` | 返回最近 6 条会话摘要 |
| `search_knowledge` | RAG 检索，仅在向量库可用时挂载 |

注册器按 `USE_RAG / SKIP_MILVUS / 向量库可用性`**条件装配**工具集。

---

## 七、RAG 引擎（`rag/`）— 已实现混合检索

- **自研 `SimpleMilvusVectorStore`**（直接用 pymilvus `MilvusClient`）：建集合（AUTOINDEX + COSINE）、`add_documents`、`similarity_search`（向量 ANN）、`keyword_search`
- **自研 BM25**：含 **CJK 二元组分词**，k1=1.5 / b=0.75
- **混合召回 pipeline**（`rag_tool.py`）：向量 + 关键词候选 → 去重 → **词法重排**（对已知知识库文档做来源加权，对噪声源降权）→ 取 Top-N 注入
- **Embedding** 多 provider（ollama `nomic-embed-text` / openai `text-embedding-3-small`）
- 文本切分：`RecursiveCharacterTextSplitter`（chunk 500 / overlap 100）
- **优雅降级**：Milvus 初始化失败 → 缓存错误、RAG 自动关闭、不阻塞主流程
- **引用追踪**：contextvars 收集，预检索引用 + 工具引用**合并去重**，流式 / 非流式统一

---

## 八、记忆与持久化

### 记忆（`memory/`）

- **短期**：窗口裁剪（最近 N 轮）
- **会话**：Redis JSON 存储（TTL 7 天），**Redis 不可用自动降级到进程内 `InMemoryRedis`**
- **滚动摘要**：后台异步用 LLM 把最近 20 条压成 ≤120 字中文摘要

### PostgreSQL 持久化（`persistence/postgres.py`，可选开关）

- 4 张表：`agent_sessions` / `agent_chat_turns` / `agent_chat_stream_events` / `agent_ingest_jobs`
- 落库内容：对话轮次（含 tools_used、citations、response_mode、stream_format）、摘要、入库任务状态、流式事件
- **Redis-miss → 从 PG 快照恢复会话**并回写 Redis（可配 turn limit / TTL）
- 自动建表 + `ADD COLUMN IF NOT EXISTS` 轻量迁移；缺 psycopg / DSN / 开关时静默禁用

### 后台 Worker（`worker/background.py`）

- `asyncio.Queue` 驱动，处理 `ingest` 与 `summarize` 两类任务，任务状态落 PG

---

## 九、输出与配置

- **结构化输出**（`output_schema.py`）：`ChatOutput{answer, source, citations[], tools_used[]}`；**二段式**——自由文本再过一次 `with_structured_output` 约束，失败回退原文
- **流式输出**：支持 `plain` 与 `sse` 两种格式，SSE 含 start / chunk / done 事件
- **配置**（`config.py`）：pydantic-settings，**分层加载** `.env / backend/.env / *.local / *.private`；覆盖 LLM、RAG、输出模式、ReAct、Postgres、Redis、Milvus、日志等数十个开关

---

## 十、前端控制台（`frontend/src/pages/index.tsx`）

单页工作台，已实现：

- **健康仪表盘**（LLM / Redis / Milvus / Postgres 四卡片，30s 自动刷新）
- **对话 Playground**：流式渲染（SSE + 纯文本两种解析）、引用片段展示、tools_used、Stop 中止、Clear、prompt 快捷词
- **Session ID** 管理 / 新建
- **知识入库面板**（调 `/ingest`）
- **运行事件 trace** 日志面板

---

## 十一、测试与部署

- `tests/test_smoke.py`：**约 40 个测试**，覆盖 health、配置、执行器选择、ReAct 有界停止、LLM provider 解析、流式、内存会话、PG 持久化、ingest、RAG 引用、BM25 排序、PG 回填恢复
- `docker-compose.yml`：Redis + Postgres + Milvus standalone（etcd + minio + milvus）一键起
- `requirements.txt`：锁定 LangChain 0.3.x 系
- `local_run.py`：本地 LLM 探针

---

## 十二、已标注但尚未实现的方向（代码内 TODO）

代码里明确留了几处待办，说明这些**目前还未做**：

- **提示词三层抽象**（系统基线 → 项目 → 会话策略，首轮引导生成 `session_policy`）
- **存储统一接口**（PG + Redis + Milvus 抽象顶层接口）
- **LLM / Prompt / Tool 顶层可插拔接口**、工具配置化注册
- **检索增强**：向量 ANN + BM25 + **Rerank 重排序**（目前只到词法重排，缺独立 rerank 模型）
- **ReAct 行为约束细化**（思考长度 / 调用次数策略，可观测 thinking / act / observe 阶段）

---

## 附：核心模块速查

| 模块 | 路径 | 职责 |
|------|------|------|
| API 入口 | `main.py` | 路由、CORS、日志、生命周期、对话编排 |
| 配置 | `config.py` | 分层 env 加载、全部开关 |
| 依赖装配 | `deps.py` | Redis 客户端 + 内存降级 |
| LLM | `llm.py` | 多 provider 模型构建 |
| 提示词 | `prompt.py` | system prompt、agent / basic prompt |
| Agent 执行 | `agent/agent_executor.py` | 三路径执行器选择 |
| 显式 ReAct | `agent/react_runtime.py` | 有界 ReAct 循环 |
| 工具 | `tools/` | time / weather / session / rag |
| RAG 存储 | `rag/milvus_store.py` | Milvus 封装 + BM25 |
| RAG 检索 | `tools/rag_tool.py` | 混合召回 + 重排 + 引用 |
| Embedding | `rag/embedding.py` | 多 provider 向量化 |
| 记忆 | `memory/` | 短期 / 会话 / 摘要 |
| 持久化 | `persistence/postgres.py` | PG 四表落库与恢复 |
| 后台任务 | `worker/background.py` | ingest / summarize |
| 结构化输出 | `output_schema.py` / `output_parser.py` | ChatOutput + 二段式 |
| 前端 | `frontend/src/pages/index.tsx` | 控制台 UI |
