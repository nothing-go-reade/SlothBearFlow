# SlothBearFlow

SlothBearFlow 是一个本地优先、可生产加固的单 Agent 应用。项目把 Agent 执行、
Tool/Function Call、RAG、MCP Client、分层记忆、评估与可观测性、安全护栏以及
React 管理控制台整合在同一个仓库中。

当前状态：后端与前端均可运行；Multi-Agent 产品能力按当前路线暂缓。

## 核心能力

- Agent：Basic、LangChain Tool Calling、Explicit ReAct 三条执行路径，共用统一结果、
  总 deadline、停止原因、工具轨迹和持久化收尾。
- Tool/Function Call：严格 schema、默认拒绝、参数约束、调用配额、超时、有限重试、
  熔断、幂等键、一次性人工审批和输出脱敏。
- RAG：结构化 Markdown 切分、Embedding、Milvus、向量 + BM25 混合检索、RRF、
  rerank、阈值过滤、上下文预算、ACL、Manifest、引用和 groundedness 校验。
- MCP：Streamable HTTP Client、协议协商、分页发现、租户/用户/作用域隔离缓存、
  SSRF 防护、凭据隔离、绝对 deadline 和工具调用幂等元数据。
- 记忆：Redis 短期窗口、异步摘要、PostgreSQL 持久化与恢复、删除 tombstone、
  PII/密钥脱敏和可选 Background Reflection。
- 可观测性：本地 Trace Store、Agent/RAG/LLM/Memory/PostgreSQL spans、Langfuse
  bridge、自托管 Compose、Prometheus/Grafana 配置和防篡改审计链。
- 工程与安全：HttpOnly Cookie 登录、JWT、RBAC、租户隔离、限流、并发/请求大小
  限制、CORS、本地匿名边界、命令/路径/网络护栏、Alembic、CI 和非 root 镜像。
- 前端：React + Umi + TypeScript 三栏工作台，支持 SSE 聊天、知识入库、记忆删除、
  审批、审计和 Trace 查看，并适配桌面、390px 与 320px。

## 目录

```text
.
|-- backend/
|   |-- config/                    # Tool Guard 策略
|   |-- migrations/                # Alembic migrations
|   |-- scripts/                   # 密码、评估、Milvus 鉴权初始化脚本
|   |-- src/slothbearflow_backend/
|   |   |-- agent/                 # Agent 执行与统一结果
|   |   |-- evaluation/            # 版本化评估数据与评估器
|   |   |-- learning/              # Background Reflection
|   |   |-- mcp/                   # MCP Client/Manager
|   |   |-- memory/                # Redis/摘要/隐私
|   |   |-- observability/         # Trace、Langfuse、metrics facade
|   |   |-- persistence/           # PostgreSQL
|   |   |-- rag/                   # Chunk、Milvus、rerank、citation、ACL
|   |   |-- security/              # Auth、审批、审计和执行护栏
|   |   |-- tools/                 # 内置工具与安全包装
|   |   `-- main.py                # FastAPI 入口
|   `-- tests/
|-- frontend/                      # React + Umi 控制台
|-- docs/                          # 架构、优化、审查和验收文档
|-- .github/workflows/ci.yml
`-- pyproject.toml
```

## 本地启动

### 1. Python 环境

目标运行时为 Python 3.12。

```bash
python3.12 -m venv .venv
./.venv/bin/python -m pip install -r backend/requirements.txt
cp backend/.env.example .env
```

根据本机模型修改 `.env` 中的 `OLLAMA_MODEL`、`OLLAMA_EMBED_MODEL` 等配置。

### 2. 基础依赖

```bash
docker compose -f backend/docker-compose.yml up -d redis postgres etcd minio milvus
```

### 3. 数据库迁移

首次生产式运行建议执行 Alembic；本地环境仍保留兼容性的 runtime schema 初始化。

```bash
./.venv/bin/python -m alembic \
  -c backend/migrations/alembic.ini upgrade head
```

### 4. 后端

```bash
./.venv/bin/python -m uvicorn \
  backend.src.slothbearflow_backend.main:app \
  --host 127.0.0.1 --port 8000 --no-proxy-headers
```

- API 文档：`http://127.0.0.1:8000/docs`
- Liveness：`http://127.0.0.1:8000/health`
- Readiness：`http://127.0.0.1:8000/ready`

### 5. 前端

```bash
cd frontend
pnpm install --frozen-lockfile
pnpm dev --port 8001
```

打开 `http://localhost:8001`。开发代理会把 `/api` 转发到后端 `8000` 端口。

## Docker 运行

仅启动基础设施：

```bash
docker compose -f backend/docker-compose.yml up -d
```

构建并启动完整应用：

```bash
docker compose -f backend/docker-compose.yml --profile app up -d --build
```

生产覆盖文件会强制登录、非默认密码、完整 PostgreSQL DSN、Milvus 鉴权初始化、
LLM 探针、受保护 metrics、严格 CORS 和关闭 API 文档：

```bash
docker compose \
  -f backend/docker-compose.yml \
  -f backend/docker-compose.production.yml \
  --profile app up -d --build
```

生产环境需显式注入 CRUD-only 的 `PRODUCTION_POSTGRES_DSN`、仅供迁移任务使用的
`MIGRATION_POSTGRES_DSN`、`MILVUS_TOKEN`、
`MILVUS_BOOTSTRAP_TOKEN`、`REDIS_PASSWORD`、`AUTH_SECRET`、`AUTH_USERS_JSON`
等私密配置；生产 `REDIS_PASSWORD` 至少 12 个字符。
不要提交真实 `.env`。

## 可观测性

生成本地私密配置后，可启动 Langfuse、Prometheus 与 Grafana：

```bash
./.venv/bin/python backend/scripts/generate_observability_env.py
docker compose \
  --env-file backend/.env.observability \
  -f backend/docker-compose.yml \
  -f backend/docker-compose.observability.yml \
  --profile observability up -d
```

即使未启动 Langfuse，本地 Trace Store 也会保留最近 Trace，可在前端“观测”面板查看。

## 主要 API

| 方法 | 路径 | 作用 |
| --- | --- | --- |
| `GET` | `/health` | 浅 liveness；staging/production 不访问外部依赖 |
| `GET` | `/ready` | 最小化部署 readiness（依赖降级时返回 503） |
| `GET` | `/runtime/status` | 鉴权后的完整依赖、能力和降级状态 |
| `POST` | `/auth/login` | 建立 HttpOnly Cookie 会话 |
| `POST` | `/auth/logout` | 清理服务端登录态 Cookie |
| `POST` | `/chat` | Agent 对话，支持 SSE |
| `POST` | `/ingest` | 创建持久化知识入库任务 |
| `GET` | `/ingest/{job_id}` | 查询入库任务状态 |
| `GET` | `/knowledge/documents` | 查询当前租户可见 Manifest |
| `GET` | `/memory/{session_id}` | 查询会话记忆 |
| `DELETE` | `/memory/{session_id}` | 删除会话并写 tombstone |
| `GET` | `/security/approvals` | 查询工具审批 |
| `POST` | `/security/approvals/{id}` | 同意或拒绝一次性审批 |
| `GET` | `/security/audit` | 查询租户审计事件并验证哈希链 |
| `GET` | `/observability/traces` | 查询本地 Trace |
| `GET` | `/metrics` | Prometheus 指标（可配置 Bearer Token） |

## 质量门禁

```bash
./.venv/bin/python -m pytest -q
python3.12 -m compileall -q backend/src backend/tests backend/migrations backend/scripts
ruff check backend/src backend/tests backend/migrations backend/scripts
cd frontend && pnpm build
```

CI 还会执行 mypy、coverage、Alembic SQL 渲染、Compose 校验和前后端镜像构建。

## 文档

- [全面优化实施与验收报告](docs/SlothBearFlow-全面优化实施与验收报告.md)
- [Agent 全面优化清单](docs/SlothBearFlow-Agent全面优化清单.md)
- [Agent 历史能力基线](docs/SlothBearFlow-Agent能力全景梳理.md)
- [工程化与安全护栏方案](docs/SlothBearFlow-工程化与安全护栏优化方案.md)
- [评估与可观测性方案](docs/SlothBearFlow-评估与可观测性-Langfuse优化方案.md)
- [Multi-Agent 未来方案](docs/SlothBearFlow-Multi-Agent未来优化方案.md)
