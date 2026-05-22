# SlothBearFlow 项目知识库种子数据

更新时间：2026-05-23

本文档用于初始化 SlothBearFlow 的本地 RAG 知识库。内容面向向量检索和 Agent 问答，覆盖项目定位、架构、运行方式、API、数据引擎、前端页面和常见问题。

## 1. 项目定位

SlothBearFlow 是一个本地优先的 AI Agent 服务脚手架，用于把本地大模型、工具调用、短期会话记忆、元数据持久化和 RAG 知识检索整合成一个可运行的开发工作台。

当前项目由后端和前端两部分组成：

- 后端使用 FastAPI 提供 `/chat`、`/ingest`、`/health` 等接口。
- 前端使用 React + Umi + TypeScript 提供本地 Agent Playground。
- 本地模型通过 Ollama 调用，当前对话模型为 `deepseek-r1:7b`。
- 本地 embedding 通过 Ollama 的 `nomic-embed-text` 模型生成。
- Redis 负责短期会话状态。
- PostgreSQL 负责元数据和会话持久化。
- Milvus 负责向量数据存储和相似度检索。

## 2. 核心架构

SlothBearFlow 的主要链路如下：

1. 用户在前端页面输入问题。
2. 前端请求 FastAPI 后端 `/chat` 接口。
3. 后端加载当前 session 的 Redis 会话记忆。
4. 如果启用 RAG，Agent 可通过 `search_knowledge` 工具查询 Milvus 中的项目知识。
5. 后端调用 Ollama LLM 生成回答。
6. 回答、流式事件和会话元数据会按配置写入 PostgreSQL。
7. Redis 保存最近会话窗口，便于下一轮快速恢复上下文。

主要代码目录：

- `backend/src/slothbearflow_backend/main.py`：FastAPI 入口，注册 `/health`、`/ingest`、`/chat`。
- `backend/src/slothbearflow_backend/config.py`：Pydantic Settings 配置入口。
- `backend/src/slothbearflow_backend/agent/`：AgentExecutor 和显式 ReAct runtime。
- `backend/src/slothbearflow_backend/memory/`：Redis 记忆、短期记忆、摘要记忆。
- `backend/src/slothbearflow_backend/persistence/postgres.py`：PostgreSQL 元数据持久化。
- `backend/src/slothbearflow_backend/rag/`：Milvus 向量库、embedding、文本切块和 ingest。
- `backend/src/slothbearflow_backend/tools/`：Agent 内置工具。
- `backend/src/slothbearflow_backend/worker/background.py`：后台任务队列，处理 ingest 和摘要任务。
- `frontend/src/pages/index.tsx`：前端 Agent 控制台页面。
- `frontend/config/config.ts`：Umi 配置和 API 代理。

## 3. 本地依赖组件

本地依赖通过 `backend/docker-compose.yml` 编排：

- Redis：端口 `6379`，负责短期会话记忆。
- PostgreSQL：端口 `5432`，数据库名 `slothbearflow`，默认用户和密码均为 `postgres`。
- Milvus Standalone：端口 `19530`，负责向量库。
- etcd：Milvus 的元数据依赖。
- MinIO：Milvus 的对象存储依赖，端口 `9000` 和 `9001`。

启动命令：

```bash
docker compose -f backend/docker-compose.yml up -d
```

健康检查可访问：

```text
http://127.0.0.1:8000/health
```

当前健康状态中如果出现以下字段，说明三组件集成正常：

- `redis.ok=true`
- `session_store.backend=redis`
- `milvus.enabled=true`
- `milvus.collection=chat_knowledge`
- `postgres_persistence.enabled=true`
- `postgres_persistence.ready=true`
- `embedding.provider=ollama`
- `embedding.model=nomic-embed-text`

## 4. 记忆、元数据和向量数据的分工

Redis 是记忆引擎：

- 保存会话最近消息窗口。
- 用于快速读取当前 session 的上下文。
- Redis 不适合保存长期、结构化审计数据。

PostgreSQL 是元数据引擎：

- 保存会话信息。
- 保存对话轮次。
- 保存流式输出事件。
- 保存 `/ingest` 知识写入任务的 job 状态。
- Redis miss 时可以按配置从 PostgreSQL 恢复最近轮次。

Milvus 是向量数据引擎：

- 保存 RAG 文档切块后的 embedding 向量。
- 默认集合为 `chat_knowledge`。
- 文档通过 `/ingest` 进入后台任务队列。
- 后台 worker 会将文本切成 chunk，再调用 embedding 模型并写入 Milvus。
- Agent 查询知识时会通过 `search_knowledge` 工具执行相似度检索。

## 5. API 说明

`GET /health`

- 用于检查 Redis、Milvus、PostgreSQL、LLM、embedding 的状态。
- 前端页面会使用该接口展示服务健康状态。

`POST /ingest`

- 用于把纯文本写入知识库。
- 请求体包含 `source` 和 `text`。
- 接口返回 `accepted=true` 和 `job_id`。
- 实际写入由后台 worker 异步完成。

示例：

```json
{
  "source": "docs/project-overview.md",
  "text": "SlothBearFlow 是一个本地 AI Agent 服务..."
}
```

`POST /chat`

- 统一聊天入口。
- 请求体包含 `session_id` 和 `message`。
- 支持普通输出、结构化输出和流式输出。
- 当 RAG 可用时，Agent 可检索 Milvus 中的项目知识。

示例：

```json
{
  "session_id": "local-dev",
  "message": "SlothBearFlow 的 Redis、Milvus、Postgres 分别负责什么？"
}
```

## 6. 前端页面

前端目录为 `frontend/`，技术栈为 React + Umi + TypeScript。

启动命令：

```bash
cd frontend
pnpm dev
```

或指定端口：

```bash
PORT=5173 ./node_modules/.bin/umi dev --host 127.0.0.1
```

访问地址：

```text
http://127.0.0.1:5173/
```

Umi 代理配置：

- `/api/*` 代理到 `http://127.0.0.1:8000`
- `/openapi.json` 代理到 `http://127.0.0.1:8000/openapi.json`

前端页面能力：

- 展示后端健康状态。
- 展示 LLM、Redis、Milvus、PostgreSQL 状态。
- 提供 Agent Playground，调用 `/api/chat`。
- 提供 Knowledge source 输入区，调用 `/api/ingest` 写入向量库。
- 提供 API Docs 入口，打开后端 Swagger 文档。

## 7. 配置说明

配置文件按顺序加载：

```text
.env
backend/.env
.env.local
backend/.env.local
.env.private
backend/.env.private
```

常用配置：

- `LLM_PROVIDER=ollama`
- `OLLAMA_MODEL=deepseek-r1:7b`
- `OLLAMA_BASE_URL=http://127.0.0.1:11434`
- `OLLAMA_EMBED_MODEL=nomic-embed-text`
- `USE_RAG=true`
- `SKIP_MILVUS=false`
- `MILVUS_URI=http://127.0.0.1:19530`
- `MILVUS_COLLECTION=chat_knowledge`
- `ENABLE_POSTGRES_PERSISTENCE=true`
- `POSTGRES_RESTORE_ON_REDIS_MISS=true`
- `REDIS_HOST=127.0.0.1`
- `REDIS_PORT=6379`

## 8. 启动顺序

推荐本地完整启动顺序：

1. 启动 Docker Desktop。
2. 启动 Redis、Milvus、PostgreSQL：

```bash
docker compose -f backend/docker-compose.yml up -d
```

3. 确认 Ollama 已启动，并已拉取模型：

```bash
ollama pull deepseek-r1:7b
ollama pull nomic-embed-text
```

4. 启动后端：

```bash
./.venv/bin/python -m uvicorn backend.src.slothbearflow_backend.main:app --host 127.0.0.1 --port 8000
```

5. 启动前端：

```bash
cd frontend
PORT=5173 ./node_modules/.bin/umi dev --host 127.0.0.1
```

6. 打开前端：

```text
http://127.0.0.1:5173/
```

## 9. 常见问题

如果前端打不开：

- 确认 `5173` 端口是否有 Umi dev server。
- 使用 `lsof -nP -iTCP:5173 -sTCP:LISTEN` 检查。
- 前端地址应为 `http://127.0.0.1:5173/`，不是后端 `/health` 地址。

如果 API 文档显示 `Failed to load API definition`：

- 确认后端 `/openapi.json` 能访问。
- 确认 Umi 已代理 `/openapi.json` 到后端。
- 直接访问 `http://127.0.0.1:8000/docs` 也可以查看后端 Swagger。

如果 RAG 不生效：

- 检查 `.env` 中 `USE_RAG=true`。
- 检查 `.env` 中 `SKIP_MILVUS=false`。
- 检查 Milvus 容器是否运行。
- 检查 `/health` 中 `milvus.enabled=true`。
- 检查 Ollama embedding 模型 `nomic-embed-text` 是否可用。

如果 Redis 记忆不生效：

- 检查 Redis 容器是否运行。
- 检查 `/health` 中 `redis.ok=true`。
- 检查 `session_store.backend=redis`。

如果 PostgreSQL 持久化不生效：

- 检查 `.env` 中 `ENABLE_POSTGRES_PERSISTENCE=true`。
- 检查 PostgreSQL 容器是否运行。
- 检查 `/health` 中 `postgres_persistence.ready=true`。

## 10. 已完成的本地优化

本项目当前已经完成以下优化：

- 修复后端工具函数 docstring，保证 LangChain tool 正常注册。
- 扩展私有环境配置加载路径，支持 `.env.local` 和 `.env.private`。
- 新增 React + Umi + TypeScript 前端控制台。
- 优化 API 文档入口，修复 `/openapi.json` 代理。
- 优化前端视觉样式，采用简洁的本地云控制台风格。
- 安装并调试 Docker Desktop。
- 通过 Docker Compose 启动 Redis、Milvus、PostgreSQL。
- 拉取并验证 Ollama embedding 模型。
- 修复 Milvus 写入适配，使用 `pymilvus.MilvusClient` 实现简单向量存储。
- 修复 PostgreSQL 会话恢复 SQL。
- 修复 uvicorn access log formatter。
- 验证 Redis、Milvus、PostgreSQL 三组件联通。
- 创建完整本地运行与三组件集成优化记录文档。

## 11. 适合向量库召回的问题

写入本知识库后，Agent 应能回答以下问题：

- SlothBearFlow 是什么？
- SlothBearFlow 的前后端分别用什么技术？
- 本地怎么启动 SlothBearFlow？
- Redis、Milvus、PostgreSQL 分别负责什么？
- `/ingest` 是怎么把文档写入向量库的？
- 为什么需要 Ollama 的 `nomic-embed-text`？
- 前端页面打不开应该怎么检查？
- API 文档打不开应该怎么修？
- 如何确认 RAG 已启用？
- 项目当前完成过哪些优化？
