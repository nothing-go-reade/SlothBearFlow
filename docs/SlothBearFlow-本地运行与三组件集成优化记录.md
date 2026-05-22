# SlothBearFlow 本地运行与三组件集成优化记录

更新时间：2026-05-22 23:47:55 CST

## 1. 背景与目标

本轮优化从“项目之前可以跑通，现在需要先跑起来”开始，逐步扩展到前端页面、API 文档、界面体验、本地依赖服务和数据引擎集成。

最终目标是让 SlothBearFlow 在本地开发环境中完整跑通，并保证以下能力可用：

- 记忆引擎：Redis 负责会话记忆缓存。
- 元数据引擎：PostgreSQL 负责会话、对话轮次、流式事件、知识库写入任务等元数据持久化。
- 向量数据引擎：Milvus 负责 RAG 知识向量存储和相似度检索。
- LLM/Embedding：本地 Ollama 提供对话模型和 embedding 模型。
- 前端：React + Umi + TypeScript 提供本地 Agent Playground 页面。

## 2. 项目现状梳理

项目主要由后端和前端两部分组成：

- 后端：FastAPI，入口为 `backend/src/slothbearflow_backend/main.py`。
- 配置：Pydantic Settings，入口为 `backend/src/slothbearflow_backend/config.py`。
- 会话记忆：`backend/src/slothbearflow_backend/memory/redis_memory.py`。
- 元数据持久化：`backend/src/slothbearflow_backend/persistence/postgres.py`。
- RAG/向量库：`backend/src/slothbearflow_backend/rag/`。
- 本地依赖编排：`backend/docker-compose.yml`。
- 前端：`frontend/`，使用 React + Umi + TypeScript。

## 3. 已完成优化内容

### 3.1 后端启动修复

修复了 LangChain `@tool` 对函数描述的要求。部分工具函数缺少 docstring，会导致工具注册或运行阶段出错。

涉及文件：

- `backend/src/slothbearflow_backend/tools/time_tool.py`
- `backend/src/slothbearflow_backend/tools/weather_tool.py`
- `backend/src/slothbearflow_backend/tools/session_tool.py`
- `backend/src/slothbearflow_backend/tools/rag_tool.py`

修复后，后端测试通过：

```bash
./.venv/bin/python -m pytest -q backend/tests
```

结果：

```text
25 passed
```

### 3.2 私有环境配置加载优化

用户说明项目之前是 source 到私有 env 环境后启动。为兼容本地私有配置，扩展了配置加载路径。

涉及文件：

- `backend/src/slothbearflow_backend/config.py`

现在会按顺序读取：

```text
.env
backend/.env
.env.local
backend/.env.local
.env.private
backend/.env.private
```

这样既能保留默认 `.env`，也能通过 `.env.local` 或 `.env.private` 做本机私有覆盖。

### 3.3 前端从占位页面升级为 Agent Playground

根据“类似 deer-flow 的页面，可以前端展示调用”的需求，前端改造为 React + Umi + TypeScript。

涉及文件：

- `frontend/package.json`
- `frontend/config/config.ts`
- `frontend/src/pages/index.tsx`
- `frontend/src/pages/index.css`
- `frontend/tsconfig.json`

主要能力：

- 展示后端健康状态。
- 创建/维护 session。
- 调用 `/chat` 进行对话。
- 调用 `/ingest` 写入知识文本。
- 展示运行事件和基础状态。

### 3.4 API 文档代理修复

用户截图中 API 文档报错：

```text
Failed to load API definition.
Fetch error
Not Found /openapi.json
```

原因是 FastAPI docs 页面通过前端代理访问时，Swagger UI 默认请求 `/openapi.json`，但 Umi 前端没有把该路径代理到后端。

修复点：

- `frontend/config/config.ts`

现在代理规则包含：

- `/api` -> `http://127.0.0.1:8000`
- `/openapi.json` -> `http://127.0.0.1:8000`

因此前端侧访问 API 文档时可以正确加载 OpenAPI schema。

### 3.5 前端视觉样式优化

根据“页面有点复古，希望简洁、更好看、带渐变、参考开源作品和云厂商风格”的反馈，对页面做了云控制台风格优化。

方向：

- 从复古面板改为浅色控制台风格。
- 使用克制的蓝色、青绿色渐变。
- 强化状态栏、运行指标、卡片层级和留白。
- 页面文案改为更偏产品化的 Agent Playground。

涉及文件：

- `frontend/src/pages/index.tsx`
- `frontend/src/pages/index.css`

### 3.6 Docker Desktop 安装

用户明确要求安装 Docker Desktop，而不是 Colima + Docker CLI。

处理过程：

1. 终止已开始的 `brew install colima docker docker-compose`。
2. 尝试 `brew install --cask docker`，但 Homebrew cask 元数据请求卡住。
3. 改为下载 Docker 官方 Apple Silicon DMG。
4. 校验 SHA256，确认与 Homebrew cask 中的官方 hash 一致。
5. 安装到 `/Applications/Docker.app`。
6. 启动 Docker Desktop。
7. 为当前 shell 创建 `docker` 命令入口：

```bash
/opt/homebrew/bin/docker -> /Applications/Docker.app/Contents/Resources/bin/docker
```

当前版本：

```text
Docker version 29.4.3
Docker Compose version v5.1.4
```

### 3.7 Docker registry mirror 配置

Docker Desktop 安装完成后，拉镜像时发现 Docker Hub `registry-1.docker.io` 直连超时。

处理：

- 检测多个 registry mirror。
- 选择可用镜像源。
- 写入 `~/.docker/daemon.json`。
- 重启 Docker Desktop。

当前 mirror：

```json
{
  "registry-mirrors": [
    "https://docker.m.daocloud.io",
    "https://docker.1ms.run"
  ]
}
```

### 3.8 Redis、PostgreSQL、Milvus 本地服务启动

使用项目已有 `backend/docker-compose.yml` 启动本地依赖服务。

启动命令：

```bash
docker compose -f backend/docker-compose.yml up -d redis postgres etcd minio milvus
```

服务说明：

- Redis：会话记忆缓存，端口 `6379`。
- PostgreSQL：元数据持久化，端口 `5432`。
- Milvus：向量数据引擎，端口 `19530`。
- etcd：Milvus 依赖。
- MinIO：Milvus 依赖。

当前容器状态：

```text
backend-redis-1      Up
backend-postgres-1   Up (healthy)
backend-milvus-1     Up
backend-etcd-1       Up
backend-minio-1      Up
```

注意：当前 Milvus 镜像 `milvusdb/milvus:v2.4.4` 为 `linux/amd64`，在 Apple Silicon 上由 Docker Desktop 兼容运行。功能可用，但性能可能低于原生 arm64 镜像。

### 3.9 Ollama embedding 模型安装

项目默认使用 Ollama 作为 embedding provider 时，需要 `nomic-embed-text`。

已执行：

```bash
ollama pull nomic-embed-text
```

当前 Ollama 模型：

```text
nomic-embed-text:latest
deepseek-r1:7b
```

### 3.10 项目 `.env` 接入三组件

涉及文件：

- `.env`

当前关键配置：

```env
REDIS_HOST=127.0.0.1
REDIS_PORT=6379
REDIS_DB=0

MILVUS_URI=http://127.0.0.1:19530
MILVUS_COLLECTION=chat_knowledge
SKIP_MILVUS=false
USE_RAG=true

ENABLE_POSTGRES_PERSISTENCE=true
POSTGRES_DSN=postgresql://postgres:postgres@127.0.0.1:5432/slothbearflow
POSTGRES_RESTORE_ON_REDIS_MISS=true
POSTGRES_RESTORE_TURN_LIMIT=20
POSTGRES_RESTORE_REDIS_TTL_SEC=604800
```

含义：

- Redis 作为默认会话后端。
- Milvus/RAG 已开启。
- PostgreSQL 元数据持久化已开启。
- Redis miss 时，会尝试从 PostgreSQL 恢复会话快照并重新写回 Redis。

### 3.11 Milvus 集成兼容修复

实际联调时发现 `langchain_milvus` 与当前 `pymilvus` 版本在连接管理上不兼容。

现象：

```text
ConnectionNotExistException: should create connection first
```

原因：

- 当前 `pymilvus.MilvusClient` 使用新的连接管理方式。
- `langchain_milvus` 内部仍调用旧的 ORM `utility.has_collection(..., using=alias)`。
- 新 client 的 alias 没有注册到旧 ORM connection manager，导致初始化失败。

处理：

在项目内实现轻量 `SimpleMilvusVectorStore`，只保留 SlothBearFlow 当前真正需要的接口：

- `add_documents(documents)`
- `similarity_search(query, k=4)`

涉及文件：

- `backend/src/slothbearflow_backend/rag/milvus_store.py`

当前能力：

- 自动创建 `chat_knowledge` collection。
- 使用 Ollama embedding 生成向量。
- 写入 Milvus。
- 支持相似度检索并返回 LangChain `Document`。
- metadata 中保留 `source`。

### 3.12 PostgreSQL 会话恢复修复

实际验证 Redis miss -> PostgreSQL restore 时发现 SQL 子查询排序问题。

原问题：

子查询外层按 `id` 排序，但内层没有 select `id`，导致：

```text
column "id" does not exist
```

修复：

内层查询补充 `id` 字段，再由外层按 `id ASC` 恢复历史顺序。

涉及文件：

- `backend/src/slothbearflow_backend/persistence/postgres.py`

### 3.13 Uvicorn access log 格式修复

真实 HTTP 请求 `/health` 时发现 access log 文件 handler 使用了普通 `logging.Formatter`，但格式字段引用了 `client_addr/request_line/status_code`，会产生 logging error。

处理：

- access log 文件 handler 改用 `uvicorn.logging.AccessFormatter`。

涉及文件：

- `backend/src/slothbearflow_backend/main.py`

## 4. 当前验证结果

### 4.1 Docker 与容器状态

命令：

```bash
docker compose -f backend/docker-compose.yml ps
```

结果摘要：

```text
backend-redis-1      Up
backend-postgres-1   Up (healthy)
backend-milvus-1     Up
backend-etcd-1       Up
backend-minio-1      Up
```

### 4.2 Redis 验证

命令：

```bash
docker exec backend-redis-1 redis-cli ping
```

结果：

```text
PONG
```

进一步验证：

- Redis session key 可以写入。
- 删除 Redis session 后，可以从 PostgreSQL 恢复。
- 恢复后的 client 仍为 Redis。

结果：

```text
redis_ping= (True, None)
redis_session_exists= True
redis_restore_messages= 2
redis_restore_client= Redis
```

### 4.3 PostgreSQL 验证

命令：

```bash
docker exec backend-postgres-1 pg_isready -U postgres -d slothbearflow
```

结果：

```text
/var/run/postgresql:5432 - accepting connections
```

当前数据库：

```text
slothbearflow|postgres
```

当前表：

```text
agent_sessions
agent_chat_turns
agent_chat_stream_events
agent_ingest_jobs
```

验证结果：

```text
postgres_turns_for_check= 1
```

### 4.4 Milvus 验证

当前 collection：

```text
chat_knowledge
```

当前统计：

```text
milvus_stats= {'row_count': 3}
```

相似度检索验证：

```text
vector_search= [
  ('codex-final-check', '最终验证：Redis 负责记忆缓存，Postgres 负责元数据持久化，Milv'),
  ('codex-integration-check', 'SlothBearFlow 集成验证：Redis 用于会话记忆，PostgreS')
]
```

健康状态：

```text
milvus_status= {'enabled': True, 'collection': 'chat_knowledge'}
```

### 4.5 后端 HTTP 健康检查

命令：

```bash
curl -sf http://127.0.0.1:8000/health
```

结果摘要：

```json
{
  "ok": true,
  "redis": {
    "ok": true,
    "error": null
  },
  "session_store": {
    "backend": "redis",
    "loaded_messages": 0
  },
  "milvus": {
    "enabled": true,
    "collection": "chat_knowledge"
  },
  "postgres_persistence": {
    "enabled": true,
    "ready": true
  },
  "llm": {
    "provider": "ollama",
    "model": "deepseek-r1:7b"
  },
  "embedding": {
    "provider": "ollama",
    "model": "nomic-embed-text"
  }
}
```

### 4.6 HTTP ingest 验证

通过真实 HTTP 请求写入知识文本：

```bash
curl -X POST http://127.0.0.1:8000/ingest \
  -H 'Content-Type: application/json' \
  -d '{"text":"HTTP 集成验证：Milvus 存储向量，Postgres 记录任务状态，Redis 服务保持在线。","source":"codex-http-check"}'
```

任务最终状态：

```text
completed
```

说明：

- HTTP API 可接受知识写入。
- 后台 worker 可处理 ingest job。
- Ollama embedding 可生成向量。
- Milvus 可写入向量。
- PostgreSQL 可记录任务状态。

## 5. 当前启动方式

### 5.1 启动 Docker Desktop

```bash
open -a Docker
```

确认 Docker daemon：

```bash
docker info
```

### 5.2 启动三组件

```bash
docker compose -f backend/docker-compose.yml up -d redis postgres etcd minio milvus
```

### 5.3 启动后端

```bash
./.venv/bin/python -m uvicorn backend.src.slothbearflow_backend.main:app --host 127.0.0.1 --port 8000
```

健康检查：

```bash
curl http://127.0.0.1:8000/health
```

### 5.4 启动前端

```bash
cd frontend
pnpm dev
```

默认访问：

```text
http://127.0.0.1:5173/
```

## 6. 数据流说明

### 6.1 对话链路

1. 前端调用 `/chat`。
2. 后端读取 Redis session。
3. 如果 Redis 没有 session 且开启 `POSTGRES_RESTORE_ON_REDIS_MISS`，则从 PostgreSQL 恢复最近对话。
4. Agent 调用 LLM。
5. 对话结果写回 Redis。
6. 对话轮次、工具、引用、流式事件等元数据写入 PostgreSQL。
7. 后台任务可异步更新摘要。

### 6.2 知识库写入链路

1. 前端或 API 调用 `/ingest`。
2. 后端创建 ingest job。
3. PostgreSQL 写入 job 状态 `queued`。
4. 后台 worker 消费任务。
5. 文本切分为 documents。
6. Ollama `nomic-embed-text` 生成 embedding。
7. Milvus 写入 `chat_knowledge` collection。
8. PostgreSQL 更新 job 状态为 `completed`。

### 6.3 RAG 检索链路

1. Agent 需要知识检索时调用 `search_knowledge` 工具。
2. 工具调用 vector store 的 `similarity_search`。
3. Milvus 返回相似文本片段。
4. 工具将来源和片段写入 RAG context。
5. 最终响应中可带 citations/source。

## 7. 文件变更清单

核心后端：

- `backend/src/slothbearflow_backend/config.py`
- `backend/src/slothbearflow_backend/main.py`
- `backend/src/slothbearflow_backend/persistence/postgres.py`
- `backend/src/slothbearflow_backend/rag/milvus_store.py`
- `backend/src/slothbearflow_backend/tools/time_tool.py`
- `backend/src/slothbearflow_backend/tools/weather_tool.py`
- `backend/src/slothbearflow_backend/tools/session_tool.py`
- `backend/src/slothbearflow_backend/tools/rag_tool.py`

前端：

- `frontend/package.json`
- `frontend/config/config.ts`
- `frontend/src/pages/index.tsx`
- `frontend/src/pages/index.css`
- `frontend/tsconfig.json`

配置与文档：

- `.env`
- `.gitignore`
- `README.md`
- `docs/SlothBearFlow-本地运行与三组件集成优化记录.md`

本机环境：

- `/Applications/Docker.app`
- `/opt/homebrew/bin/docker`
- `~/.docker/daemon.json`

## 8. 当前结论

当前 SlothBearFlow 本地开发环境已经跑通：

- Redis 记忆引擎可用。
- PostgreSQL 元数据引擎可用。
- Milvus 向量数据引擎可用。
- Ollama LLM 与 embedding 模型可用。
- 后端 `/health` 显示所有核心依赖 ready。
- HTTP ingest 可完成写入。
- Milvus 可检索写入内容。
- Redis miss 可从 PostgreSQL 恢复会话。
- 后端测试通过：`25 passed`。

因此，“记忆 + 元数据 + 向量数据引擎”三组件已完成本地集成，并通过实际读写链路验证。

## 9. 后续建议

短期建议：

- 将 `.env` 中的本地默认密码仅用于开发环境，不用于生产环境。
- 前端增加更细的组件状态展示，例如 Redis/Postgres/Milvus 分别的 latency 和最近一次错误。
- 为 `/ingest` 增加 job 查询接口，前端可以展示 queued/running/completed/failed。

中期建议：

- 将 Redis、PostgreSQL、Milvus 抽象为统一 storage layer，减少业务层直接感知底层存储。
- 为 Milvus 增加 collection schema 版本管理和迁移脚本。
- 为 RAG 增加 BM25/关键词检索和 rerank，避免只依赖向量召回。

生产化建议：

- PostgreSQL 使用独立账号和强密码。
- Redis 配置密码或运行在内网。
- Milvus/MinIO/etcd 配置持久化备份策略。
- Docker Desktop 适合本地开发，生产环境应使用服务器 Docker、Kubernetes 或云托管服务。
