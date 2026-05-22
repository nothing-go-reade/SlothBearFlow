# SlothBearFlow 项目知识库问答卡片

更新时间：2026-05-23

这份文档用于增强 Milvus 向量检索召回效果。每一节都按“常见问题 + 直接答案”组织，适合 Agent 在回答项目介绍、启动、架构和故障排查时检索。

## SlothBearFlow 是什么？

SlothBearFlow 是一个本地优先的 AI Agent 服务脚手架。它把 FastAPI 后端、React + Umi + TypeScript 前端、本地 Ollama 大模型、Redis 记忆、PostgreSQL 元数据持久化、Milvus 向量检索组合成一个可运行的本地 Agent 工作台。它适合用于开发和验证本地 Agent、工具调用、RAG 知识库、会话记忆和多组件集成。

## SlothBearFlow 的后端是什么？

SlothBearFlow 后端是 FastAPI 服务，入口文件是 `backend/src/slothbearflow_backend/main.py`。后端提供 `/health`、`/ingest` 和 `/chat` 三个核心接口。`/health` 用于检查依赖服务状态，`/ingest` 用于写入知识库，`/chat` 用于执行 Agent 对话。

## SlothBearFlow 的前端是什么？

SlothBearFlow 前端是 React + Umi + TypeScript 控制台，目录是 `frontend/`。前端页面是本地 Agent Playground，可以查看服务健康状态、调用聊天接口、写入知识库并打开 API 文档。前端默认访问地址是 `http://127.0.0.1:5173/`。

## Redis 在 SlothBearFlow 里负责什么？

Redis 是 SlothBearFlow 的记忆引擎，负责保存短期会话状态和最近消息窗口。后端处理 `/chat` 时会按 `session_id` 读取 Redis 中的历史消息，让 Agent 能接上当前会话上下文。健康检查中 `redis.ok=true` 且 `session_store.backend=redis` 表示 Redis 记忆可用。

## PostgreSQL 在 SlothBearFlow 里负责什么？

PostgreSQL 是 SlothBearFlow 的元数据引擎，负责持久化会话、对话轮次、流式事件和知识库写入任务。开启 `ENABLE_POSTGRES_PERSISTENCE=true` 后，后端会把聊天和 ingest 任务元数据写入 PostgreSQL。开启 `POSTGRES_RESTORE_ON_REDIS_MISS=true` 后，Redis 缺失时可以从 PostgreSQL 恢复最近会话。

## Milvus 在 SlothBearFlow 里负责什么？

Milvus 是 SlothBearFlow 的向量数据引擎，负责保存 RAG 文档切块后的 embedding 向量，并提供相似度检索。默认集合名是 `chat_knowledge`。当 `/ingest` 接收到文本后，后台 worker 会切块、调用 Ollama embedding 模型、写入 Milvus。Agent 可通过 `search_knowledge` 工具查询 Milvus。

## Ollama 在 SlothBearFlow 里负责什么？

Ollama 为 SlothBearFlow 提供本地 LLM 和 embedding。当前对话模型是 `deepseek-r1:7b`，当前 embedding 模型是 `nomic-embed-text`。`/chat` 会调用 LLM 生成回答，`/ingest` 会调用 embedding 模型把文本转成向量。

## SlothBearFlow 怎么启动后端？

推荐先启动 Docker Desktop 和基础组件，然后启动后端：

```bash
docker compose -f backend/docker-compose.yml up -d
./.venv/bin/python -m uvicorn backend.src.slothbearflow_backend.main:app --host 127.0.0.1 --port 8000
```

后端健康检查地址是 `http://127.0.0.1:8000/health`，API 文档地址是 `http://127.0.0.1:8000/docs`。

## SlothBearFlow 怎么启动前端？

前端启动命令：

```bash
cd frontend
PORT=5173 ./node_modules/.bin/umi dev --host 127.0.0.1
```

前端页面地址是 `http://127.0.0.1:5173/`。如果浏览器停留在 `http://127.0.0.1:8000/health`，说明当前看到的是后端健康接口，不是前端页面。

## SlothBearFlow 怎么写入项目知识库？

可以调用后端 `/ingest` 接口写入知识库，请求体包含 `source` 和 `text`。后端会返回 `job_id`，并由后台 worker 异步写入 Milvus。前端页面的 Knowledge source 输入区也可以调用该能力。

## 如何确认三组件都正常？

访问 `http://127.0.0.1:8000/health`。如果返回中包含 `redis.ok=true`、`milvus.enabled=true`、`postgres_persistence.ready=true`，说明 Redis、Milvus、PostgreSQL 都正常。Docker Compose 中的服务包括 `redis`、`postgres`、`milvus`、`etcd` 和 `minio`。

## API 文档打不开怎么办？

如果 Swagger 页面提示 `Failed to load API definition`，优先检查 `/openapi.json` 是否能访问。后端直接地址是 `http://127.0.0.1:8000/openapi.json`，前端 Umi 也代理了 `/openapi.json` 到后端。现在 API 文档入口已经优化过，前端可直接打开后端文档。

## RAG 不生效怎么办？

检查 `.env` 中 `USE_RAG=true` 且 `SKIP_MILVUS=false`。检查 Milvus 容器是否运行，检查 `/health` 中 `milvus.enabled=true`。检查 Ollama embedding 模型 `nomic-embed-text` 是否已经拉取并可用。

## SlothBearFlow 当前已经完成哪些优化？

SlothBearFlow 当前已经完成后端启动修复、私有 env 配置加载、React + Umi + TypeScript 前端控制台、API 文档代理修复、前端样式优化、Docker Desktop 安装、Redis/Milvus/PostgreSQL 本地部署、Ollama embedding 集成、Milvus 写入适配、PostgreSQL 会话恢复修复、access log formatter 修复，以及三组件联通验证。
