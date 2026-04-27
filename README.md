# LangChain Single Agent Demo

这是一个基于 `FastAPI + LangChain + Ollama + Redis + Milvus` 的单 Agent 项目骨架，目标是让项目在本地第三方组件准备好的前提下可以直接运行，同时在组件缺失时尽量优雅降级。

## 功能概览

- `POST /chat`
  统一对话入口，支持短期记忆、摘要记忆、工具调用，默认直接返回自然语言结果，可按开关启用结构化输出或 SSE 流式输出。
- `POST /ingest`
  将纯文本切分后写入 Milvus，供 `search_knowledge` 工具做 RAG 检索。
- `GET /health`
  返回 Redis、Milvus 和会话存储后端状态，便于快速排障。

## 内置工具

- `get_current_time`
  处理“现在几点”“今天几号”这类时间问题。
- `get_weather`
  离线天气样例工具，适合工具调用联调；接入真实 API 后可直接替换实现。
- `get_session_context`
  让 Agent 在追问场景下主动查看最近对话上下文。
- `search_knowledge`
  启用 RAG 且 Milvus 可用时自动注册，返回带编号和来源的检索片段。

## 环境准备

推荐 Python `3.10+`，当前项目也兼容 Python `3.9`。

安装依赖：

```bash
./.venv/bin/pip install -r requirements.txt
```

准备环境变量：

```bash
cp .env.example .env
```

## 启动方式

启动 API 服务：

```bash
./.venv/bin/python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

启动后可以先检查：

```bash
curl http://127.0.0.1:8000/health
```

如果只想验证 Ollama 模型是否能访问：

```bash
./.venv/bin/python local_run.py
```

## 常见开关

- 关闭 RAG：设置 `USE_RAG=false`
- 跳过 Milvus 初始化：设置 `SKIP_MILVUS=true`
- 关闭二段式结构化整理：设置 `STRUCTURED_OUTPUT=false`
- 开启二段式结构化整理：设置 `STRUCTURED_OUTPUT=true`
- 开启流式输出：设置 `STREAM_OUTPUT=true`
- 流式输出格式：设置 `STREAM_OUTPUT_FORMAT=plain` 或 `STREAM_OUTPUT_FORMAT=sse`
- 关闭异步摘要压缩：设置 `ASYNC_SUMMARY_UPDATE=false`

## 请求示例

聊天请求：

```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H 'Content-Type: application/json' \
  -d '{"session_id":"demo-1","message":"帮我总结一下退款流程"}'
```

入库请求：

```bash
curl -X POST http://127.0.0.1:8000/ingest \
  -H 'Content-Type: application/json' \
  -d '{"source":"refund-policy.md","text":"退款申请需要先提交工单，再经过财务审核。"}'
```

## 排障建议

- `/health` 里 `session_store.backend=memory` 表示 Redis 不可用，但聊天仍会工作。
- `milvus.enabled=false` 且 `reason` 为连接错误时，主对话链路仍可运行，只是不会注册 `search_knowledge`。
- 默认 `structured_output=false`，接口直接返回模型最终文本，更适合多模型调试。
- `stream_output=true` 时，只有基础聊天链路会走流式输出；工具链路会自动回退为普通阻塞输出。
- `STREAM_OUTPUT_FORMAT=plain` 时会直接输出内容文本，适合 `curl`；`sse` 会返回 `start/chunk/done` 事件。
- 如果 `structured_output=true` 且模型不支持稳定结构化输出，系统会自动回退为原始文本。

## 测试

运行测试：

```bash
./.venv/bin/python -m pytest -q
```

当前测试覆盖：

- 基础健康检查
- 内存会话存储降级
- `/chat` 正常响应
- `/ingest` 在不同开关组合下的行为
- RAG 引用在 API 响应中的透传
