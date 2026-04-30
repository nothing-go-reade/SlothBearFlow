# SlothBearFlow

A monorepo-ready AI agent service scaffold.

Current status:
- `backend/` is fully runnable.
- `frontend/` is a placeholder for future UI work.

## Repository Layout

```text
.
|-- backend/
|   |-- src/
|   |   `-- slothbearflow_backend/
|   |       |-- agent/
|   |       |-- memory/
|   |       |-- persistence/
|   |       |-- rag/
|   |       |-- tools/
|   |       |-- worker/
|   |       `-- main.py
|   |-- tests/
|   |-- .env.example
|   |-- docker-compose.yml
|   |-- local_run.py
|   |-- pytest.ini
|   `-- requirements.txt
|-- frontend/
|   `-- .gitkeep
|-- LICENSE
`-- README.md
```

## Backend Quick Start

### 1) Install dependencies

```bash
./.venv/bin/pip install -r backend/requirements.txt
```

### 2) Prepare environment variables

```bash
cp backend/.env.example .env
```

### 3) Start infrastructure (optional for full stack)

```bash
docker compose -f backend/docker-compose.yml up -d
```

### 4) Run API server

```bash
./.venv/bin/python -m uvicorn backend.src.slothbearflow_backend.main:app --host 0.0.0.0 --port 8000
```

Then open:
- API docs: `http://127.0.0.1:8000/docs`
- Health check: `http://127.0.0.1:8000/health`

## Backend Features

- `POST /chat`
  Unified chat entrypoint with short-term memory, summary memory, and tool calling. Supports plain output, structured output, and streaming.
- `POST /ingest`
  Splits plain text and writes chunks to Milvus for RAG retrieval.
- `GET /health`
  Reports Redis, Milvus, session store, LLM, and embedding status.

## Built-in Tools

- `get_current_time`
- `get_weather`
- `get_session_context`
- `search_knowledge` (enabled when RAG is available)

## Backend Configuration

Main flags (via `.env`):
- `LLM_PROVIDER=ollama|openai`
- `USE_RAG=false|true`
- `SKIP_MILVUS=true|false`
- `STRUCTURED_OUTPUT=false|true`
- `STREAM_OUTPUT=false|true`
- `STREAM_OUTPUT_FORMAT=plain|sse`
- `ENABLE_POSTGRES_PERSISTENCE=false|true`
- `POSTGRES_RESTORE_ON_REDIS_MISS=false|true`
- `POSTGRES_RESTORE_TURN_LIMIT=20`
- `ENABLE_EXPLICIT_REACT_RUNTIME=false|true`
- `REACT_MAX_STEPS=4`
- `REACT_TOOL_TIMEOUT_SEC=15`
- `REACT_STREAM_THOUGHTS=false|true`

ReAct runtime notes:
- With `ENABLE_EXPLICIT_REACT_RUNTIME=false` (default), `/chat` keeps current LangChain AgentExecutor behavior.
- With `ENABLE_EXPLICIT_REACT_RUNTIME=true`, `/chat` uses an explicit bounded ReAct loop while keeping response schema unchanged.

See `backend/.env.example` for full options.

## Testing

```bash
./.venv/bin/python -m pytest -q backend/tests
```

## Local LLM Probe

```bash
./.venv/bin/python backend/local_run.py
```

## Frontend

`frontend/` is intentionally empty for now and reserved for future frontend implementation.
