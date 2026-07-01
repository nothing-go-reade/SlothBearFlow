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
- `POSTGRES_RESTORE_REDIS_TTL_SEC=604800`
- `ENABLE_EXPLICIT_REACT_RUNTIME=false|true`
- `REACT_MAX_STEPS=4`
- `REACT_TOOL_TIMEOUT_SEC=15`
- `REACT_STREAM_THOUGHTS=false|true`
- `ENABLE_BACKGROUND_REVIEW=false|true`
- `REVIEW_MEMORY_INTERVAL=3`
- `REVIEW_SKILLS_INTERVAL=5`
- `REVIEW_BASE_DIR=agent_learning`
- `REVIEW_MAX_ITEMS=5`
- `REVIEW_MODEL=` (empty → reuse main LLM)
- `REVIEW_FORCE_STRUCTURED=false|true`
- `REVIEW_TOOL_TRACE=false|true`
- `INJECT_LEARNING_INTO_PROMPT=false|true`
- `LEARNING_PROMPT_BUDGET_CHARS=1200`

ReAct runtime notes:
- With `ENABLE_EXPLICIT_REACT_RUNTIME=false` (default), `/chat` keeps current LangChain AgentExecutor behavior.
- With `ENABLE_EXPLICIT_REACT_RUNTIME=true`, `/chat` uses an explicit bounded ReAct loop while keeping response schema unchanged.

## Background Review (Memory / Skills self-learning)

Hermes-style background reflection. After the main answer is delivered, the
turn orchestration layer (`agent/conversation_loop.py`, `ChatTurnRunner`) may
enqueue a `"review"` job on the existing async worker. A restricted review
agent then replays a snapshot of the turn and distills:

- **Memory** — durable user preferences / identity / how-they-want-the-agent-to-work
- **Skills** — reusable task techniques (user corrections to format/tone/workflow are a first-class skill signal)

Storage: Markdown files are the **source of truth** (`agent_learning/memory/*.md`,
`agent_learning/skills/*.md`); a derived `index.sqlite` provides dedup + relevance
selection and can be fully rebuilt from disk. With `ENABLE_BACKGROUND_REVIEW=true`,
review runs every `REVIEW_MEMORY_INTERVAL` / `REVIEW_SKILLS_INTERVAL` turns (nudge cadence).

Write path is auto-selected by model capability:
- tool-capable models → Hermes-style `save_memory` / `save_skill` tool calls gated by a thread-local whitelist (only those tools execute);
- otherwise (e.g. `deepseek-r1:7b`) → structured-output JSON written by backend code.

Isolation guarantees: the review never writes the Redis session, never appends to
conversation history, and only writes through the learning store — so the main
session and its history stay clean.

Read-back (closing the loop) is opt-in via `INJECT_LEARNING_INTO_PROMPT=true`:
relevant memory/skills are injected (bounded by `LEARNING_PROMPT_BUDGET_CHARS`)
into the next turn's system prompt. Note this changes the system prompt across
turns and can reduce provider prefix-cache hits; the injected block deliberately
omits volatile fields to stay byte-stable when the learning set is unchanged.

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

`frontend/` is a React + Umi + TypeScript console for calling the backend API.

```bash
cd frontend
pnpm install
pnpm dev
```

Then open `http://127.0.0.1:5173`. The local console calls the FastAPI backend directly at `http://127.0.0.1:8000` so Server-Sent Events can stream reliably in development.
