# LangChain Single Agent

A single-agent project scaffold built with `FastAPI`, `LangChain`, `Ollama`, `Redis`, and `Milvus`.

The goal of this repository is to provide a local-first agent service that can run out of the box when its third-party dependencies are available, while degrading gracefully when some components are missing.

## Quick Start

```bash
cp .env.example .env
./.venv/bin/pip install -r requirements.txt
./.venv/bin/python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Then open:

- API docs: `http://127.0.0.1:8000/docs`
- Health check: `http://127.0.0.1:8000/health`

## Features

- `POST /chat`
  Unified chat entrypoint with short-term memory, summary memory, and tool calling. By default, it returns plain natural-language responses, with optional structured output and SSE streaming.
- `POST /ingest`
  Splits plain text into chunks and stores them in Milvus so the `search_knowledge` tool can use them for RAG retrieval.
- `GET /health`
  Reports the status of Redis, Milvus, and the session storage backend for quick troubleshooting.

## Built-in Tools

- `get_current_time`
  Handles time-related questions such as "What time is it now?" or "What's today's date?"
- `get_weather`
  An offline weather demo tool that is useful for tool-calling integration tests. You can replace it directly with a real weather API implementation.
- `get_session_context`
  Lets the agent actively inspect recent conversation context during follow-up turns.
- `search_knowledge`
  Automatically registered when RAG is enabled and Milvus is available. Returns retrieved snippets with source references and chunk indices.

## Requirements

- Python `3.10+` recommended
- Python `3.9` is also supported

## Project Structure

```text
.
|-- app/
|   |-- agent/        # Agent executor and chat workflow
|   |-- memory/       # Short-term and summary memory management
|   |-- rag/          # Milvus integration, embeddings, ingest, splitter
|   |-- tools/        # Built-in tools and tool registry
|   |-- worker/       # Background task processing
|   |-- config.py     # Environment-driven settings
|   `-- main.py       # FastAPI application entrypoint
|-- tests/            # Smoke tests and API behavior checks
|-- local_run.py      # Local Ollama connectivity check
|-- requirements.txt
`-- docker-compose.yml
```

Install dependencies:

```bash
./.venv/bin/pip install -r requirements.txt
```

Prepare environment variables:

```bash
cp .env.example .env
```

## Running the Service

Start the API server:

```bash
./.venv/bin/python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

After the server starts, you can verify the service with:

```bash
curl http://127.0.0.1:8000/health
```

If you only want to verify that the Ollama model is reachable:

```bash
./.venv/bin/python local_run.py
```

## Configuration Flags

LLM provider selection:

- `LLM_PROVIDER=ollama` uses the local Ollama deployment
- `LLM_PROVIDER=openai` uses OpenAI or an OpenAI-compatible cloud endpoint
- `LLM_MODEL` can override the provider-specific default model
- `LLM_SUPPORTS_TOOLS` can force tool-calling on or off across providers
- `OPENAI_BASE_URL` can be used with OpenAI-compatible vendors
- `OPENAI_API_KEY` is required when using `LLM_PROVIDER=openai`
- `EMBEDDING_PROVIDER` defaults to `LLM_PROVIDER`, but can be set independently
- `EMBEDDING_MODEL` can override the provider-specific default embedding model
- `OPENAI_EMBED_MODEL` controls the default OpenAI embedding model

Model parameter config (chat models):

- Global knobs:
  - `LLM_TEMPERATURE`
  - `LLM_TOP_P`
  - `LLM_MAX_TOKENS`
  - `LLM_DEEP_THINK`
  - `LLM_REASONING_EFFORT`
  - `LLM_MODEL_KWARGS_JSON`
  - `LLM_EXTRA_BODY_JSON`
- OpenAI/OpenAI-compatible overrides:
  - `OPENAI_TEMPERATURE`
  - `OPENAI_TOP_P`
  - `OPENAI_MAX_TOKENS`
  - `OPENAI_DEEP_THINK`
  - `OPENAI_REASONING_EFFORT`
  - `OPENAI_MODEL_KWARGS_JSON`
  - `OPENAI_EXTRA_BODY_JSON`

Precedence:

- `temperature`: function arg in code > OpenAI override > global > default `0.2`
- Other OpenAI params: OpenAI override > global > omitted

Deep-think behavior:

- If `*_REASONING_EFFORT` is explicitly set, it is used directly.
- Else if `*_DEEP_THINK=true`, it maps to `reasoning_effort=high`.
- Else reasoning parameters are omitted.

Notes for OpenAI-compatible vendors:

- Put standard/near-standard extras in `*_MODEL_KWARGS_JSON`.
- Put vendor-private request-body fields in `*_EXTRA_BODY_JSON`.
- Both JSON fields must be JSON objects.

- Disable RAG: set `USE_RAG=false`
- Skip Milvus initialization: set `SKIP_MILVUS=true`
- Disable the second-pass structured formatter: set `STRUCTURED_OUTPUT=false`
- Enable the second-pass structured formatter: set `STRUCTURED_OUTPUT=true`
- Enable streaming output: set `STREAM_OUTPUT=true`
- Choose streaming format: set `STREAM_OUTPUT_FORMAT=plain` or `STREAM_OUTPUT_FORMAT=sse`
- Disable asynchronous summary compression: set `ASYNC_SUMMARY_UPDATE=false`
- Enable PostgreSQL metadata persistence: set `ENABLE_POSTGRES_PERSISTENCE=true`
- Configure PostgreSQL DSN: set `POSTGRES_DSN=postgresql://...`

## Example Requests

Chat request:

```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H 'Content-Type: application/json' \
  -d '{"session_id":"demo-1","message":"Please summarize the refund process for me."}'
```

Example OpenAI-compatible configuration:

```bash
export LLM_PROVIDER=openai
export OPENAI_MODEL=gpt-4o-mini
export OPENAI_API_KEY=your_api_key
export OPENAI_EMBED_MODEL=text-embedding-3-small
./.venv/bin/python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Ingest request:

```bash
curl -X POST http://127.0.0.1:8000/ingest \
  -H 'Content-Type: application/json' \
  -d '{"source":"refund-policy.md","text":"Refund requests must be submitted through a support ticket before finance review."}'
```

## Troubleshooting

- If `/health` shows `session_store.backend=memory`, Redis is unavailable, but chat will still work.
- If `milvus.enabled=false` and the reported `reason` is a connection error, the main chat flow still works, but `search_knowledge` will not be registered.
- When `ENABLE_POSTGRES_PERSISTENCE=true`, the service will also persist chat sessions, chat turns, summaries, and ingest job metadata to PostgreSQL.
- By default, `structured_output=false`, which means the API returns the model's final plain-text response directly. This is usually better for multi-model debugging.
- When `stream_output=true`, only the basic chat path uses streaming. Tool-invoking paths automatically fall back to standard blocking responses.
- When `STREAM_OUTPUT_FORMAT=plain`, the response is returned as plain text, which is convenient for `curl`. The `sse` mode returns `start`, `chunk`, and `done` events.
- If `structured_output=true` but the model does not support stable structured output, the system automatically falls back to raw text.

## Testing

Run tests with:

```bash
./.venv/bin/python -m pytest -q
```

Current test coverage includes:

- Basic health checks
- In-memory session store fallback
- Successful `/chat` responses
- `/ingest` behavior under different feature-flag combinations
- Pass-through of RAG references in API responses

## Contributing

Issues and pull requests are welcome.

Before submitting changes, it is a good idea to:

- Keep feature changes focused and easy to review
- Update the README or examples when behavior changes
- Run the test suite locally with `./.venv/bin/python -m pytest -q`

## License

No license file has been added yet.

If you plan to open source this repository on GitHub, add a `LICENSE` file before publishing so others know how they are allowed to use, modify, and redistribute the project.
