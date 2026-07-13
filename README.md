# document-api-rag

Document ingestion + retrieval-augmented generation (RAG) API, built with FastAPI.

## Tech stack

- **Python 3.12** (managed & pinned via [uv](https://docs.astral.sh/uv/))
- **FastAPI** + Uvicorn — async web API with auto OpenAPI docs
- **Pydantic v2** / pydantic-settings — validation & 12-factor config
- **pytest** + httpx — testing
- **ruff** — linting & formatting
- **Docker** — containerized deployment

## Project structure

```
document-api-rag/
├── app/
│   ├── main.py            # FastAPI app factory & entrypoint
│   ├── api/               # Route handlers (health, and later: documents, query)
│   └── core/              # Config, and later: db, security, logging
├── tests/                 # pytest suite
├── docker/                # Dockerfile + docker-compose
├── .env.example           # Copy to .env for local config
├── pyproject.toml         # Dependencies + tool config
└── .python-version        # Pins Python 3.12
```

## Getting started

```bash
# 1. Install dependencies into a 3.12 virtual env
uv sync

# 2. Create your local env file
cp .env.example .env

# 3. Run the API (auto-reload on save)
uv run poe start

# 4. Open the interactive docs
open http://localhost:8000/docs
```

## Tasks

All project commands are defined under `[tool.poe.tasks]` in `pyproject.toml`
and run via `uv run poe <task>`:

| Command | What it does |
|---|---|
| `uv run poe start`  | Dev server with auto-reload |
| `uv run poe serve`  | Production server (host 0.0.0.0:8000, no reload) |
| `uv run poe lint`   | Lint with ruff |
| `uv run poe format` | Auto-format with ruff |
| `uv run poe test`   | Run the pytest suite |
| `uv run poe check`  | Lint + test (use this before committing / in CI) |
| `uv run poe`        | List all available tasks |

Add a dependency with `uv add <package>` (or `uv add --dev <package>` for dev tools).

## Roadmap

- **Phase 1 (done):** project foundation, FastAPI `/health` endpoint
- **Phase 2:** Postgres + pgvector, document upload/parsing, chunking, embeddings, retrieval + Claude answers
- **Phase 3:** auth, rate limiting, structured logging, full test suite
- **Phase 4:** Docker deployment, CI/CD, observability
