# document-api-rag

Document ingestion + retrieval-augmented generation API.

Built with FastAPI and MongoDB (Atlas Local). The document-ingestion and RAG
retrieval features are still being built out; the current codebase provides the
application foundation: health/readiness probes, user registration, request
correlation, structured logging, CORS, and rate limiting.

## Features

- **FastAPI application factory** (`create_app`) with a lifespan that opens the
  MongoDB client at startup and closes it at shutdown.
- **Liveness and readiness probes** — `/health` (process is up) and `/ready`
  (MongoDB reachable, returns 503 when it is not).
- **User registration** — `POST /v1/auth/register`, with bcrypt password hashing
  and duplicate email/username detection.
- **MongoDB integration** — a single shared async client, with a JSON-schema
  validator and unique indexes on the `users` collection applied idempotently at
  startup.
- **Versioned API** — feature endpoints are mounted under `/v1`; operational
  endpoints (health, readiness) stay unversioned for stable orchestrator paths.
- **Request correlation** — middleware assigns an `X-Request-ID` to every request
  (reusing an incoming one if present) and echoes it back in the response.
- **Structured logging** — human-readable console logs in development, single-line
  JSON logs in production; every log line is tagged with the request id.
- **Rate limiting** — via slowapi; registration is limited to 3 requests/minute
  per client IP.
- **CORS** — restricted to an explicit list of allowed origins (never `*`).

## Project Structure

```
document-api-rag/
├── app/
│   ├── main.py              # App factory, middleware wiring, lifespan, entrypoint
│   ├── middleware.py        # RequestContextMiddleware — assigns/echoes X-Request-ID
│   ├── dependencies.py      # Shared FastAPI dependency aliases (DbDependency)
│   ├── logging_config.py    # Logging setup: console/JSON formatters, request-id filter
│   ├── routers/
│   │   ├── __init__.py      # Aggregates root_router (ops) and api_router (/v1)
│   │   ├── auth.py          # POST /auth/register
│   │   └── health.py        # GET /health, GET /ready
│   ├── core/
│   │   ├── config.py        # Pydantic-settings Settings loaded from env/.env
│   │   ├── security.py      # bcrypt hash_password / verify_password
│   │   └── rate_limit.py    # Shared slowapi Limiter
│   ├── models/
│   │   └── user.py          # users document shape, indexes, $jsonSchema validator
│   ├── schemas/
│   │   ├── auth.py          # CreateUserRequest, UserResponse, UserRole
│   │   └── health.py        # HealthResponse, ReadinessResponse
│   ├── db/
│   │   └── mongodb.py       # Async client lifecycle, schema/index setup, ping
│   └── services/
│       ├── user_service.py  # create_user (user lifecycle)
│       └── auth_service.py  # Auth/token logic (planned — currently a stub)
├── tests/                   # pytest suite (health, readiness, CORS, request-id)
├── docker/
│   ├── Dockerfile           # Multi-stage production image
│   └── docker-compose.yml   # Local stack: API + MongoDB Atlas Local
├── .env.example             # Copy to .env for local config
├── pyproject.toml           # Dependencies, poe tasks, ruff/pytest config
└── .python-version          # Pins Python 3.12
```

## Requirements

- **Python 3.12+**
- **[uv](https://docs.astral.sh/uv/)** for dependency and environment management
- **Docker** (for the local MongoDB Atlas Local instance)

Key runtime dependencies: `fastapi`, `uvicorn[standard]`, `pydantic[email]`,
`pydantic-settings`, `pymongo`, `bcrypt`, `slowapi`.

Dev dependencies: `pytest`, `httpx`, `ruff`, `poethepoet`.

## Installation

```bash
# 1. Install dependencies into a Python 3.12 virtual env
uv sync

# 2. Create your local env file
cp .env.example .env

# 3. Start MongoDB (Atlas Local) on host port 27018
uv run poe db-up
```

## Running the App

```bash
uv run poe start
```

This runs `uvicorn app.main:app --reload` on http://localhost:8000. Interactive
API docs are served at http://localhost:8000/docs.

Other tasks (defined under `[tool.poe.tasks]` in `pyproject.toml`):

| Command | What it does |
|---|---|
| `uv run poe start`   | Dev server with auto-reload |
| `uv run poe serve`   | Production server (0.0.0.0:8000, no reload) |
| `uv run poe lint`    | Lint with ruff |
| `uv run poe format`  | Auto-format with ruff |
| `uv run poe test`    | Run the pytest suite |
| `uv run poe check`   | Lint + test |
| `uv run poe db-up`   | Start MongoDB (Atlas Local) |
| `uv run poe db-down` | Stop the Docker stack |
| `uv run poe db-logs` | Tail MongoDB logs |

## API Endpoints

| Method | Path                | Description |
|--------|---------------------|-------------|
| GET    | `/health`           | Liveness probe; returns 200 if the process is up (no dependency checks). |
| GET    | `/ready`            | Readiness probe; returns 200 if MongoDB is reachable, 503 otherwise. |
| POST   | `/v1/auth/register` | Register a new user. Returns 201, or 400 if the email/username is taken. Rate limited to 3/minute per client IP. |

## Running Tests

```bash
uv run poe test
```

Or directly: `.venv/bin/python -m pytest -q`. The suite covers the health and
readiness probes, CORS behavior, and request-id correlation. It uses FastAPI's
`TestClient`, so the app lifespan does not run and MongoDB is not required —
which is why the readiness probe is expected to report 503 under test.

## Environment Variables

Configuration is loaded from environment variables (and a local `.env` file in
development) via pydantic-settings. Copy `.env.example` to `.env` to get started.

| Variable          | Default                                             | Description |
|-------------------|-----------------------------------------------------|-------------|
| `APP_NAME`        | `document-api-rag`                                  | Application name (shown in health responses and logs). |
| `ENVIRONMENT`     | `development`                                        | Deployment environment label. |
| `DEBUG`           | `false`                                              | Debug flag. |
| `API_HOST`        | `0.0.0.0`                                            | API server bind host. |
| `API_PORT`        | `8000`                                               | API server port. |
| `MONGODB_URI`     | `mongodb://localhost:27018/?directConnection=true`  | MongoDB connection string. |
| `MONGODB_DB_NAME` | `document_rag`                                       | Database name. |
| `LOG_LEVEL`       | `INFO`                                               | Log level: DEBUG / INFO / WARNING / ERROR / CRITICAL. |
| `LOG_JSON`        | `false`                                              | `true` for structured JSON logs (production), `false` for readable dev logs. |
| `CORS_ORIGINS`    | `http://localhost:3000`                              | Comma-separated list of browser origins allowed to call the API. |
