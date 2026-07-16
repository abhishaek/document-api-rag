# document-api-rag

Document ingestion + retrieval-augmented generation API.

Built with FastAPI and MongoDB (Atlas Local). The document-ingestion and RAG
retrieval features are still being built out; the current codebase provides the
application foundation and a complete authentication layer: health/readiness
probes, user registration, JWT login with refresh-token rotation, logout with
immediate token revocation, request correlation, structured logging, CORS, and
rate limiting.

## Features

- **FastAPI application factory** (`create_app`) with a lifespan that opens the
  MongoDB client at startup and closes it at shutdown.
- **Liveness and readiness probes** — `/health` (process is up, no dependency
  checks) and `/ready` (MongoDB reachable, returns 503 when it is not).
- **User registration** — `POST /v1/auth/register`, with bcrypt password hashing
  and duplicate email/username detection enforced by unique indexes.
- **JWT authentication** — `POST /v1/auth/login` issues a short-lived access
  token (15 minutes by default) plus a long-lived refresh token. Login is
  hardened against username enumeration by timing.
- **Refresh-token rotation** — `POST /v1/auth/refresh` exchanges a refresh token
  for a new access/refresh pair and invalidates the old one, so each refresh
  token is single-use. Refresh tokens are stored only as SHA-256 hashes.
- **Logout with immediate revocation** — `POST /v1/auth/logout` deletes the
  refresh token and adds the access token's `jti` to a denylist, so the access
  token stops working immediately instead of remaining valid until expiry.
- **Account deactivation is enforced on refresh** — an inactive user's refresh
  token is revoked and refused, so they cannot mint new access tokens.
- **MongoDB integration** — a single shared async client, with `$jsonSchema`
  validators and indexes (including TTL indexes for automatic token cleanup)
  applied idempotently at startup.
- **Versioned API** — feature endpoints are mounted under `/v1`; operational
  endpoints (health, readiness) stay unversioned for stable orchestrator paths.
- **Request correlation** — middleware assigns an `X-Request-ID` to every request
  (reusing an incoming one if present) and echoes it back in the response.
- **Structured logging** — human-readable console logs in development, single-line
  JSON logs in production; every log line is tagged with the request id.
- **Rate limiting** — via slowapi, per client IP: 3/minute on register, 5/minute
  on login, 10/minute on refresh.
- **CORS** — restricted to an explicit list of allowed origins (never `*`).

## Project Structure

```
document-api-rag/
├── app/
│   ├── main.py                  # App factory, middleware wiring, lifespan, entrypoint
│   ├── middleware.py            # RequestContextMiddleware — assigns/echoes X-Request-ID
│   ├── dependencies.py          # DbDependency, get_current_user / UserDependency
│   ├── logging_config.py        # Logging setup: console/JSON formatters, request-id filter
│   ├── routers/
│   │   ├── __init__.py          # Aggregates root_router (ops) and api_router (/v1)
│   │   ├── auth.py              # register, login, refresh, logout
│   │   └── health.py            # GET /health, GET /ready
│   ├── core/
│   │   ├── config.py            # Pydantic-settings Settings loaded from env/.env
│   │   ├── security.py          # bcrypt hash_password / verify_password
│   │   └── rate_limit.py        # Shared slowapi Limiter
│   ├── models/
│   │   ├── user.py              # users document shape, unique indexes, validator
│   │   ├── refresh_token.py     # refresh_tokens shape, unique + TTL indexes, validator
│   │   └── revoked_token.py     # revoked_tokens (access-token jti denylist), TTL index
│   ├── schemas/
│   │   ├── auth.py              # CreateUserRequest, UserResponse, UserRole, Token, RefreshRequest
│   │   └── health.py            # HealthResponse, ReadinessResponse
│   ├── db/
│   │   └── mongodb.py           # Async client lifecycle, schema/index setup, ping
│   └── services/
│       ├── user_service.py      # create_user, get_user_by_id (user lifecycle)
│       └── auth_service.py      # Credential verification and the full token lifecycle
├── tests/
│   ├── conftest.py              # In-memory fake DB, async client, rate-limit disabling
│   ├── test_auth.py             # Auth routes: register, login, refresh, logout
│   ├── test_auth_service.py     # Token creation, verification, rotation, revocation
│   ├── test_config.py           # Settings loading and validation
│   ├── test_cors.py             # Allowed/disallowed origin behavior
│   ├── test_health.py           # Liveness probe
│   ├── test_readiness.py        # Readiness probe (200 / 503)
│   └── test_request_id.py       # Request-id generation and passthrough
├── docker/
│   ├── Dockerfile               # Multi-stage production image
│   └── docker-compose.yml       # Local stack: API + MongoDB Atlas Local
├── .env.example                 # Copy to .env for local config
├── pyproject.toml               # Dependencies, poe tasks, ruff/pytest config
└── .python-version              # Pins Python 3.12
```

## Requirements

- **Python 3.12+**
- **[uv](https://docs.astral.sh/uv/)** for dependency and environment management
- **Docker** (for the local MongoDB Atlas Local instance)

Key runtime dependencies: `fastapi`, `uvicorn[standard]`, `pydantic[email]`,
`pydantic-settings`, `pymongo`, `bcrypt`, `pyjwt`, `python-multipart`, `slowapi`.

Dev dependencies: `pytest`, `pytest-asyncio`, `httpx`, `ruff`, `poethepoet`.

## Installation

```bash
# 1. Install dependencies into a Python 3.12 virtual env
uv sync

# 2. Create your local env file
cp .env.example .env

# 3. Generate a SECRET_KEY and add it to .env
#    SECRET_KEY has no default — the app will not start without it.
echo "SECRET_KEY=$(openssl rand -hex 32)" >> .env

# 4. Start MongoDB (Atlas Local) on host port 27018
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
| GET    | `/health`           | Liveness probe. Returns 200 if the process is up; performs no dependency checks. |
| GET    | `/ready`            | Readiness probe. Returns 200 if MongoDB is reachable, 503 otherwise. |
| POST   | `/v1/auth/register` | Register a new user (JSON body: email, username, password, optional role). Returns 201, or 400 if the email/username is taken. Rate limited to 3/minute per IP. |
| POST   | `/v1/auth/login`    | Log in with an OAuth2 password form (`username`, `password`, form-encoded). Returns 200 with an access token and refresh token, or 401 on bad credentials. Rate limited to 5/minute per IP. |
| POST   | `/v1/auth/refresh`  | Exchange a refresh token (JSON body: `refresh_token`) for a new access/refresh pair, invalidating the old refresh token. Returns 401 if the token is invalid, expired, or the account is inactive. Rate limited to 10/minute per IP. |
| POST   | `/v1/auth/logout`   | Revoke the refresh token (JSON body: `refresh_token`) and denylist the current access token. Requires a valid `Authorization: Bearer` header. Returns 204. |

Usernames are deliberately case-sensitive: `Abhi` and `abhi` are distinct
accounts. Email addresses are normalized to lowercase, so they are not.

## Running Tests

```bash
uv run poe test
```

Or directly: `.venv/bin/python -m pytest -q`.

The suite covers the auth routes and auth service, settings loading, the health
and readiness probes, CORS behavior, and request-id correlation. MongoDB is not
required: `conftest.py` overrides the database dependency with an in-memory fake
that emulates the unique indexes and the query operators the services use, and
rate limiting is disabled for the duration of each test.

## Environment Variables

Configuration is loaded from environment variables (and a local `.env` file in
development) via pydantic-settings. Copy `.env.example` to `.env` to get started.

`SECRET_KEY` is required and has no default, so that a real secret is never
committed. Generate one with `openssl rand -hex 32`. Every other variable falls
back to the default below if unset.

| Variable                      | Default                                             | Description |
|-------------------------------|-----------------------------------------------------|-------------|
| `SECRET_KEY`                  | _(required)_                                        | Secret used to sign JWTs. The app fails to start if unset. |
| `APP_NAME`                    | `document-api-rag`                                  | Application name (shown in health responses and logs). |
| `ENVIRONMENT`                 | `development`                                       | Deployment environment label. |
| `DEBUG`                       | `false`                                             | Debug flag. |
| `API_HOST`                    | `0.0.0.0`                                           | API server bind host. |
| `API_PORT`                    | `8000`                                              | API server port. |
| `MONGODB_URI`                 | `mongodb://localhost:27018/?directConnection=true`  | MongoDB connection string. |
| `MONGODB_DB_NAME`             | `document_rag`                                      | Database name. |
| `JWT_ALGORITHM`               | `HS256`                                             | Algorithm used to sign access tokens. |
| `ACCESS_TOKEN_EXPIRE_MINUTES` | `15`                                                | Access token lifetime in minutes. |
| `REFRESH_TOKEN_EXPIRE_DAYS`   | `7`                                                 | Refresh token lifetime in days. |
| `LOG_LEVEL`                   | `INFO`                                              | Log level: DEBUG / INFO / WARNING / ERROR / CRITICAL. |
| `LOG_JSON`                    | `false`                                             | `true` for structured JSON logs (production), `false` for readable dev logs. |
| `CORS_ORIGINS`                | `http://localhost:3000`                             | Comma-separated list of browser origins allowed to call the API. |
</content>
