# Multi-stage build for a small, reproducible production image.
# Stage 1: install dependencies with uv into a virtual env.
FROM python:3.12-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/app/.venv

WORKDIR /app

# Install deps first (cached layer), then the project.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY . .
RUN uv sync --frozen --no-dev

# Stage 2: minimal runtime image.
FROM python:3.12-slim AS runtime

WORKDIR /app

# Run as a non-root user for security.
RUN useradd --create-home --uid 1000 appuser
COPY --from=builder --chown=appuser:appuser /app /app
USER appuser

ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
