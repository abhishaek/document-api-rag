"""FastAPI application entrypoint.

Run in development with:
    uv run poe start
"""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from app.core.config import get_settings
from app.core.rate_limit import limiter
from app.db.mongodb import close_mongo_connection, connect_to_mongo
from app.logging_config import configure_logging
from app.middleware import RequestContextMiddleware
from app.routers import api_router, root_router

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Manage resources tied to the app's life: open the MongoDB client at
    startup, close it at shutdown. Code before `yield` runs on startup; code
    after runs on shutdown."""
    # Settings were stashed on app.state by create_app — reuse them here.
    await connect_to_mongo(app.state.settings)
    logger.info("startup complete")
    yield
    await close_mongo_connection()
    logger.info("shutdown complete")


def create_app() -> FastAPI:
    """Application factory. Keeping this as a function makes testing and
    alternative configurations (e.g. per-environment) straightforward."""
    settings = get_settings()

    # Configure logging before anything else so all subsequent logs are formatted.
    configure_logging(settings)

    app = FastAPI(
        title=settings.app_name,
        description="Document ingestion + retrieval-augmented generation API.",
        version="0.1.0",
        lifespan=lifespan,
    )
    # Share the single settings instance with the lifespan handler (and anything
    # else that has the app) instead of calling get_settings() again.
    app.state.settings = settings

    # Rate limiting: slowapi reads limiter off app.state and returns 429 via this
    # handler when a @limiter.limit(...) is exceeded.
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

    # Middleware runs in reverse order of registration, so the request-context
    # middleware is added last to sit outermost — the request_id is set before
    # anything else (including CORS) runs and is available to all downstream logs.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,  # explicit origins, never "*"
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(RequestContextMiddleware)

    # Operational endpoints (health, ...) stay unversioned; feature API is /v1.
    app.include_router(root_router)
    app.include_router(api_router)

    logger.info(
        "application configured",
        extra={"environment": settings.environment, "log_json": settings.log_json},
    )
    return app


app = create_app()
