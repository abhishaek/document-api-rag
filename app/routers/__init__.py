"""Router aggregation.

Two top-level routers are exposed:

* ``root_router``  – unversioned operational endpoints (health, readiness,
  metrics). These live outside ``/v1`` so orchestrators and uptime checks have
  a stable path that never changes across API versions.
* ``api_router``   – the versioned feature API. Everything user-facing is
  mounted under ``/v1`` so we can ship ``/v2`` later without breaking clients.

To add a feature: create ``app/routers/<feature>.py`` with its own
``router = APIRouter()``, then include it below. ``app.main`` never changes.
"""

from fastapi import APIRouter

from app.routers import auth, health

# --- Unversioned operational endpoints ---
root_router = APIRouter()
root_router.include_router(health.router)

# --- Versioned feature API (everything user-facing) ---
# auth.router already carries its own "/auth" prefix, so it becomes /v1/auth/...
api_router = APIRouter(prefix="/v1")
api_router.include_router(auth.router)
# api_router.include_router(summary.router)       # -> /v1/summary/...
# api_router.include_router(documents.router)     # -> /v1/documents/...
# api_router.include_router(retrieval.router)     # -> /v1/retrieval/...
