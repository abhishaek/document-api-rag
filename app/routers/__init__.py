"""Router aggregation.

Two top-level routers are exposed:

* ``root_router``  – unversioned operational endpoints (health, readiness,
  metrics). These live outside ``/v1`` so orchestrators and uptime checks have
  a stable path that never changes across API versions.
* ``api_router``   – the versioned feature API. Everything user-facing is
  mounted under ``/v1`` so we can ship ``/v2`` later without breaking clients.

Route map — where each area lives (open the file for the individual endpoints,
or run the app and browse ``/docs`` for the live, per-tag list):

    Base path         Module                     Responsibility
    ----------------  -------------------------  ------------------------------
    /health, /ready   app/routers/health.py      Liveness / readiness probes
    /v1/auth/*        app/routers/auth.py        Register, login, refresh, logout
    /v1/documents/*   app/routers/documents.py   Upload, list, fetch documents

Keep this table at the domain level (one row per router), not per-endpoint — a
per-endpoint list would drift out of sync, whereas ``/docs`` already is the
always-accurate per-endpoint map.

To add a feature: create ``app/routers/<feature>.py`` with its own
``router = APIRouter()``, include it below, and add a row above. ``app.main``
never changes.
"""

from fastapi import APIRouter

from app.routers import auth, documents, health

# --- Unversioned operational endpoints ---
root_router = APIRouter()
root_router.include_router(health.router)

# --- Versioned feature API (everything user-facing) ---
# auth.router already carries its own "/auth" prefix, so it becomes /v1/auth/...
api_router = APIRouter(prefix="/v1")
api_router.include_router(auth.router)
# api_router.include_router(summary.router)       # -> /v1/summary/...
api_router.include_router(documents.router)     # -> /v1/documents/...
# api_router.include_router(retrieval.router)     # -> /v1/retrieval/...
