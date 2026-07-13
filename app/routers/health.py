"""Health check endpoints used by load balancers, orchestrators, and uptime checks."""

import logging

from fastapi import APIRouter, Response, status

from app.core.config import get_settings
from app.db import mongodb
from app.schemas import HealthResponse, ReadinessResponse

logger = logging.getLogger(__name__)

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse, status_code=status.HTTP_200_OK)
def health() -> HealthResponse:
    """Liveness probe: returns 200 if the process is up.

    Deliberately does NOT touch external dependencies — an orchestrator uses
    this to decide whether to restart the container, and a transient DB blip
    should not cause a restart loop.
    """
    logger.debug("health check requested")
    settings = get_settings()
    return HealthResponse(
        status="Ok. Application is Running.",
        app=settings.app_name,
        environment=settings.environment,
    )


@router.get("/ready", response_model=ReadinessResponse, status_code=status.HTTP_200_OK)
async def ready(response: Response) -> ReadinessResponse:
    """Readiness probe: returns 200 only if dependencies (MongoDB) are reachable.

    An orchestrator uses this to decide whether to route traffic to the pod.
    Returns 503 when a dependency is down so traffic is withheld until healthy.
    """
    mongo_ok = await mongodb.ping()
    if not mongo_ok:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return ReadinessResponse(
        ready=mongo_ok,
        checks={"mongodb": "ok" if mongo_ok else "unavailable"},
    )
