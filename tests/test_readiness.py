"""Readiness probe behavior.

Under TestClient the app's lifespan does not run, so MongoDB is never connected.
The readiness probe must therefore report "not ready" (503), while liveness
(/health) stays 200 because it does not depend on external services.
"""

import pytest
from fastapi.testclient import TestClient

from app.db import mongodb
from app.main import app

client = TestClient(app)


def test_liveness_is_ok_without_db():
    assert client.get("/health").status_code == 200


def test_readiness_is_200_when_db_reachable(monkeypatch: pytest.MonkeyPatch):
    """The 503 path is covered below; this pins the healthy path so a probe that
    always reports "not ready" would fail rather than pass silently."""

    async def fake_ping() -> bool:
        return True

    monkeypatch.setattr(mongodb, "ping", fake_ping)

    response = client.get("/ready")

    assert response.status_code == 200
    body = response.json()
    assert body["ready"] is True
    assert body["checks"]["mongodb"] == "ok"


def test_readiness_is_503_when_db_unavailable():
    response = client.get("/ready")

    assert response.status_code == 503
    body = response.json()
    assert body["ready"] is False
    assert body["checks"]["mongodb"] == "unavailable"
