"""Readiness probe behavior.

Under TestClient the app's lifespan does not run, so MongoDB is never connected.
The readiness probe must therefore report "not ready" (503), while liveness
(/health) stays 200 because it does not depend on external services.
"""

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_liveness_is_ok_without_db():
    assert client.get("/health").status_code == 200


def test_readiness_is_503_when_db_unavailable():
    response = client.get("/ready")

    assert response.status_code == 503
    body = response.json()
    assert body["ready"] is False
    assert body["checks"]["mongodb"] == "unavailable"
