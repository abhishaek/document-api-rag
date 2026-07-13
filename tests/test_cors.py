"""Verify CORS is configured for allowed origins and rejects unknown ones."""

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_allowed_origin_gets_cors_header():
    response = client.get("/health", headers={"Origin": "http://localhost:3000"})

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://localhost:3000"


def test_unknown_origin_is_not_allowed():
    response = client.get("/health", headers={"Origin": "http://evil.example.com"})

    # Request still succeeds server-side, but the browser gets no allow header,
    # so it will block the response.
    assert response.status_code == 200
    assert "access-control-allow-origin" not in response.headers
