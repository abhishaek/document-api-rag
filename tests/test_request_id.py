"""Verify request-id correlation middleware."""

from fastapi.testclient import TestClient

from app.main import app
from app.middleware import REQUEST_ID_HEADER

client = TestClient(app)


def test_response_includes_generated_request_id():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.headers.get(REQUEST_ID_HEADER)  # non-empty id was generated


def test_incoming_request_id_is_preserved():
    response = client.get("/health", headers={REQUEST_ID_HEADER: "trace-123"})

    assert response.headers.get(REQUEST_ID_HEADER) == "trace-123"
