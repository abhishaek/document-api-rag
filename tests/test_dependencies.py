"""Tests for get_current_user (app/dependencies.py).

get_current_user gates every authenticated request and rejects for five distinct
reasons, each with its own message. The auth-route tests cover only the
missing-header case, so these exercise the remaining branches directly by
signing tokens with the app's real secret and varying one thing at a time.

/v1/auth/logout is the probe: it is currently the only route behind
UserDependency.
"""

from datetime import UTC, datetime, timedelta

import jwt

from app.core.config import get_settings

LOGOUT_URL = "/v1/auth/logout"


def _token(**overrides) -> str:
    """Sign a valid token with the app secret. Pass a claim as None to omit it."""
    settings = get_settings()
    payload = {
        "sub": "alice",
        "id": "507f1f77bcf86cd799439011",
        "role": "user",
        "jti": "test-jti",
        "exp": datetime.now(UTC) + timedelta(minutes=5),
    }
    payload.update(overrides)
    payload = {k: v for k, v in payload.items() if v is not None}
    return jwt.encode(payload, settings.secret_key, algorithm=settings.jwt_algorithm)


async def _logout_with(client, token: str):
    return await client.post(
        LOGOUT_URL,
        json={"refresh_token": "irrelevant-for-these-tests"},
        headers={"Authorization": f"Bearer {token}"},
    )


async def test_valid_token_is_accepted(client):
    response = await _logout_with(client, _token())

    assert response.status_code == 204


async def test_expired_token_is_rejected(client):
    response = await _logout_with(
        client, _token(exp=datetime.now(UTC) - timedelta(seconds=1))
    )

    assert response.status_code == 401
    assert "expired" in response.json()["detail"].lower()


async def test_malformed_token_is_rejected(client):
    response = await _logout_with(client, "this-is-not-a-jwt")

    assert response.status_code == 401
    assert response.json()["detail"] == "Could not validate credentials."


async def test_token_signed_with_wrong_secret_is_rejected(client):
    """A well-formed token with a valid-looking payload must still fail if it was
    not signed by us — otherwise anyone could mint their own."""
    settings = get_settings()
    forged = jwt.encode(
        {
            "sub": "alice",
            "id": "507f1f77bcf86cd799439011",
            "role": "admin",
            "jti": "forged",
            "exp": datetime.now(UTC) + timedelta(minutes=5),
        },
        "an-entirely-different-secret-not-known-to-the-app",
        algorithm=settings.jwt_algorithm,
    )

    response = await _logout_with(client, forged)

    assert response.status_code == 401
    assert response.json()["detail"] == "Could not validate credentials."


async def test_token_missing_sub_claim_is_rejected(client):
    response = await _logout_with(client, _token(sub=None))

    assert response.status_code == 401


async def test_token_missing_id_claim_is_rejected(client):
    response = await _logout_with(client, _token(id=None))

    assert response.status_code == 401
