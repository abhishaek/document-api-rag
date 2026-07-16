"""Tests for the auth router (app/routers/auth.py).

Covers POST /v1/auth/register: the happy path, response shape, email
normalization, duplicate handling, and input validation. Uses the async httpx
client and in-memory fake database from conftest.py.
"""

REGISTER_URL = "/v1/auth/register"
LOGIN_URL = "/v1/auth/login"
REFRESH_URL = "/v1/auth/refresh"
LOGOUT_URL = "/v1/auth/logout"


async def register_and_login(client) -> dict:
    """Register the default user and log in, returning the token payload."""
    await client.post(REGISTER_URL, json=valid_payload())
    response = await client.post(
        LOGIN_URL, data={"username": "alice", "password": "supersecret"}
    )
    assert response.status_code == 200
    return response.json()


def valid_payload(**overrides) -> dict:
    payload = {
        "email": "alice@example.com",
        "username": "alice",
        "password": "supersecret",
    }
    payload.update(overrides)
    return payload


async def test_register_returns_201_on_valid_input(client):
    response = await client.post(REGISTER_URL, json=valid_payload())

    assert response.status_code == 201
    body = response.json()
    assert body["email"] == "alice@example.com"
    assert body["username"] == "alice"
    assert body["role"] == "user"
    assert body["is_active"] is True
    assert "id" in body


async def test_register_response_excludes_password(client):
    response = await client.post(REGISTER_URL, json=valid_payload())

    body = response.json()
    assert "password" not in body
    assert "hashed_password" not in body


async def test_register_lowercases_email(client):
    response = await client.post(
        REGISTER_URL, json=valid_payload(email="Alice@Example.com")
    )

    assert response.status_code == 201
    assert response.json()["email"] == "alice@example.com"


async def test_register_defaults_role_to_user(client):
    response = await client.post(REGISTER_URL, json=valid_payload())

    assert response.json()["role"] == "user"


async def test_register_accepts_admin_role(client):
    response = await client.post(REGISTER_URL, json=valid_payload(role="admin"))

    assert response.status_code == 201
    assert response.json()["role"] == "admin"


async def test_register_returns_400_on_duplicate_email(client):
    await client.post(REGISTER_URL, json=valid_payload(username="alice"))
    response = await client.post(
        REGISTER_URL, json=valid_payload(username="alice2")
    )

    assert response.status_code == 400


async def test_register_returns_400_on_duplicate_username(client):
    await client.post(REGISTER_URL, json=valid_payload(email="a@example.com"))
    response = await client.post(
        REGISTER_URL, json=valid_payload(email="b@example.com")
    )

    assert response.status_code == 400


async def test_register_returns_422_on_invalid_email(client):
    response = await client.post(
        REGISTER_URL, json=valid_payload(email="not-an-email")
    )

    assert response.status_code == 422


async def test_register_returns_422_on_short_username(client):
    response = await client.post(REGISTER_URL, json=valid_payload(username="ab"))

    assert response.status_code == 422


async def test_register_strips_surrounding_whitespace_from_username(client):
    response = await client.post(REGISTER_URL, json=valid_payload(username="  alice  "))

    assert response.status_code == 201
    assert response.json()["username"] == "alice"


async def test_register_checks_username_length_after_stripping(client):
    """"  ab  " is 6 characters as sent but 2 once stripped, so it must fail the
    3-character minimum rather than sneak past it."""
    response = await client.post(REGISTER_URL, json=valid_payload(username="  ab  "))

    assert response.status_code == 422


async def test_register_usernames_are_case_sensitive(client):
    """Deliberate: "Abhi" and "abhi" are distinct accounts. Contrast with email,
    which is normalized to lowercase — the two fields differ on purpose."""
    first = await client.post(
        REGISTER_URL, json=valid_payload(username="Abhi", email="a@example.com")
    )
    second = await client.post(
        REGISTER_URL, json=valid_payload(username="abhi", email="b@example.com")
    )

    assert first.status_code == 201
    assert second.status_code == 201
    assert first.json()["username"] == "Abhi"
    assert second.json()["username"] == "abhi"
    assert first.json()["id"] != second.json()["id"]


async def test_register_returns_422_on_short_password(client):
    response = await client.post(REGISTER_URL, json=valid_payload(password="short"))

    assert response.status_code == 422


async def test_register_returns_422_on_invalid_role(client):
    response = await client.post(
        REGISTER_URL, json=valid_payload(role="superadmin")
    )

    assert response.status_code == 422


async def test_register_returns_422_on_missing_fields(client):
    response = await client.post(REGISTER_URL, json={"email": "a@example.com"})

    assert response.status_code == 422


# --- POST /v1/auth/login (OAuth2 form: username + password) ---


async def test_login_returns_tokens_on_valid_credentials(client):
    await client.post(REGISTER_URL, json=valid_payload())

    response = await client.post(
        LOGIN_URL, data={"username": "alice", "password": "supersecret"}
    )

    assert response.status_code == 200
    body = response.json()
    assert body["token_type"] == "bearer"
    assert body["access_token"]
    assert body["refresh_token"]


async def test_login_returns_401_on_wrong_password(client):
    await client.post(REGISTER_URL, json=valid_payload())

    response = await client.post(
        LOGIN_URL, data={"username": "alice", "password": "wrongpass"}
    )

    assert response.status_code == 401


async def test_login_returns_401_on_unknown_user(client):
    response = await client.post(
        LOGIN_URL, data={"username": "ghost", "password": "whatever"}
    )

    assert response.status_code == 401


# --- POST /v1/auth/refresh ---


async def test_refresh_returns_new_tokens(client):
    tokens = await register_and_login(client)

    response = await client.post(
        REFRESH_URL, json={"refresh_token": tokens["refresh_token"]}
    )

    assert response.status_code == 200
    body = response.json()
    assert body["access_token"]
    assert body["refresh_token"]
    # Rotation: the new refresh token differs from the one we sent.
    assert body["refresh_token"] != tokens["refresh_token"]


async def test_refresh_invalidates_old_token(client):
    tokens = await register_and_login(client)

    # First refresh succeeds and rotates the token...
    await client.post(REFRESH_URL, json={"refresh_token": tokens["refresh_token"]})
    # ...so reusing the original refresh token now fails.
    second = await client.post(
        REFRESH_URL, json={"refresh_token": tokens["refresh_token"]}
    )

    assert second.status_code == 401


async def test_refresh_returns_401_on_unknown_token(client):
    response = await client.post(
        REFRESH_URL, json={"refresh_token": "not-a-real-token"}
    )

    assert response.status_code == 401


async def test_refresh_rejected_after_user_deactivated(client, fake_db):
    from app.models.user import COLLECTION_NAME as USERS_COLLECTION

    tokens = await register_and_login(client)

    # Simulate the account being deactivated after the token was issued.
    for doc in fake_db[USERS_COLLECTION].docs:
        doc["is_active"] = False

    response = await client.post(
        REFRESH_URL, json={"refresh_token": tokens["refresh_token"]}
    )
    assert response.status_code == 401

    # The refresh token was also revoked, so a retry still fails (no infinite
    # refresh loop for a disabled account).
    retry = await client.post(
        REFRESH_URL, json={"refresh_token": tokens["refresh_token"]}
    )
    assert retry.status_code == 401


# --- POST /v1/auth/logout ---


async def test_logout_revokes_refresh_token(client):
    tokens = await register_and_login(client)
    auth_header = {"Authorization": f"Bearer {tokens['access_token']}"}

    response = await client.post(
        LOGOUT_URL,
        json={"refresh_token": tokens["refresh_token"]},
        headers=auth_header,
    )

    assert response.status_code == 204
    # The refresh token is gone: it can no longer be exchanged.
    refresh = await client.post(
        REFRESH_URL, json={"refresh_token": tokens["refresh_token"]}
    )
    assert refresh.status_code == 401


async def test_logout_requires_authentication(client):
    tokens = await register_and_login(client)

    # No Authorization header → the UserDependency rejects the request.
    response = await client.post(
        LOGOUT_URL, json={"refresh_token": tokens["refresh_token"]}
    )

    assert response.status_code == 401


async def test_logout_revokes_access_token_immediately(client):
    tokens = await register_and_login(client)
    auth_header = {"Authorization": f"Bearer {tokens['access_token']}"}

    logout = await client.post(
        LOGOUT_URL,
        json={"refresh_token": tokens["refresh_token"]},
        headers=auth_header,
    )
    assert logout.status_code == 204

    # The same access token is now denylisted: reusing it on a protected route
    # (logout itself) is rejected before the handler runs.
    reused = await client.post(
        LOGOUT_URL,
        json={"refresh_token": "anything"},
        headers=auth_header,
    )
    assert reused.status_code == 401
