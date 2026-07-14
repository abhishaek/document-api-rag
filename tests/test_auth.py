"""Tests for the auth router (app/routers/auth.py).

Covers POST /v1/auth/register: the happy path, response shape, email
normalization, duplicate handling, and input validation. Uses the async httpx
client and in-memory fake database from conftest.py.
"""

REGISTER_URL = "/v1/auth/register"


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
