"""Rate limiting on the auth routes (slowapi, per client IP).

The rest of the suite runs with limits disabled via the autouse fixture in
conftest.py, so nothing else asserts that the limits actually fire. These tests
re-enable them by shadowing that fixture, and reset the limiter's in-memory
store around each test so counts never leak between them.

Limits under test come from the decorators in app/routers/auth.py:
register 3/minute, login 5/minute.
"""

import pytest

from app.core.rate_limit import limiter

REGISTER_URL = "/v1/auth/register"
LOGIN_URL = "/v1/auth/login"


@pytest.fixture(autouse=True)
def disable_rate_limiting():
    """Shadow conftest's autouse fixture of the same name, which turns limits off.

    These tests need them on. The store is reset on both sides so neither this
    module nor the rest of the suite inherits a partially-consumed budget.
    """
    limiter.reset()
    limiter.enabled = True
    yield
    limiter.reset()


def _payload(n: int) -> dict:
    """A distinct, valid registration payload, so 4xx can only come from the
    limiter and never from a duplicate-user conflict."""
    return {
        "email": f"user{n}@example.com",
        "username": f"user{n}",
        "password": "supersecret",
    }


async def test_register_allows_three_requests_per_minute(client):
    codes = [
        (await client.post(REGISTER_URL, json=_payload(i))).status_code
        for i in range(3)
    ]

    assert codes == [201, 201, 201]


async def test_register_returns_429_on_the_fourth_request(client):
    for i in range(3):
        await client.post(REGISTER_URL, json=_payload(i))

    response = await client.post(REGISTER_URL, json=_payload(99))

    assert response.status_code == 429
    assert "rate limit exceeded" in response.json()["error"].lower()


async def test_login_returns_429_after_five_attempts(client):
    """The login limit (5/minute) is what bounds password guessing, so it must
    fire on wrong passwords too, not only on successful logins."""
    await client.post(REGISTER_URL, json=_payload(0))

    codes = [
        (
            await client.post(
                LOGIN_URL, data={"username": "user0", "password": "wrong-password"}
            )
        ).status_code
        for _ in range(6)
    ]

    assert codes[:5] == [401] * 5
    assert codes[5] == 429


async def test_register_and_login_limits_are_counted_separately(client):
    """Each endpoint has its own budget: exhausting register must not lock out
    login."""
    for i in range(4):
        await client.post(REGISTER_URL, json=_payload(i))

    response = await client.post(
        LOGIN_URL, data={"username": "user0", "password": "supersecret"}
    )

    assert response.status_code == 200
