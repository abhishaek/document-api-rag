"""Shared pytest fixtures.

The auth endpoints depend on MongoDB and on the slowapi rate limiter. Rather
than spin up a real database, these fixtures inject a lightweight in-memory
fake via FastAPI's dependency overrides, and disable rate limiting so repeated
requests across the suite don't trip the per-IP limit.
"""

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from pymongo.errors import DuplicateKeyError

from app.core.rate_limit import limiter
from app.db.mongodb import get_database
from app.main import app


class FakeCollection:
    """Minimal async stand-in for a pymongo collection.

    Implements just enough for user_service.create_user: insert_one, plus
    emulation of the unique indexes on ``email`` and ``username`` (raising the
    same DuplicateKeyError the real driver would).
    """

    def __init__(self) -> None:
        self.docs: list[dict] = []
        self._counter = 0

    async def insert_one(self, document: dict):
        for existing in self.docs:
            for field in ("email", "username"):
                if existing[field] == document[field]:
                    raise DuplicateKeyError(
                        "E11000 duplicate key error",
                        11000,
                        {"keyPattern": {field: 1}},
                    )
        self._counter += 1
        inserted_id = f"fakeid{self._counter}"
        self.docs.append({**document, "_id": inserted_id})

        class _Result:
            pass

        result = _Result()
        result.inserted_id = inserted_id
        return result


class FakeDatabase:
    """Async stand-in for a pymongo database: db[name] -> FakeCollection."""

    def __init__(self) -> None:
        self._collections: dict[str, FakeCollection] = {}

    def __getitem__(self, name: str) -> FakeCollection:
        return self._collections.setdefault(name, FakeCollection())


@pytest.fixture
def fake_db() -> FakeDatabase:
    """A fresh in-memory database per test (no state leaks between tests)."""
    return FakeDatabase()


@pytest.fixture(autouse=True)
def disable_rate_limiting():
    """Disable slowapi limits for the duration of a test, then restore them."""
    limiter.enabled = False
    yield
    limiter.enabled = True


@pytest_asyncio.fixture
async def client(fake_db: FakeDatabase):
    """Async httpx client wired to the app, with the DB dependency overridden
    to use the in-memory fake."""
    app.dependency_overrides[get_database] = lambda: fake_db
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac
    app.dependency_overrides.clear()
