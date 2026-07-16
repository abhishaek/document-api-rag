"""Shared pytest fixtures.

The auth endpoints depend on MongoDB and on the slowapi rate limiter. Rather
than spin up a real database, these fixtures inject a lightweight in-memory
fake via FastAPI's dependency overrides, and disable rate limiting so repeated
requests across the suite don't trip the per-IP limit.
"""

import pytest
import pytest_asyncio
from bson import ObjectId
from httpx import ASGITransport, AsyncClient
from pymongo.errors import DuplicateKeyError

from app.core.rate_limit import limiter
from app.db.mongodb import get_database
from app.main import app


class _SimpleResult:
    """Stand-in for pymongo's InsertOneResult / DeleteResult (just the
    attributes our code reads)."""

    def __init__(self, inserted_id=None, deleted_count=0) -> None:
        self.inserted_id = inserted_id
        self.deleted_count = deleted_count


class FakeCollection:
    """Minimal async stand-in for a pymongo collection.

    Implements just enough for the services under test: insert_one (with
    emulation of the unique indexes on ``email`` and ``username``, raising the
    same DuplicateKeyError the real driver would) and a find_one that matches
    documents by exact field equality.
    """

    def __init__(self) -> None:
        self.docs: list[dict] = []
        self._counter = 0

    @staticmethod
    def _matches(doc: dict, query: dict) -> bool:
        """Exact-equality match, plus the ``{"$gt": value}`` operator form used
        by the refresh-token lookup. Extend here as tests need more operators."""
        for field, condition in query.items():
            actual = doc.get(field)
            if isinstance(condition, dict):
                for op, operand in condition.items():
                    if op == "$gt" and not (actual is not None and actual > operand):
                        return False
                    if op == "$lt" and not (actual is not None and actual < operand):
                        return False
            elif actual != condition:
                return False
        return True

    async def find_one(self, query: dict) -> dict | None:
        for doc in self.docs:
            if self._matches(doc, query):
                return doc
        return None

    async def delete_one(self, query: dict):
        for i, doc in enumerate(self.docs):
            if self._matches(doc, query):
                del self.docs[i]
                return _SimpleResult(deleted_count=1)
        return _SimpleResult(deleted_count=0)

    async def insert_one(self, document: dict):
        # Emulate the unique indexes only for fields the document actually has,
        # so collections without email/username (e.g. refresh_tokens) work too.
        # Mirrors the real indexes: users.email, users.username,
        # refresh_tokens.token_hash, revoked_tokens.jti.
        for existing in self.docs:
            for field in ("email", "username", "token_hash", "jti"):
                if field in document and existing.get(field) == document[field]:
                    raise DuplicateKeyError(
                        "E11000 duplicate key error",
                        11000,
                        {"keyPattern": {field: 1}},
                    )
        # Use a real ObjectId (like the driver) so str(_id) -> ObjectId(str)
        # roundtrips the way get_user_by_id relies on.
        inserted_id = document.get("_id", ObjectId())
        self.docs.append({**document, "_id": inserted_id})
        return _SimpleResult(inserted_id=inserted_id)


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
