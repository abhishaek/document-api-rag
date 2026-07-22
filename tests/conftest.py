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

    def __init__(
        self,
        inserted_id=None,
        deleted_count=0,
        modified_count=0,
        inserted_ids=None,
    ) -> None:
        self.inserted_id = inserted_id
        self.deleted_count = deleted_count
        self.modified_count = modified_count
        self.inserted_ids = inserted_ids or []


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
                    if op == "$in" and actual not in operand:
                        return False
            elif actual != condition:
                return False
        return True

    async def find_one(self, query: dict) -> dict | None:
        for doc in self.docs:
            if self._matches(doc, query):
                return doc
        return None

    def find(self, query: dict) -> "FakeCursor":
        """Return a cursor over every matching doc (list_documents uses this)."""
        return FakeCursor([doc for doc in self.docs if self._matches(doc, query)])

    async def delete_one(self, query: dict):
        for i, doc in enumerate(self.docs):
            if self._matches(doc, query):
                del self.docs[i]
                return _SimpleResult(deleted_count=1)
        return _SimpleResult(deleted_count=0)

    async def update_one(self, query: dict, update: dict):
        """Apply ``$set`` and/or ``$inc`` to the first matching doc.

        Just the operators the services use (status transitions and the attempts
        counter). Extend here as more are needed."""
        for doc in self.docs:
            if self._matches(doc, query):
                if "$set" in update:
                    doc.update(update["$set"])
                for field, amount in update.get("$inc", {}).items():
                    doc[field] = doc.get(field, 0) + amount
                return _SimpleResult(modified_count=1)
        return _SimpleResult(modified_count=0)

    async def insert_many(self, documents: list[dict]):
        """Insert each document, reusing insert_one (so unique-index emulation and
        _id assignment apply). Returns the generated ids."""
        ids = [(await self.insert_one(doc)).inserted_id for doc in documents]
        return _SimpleResult(inserted_ids=ids)

    async def delete_many(self, query: dict):
        """Remove every matching doc; returns how many were deleted."""
        before = len(self.docs)
        self.docs = [doc for doc in self.docs if not self._matches(doc, query)]
        return _SimpleResult(deleted_count=before - len(self.docs))

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
        # Compound unique index (user_id, content_hash) on the documents
        # collection — what makes a re-upload idempotent. Emulated here so router
        # tests exercise the real DuplicateKeyError path rather than silently
        # inserting a second record.
        if "user_id" in document and "content_hash" in document:
            for existing in self.docs:
                if (
                    existing.get("user_id") == document["user_id"]
                    and existing.get("content_hash") == document["content_hash"]
                ):
                    raise DuplicateKeyError(
                        "E11000 duplicate key error",
                        11000,
                        {"keyPattern": {"user_id": 1, "content_hash": 1}},
                    )
        # Use a real ObjectId (like the driver) so str(_id) -> ObjectId(str)
        # roundtrips the way get_user_by_id relies on.
        inserted_id = document.get("_id", ObjectId())
        self.docs.append({**document, "_id": inserted_id})
        return _SimpleResult(inserted_id=inserted_id)


class FakeCursor:
    """Minimal async stand-in for a pymongo cursor: sort() then to_list().

    Only the surface list_documents uses. sort() reorders in place and returns
    self so the fluent ``find(...).sort(...)`` chain works.
    """

    def __init__(self, docs: list[dict]) -> None:
        self._docs = docs

    def sort(self, field: str, direction: int) -> "FakeCursor":
        self._docs.sort(key=lambda d: d.get(field), reverse=direction < 0)
        return self

    async def to_list(self, length: int | None = None) -> list[dict]:
        return self._docs[:length] if length is not None else list(self._docs)


class FakeDatabase:
    """Async stand-in for a pymongo database: db[name] -> FakeCollection."""

    def __init__(self) -> None:
        self._collections: dict[str, FakeCollection] = {}

    def __getitem__(self, name: str) -> FakeCollection:
        return self._collections.setdefault(name, FakeCollection())


class FakeEmbedder:
    """Deterministic ``Embedder`` double: a fixed-width vector per text, no API.

    The vector is derived from the text length so different chunks get different
    vectors, and records each call so tests can assert what was embedded.
    """

    def __init__(self, dimensions: int = 8) -> None:
        self.dimensions = dimensions
        self.calls: list[list[str]] = []

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return [[float(len(t))] * self.dimensions for t in texts]


@pytest.fixture
def fake_embedder() -> FakeEmbedder:
    """A deterministic embedder that never touches the Voyage API."""
    return FakeEmbedder()


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
