"""Tests for the search router (app/routers/search.py).

The router's own responsibilities are what's covered here: it requires auth,
validates the request body, scopes the search to the caller, and shapes the
response. The vector search itself is not executed — a local fake collection
returns canned rows (ranking is proven by the Atlas Local integration test),
because Atlas ``$vectorSearch`` can't be faithfully emulated in-memory.
"""

from datetime import UTC, datetime, timedelta

import jwt
import pytest
import pytest_asyncio
from bson import ObjectId

from app.core.config import get_settings
from app.db.mongodb import get_database
from app.dependencies import get_embedder
from app.main import app
from app.models.chunk import COLLECTION_NAME as CHUNKS_COLLECTION
from tests.conftest import FakeDatabase, FakeEmbedder

SEARCH_URL = "/v1/search"
USER_A = "507f1f77bcf86cd799439011"


def _token(user_id: str) -> str:
    settings = get_settings()
    payload = {
        "sub": "tester",
        "id": user_id,
        "role": "user",
        "jti": f"jti-{user_id}",
        "exp": datetime.now(UTC) + timedelta(minutes=5),
    }
    return jwt.encode(payload, settings.secret_key, algorithm=settings.jwt_algorithm)


def _auth(user_id: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {_token(user_id)}"}


class _CannedCursor:
    def __init__(self, rows):
        self._rows = rows

    async def to_list(self, length=None):
        return self._rows[:length] if length is not None else list(self._rows)


class _CannedChunks:
    """Returns fixed rows for any aggregate, and records the pipeline it was
    given so a test can assert the tenant filter — it does NOT rank."""

    def __init__(self, rows):
        self._rows = rows
        self.pipeline = None

    async def aggregate(self, pipeline):
        self.pipeline = pipeline
        return _CannedCursor(self._rows)


class _CannedDb:
    """Stubs only the chunks collection; every other collection (e.g. the
    revoked_tokens lookup auth performs) delegates to a real in-memory fake, so
    replacing the DB doesn't break the auth path."""

    def __init__(self, chunks):
        self._chunks = chunks
        self._rest = FakeDatabase()

    def __getitem__(self, name):
        if name == CHUNKS_COLLECTION:
            return self._chunks
        return self._rest[name]


@pytest_asyncio.fixture
async def search_client(client):
    """The conftest client with the chunks collection returning one canned row and
    a fake embedder, so a search reaches the router without Atlas or Voyage. The
    fake collection is exposed on the client so tests can read the pipeline back."""
    chunks = _CannedChunks(
        [
            {
                "_id": ObjectId(),
                "document_id": ObjectId(),
                "chunk_index": 0,
                "text": "a matching chunk",
                "score": 0.91,
            }
        ]
    )
    app.dependency_overrides[get_database] = lambda: _CannedDb(chunks)
    app.dependency_overrides[get_embedder] = FakeEmbedder
    client.chunks = chunks  # let tests inspect the recorded pipeline
    yield client
    # client's own teardown clears every override.


async def test_search_requires_auth(client):
    response = await client.post(SEARCH_URL, json={"query": "anything"})

    assert response.status_code == 401


async def test_empty_query_is_rejected(search_client):
    """min_length=1 on the query means a blank search is a 422, not an empty
    vector search."""
    response = await search_client.post(
        SEARCH_URL, json={"query": ""}, headers=_auth(USER_A)
    )

    assert response.status_code == 422


async def test_search_returns_ranked_results(search_client):
    response = await search_client.post(
        SEARCH_URL, json={"query": "what is the policy?"}, headers=_auth(USER_A)
    )

    assert response.status_code == 200
    body = response.json()
    assert body["query"] == "what is the policy?"
    assert len(body["results"]) == 1
    result = body["results"][0]
    assert result["text"] == "a matching chunk"
    assert result["chunk_index"] == 0
    assert result["score"] == 0.91


async def test_search_is_scoped_to_the_caller(search_client):
    """The pipeline the router runs must filter to the token's user, so a query
    can't reach another tenant's chunks."""
    await search_client.post(
        SEARCH_URL, json={"query": "x"}, headers=_auth(USER_A)
    )

    vector_stage = search_client.chunks.pipeline[0]["$vectorSearch"]
    assert vector_stage["filter"] == {"user_id": ObjectId(USER_A)}


@pytest.mark.parametrize("limit", [0, -1])
async def test_non_positive_limit_is_rejected(search_client, limit):
    """ge=1 on limit rejects a nonsensical count at the HTTP boundary."""
    response = await search_client.post(
        SEARCH_URL, json={"query": "x", "limit": limit}, headers=_auth(USER_A)
    )

    assert response.status_code == 422


async def test_search_scoped_to_a_document(search_client):
    """A document_id in the body narrows the pipeline filter to that document,
    combined with the caller's tenant scope."""
    document_id = str(ObjectId())

    response = await search_client.post(
        SEARCH_URL,
        json={"query": "x", "document_id": document_id},
        headers=_auth(USER_A),
    )

    assert response.status_code == 200
    vector_stage = search_client.chunks.pipeline[0]["$vectorSearch"]
    assert vector_stage["filter"] == {
        "user_id": ObjectId(USER_A),
        "document_id": ObjectId(document_id),
    }


async def test_malformed_document_id_is_rejected(search_client):
    """A document_id that isn't a valid ObjectId is a 422 at the boundary, never a
    500 from ObjectId(...) raising in the service."""
    response = await search_client.post(
        SEARCH_URL,
        json={"query": "x", "document_id": "not-an-object-id"},
        headers=_auth(USER_A),
    )

    assert response.status_code == 422
