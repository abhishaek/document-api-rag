"""Tests for the chunks vector search index (app/db/mongodb._ensure_vector_index).

The Atlas Search index API needs a real Atlas server, which unit tests don't have,
so a fake collection stands in. What's covered is *our* logic: build the right
definition, create the index only when missing, and never crash startup when the
server doesn't support vector search.
"""

from pymongo.errors import OperationFailure

from app.db.mongodb import _ensure_vector_index
from app.models.chunk import (
    VECTOR_INDEX_NAME,
    build_vector_index_definition,
)


def test_build_vector_index_definition():
    definition = build_vector_index_definition(1024)

    assert definition == {
        "fields": [
            {
                "type": "vector",
                "path": "embedding",
                "numDimensions": 1024,
                "similarity": "cosine",
            }
        ]
    }


class _FakeSearchCursor:
    """Async-iterable list of existing search indexes."""

    def __init__(self, names):
        self._names = names

    def __aiter__(self):
        return self._agen()

    async def _agen(self):
        for name in self._names:
            yield {"name": name}


class _FakeVectorCollection:
    def __init__(self, existing=(), list_raises=None):
        self._existing = list(existing)
        self._list_raises = list_raises
        self.created = []

    async def list_search_indexes(self):
        # A coroutine returning the cursor — matches pymongo's async API, where a
        # server without Atlas Search raises here, at the command, not on iteration.
        if self._list_raises:
            raise self._list_raises
        return _FakeSearchCursor(self._existing)

    async def create_search_index(self, model):
        self.created.append(model)


class _FakeDb:
    def __init__(self, collection):
        self._collection = collection

    def __getitem__(self, _name):
        return self._collection


async def test_creates_index_when_missing():
    collection = _FakeVectorCollection(existing=[])

    await _ensure_vector_index(_FakeDb(collection))

    assert len(collection.created) == 1


async def test_skips_when_index_already_exists():
    collection = _FakeVectorCollection(existing=[VECTOR_INDEX_NAME])

    await _ensure_vector_index(_FakeDb(collection))

    assert collection.created == []


async def test_best_effort_when_server_lacks_vector_search():
    """A server without Atlas Search raises when listing indexes; startup must
    survive it (warn and move on), not crash."""
    collection = _FakeVectorCollection(list_raises=OperationFailure("no such command"))

    await _ensure_vector_index(_FakeDb(collection))  # must not raise

    assert collection.created == []
