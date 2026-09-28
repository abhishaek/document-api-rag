"""Tests for the two chunks search indexes (app/db/mongodb).

Two Atlas Search indexes back retrieval: the vector one over ``embedding`` (dense
leg) and the lexical one over ``text`` (BM25 leg of hybrid search). Both go through
the same ``_ensure_search_index`` machinery, so both are covered here.

The Atlas Search index API needs a real Atlas server, which unit tests don't have,
so a fake collection stands in. What's covered is *our* logic: build the right
definition, create an index only when missing, rebuild it when the definition
drifted, and never crash startup when the server doesn't support Atlas Search.
"""

from pymongo.errors import OperationFailure

from app.db.mongodb import (
    _ensure_text_index,
    _ensure_vector_index,
    _text_index_matches,
    _vector_index_matches,
)
from app.models.chunk import (
    TEXT_INDEX_NAME,
    VECTOR_INDEX_NAME,
    build_text_index_definition,
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
            },
            {
                "type": "filter",
                "path": "user_id",
            },
            {
                "type": "filter",
                "path": "document_id",
            },
        ]
    }


# The dimensions _ensure_vector_index reads from settings in the test env.
_CURRENT_DEF = build_vector_index_definition(1024)
# A pre-drift definition: the vector field only, before the filter fields were
# added. Standing in for an index created by an older version of the code.
_STALE_DEF = {
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
    """Async-iterable of existing search indexes: name -> live definition."""

    def __init__(self, indexes):
        self._indexes = indexes

    def __aiter__(self):
        return self._agen()

    async def _agen(self):
        for name, definition in self._indexes.items():
            yield {"name": name, "latestDefinition": definition}


class _FakeVectorCollection:
    def __init__(self, existing=None, list_raises=None):
        # existing maps an index name to its current definition.
        self._existing = dict(existing or {})
        self._list_raises = list_raises
        self.created = []
        self.dropped = []

    async def list_search_indexes(self):
        # A coroutine returning the cursor — matches pymongo's async API, where a
        # server without Atlas Search raises here, at the command, not on iteration.
        if self._list_raises:
            raise self._list_raises
        return _FakeSearchCursor(self._existing)

    async def create_search_index(self, model):
        self.created.append(model)

    async def drop_search_index(self, name):
        self.dropped.append(name)
        # The drop settles immediately in the fake, so the recreate wait returns
        # at once rather than sleeping.
        self._existing.pop(name, None)


class _FakeDb:
    def __init__(self, collection):
        self._collection = collection

    def __getitem__(self, _name):
        return self._collection


def test_vector_index_matches_ignores_extra_keys():
    """A definition Atlas echoes back with extra default keys still counts as a
    match, so a healthy index isn't rebuilt on every startup."""
    echoed = build_vector_index_definition(1024)
    echoed["fields"][0]["quantization"] = "none"  # a default Atlas may add

    assert _vector_index_matches(echoed, build_vector_index_definition(1024))


def test_vector_index_matches_detects_missing_filter():
    """Dropping a filter field is drift that must be detected."""
    assert not _vector_index_matches(_STALE_DEF, build_vector_index_definition(1024))


async def test_creates_index_when_missing():
    collection = _FakeVectorCollection(existing={})

    await _ensure_vector_index(_FakeDb(collection))

    assert len(collection.created) == 1
    assert collection.dropped == []


async def test_skips_when_definition_matches():
    collection = _FakeVectorCollection(existing={VECTOR_INDEX_NAME: _CURRENT_DEF})

    await _ensure_vector_index(_FakeDb(collection))

    assert collection.created == []
    assert collection.dropped == []


async def test_recreates_when_definition_drifted():
    """An index left by an older code version (no filter fields) is dropped and
    recreated with the current definition — the fix for the manual drop/recreate
    this used to require. update_search_index isn't used: it's rejected for
    vectorSearch indexes on some Atlas builds."""
    collection = _FakeVectorCollection(existing={VECTOR_INDEX_NAME: _STALE_DEF})

    await _ensure_vector_index(_FakeDb(collection))

    assert collection.dropped == [VECTOR_INDEX_NAME]
    assert len(collection.created) == 1
    model = collection.created[0]
    assert model.document["name"] == VECTOR_INDEX_NAME
    assert model.document["definition"] == _CURRENT_DEF
    assert model.document["type"] == "vectorSearch"


async def test_best_effort_when_server_lacks_vector_search():
    """A server without Atlas Search raises when listing indexes; startup must
    survive it (warn and move on), not crash."""
    collection = _FakeVectorCollection(list_raises=OperationFailure("no such command"))

    await _ensure_vector_index(_FakeDb(collection))  # must not raise

    assert collection.created == []
    assert collection.dropped == []


# --- The lexical (BM25) index backing hybrid search ---------------------------

_TEXT_DEF = build_text_index_definition()
# A pre-drift definition: an index built before document_id was a filterable
# field, so a single-document hybrid search couldn't be scoped inside the search.
_STALE_TEXT_DEF = {
    "mappings": {
        "dynamic": False,
        "fields": {
            "text": {"type": "string", "analyzer": "lucene.standard"},
            "user_id": {"type": "objectId"},
        },
    }
}


def test_build_text_index_definition():
    """`text` is analyzed for BM25; the two scope fields are objectId so they can
    be used in a compound.filter `equals` clause *inside* the search."""
    assert _TEXT_DEF == {
        "mappings": {
            "dynamic": False,
            "fields": {
                "text": {"type": "string", "analyzer": "lucene.standard"},
                "user_id": {"type": "objectId"},
                "document_id": {"type": "objectId"},
            },
        }
    }


def test_text_index_does_not_map_the_embedding():
    """dynamic:False is load-bearing — dynamic mapping would index the 1024-float
    embedding into a lexical index that can never match on it."""
    mappings = _TEXT_DEF["mappings"]

    assert mappings["dynamic"] is False
    assert "embedding" not in mappings["fields"]


def test_text_index_matches_ignores_atlas_defaults():
    """Atlas echoes a string field back stuffed with its own defaults. If those
    counted as drift, a healthy index would be dropped and rebuilt every startup."""
    echoed = build_text_index_definition()
    echoed["mappings"]["fields"]["text"].update({"indexOptions": "offsets", "norms": "include"})

    assert _text_index_matches(echoed, _TEXT_DEF)


def test_text_index_matches_detects_missing_field():
    """Dropping a mapped field is real drift and must be detected."""
    assert not _text_index_matches(_STALE_TEXT_DEF, _TEXT_DEF)


def test_text_index_matches_detects_changed_analyzer():
    """Swapping the analyzer changes what a query matches, so it counts as drift
    even though the field names are identical."""
    changed = build_text_index_definition()
    changed["mappings"]["fields"]["text"]["analyzer"] = "lucene.english"

    assert not _text_index_matches(changed, _TEXT_DEF)


async def test_creates_text_index_when_missing():
    collection = _FakeVectorCollection(existing={})

    await _ensure_text_index(_FakeDb(collection))

    assert len(collection.created) == 1
    model = collection.created[0]
    assert model.document["name"] == TEXT_INDEX_NAME
    # type "search", not "vectorSearch" — a different Atlas index kind entirely.
    assert model.document["type"] == "search"
    assert model.document["definition"] == _TEXT_DEF


async def test_skips_text_index_when_definition_matches():
    collection = _FakeVectorCollection(existing={TEXT_INDEX_NAME: _TEXT_DEF})

    await _ensure_text_index(_FakeDb(collection))

    assert collection.created == []
    assert collection.dropped == []


async def test_recreates_text_index_when_definition_drifted():
    collection = _FakeVectorCollection(existing={TEXT_INDEX_NAME: _STALE_TEXT_DEF})

    await _ensure_text_index(_FakeDb(collection))

    assert collection.dropped == [TEXT_INDEX_NAME]
    assert collection.created[0].document["definition"] == _TEXT_DEF


async def test_text_index_best_effort_when_server_lacks_atlas_search():
    """Same contract as the vector index: a server without Atlas Search must not
    take startup down — the app still ingests, only search is unavailable."""
    collection = _FakeVectorCollection(list_raises=OperationFailure("no such command"))

    await _ensure_text_index(_FakeDb(collection))  # must not raise

    assert collection.created == []


async def test_the_two_indexes_are_independent():
    """An existing vector index doesn't satisfy the text index (or vice versa) —
    they're separate Atlas indexes and hybrid search needs both present."""
    collection = _FakeVectorCollection(existing={VECTOR_INDEX_NAME: _CURRENT_DEF})

    await _ensure_vector_index(_FakeDb(collection))
    await _ensure_text_index(_FakeDb(collection))

    created = [m.document["name"] for m in collection.created]
    assert created == [TEXT_INDEX_NAME]
