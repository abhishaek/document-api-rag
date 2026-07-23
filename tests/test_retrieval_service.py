"""Tests for the retrieval service (app/services/retrieval_service.py).

Atlas ``$vectorSearch`` needs a real Atlas server, so these tests don't execute
a search. They cover *our* logic — the pipeline we build, the limit clamping, the
query-embedding call, and tenant scoping — with the actual ranking left to an
integration test against Atlas Local (test_search_integration.py).
"""

from bson import ObjectId

from app.core.config import get_settings
from app.models.chunk import VECTOR_INDEX_NAME
from app.services.retrieval_service import (
    _resolve_limit,
    build_vector_search_pipeline,
    search,
)
from tests.conftest import FakeEmbedder

settings = get_settings()


def test_pipeline_has_vector_search_and_projection():
    user_id = ObjectId()
    pipeline = build_vector_search_pipeline([0.1, 0.2, 0.3], user_id, limit=5)

    vector_stage = pipeline[0]["$vectorSearch"]
    assert vector_stage["index"] == VECTOR_INDEX_NAME
    assert vector_stage["path"] == "embedding"
    assert vector_stage["queryVector"] == [0.1, 0.2, 0.3]
    assert vector_stage["limit"] == 5
    # numCandidates is limit * the configured multiplier.
    assert vector_stage["numCandidates"] == 5 * settings.search_num_candidates_multiplier
    # The projection returns the score and omits the embedding.
    projection = pipeline[1]["$project"]
    assert projection["score"] == {"$meta": "vectorSearchScore"}
    assert "embedding" not in projection


def test_pipeline_filters_to_the_tenant():
    """The search must be scoped to one user inside $vectorSearch, so a query
    can never reach another tenant's chunks."""
    user_id = ObjectId()
    pipeline = build_vector_search_pipeline([0.0], user_id, limit=3)

    assert pipeline[0]["$vectorSearch"]["filter"] == {"user_id": user_id}


def test_pipeline_filters_to_a_document_when_given():
    """A document_id narrows the search to one document, combined with the tenant
    scope so it can't be used to reach another user's document."""
    user_id = ObjectId()
    document_id = ObjectId()
    pipeline = build_vector_search_pipeline(
        [0.0], user_id, limit=3, document_id=document_id
    )

    assert pipeline[0]["$vectorSearch"]["filter"] == {
        "user_id": user_id,
        "document_id": document_id,
    }


def test_pipeline_omits_document_filter_when_not_given():
    """Without a document_id the filter is user-only, so the search spans all of
    the caller's documents."""
    user_id = ObjectId()
    pipeline = build_vector_search_pipeline([0.0], user_id, limit=3)

    assert pipeline[0]["$vectorSearch"]["filter"] == {"user_id": user_id}
    assert "document_id" not in pipeline[0]["$vectorSearch"]["filter"]


def test_num_candidates_capped_at_atlas_ceiling():
    """A large limit can't push numCandidates past Atlas's 10,000 cap."""
    pipeline = build_vector_search_pipeline([0.0], ObjectId(), limit=10_000)

    assert pipeline[0]["$vectorSearch"]["numCandidates"] == 10_000


def test_resolve_limit_defaults_and_clamps():
    assert _resolve_limit(None) == settings.search_default_limit
    assert _resolve_limit(3) == 3
    # Above the ceiling clamps down; below 1 clamps up.
    assert _resolve_limit(9999) == settings.search_max_limit
    assert _resolve_limit(0) == 1


class _RecordingCursor:
    def __init__(self, rows):
        self._rows = rows

    async def to_list(self, length=None):
        return self._rows[:length] if length is not None else list(self._rows)


class _RecordingCollection:
    """Captures the aggregate pipeline and returns canned rows — stands in for
    the chunks collection so `search` can be tested without a real Atlas."""

    def __init__(self, rows):
        self._rows = rows
        self.pipeline = None

    async def aggregate(self, pipeline):
        self.pipeline = pipeline
        return _RecordingCursor(self._rows)


class _RecordingDb:
    def __init__(self, collection):
        self._collection = collection

    def __getitem__(self, _name):
        return self._collection


async def test_search_embeds_query_and_maps_rows():
    """search() must embed the query as a *query*, run the tenant-scoped pipeline,
    and map Mongo rows onto SearchResult."""
    doc_id = ObjectId()
    chunk_id = ObjectId()
    collection = _RecordingCollection(
        [
            {
                "_id": chunk_id,
                "document_id": doc_id,
                "chunk_index": 2,
                "text": "the answer",
                "score": 0.87,
            }
        ]
    )
    embedder = FakeEmbedder()
    user_id = str(ObjectId())

    results = await search(_RecordingDb(collection), embedder, user_id, "a question")

    # The query went through embed_query (not embed_documents).
    assert embedder.query_calls == ["a question"]
    assert embedder.calls == []
    # The pipeline was scoped to this tenant.
    assert collection.pipeline[0]["$vectorSearch"]["filter"] == {
        "user_id": ObjectId(user_id)
    }
    # Rows are mapped, ObjectIds stringified, score carried through.
    assert len(results) == 1
    assert results[0].id == str(chunk_id)
    assert results[0].document_id == str(doc_id)
    assert results[0].chunk_index == 2
    assert results[0].text == "the answer"
    assert results[0].score == 0.87


async def test_search_scopes_to_document_when_given():
    """A document_id string reaches the pipeline as an ObjectId filter alongside
    the tenant scope."""
    collection = _RecordingCollection([])
    embedder = FakeEmbedder()
    user_id = str(ObjectId())
    document_id = str(ObjectId())

    await search(
        _RecordingDb(collection),
        embedder,
        user_id,
        "a question",
        document_id=document_id,
    )

    assert collection.pipeline[0]["$vectorSearch"]["filter"] == {
        "user_id": ObjectId(user_id),
        "document_id": ObjectId(document_id),
    }
