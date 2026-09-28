"""Tests for the retrieval service (app/services/retrieval_service.py).

Atlas ``$vectorSearch`` and ``$search`` need a real Atlas server, so these tests
don't execute a search. They cover *our* logic — the pipelines we build for each
mode, the RRF fusion stages, the limit/mode clamping, the query-embedding call,
and tenant scoping in every leg — with the actual ranking left to an integration
test against Atlas Local (test_search_integration.py).
"""

import pytest
from bson import ObjectId

from app.core.config import get_settings
from app.models.chunk import COLLECTION_NAME, TEXT_INDEX_NAME, VECTOR_INDEX_NAME
from app.services.retrieval_service import (
    _resolve_fetch_k,
    _resolve_limit,
    build_hybrid_search_pipeline,
    build_text_search_pipeline,
    build_vector_search_pipeline,
    resolve_mode,
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


# --- Lexical leg --------------------------------------------------------------


def test_text_pipeline_searches_and_limits():
    user_id = ObjectId()
    pipeline = build_text_search_pipeline("aws lambda", user_id, limit=5)

    search_stage = pipeline[0]["$search"]
    assert search_stage["index"] == TEXT_INDEX_NAME
    assert search_stage["compound"]["must"] == [
        {"text": {"query": "aws lambda", "path": "text"}}
    ]
    # $search has no limit option of its own, so the pipeline must apply one.
    assert pipeline[1] == {"$limit": 5}
    projection = pipeline[2]["$project"]
    assert projection["score"] == {"$meta": "searchScore"}
    assert "embedding" not in projection


def test_text_pipeline_scopes_inside_the_search_not_after_it():
    """The tenant scope must be a compound.filter clause. As a later $match it
    would run after $limit had already been spent and could drop the caller's own
    results — the same trap the vector leg's index filter fields exist to avoid."""
    user_id = ObjectId()
    pipeline = build_text_search_pipeline("q", user_id, limit=5)

    assert pipeline[0]["$search"]["compound"]["filter"] == [
        {"equals": {"path": "user_id", "value": user_id}}
    ]
    assert not any("$match" in stage for stage in pipeline)


def test_text_pipeline_scopes_to_a_document_when_given():
    user_id = ObjectId()
    document_id = ObjectId()
    pipeline = build_text_search_pipeline("q", user_id, limit=5, document_id=document_id)

    assert pipeline[0]["$search"]["compound"]["filter"] == [
        {"equals": {"path": "user_id", "value": user_id}},
        {"equals": {"path": "document_id", "value": document_id}},
    ]


# --- Hybrid (RRF) -------------------------------------------------------------


def _hybrid(user_id=None, limit=5, document_id=None):
    return build_hybrid_search_pipeline(
        [0.1, 0.2], "aws lambda", user_id or ObjectId(), limit, document_id
    )


def _stage(pipeline, name):
    """The first stage in `pipeline` with the given operator key."""
    return next(stage[name] for stage in pipeline if name in stage)


def test_hybrid_runs_both_legs():
    """The dense leg leads the pipeline (a search stage must come first) and the
    lexical leg is appended through $unionWith over the same collection — the only
    way to run two search stages in one aggregation."""
    pipeline = _hybrid()

    assert "$vectorSearch" in pipeline[0]
    union = _stage(pipeline, "$unionWith")
    assert union["coll"] == COLLECTION_NAME
    assert "$search" in union["pipeline"][0]


def test_hybrid_over_fetches_both_legs():
    """Fusion can only reorder what the legs handed it, so each leg fetches well
    past the final limit — a chunk outside both fetch windows can never surface."""
    settings_fetch = get_settings().search_fetch_multiplier
    pipeline = _hybrid(limit=5)

    fetch_k = 5 * settings_fetch
    assert pipeline[0]["$vectorSearch"]["limit"] == fetch_k
    assert {"$limit": fetch_k} in _stage(pipeline, "$unionWith")["pipeline"]
    # ...but only `limit` rows come back after fusion. (The outer pipeline's only
    # $limit is the final one — the fetch_k limits live inside the legs.)
    assert _stage(pipeline, "$limit") == 5


def test_hybrid_scopes_both_legs_to_the_tenant():
    """Tenant isolation has to hold in *each* leg. A leg left unscoped would pour
    another tenant's chunks straight into the fusion."""
    user_id = ObjectId()
    pipeline = _hybrid(user_id=user_id)

    assert pipeline[0]["$vectorSearch"]["filter"] == {"user_id": user_id}
    text_leg = _stage(pipeline, "$unionWith")["pipeline"]
    assert text_leg[0]["$search"]["compound"]["filter"] == [
        {"equals": {"path": "user_id", "value": user_id}}
    ]


def test_hybrid_scopes_both_legs_to_a_document():
    user_id = ObjectId()
    document_id = ObjectId()
    pipeline = _hybrid(user_id=user_id, document_id=document_id)

    assert pipeline[0]["$vectorSearch"]["filter"] == {
        "user_id": user_id,
        "document_id": document_id,
    }
    text_leg = _stage(pipeline, "$unionWith")["pipeline"]
    assert {"equals": {"path": "document_id", "value": document_id}} in (
        text_leg[0]["$search"]["compound"]["filter"]
    )


def test_hybrid_materializes_rank_in_each_leg():
    """There's no $rank operator, so each leg's rank is materialized by pushing the
    already-ordered rows into one array and unwinding it with includeArrayIndex."""
    pipeline = _hybrid()
    text_leg = _stage(pipeline, "$unionWith")["pipeline"]

    for leg in (pipeline, text_leg):
        assert {"$group": {"_id": None, "docs": {"$push": "$$ROOT"}}} in leg
        assert {"$unwind": {"path": "$docs", "includeArrayIndex": "rank"}} in leg


def test_hybrid_rrf_terms_use_weight_over_k_plus_rank():
    """Each leg contributes weight / (k + rank + 1). The +1 makes the rank 1-based,
    so the top hit scores weight/(k+1) rather than weight/k."""
    settings = get_settings()
    pipeline = _hybrid()
    text_leg = _stage(pipeline, "$unionWith")["pipeline"]

    vector_term = _stage(pipeline, "$replaceWith")["$mergeObjects"][1]
    text_term = _stage(text_leg, "$replaceWith")["$mergeObjects"][1]

    assert vector_term == {
        "rrf_vector": {
            "$divide": [
                settings.search_vector_weight,
                {"$add": ["$rank", settings.search_rrf_k, 1]},
            ]
        }
    }
    assert text_term == {
        "rrf_text": {
            "$divide": [
                settings.search_text_weight,
                {"$add": ["$rank", settings.search_rrf_k, 1]},
            ]
        }
    }


def test_hybrid_fuses_on_rank_not_on_raw_score():
    """The final score is the sum of the two RRF terms and nothing else. Adding the
    native scores instead would be meaningless — cosine in [0,1] and unbounded BM25
    aren't on the same scale."""
    pipeline = _hybrid()

    fused = _stage(pipeline, "$addFields")["score"]
    assert fused == {
        "$add": [{"$ifNull": ["$rrf_vector", 0]}, {"$ifNull": ["$rrf_text", 0]}]
    }


def test_hybrid_merges_a_chunk_found_by_both_legs():
    """A chunk both legs found arrives twice with the same _id and must collapse to
    one row carrying both RRF terms — that $group *is* the fusion point."""
    pipeline = _hybrid()

    fusion = next(
        stage["$group"]
        for stage in pipeline
        if "$group" in stage and stage["$group"]["_id"] == "$_id"
    )
    assert fusion["rrf_vector"] == {"$max": "$rrf_vector"}
    assert fusion["rrf_text"] == {"$max": "$rrf_text"}
    # A leg that missed the chunk leaves its native score absent, which is how a
    # caller can tell which leg found it.
    assert fusion["vector_score"] == {"$max": "$vector_score"}
    assert fusion["text_score"] == {"$max": "$text_score"}


def test_hybrid_sort_has_a_deterministic_tiebreak():
    """RRF ties are routine — with equal weights each leg's top hit scores the same
    — so without a tiebreak identical queries would return different orders."""
    pipeline = _hybrid()

    assert _stage(pipeline, "$sort") == {"score": -1, "_id": 1}


def test_hybrid_never_carries_the_embedding_through_fusion():
    """Each leg projects before its $group: $$ROOT would otherwise push 1024 floats
    per chunk through the whole fusion machinery."""
    pipeline = _hybrid()
    text_leg = _stage(pipeline, "$unionWith")["pipeline"]

    for leg in (pipeline, text_leg):
        project_index = next(i for i, stage in enumerate(leg) if "$project" in stage)
        group_index = next(i for i, stage in enumerate(leg) if "$group" in stage)
        assert project_index < group_index
        assert "embedding" not in leg[project_index]["$project"]
    # And the response projection drops the fusion internals.
    assert "rrf_vector" not in pipeline[-1]["$project"]
    assert "rrf_text" not in pipeline[-1]["$project"]


# --- Mode and fetch resolution -----------------------------------------------


def test_resolve_mode_defaults_and_validates():
    assert resolve_mode(None) == get_settings().search_default_mode
    assert resolve_mode("vector") == "vector"
    assert resolve_mode("hybrid") == "hybrid"


def test_resolve_mode_rejects_an_unknown_mode():
    """The service can be called from code that skips request validation, so it
    must fail loudly rather than silently building the wrong pipeline."""
    with pytest.raises(ValueError, match="unknown search mode"):
        resolve_mode("semantic")


def test_resolve_fetch_k_is_at_least_the_limit():
    settings = get_settings()
    assert _resolve_fetch_k(5) == 5 * settings.search_fetch_multiplier
    # Bounded, so a large limit x multiplier can't drag back hundreds of texts.
    assert _resolve_fetch_k(10_000) == 200


# --- search() dispatch --------------------------------------------------------


async def test_search_defaults_to_the_configured_mode():
    collection = _RecordingCollection([])

    await search(_RecordingDb(collection), FakeEmbedder(), str(ObjectId()), "q")

    expected = get_settings().search_default_mode
    if expected == "hybrid":
        assert any("$unionWith" in stage for stage in collection.pipeline)
    elif expected == "text":
        assert "$search" in collection.pipeline[0]
    else:
        assert "$vectorSearch" in collection.pipeline[0]


async def test_search_in_text_mode_skips_embedding():
    """There's no vector leg to embed for, so paying for a Voyage call would be
    pure waste."""
    collection = _RecordingCollection([])
    embedder = FakeEmbedder()

    await search(
        _RecordingDb(collection), embedder, str(ObjectId()), "q", mode="text"
    )

    assert embedder.query_calls == []
    assert "$search" in collection.pipeline[0]


async def test_search_in_hybrid_mode_embeds_once_and_runs_both_legs():
    collection = _RecordingCollection([])
    embedder = FakeEmbedder()

    await search(
        _RecordingDb(collection), embedder, str(ObjectId()), "q", mode="hybrid"
    )

    assert embedder.query_calls == ["q"]
    assert "$vectorSearch" in collection.pipeline[0]
    assert any("$unionWith" in stage for stage in collection.pipeline)


async def test_search_maps_per_leg_scores_when_present():
    """Hybrid rows carry each leg's native score; a missing one means that leg
    didn't retrieve the chunk, and must map to None rather than raising."""
    collection = _RecordingCollection(
        [
            {
                "_id": "chunk-a",
                "document_id": ObjectId(),
                "chunk_index": 0,
                "text": "found by both",
                "score": 0.032,
                "vector_score": 0.81,
                "text_score": 6.4,
            },
            {
                "_id": "chunk-b",
                "document_id": ObjectId(),
                "chunk_index": 1,
                "text": "found by the lexical leg only",
                "score": 0.016,
                "text_score": 5.1,
            },
        ]
    )

    results = await search(
        _RecordingDb(collection), FakeEmbedder(), str(ObjectId()), "q", mode="hybrid"
    )

    assert (results[0].vector_score, results[0].text_score) == (0.81, 6.4)
    assert results[1].vector_score is None
    assert results[1].text_score == 5.1
