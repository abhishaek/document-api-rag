"""Search over a user's chunks — dense, lexical, or the two fused with RRF.

Retrieval is the read side of the RAG pipeline. There are two independent ways to
find a chunk, and they fail on opposite kinds of query:

* **dense** (``$vectorSearch``) — embed the query into the same space the chunks
  were embedded into and take the nearest vectors. Strong on paraphrase ("run code
  without servers" finds the Lambda chunk); blind to rare literal tokens, because
  1024 floats averaged over a 1200-char chunk don't preserve that ``S3:GetObject``
  appeared once.
* **lexical** (``$search``, BM25) — match analyzed terms through an inverted index.
  The mirror image: nails identifiers, error codes, version numbers and acronyms;
  misses anything phrased differently from the document.

**Hybrid** runs both and fuses them with Reciprocal Rank Fusion, so either signal
alone can surface a chunk and agreement between them compounds.

Why RRF and not adding the scores: the two legs' scores are not on the same scale.
``vectorSearchScore`` is cosine in ~[0, 1]; ``searchScore`` is raw BM25 — unbounded,
and dependent on the corpus and the query's term rarity. A BM25 of 7.3 and a cosine
of 0.82 cannot be compared, and per-query normalization is unstable (it swings with
whatever happened to be in the result set). RRF throws the magnitudes away and keeps
only *position*::

    score(chunk) = sum over legs of  weight_leg / (k + rank_in_leg)

with ``k`` (``settings.search_rrf_k``, 60 by default) damping how much the very top
ranks dominate. A chunk ranked 1st in one leg and absent from the other still beats
a chunk sitting 8th in both — which is the property that makes fusion worth doing.

The work splits in two so it can be tested without a real Atlas server (neither
``$vectorSearch`` nor ``$search`` can be emulated in-memory — see the tests):

* ``build_*_pipeline`` — *pure* functions returning aggregation pipelines. No I/O,
  so a unit test can assert every stage is shaped correctly.
* ``search`` — embeds the query, runs the pipeline, maps rows to the response
  schema. Actual execution is covered by an integration test against Atlas Local.

Tenant isolation is enforced *inside* every leg — the vector index's ``user_id``
filter field, and a ``compound.filter`` ``equals`` clause for the lexical leg —
never by a post-filter. A post-filter would run after the leg's ``limit`` had
already been spent and could drop a user's own results.

Note on the fusion mechanics: MongoDB 8.1+ has a ``$rankFusion`` stage that does
RRF natively in one stage. This module hand-rolls it with ``$unionWith`` instead so
it runs on any Atlas that supports ``$vectorSearch`` and ``$search`` — and so the
fusion is visible and unit-testable rather than opaque. If the deployment is pinned
to 8.1+, the two legs here map onto ``$rankFusion``'s named input pipelines directly.
"""

import logging

from bson import ObjectId
from pymongo.asynchronous.database import AsyncDatabase

from app.core.config import get_settings
from app.models.chunk import COLLECTION_NAME, TEXT_INDEX_NAME, VECTOR_INDEX_NAME
from app.schemas.search import SearchResult
from app.services.embedding_service import Embedder

logger = logging.getLogger(__name__)

settings = get_settings()

# Atlas caps numCandidates at 10,000; the multiplier can't push us past it.
_MAX_NUM_CANDIDATES = 10_000

# Ceiling on how many candidates one leg fetches before fusion. The multiplier is
# a per-request lever; this stops a large limit combined with a large multiplier
# from making each leg drag back hundreds of chunk texts to rank a handful.
_MAX_FETCH_K = 200

SEARCH_MODES = ("vector", "text", "hybrid")

# The fields every mode returns. Never the 1024-float embedding — it would bloat
# every row, and in the fused pipeline it would also be pushed through a $group.
_RESULT_FIELDS = ("_id", "document_id", "chunk_index", "text")


def _resolve_limit(limit: int | None) -> int:  # type: ignore
    """Clamp the requested limit into [1, search_max_limit], defaulting when None.

    Done in the service (not only the schema) because the service is also called
    from code paths that don't pass through request validation — the service must
    never trust its caller to have bounded this.
    """
    if limit is None:
        return settings.search_default_limit
    return max(1, min(limit, settings.search_max_limit))


def resolve_mode(mode: str | None) -> str:
    """Fall back to the configured mode, and refuse an unknown one.

    Same reasoning as ``_resolve_limit``: the schema validates the HTTP boundary,
    but the service is callable from code that skips it, so an unknown mode must
    fail here loudly rather than silently building the wrong pipeline.

    Public (unlike the other resolvers) because the router needs the same answer
    to report which strategy ran — deriving it there from settings separately
    would be a second copy of this rule, free to drift from this one.
    """
    if mode is None:
        return settings.search_default_mode
    if mode not in SEARCH_MODES:
        raise ValueError(f"unknown search mode {mode!r}; expected one of {list(SEARCH_MODES)}")
    return mode


def _resolve_fetch_k(limit: int) -> int:
    """How many candidates *each* leg fetches before fusion.

    Deliberately well above the final limit: RRF can only reorder what the legs
    handed it, so a chunk outside both legs' fetch window can never surface no
    matter how well it would have fused. Bounded by ``_MAX_FETCH_K``.
    """
    return min(max(limit, limit * settings.search_fetch_multiplier), _MAX_FETCH_K)


def _search_filter(user_id: ObjectId, document_id: ObjectId | None) -> dict:
    """The tenant (and optionally document) scope, as ``$vectorSearch`` wants it.

    ``user_id`` is always applied, so one tenant never sees another's chunks.
    ``document_id``, when given, narrows to a single document — combined with
    ``user_id``, so a caller passing another user's document id gets nothing
    rather than a cross-tenant leak.
    """
    search_filter: dict = {"user_id": user_id}
    if document_id is not None:
        search_filter["document_id"] = document_id
    return search_filter


def _vector_search_stage(
    query_embedding: list[float],
    user_id: ObjectId,
    limit: int,
    document_id: ObjectId | None = None,
) -> dict:
    """The ``$vectorSearch`` stage, shared by the vector-only and hybrid pipelines.

    ``numCandidates`` is how many nearest neighbours the HNSW search explores
    before returning the top ``limit`` — larger means better recall for slightly
    more work. The ``filter`` scopes the search *before* ranking, so the ``limit``
    isn't spent on rows that will be discarded.
    """
    num_candidates = min(limit * settings.search_num_candidates_multiplier, _MAX_NUM_CANDIDATES)
    return {
        "$vectorSearch": {
            "index": VECTOR_INDEX_NAME,
            "path": "embedding",
            "queryVector": query_embedding,
            "numCandidates": num_candidates,
            "limit": limit,
            "filter": _search_filter(user_id, document_id),
        }
    }


def _text_search_stage(
    query: str,
    user_id: ObjectId,
    document_id: ObjectId | None = None,
) -> dict:
    """The ``$search`` stage, shared by the text-only and hybrid pipelines.

    The scope goes in ``compound.filter`` rather than a later ``$match`` for the
    same reason the vector leg uses index filter fields: a filter clause narrows
    the search itself, while a ``$match`` after the leg's ``$limit`` would throw
    away results the caller was entitled to. ``filter`` clauses (unlike ``must``)
    don't contribute to the BM25 score, so scoping doesn't distort the ranking.

    ``$search`` has no ``limit`` option of its own — unlike ``$vectorSearch`` — so
    the callers below follow it with an explicit ``$limit``.
    """
    scope: list[dict] = [{"equals": {"path": "user_id", "value": user_id}}]
    if document_id is not None:
        scope.append({"equals": {"path": "document_id", "value": document_id}})
    return {
        "$search": {
            "index": TEXT_INDEX_NAME,
            "compound": {
                "must": [{"text": {"query": query, "path": "text"}}],
                "filter": scope,
            },
        }
    }


def _projection(score_field: str, score_meta: str) -> dict:
    """Keep only the response fields plus this leg's native relevance score.

    Applied *before* any ``$group`` in the fused pipeline: ``$$ROOT`` would
    otherwise carry the 1024-float embedding through the fusion machinery.
    ``$meta`` is only readable in the stage right after its search stage, which is
    why each leg captures its score here rather than later.
    """
    projection: dict = dict.fromkeys(_RESULT_FIELDS, 1)
    projection[score_field] = {"$meta": score_meta}
    return projection


def _rrf_stages(rrf_field: str, weight: float) -> list[dict]:
    """Turn a leg's *ordered* results into its RRF contribution.

    A search stage returns rows already sorted by relevance but with no rank
    attached, and there is no ``$rank`` operator — so the rank is materialized the
    only way the aggregation framework allows: collect the rows into one array
    (``$push`` preserves the incoming order), then unwind it asking for the array
    index. That index *is* the 0-based rank.

    ``weight / (k + rank + 1)`` is one leg's RRF term; the ``+ 1`` makes the rank
    1-based so the top hit contributes ``weight / (k + 1)`` rather than ``weight / k``.
    """
    return [
        {"$group": {"_id": None, "docs": {"$push": "$$ROOT"}}},
        {"$unwind": {"path": "$docs", "includeArrayIndex": "rank"}},
        {
            "$replaceWith": {
                "$mergeObjects": [
                    "$docs",
                    {
                        rrf_field: {
                            "$divide": [weight, {"$add": ["$rank", settings.search_rrf_k, 1]}]
                        }
                    },
                ]
            }
        },
    ]


def build_vector_search_pipeline(
    query_embedding: list[float],
    user_id: ObjectId,
    limit: int,
    document_id: ObjectId | None = None,
) -> list[dict]:
    """Dense-only pipeline: nearest vectors for one tenant, most similar first.

    ``score`` is the cosine similarity, available only via ``$meta`` in the stage
    immediately after ``$vectorSearch``.
    """
    return [
        _vector_search_stage(query_embedding, user_id, limit, document_id),
        {"$project": _projection("score", "vectorSearchScore")},
    ]


def build_text_search_pipeline(
    query: str,
    user_id: ObjectId,
    limit: int,
    document_id: ObjectId | None = None,
) -> list[dict]:
    """Lexical-only (BM25) pipeline, best matching terms first.

    Kept as a first-class mode mainly so an eval set can measure one leg against
    the other and against the fusion — you can't tune the RRF weights without
    being able to see what each leg contributes on its own.

    ``score`` here is the raw ``searchScore``: unbounded and not comparable with
    the cosine score the vector mode returns under the same name.
    """
    return [
        _text_search_stage(query, user_id, document_id),
        {"$limit": limit},
        {"$project": _projection("score", "searchScore")},
    ]


def build_hybrid_search_pipeline(
    query_embedding: list[float],
    query: str,
    user_id: ObjectId,
    limit: int,
    document_id: ObjectId | None = None,
) -> list[dict]:
    """Both legs, fused with Reciprocal Rank Fusion.

    Shape of the pipeline::

        $vectorSearch (scoped, fetch_k)   -> project -> rank -> rrf_vector
        $unionWith chunks:
            $search (scoped, fetch_k)     -> project -> rank -> rrf_text
        $group by _id                     -> one row per chunk, both legs merged
        score = rrf_vector + rrf_text     -> $sort -> $limit

    ``$unionWith`` is what makes two searches possible in one pipeline: both
    ``$vectorSearch`` and ``$search`` must be the *first* stage of the pipeline
    they run in, so the lexical leg runs as a sub-pipeline over the same
    collection and its rows are appended to the dense leg's.

    The ``$group`` is the fusion point. A chunk found by both legs arrives twice
    with the same ``_id`` (content-addressed, so identical across legs) and is
    merged into one row carrying both RRF terms; a chunk found by one leg has the
    other term missing, which ``$ifNull`` reads as a zero contribution. ``$max``
    is a picker rather than a real maximum — each field is present at most once
    per chunk, so it just takes whichever copy has it.

    ``_id`` is the sort tiebreak. RRF produces exact ties routinely (with equal
    weights, the top hit of each leg scores identically), and without a tiebreak
    those would order arbitrarily and shuffle between identical queries.

    The raw ``vector_score``/``text_score`` are carried through to the response.
    They play no part in the ranking — that's the whole point of fusing on rank —
    but they show *why* a chunk placed where it did, and which leg found it.
    """
    fetch_k = _resolve_fetch_k(limit)

    vector_leg: list[dict] = [
        _vector_search_stage(query_embedding, user_id, fetch_k, document_id),
        {"$project": _projection("vector_score", "vectorSearchScore")},
        *_rrf_stages("rrf_vector", settings.search_vector_weight),
    ]
    text_leg: list[dict] = [
        _text_search_stage(query, user_id, document_id),
        {"$limit": fetch_k},
        {"$project": _projection("text_score", "searchScore")},
        *_rrf_stages("rrf_text", settings.search_text_weight),
    ]

    return [
        *vector_leg,
        {"$unionWith": {"coll": COLLECTION_NAME, "pipeline": text_leg}},
        {
            "$group": {
                "_id": "$_id",
                "document_id": {"$first": "$document_id"},
                "chunk_index": {"$first": "$chunk_index"},
                "text": {"$first": "$text"},
                "vector_score": {"$max": "$vector_score"},
                "text_score": {"$max": "$text_score"},
                "rrf_vector": {"$max": "$rrf_vector"},
                "rrf_text": {"$max": "$rrf_text"},
            }
        },
        {
            "$addFields": {
                "score": {
                    "$add": [{"$ifNull": ["$rrf_vector", 0]}, {"$ifNull": ["$rrf_text", 0]}]
                }
            }
        },
        {"$sort": {"score": -1, "_id": 1}},
        {"$limit": limit},
        # Drop the per-leg RRF terms; they're fusion internals, not response data.
        {
            "$project": {
                **dict.fromkeys(_RESULT_FIELDS, 1),
                "score": 1,
                "vector_score": 1,
                "text_score": 1,
            }
        },
    ]


async def search(
    db: AsyncDatabase,
    embedder: Embedder,
    user_id: str,
    query: str,
    limit: int | None = None,
    document_id: str | None = None,
    mode: str | None = None,
) -> list[SearchResult]:
    """Return the caller's chunks most relevant to ``query``, best first.

    ``mode`` picks the strategy (see ``SEARCH_MODES``), defaulting to
    ``settings.search_default_mode``. Text-only mode skips embedding entirely —
    there's no vector leg to embed for, so paying for a Voyage call would be pure
    waste. The other two embed with ``input_type="query"`` (that lives in the
    embedder) so the query lands in the same space as the ``input_type="document"``
    chunks it's matched against.

    ``user_id`` comes from the authenticated token; the caller (get_current_user)
    has already checked it parses as an ObjectId. ``document_id``, when given,
    restricts the search to that one document and is likewise already validated at
    the schema boundary, so both convert here without re-checking.
    """
    resolved_limit = _resolve_limit(limit)
    resolved_mode = resolve_mode(mode)
    owner = ObjectId(user_id)
    document = ObjectId(document_id) if document_id is not None else None

    if resolved_mode == "text":
        pipeline = build_text_search_pipeline(query, owner, resolved_limit, document)
    else:
        query_embedding = await embedder.embed_query(query)
        if resolved_mode == "hybrid":
            pipeline = build_hybrid_search_pipeline(
                query_embedding, query, owner, resolved_limit, document
            )
        else:
            pipeline = build_vector_search_pipeline(
                query_embedding, owner, resolved_limit, document
            )

    cursor = await db[COLLECTION_NAME].aggregate(pipeline)
    rows = await cursor.to_list(length=resolved_limit)

    logger.info(
        "chunk search",
        extra={"user_id": user_id, "search_mode": resolved_mode, "result_count": len(rows)},
    )
    return [
        SearchResult(
            id=str(row["_id"]),
            document_id=str(row["document_id"]),
            chunk_index=row["chunk_index"],
            text=row["text"],
            score=row["score"],
            # Populated in hybrid mode only; a single-leg search has one score and
            # it's already `score`. `.get` because a leg that didn't find this
            # chunk leaves its field missing entirely.
            vector_score=row.get("vector_score"),
            text_score=row.get("text_score"),
        )
        for row in rows
    ]
