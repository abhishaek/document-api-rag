"""Vector search over a user's chunks.

Retrieval is the read side of the RAG pipeline: embed the caller's query into the
same space the chunks were embedded into, then ask Atlas for the nearest vectors.

The work splits in two so it can be tested without a real Atlas server (its
``$vectorSearch`` can't be emulated by an in-memory fake — see the tests):

* ``build_vector_search_pipeline`` — a *pure* function that returns the aggregation
  pipeline. No I/O, so a unit test can assert every stage is shaped correctly.
* ``search`` — embeds the query, runs the pipeline, and maps rows to the response
  schema. The actual ``$vectorSearch`` execution is covered by an integration test
  against Atlas Local.

Tenant isolation is enforced *inside* the search via the index's ``user_id``
filter field, not by a post-filter: a post-filter would run after ``limit`` had
already been applied and could drop a user's own results. This is why Phase 0
declared ``user_id`` as a filter field on the vector index.
"""

import logging

from bson import ObjectId
from pymongo.asynchronous.database import AsyncDatabase

from app.core.config import get_settings
from app.models.chunk import COLLECTION_NAME, VECTOR_INDEX_NAME
from app.schemas.search import SearchResult
from app.services.embedding_service import Embedder

logger = logging.getLogger(__name__)

settings = get_settings()

# Atlas caps numCandidates at 10,000; the multiplier can't push us past it.
_MAX_NUM_CANDIDATES = 10_000


def _resolve_limit(limit: int | None) -> int:
    """Clamp the requested limit into [1, search_max_limit], defaulting when None.

    Done in the service (not only the schema) because the service is also called
    from code paths that don't pass through request validation — the service must
    never trust its caller to have bounded this.
    """
    if limit is None:
        return settings.search_default_limit
    return max(1, min(limit, settings.search_max_limit))


def build_vector_search_pipeline(
    query_embedding: list[float],
    user_id: ObjectId,
    limit: int,
    document_id: ObjectId | None = None,
) -> list[dict]:
    """Build the ``$vectorSearch`` aggregation pipeline for one tenant's query.

    ``numCandidates`` is how many nearest neighbours the HNSW search explores
    before returning the top ``limit`` — larger means better recall for slightly
    more work. The ``filter`` scopes the search *before* ranking, so the ``limit``
    isn't spent on rows that will be discarded:

    * ``user_id`` is always applied, so one tenant never sees another's chunks.
    * ``document_id``, when given, narrows the search to a single document. It's
      combined with ``user_id``, so even a caller passing another user's document
      id gets nothing rather than a cross-tenant leak.
    """
    num_candidates = min(
        limit * settings.search_num_candidates_multiplier, _MAX_NUM_CANDIDATES
    )
    search_filter: dict = {"user_id": user_id}
    if document_id is not None:
        search_filter["document_id"] = document_id
    return [
        {
            "$vectorSearch": {
                "index": VECTOR_INDEX_NAME,
                "path": "embedding",
                "queryVector": query_embedding,
                "numCandidates": num_candidates,
                "limit": limit,
                "filter": search_filter,
            }
        },
        {
            # Return only what the response needs — never the 1024-float embedding,
            # which would bloat every row. `vectorSearchScore` is the similarity,
            # available only via $meta after a $vectorSearch stage.
            "$project": {
                "_id": 1,
                "document_id": 1,
                "chunk_index": 1,
                "text": 1,
                "score": {"$meta": "vectorSearchScore"},
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
) -> list[SearchResult]:
    """Return the caller's chunks most similar to ``query``, most similar first.

    The query is embedded with ``input_type="query"`` (that lives in the embedder)
    so it lands in the same space as the ``input_type="document"`` chunks it's
    matched against. ``user_id`` comes from the authenticated token; the caller
    (get_current_user) has already checked it parses as an ObjectId.

    ``document_id``, when given, restricts the search to that one document. It's
    already validated as an ObjectId at the schema boundary, so it converts here
    without re-checking (mirroring how user_id is trusted).
    """
    resolved_limit = _resolve_limit(limit)

    query_embedding = await embedder.embed_query(query)

    pipeline = build_vector_search_pipeline(
        query_embedding,
        ObjectId(user_id),
        resolved_limit,
        document_id=ObjectId(document_id) if document_id is not None else None,
    )
    cursor = await db[COLLECTION_NAME].aggregate(pipeline)
    rows = await cursor.to_list(length=resolved_limit)

    logger.info(
        "vector search",
        extra={"user_id": user_id, "result_count": len(rows)},
    )
    return [
        SearchResult(
            id=str(row["_id"]),
            document_id=str(row["document_id"]),
            chunk_index=row["chunk_index"],
            text=row["text"],
            score=row["score"],
        )
        for row in rows
    ]
