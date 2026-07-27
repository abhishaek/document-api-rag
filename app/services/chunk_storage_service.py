"""Persist a document's chunks and their embeddings — incrementally.

Re-ingesting a document **reconciles** the stored chunks against the freshly
computed set rather than blindly replacing them. Each chunk has a deterministic,
content-addressed, position-independent id (see ``build_chunk_id``), so the sync
can tell exactly what changed:

* a chunk whose id already exists is **not** re-embedded — if it merely moved to a
  new position its ``chunk_index`` is updated in place (cheap), otherwise it is
  left untouched;
* only genuinely new or changed chunks are embedded and inserted;
* chunks whose ids no longer appear are deleted.

The payoff is twofold. A recovery re-run of an unchanged document costs *zero*
embedding calls. And because the id ignores position, an in-place document edit
(a paragraph inserted near the top, shifting everything below it) re-embeds only
the chunks whose *text* actually changed — the shifted-but-unchanged chunks keep
their ids and are merely re-indexed.

One ordering is load-bearing: **embed before any write.** A failure while
embedding leaves the current chunks exactly as they were, never a half-written
set. Deletes, re-indexes, and inserts can run in any order because ``chunk_index``
is no longer unique — the content-addressed ``_id`` is the collection's real key.
"""

import logging
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime

from bson import ObjectId
from pymongo.asynchronous.database import AsyncDatabase

from app.models.chunk import COLLECTION_NAME, build_chunk_id
from app.services.embedding_service import Embedder

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ChunkSyncResult:
    """What a sync changed. ``embedded + reused == total``; ``reindexed`` is the
    subset of reused chunks that moved; ``deleted`` is separate."""

    total: int  # chunks the document has after the sync
    embedded: int  # new/changed chunks embedded and inserted this run
    reused: int  # chunks whose id already existed (no re-embedding)
    reindexed: int  # reused chunks whose position changed (chunk_index updated)
    deleted: int  # stale chunks removed


def _desired_rows(
    user_id: ObjectId, document_id: ObjectId, chunks: list[str]
) -> list[dict]:
    """The chunk rows a document should have, keyed by content-addressed id. No
    embedding or timestamp yet — those are added only for rows actually written.

    ``occurrence`` counts earlier identical-text chunks so two copies of the same
    text get distinct ids without pinning either id to a position."""
    seen: Counter[str] = Counter()
    rows: list[dict] = []
    for index, text in enumerate(chunks):
        occurrence = seen[text]
        seen[text] += 1
        rows.append(
            {
                "_id": build_chunk_id(user_id, document_id, text, occurrence),
                "document_id": document_id,
                "user_id": user_id,
                "chunk_index": index,
                "text": text,
            }
        )
    return rows


async def _existing_chunks(db: AsyncDatabase, document_id: ObjectId) -> dict[str, int]:
    """Map a document's stored chunk ids to their current ``chunk_index``. Projects
    to id + index only, so this never pulls the (large) embedding vectors."""
    rows = await db[COLLECTION_NAME].find(
        {"document_id": document_id}, {"_id": 1, "chunk_index": 1}
    ).to_list(length=None)
    return {row["_id"]: row["chunk_index"] for row in rows}


async def sync_document_chunks(
    db: AsyncDatabase,
    document_id: ObjectId,
    user_id: ObjectId,
    chunks: list[str],
    embedder: Embedder,
) -> ChunkSyncResult:
    """Reconcile a document's stored chunks to ``chunks``, embedding only what's
    new and re-indexing what merely moved. Returns counts of what changed.

    ``chunks`` is the ordered list of chunk texts from the chunker; positions
    become ``chunk_index``. An empty list clears the document's chunks.
    """
    rows = _desired_rows(user_id, document_id, chunks)
    desired_ids = {row["_id"] for row in rows}
    existing = await _existing_chunks(db, document_id)

    to_insert = [row for row in rows if row["_id"] not in existing]
    # Reused chunks that sit at a new position: keep the row (and its embedding),
    # just correct chunk_index. Position-independent ids make this possible.
    to_reindex = [
        row
        for row in rows
        if row["_id"] in existing and existing[row["_id"]] != row["chunk_index"]
    ]
    stale_ids = set(existing) - desired_ids

    # Embed the new/changed chunks first, before touching the DB, so an embedding
    # failure raises with the current chunks still intact. A count mismatch from
    # the embedder trips zip(strict=True) here rather than storing misaligned rows.
    if to_insert:
        embeddings = await embedder.embed_documents([row["text"] for row in to_insert])
        now = datetime.now(UTC)
        for row, embedding in zip(to_insert, embeddings, strict=True):
            row["embedding"] = embedding
            row["created_at"] = now

    if stale_ids:
        await db[COLLECTION_NAME].delete_many({"_id": {"$in": list(stale_ids)}})
    for row in to_reindex:
        await db[COLLECTION_NAME].update_one(
            {"_id": row["_id"]}, {"$set": {"chunk_index": row["chunk_index"]}}
        )
    if to_insert:
        await db[COLLECTION_NAME].insert_many(to_insert)

    result = ChunkSyncResult(
        total=len(rows),
        embedded=len(to_insert),
        reused=len(rows) - len(to_insert),
        reindexed=len(to_reindex),
        deleted=len(stale_ids),
    )
    logger.info(
        "synced chunks",
        extra={
            "document_id": str(document_id),
            "total": result.total,
            "embedded": result.embedded,
            "reused": result.reused,
            "reindexed": result.reindexed,
            "deleted": result.deleted,
        },
    )
    return result
