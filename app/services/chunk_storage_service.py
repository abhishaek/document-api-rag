"""Persist a document's chunks and their embeddings.

The write is **replace, not append**: every ingest of a document deletes its
existing chunks and inserts the new set. That keeps re-ingestion idempotent — a
document reprocessed after a chunking or embedding change ends up with exactly one
current set of chunks, never a pile-up of stale ones alongside fresh.

Callers must embed *before* calling this, so the delete-then-insert here is the
only place chunks change: a failure during embedding leaves the previous chunks
untouched rather than deleting them and then failing to write replacements.
"""

import logging
from datetime import UTC, datetime

from bson import ObjectId
from pymongo.asynchronous.database import AsyncDatabase

from app.models.chunk import COLLECTION_NAME

logger = logging.getLogger(__name__)


async def replace_document_chunks(
    db: AsyncDatabase,
    document_id: ObjectId,
    user_id: ObjectId,
    chunks: list[str],
    embeddings: list[list[float]],
) -> int:
    """Replace all stored chunks for a document with a new set. Returns the count.

    ``chunks`` and ``embeddings`` are positional: chunk *i* embeds to vector *i*.
    A length mismatch is a caller bug (the embedder returned the wrong number of
    vectors) and raises rather than storing misaligned rows.
    """
    if len(chunks) != len(embeddings):
        raise ValueError(
            f"chunks ({len(chunks)}) and embeddings ({len(embeddings)}) "
            "must be the same length"
        )

    # Clear the old set first. Runs even when there are no new chunks, so a
    # reprocess that now yields nothing doesn't leave the previous set behind.
    await db[COLLECTION_NAME].delete_many({"document_id": document_id})

    if not chunks:
        return 0

    now = datetime.now(UTC)
    rows = [
        {
            "document_id": document_id,
            "user_id": user_id,
            "chunk_index": index,
            "text": text,
            "embedding": embedding,
            "created_at": now,
        }
        for index, (text, embedding) in enumerate(zip(chunks, embeddings, strict=True))
    ]
    await db[COLLECTION_NAME].insert_many(rows)
    logger.info(
        "stored chunks",
        extra={"document_id": str(document_id), "chunk_count": len(rows)},
    )
    return len(rows)
