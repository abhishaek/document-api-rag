"""Ingestion pipeline: turn an uploaded document into retrievable chunks.

Triggered — not awaited — by the upload route once a document record exists. It
runs the slow work off the request path (parse -> chunk -> embed) and records
progress through the document's status:

    pending -> processing -> ready
                          -> failed

Two properties matter here:

* **It never lets an exception escape.** It runs as a fire-and-forget background
  task, and a raise in such a task is swallowed by the event loop — the document
  would be stranded in `processing` with no signal. So every failure is caught
  and written to the record as `failed` + a reason, which the client sees when it
  polls GET /documents/{id}.

* **It re-reads the document from the DB** rather than trusting data passed in,
  and takes only the ``document_id``. That keeps the trigger cheap (an id, not a
  payload) and means the swap to a durable queue later — where the worker only
  ever receives an id — needs no change to this function.

All stages are real now: parse (step 3) -> chunk (step 4) -> embed + store (step
5). The status transitions and never-raise error handling have held since the
skeleton and don't change as stages are filled in.
"""

import logging

from pymongo.asynchronous.database import AsyncDatabase

from app.core.config import get_settings
from app.models.document import COLLECTION_NAME
from app.schemas.document import DocumentStatus
from app.services.chunk_storage_service import replace_document_chunks
from app.services.chunking_service import chunk_markdown
from app.services.document_service import (
    claim_document_for_processing,
    get_document_record,
    set_document_status,
)
from app.services.embedding_service import Embedder
from app.services.parsing_service import parse_to_markdown
from app.services.storage_service import Storage

logger = logging.getLogger(__name__)
settings = get_settings()


async def process_document(
    db: AsyncDatabase, storage: Storage, embedder: Embedder, document_id: str
) -> None:
    """Run the ingestion pipeline for one document, updating its status.

    Fire-and-forget: returns None and never raises. Called via
    ``BackgroundTasks.add_task`` from the upload route once the record exists.
    """
    logger.info("ingestion started", extra={"document_id": document_id})
    try:
        record = await get_document_record(db, document_id)
        if record is None:
            # Deleted between upload and this task running. Nothing to process —
            # and nothing to mark, since the record it would point at is gone.
            logger.warning(
                "ingestion skipped: document not found",
                extra={"document_id": document_id},
            )
            return

        # Claim the work: pending -> processing (and count the attempt), so a
        # poller sees movement and the recovery sweep can bound its retries.
        await claim_document_for_processing(db, document_id)

        raw = await storage.get(record["storage_key"])
        markdown = await parse_to_markdown(
            raw, record["mime_type"], record["original_filename"]
        )
        chunks = chunk_markdown(
            markdown, chunk_size=settings.chunk_size, overlap=settings.chunk_overlap
        )
        # Embed everything before writing, so a failure mid-embedding leaves the
        # existing chunks (if any) intact rather than half-replaced.
        embeddings = await embedder.embed_documents(chunks)
        await replace_document_chunks(
            db, record["_id"], record["user_id"], chunks, embeddings
        )

        await set_document_status(db, document_id, DocumentStatus.ready)
        logger.info(
            "ingestion complete",
            extra={"document_id": document_id, "chunk_count": len(chunks)},
        )
    except Exception as exc:
        # Any failure — a missing blob, an unparseable document, (later) an
        # embedding error — records the reason on the document instead of
        # crashing a task whose exception nobody is awaiting. str(exc) on our
        # domain errors (e.g. UnparseableDocumentError) is already user-legible.
        logger.warning(
            "ingestion failed", extra={"document_id": document_id}, exc_info=exc
        )
        await set_document_status(
            db, document_id, DocumentStatus.failed, error=str(exc)
        )


async def recover_incomplete_documents(
    db: AsyncDatabase, storage: Storage, embedder: Embedder, max_attempts: int
) -> int:
    """Re-run documents left in a non-terminal or failed state. Returns how many.

    Runs at startup. Two cases it heals, so the user never has to re-upload:

    * ``processing`` — a crash (or restart) left a document mid-ingest, with no
      task still working it.
    * ``failed`` — ingestion failed for a reason that may now be fixed. The
      motivating case: the embedding API key was missing, the document failed,
      the key has since been added, and this restart picks it back up.

    Bounded by ``max_attempts``: a document already attempted that many times is
    left ``failed`` rather than retried forever, so a genuinely broken file (truly
    unparseable, permanently rejected) doesn't loop. Attempts are counted by
    ``claim_document_for_processing`` on each run.
    """
    try:
        candidates = await db[COLLECTION_NAME].find(
            {
                "status": {
                    "$in": [
                        DocumentStatus.failed.value,
                        DocumentStatus.processing.value,
                    ]
                }
            }
        ).to_list(length=1000)
    except Exception as exc:
        # Recovery is best-effort background work; a query failure must not take
        # down startup or surface as an unretrieved task exception.
        logger.warning("recovery sweep could not list documents", exc_info=exc)
        return 0
    reprocessed = 0
    abandoned = 0
    for doc in candidates:
        document_id = str(doc["_id"])
        # Missing `attempts` (a document written before the field existed) counts
        # as 0, so it's still eligible.
        attempts = doc.get("attempts", 0)
        if attempts < max_attempts:
            # Sequential, not concurrent: keeps startup load bounded and avoids a
            # burst of embedding calls. process_document never raises, so one bad
            # document can't stop the rest.
            await process_document(db, storage, embedder, document_id)
            reprocessed += 1
        elif doc.get("status") == DocumentStatus.processing.value:
            # Out of retries AND stranded mid-processing — its worker was killed
            # before it could finish or fail (e.g. the process was restarted). Make
            # it terminal so it stops masquerading as in-flight forever; a `failed`
            # status is honest and lets a poller (or a human) see a real outcome.
            await set_document_status(
                db,
                document_id,
                DocumentStatus.failed,
                error=f"Ingestion abandoned after {attempts} attempts without "
                "completing (likely interrupted mid-processing).",
            )
            abandoned += 1
        # else: already `failed` and out of retries — leave it.

    if reprocessed or abandoned:
        logger.info(
            "recovery sweep complete",
            extra={"reprocessed": reprocessed, "abandoned": abandoned},
        )
    return reprocessed
