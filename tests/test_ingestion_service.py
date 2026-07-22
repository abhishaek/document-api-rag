"""Tests for the ingestion pipeline (app/services/ingestion_service.py).

These drive ``process_document`` the way the background task does — directly,
against a fake DB, a real FilesystemStorage, and a fake embedder (no Voyage API)
— after seeding a document with ``create_document`` (the same pairing the upload
route uses). The pipeline's contract is about *document state*: a good document
reaches `ready` with its chunks stored, any failure reaches `failed` with a
reason, and nothing ever raises out of it.
"""

from pathlib import Path

from bson import ObjectId

from app.models.chunk import COLLECTION_NAME as CHUNKS
from app.models.document import COLLECTION_NAME as DOCS
from app.schemas.document import DocumentStatus
from app.services.document_service import (
    create_document,
    get_document_record,
    set_document_status,
)
from app.services.ingestion_service import (
    process_document,
    recover_incomplete_documents,
)
from app.services.storage_service import FilesystemStorage

USER_ID = "507f1f77bcf86cd799439011"


def _storage(tmp_path: Path) -> FilesystemStorage:
    return FilesystemStorage(root=tmp_path / "storage")


async def test_process_document_marks_ready_and_stores_chunks(
    fake_db, fake_embedder, tmp_path
):
    storage = _storage(tmp_path)
    document, created = await create_document(
        fake_db, storage, USER_ID, "notes.txt", b"some readable text content\n"
    )
    assert created

    await process_document(fake_db, storage, fake_embedder, document.id)

    record = await get_document_record(fake_db, document.id)
    assert record["status"] == "ready"
    assert record["error"] is None

    # Chunks were persisted, tied to the document and its owner, each with a vector.
    stored = await fake_db[CHUNKS].find({}).to_list(length=100)
    assert stored
    for chunk in stored:
        assert chunk["document_id"] == ObjectId(document.id)
        assert chunk["user_id"] == ObjectId(USER_ID)
        assert chunk["text"]
        assert len(chunk["embedding"]) == fake_embedder.dimensions


async def test_process_document_marks_failed_when_blob_missing(
    fake_db, fake_embedder, tmp_path
):
    """If the blob can't be read, the document is marked failed with a reason —
    and process_document must not raise (its exception would be lost)."""
    storage = _storage(tmp_path)
    document, _ = await create_document(
        fake_db, storage, USER_ID, "notes.txt", b"text\n"
    )
    record = await get_document_record(fake_db, document.id)
    await storage.delete(record["storage_key"])

    await process_document(fake_db, storage, fake_embedder, document.id)  # no raise

    record = await get_document_record(fake_db, document.id)
    assert record["status"] == "failed"
    assert record["error"]


async def test_process_document_missing_record_is_noop(
    fake_db, fake_embedder, tmp_path
):
    """A document deleted between upload and processing is skipped silently — no
    record to update, and no exception."""
    storage = _storage(tmp_path)

    await process_document(fake_db, storage, fake_embedder, "507f1f77bcf86cd799439099")


async def test_processing_increments_attempts(fake_db, fake_embedder, tmp_path):
    """Each pipeline run counts an attempt — the number the recovery sweep bounds
    its retries on."""
    storage = _storage(tmp_path)
    document, _ = await create_document(
        fake_db, storage, USER_ID, "notes.txt", b"text content\n"
    )
    assert (await get_document_record(fake_db, document.id))["attempts"] == 0

    await process_document(fake_db, storage, fake_embedder, document.id)

    assert (await get_document_record(fake_db, document.id))["attempts"] == 1


async def _seed_failed(fake_db, storage, *, attempts: int):
    """Upload a (recoverable) document, then mark it failed with a given attempt
    count — the state the recovery sweep is meant to pick up."""
    document, _ = await create_document(
        fake_db, storage, USER_ID, "notes.txt", b"recoverable text content\n"
    )
    await fake_db[DOCS].update_one(
        {"_id": ObjectId(document.id)},
        {"$set": {"status": DocumentStatus.failed.value, "attempts": attempts}},
    )
    return document


async def test_recovery_reprocesses_a_failed_document(
    fake_db, fake_embedder, tmp_path
):
    """The motivating case: a document failed (e.g. missing API key), the cause is
    now fixed, and the sweep re-ingests it to `ready` — no re-upload."""
    storage = _storage(tmp_path)
    document = await _seed_failed(fake_db, storage, attempts=1)

    count = await recover_incomplete_documents(
        fake_db, storage, fake_embedder, max_attempts=3
    )

    assert count == 1
    assert (await get_document_record(fake_db, document.id))["status"] == "ready"


async def test_recovery_skips_documents_at_the_attempt_limit(
    fake_db, fake_embedder, tmp_path
):
    """A document already tried max_attempts times is left failed, not retried
    forever."""
    storage = _storage(tmp_path)
    document = await _seed_failed(fake_db, storage, attempts=3)

    count = await recover_incomplete_documents(
        fake_db, storage, fake_embedder, max_attempts=3
    )

    assert count == 0
    assert (await get_document_record(fake_db, document.id))["status"] == "failed"


async def test_recovery_marks_exhausted_processing_as_failed(
    fake_db, fake_embedder, tmp_path
):
    """A document stranded in `processing` at the attempt limit (its worker kept
    getting killed) is made terminal `failed`, not left masquerading as in-flight
    forever."""
    storage = _storage(tmp_path)
    document, _ = await create_document(
        fake_db, storage, USER_ID, "notes.txt", b"text content\n"
    )
    await fake_db[DOCS].update_one(
        {"_id": ObjectId(document.id)},
        {"$set": {"status": DocumentStatus.processing.value, "attempts": 3}},
    )

    count = await recover_incomplete_documents(
        fake_db, storage, fake_embedder, max_attempts=3
    )

    assert count == 0  # not reprocessed (out of retries)
    record = await get_document_record(fake_db, document.id)
    assert record["status"] == "failed"
    assert "abandoned" in record["error"]


async def test_recovery_ignores_ready_documents(fake_db, fake_embedder, tmp_path):
    """Only failed/processing documents are swept; a ready one is left alone."""
    storage = _storage(tmp_path)
    document, _ = await create_document(
        fake_db, storage, USER_ID, "notes.txt", b"already done\n"
    )
    await set_document_status(fake_db, document.id, DocumentStatus.ready)

    count = await recover_incomplete_documents(
        fake_db, storage, fake_embedder, max_attempts=3
    )

    assert count == 0
