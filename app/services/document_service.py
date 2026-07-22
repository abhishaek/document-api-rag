import logging
from datetime import UTC, datetime

import magic
from bson import ObjectId
from bson.errors import InvalidId
from pymongo.asynchronous.database import AsyncDatabase
from pymongo.errors import DuplicateKeyError

from app.core.config import get_settings
from app.models.document import COLLECTION_NAME
from app.schemas.document import DocumentResponse, DocumentStatus
from app.services.storage_service import Storage

logger = logging.getLogger(__name__)

settings = get_settings()


# Markdown has no magic bytes — it is plain text — so libmagic never reports
# this. _sniff_mime derives it from the extension; see there for why that is
# safe.
MARKDOWN_MIME_TYPE = "text/markdown"
MARKDOWN_EXTENSIONS = (".md", ".markdown")

# What the pipeline accepts. This mirrors the parsers available downstream: a
# type in here is a promise that ingestion can process it, so entries are added
# alongside their parser, never ahead of one.
ALLOWED_MIME_TYPES = {
    "application/pdf",  # .pdf
    "text/html",  # .html
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",  # .docx
    "text/plain",  # .txt
    MARKDOWN_MIME_TYPE,  # .md / .markdown
}
# csv (text/csv) and xlsx are intentionally absent: the tabular (structured /
# text-to-SQL) lane isn't built, and the allow-list must never promise a type the
# pipeline can't process — so uploading one returns 415 rather than getting stuck
# pending. Re-add both here, alongside their parser, when the tabular lane ships.
# This set mirrors parsing_service's registry; the two move together.


class FileTooLargeError(Exception):
    """Raised when an upload exceeds the configured size cap.

    Carries both numbers so the caller can tell the user what the limit is.
    """

    def __init__(self, size_bytes: int, limit_bytes: int) -> None:
        self.size_bytes = size_bytes
        self.limit_bytes = limit_bytes
        super().__init__(f"File is {size_bytes} bytes; limit is {limit_bytes}")


class UnsupportedFileTypeError(Exception):
    """Raised when a file's sniffed MIME type is not in ALLOWED_MIME_TYPES.

    ``mime`` is the type that was detected, so the router can tell the user
    what was rejected.
    """

    def __init__(self, mime: str) -> None:
        self.mime = mime
        super().__init__(f"Unsupported file type: {mime}")


class DocumentNotFoundError(Exception):
    """Raised when no document with the given id exists for this user."""

    def __init__(self, document_id: str) -> None:
        self.document_id = document_id
        super().__init__(f"No document found: {document_id}")


def _to_response(doc: dict) -> DocumentResponse:
    """Map a stored Mongo document to its API representation.

    Kept in one place so the `_id` (ObjectId) -> `id` (str) conversion exists
    exactly once rather than in every caller.
    """
    return DocumentResponse(
        id=str(doc["_id"]),
        original_filename=doc["original_filename"],
        mime_type=doc["mime_type"],
        size_bytes=doc["size_bytes"],
        status=doc["status"],
        error=doc["error"],
        created_at=doc["created_at"],
    )


def _sniff_mime(raw: bytes, filename: str) -> str:
    """Identify a file from its bytes, using the extension only where bytes cannot decide.

    Binary formats (PDF, DOCX, XLSX) carry magic bytes, so libmagic identifies
    them outright and the filename is irrelevant — a PNG named ``invoice.pdf`` is
    still a PNG, and that is the point of sniffing at all.

    Markdown has no signature: it *is* plain text, so libmagic reports
    text/plain and the extension is the only thing separating a .md from a .txt.
    Narrowing text/plain by extension is safe precisely because it is a
    narrowing — it can only reclassify a file already accepted as text, and
    never lets a filename override what the bytes proved.
    """
    mime = magic.from_buffer(raw, mime=True)
    if mime == "text/plain" and filename.lower().endswith(MARKDOWN_EXTENSIONS):
        return MARKDOWN_MIME_TYPE
    return mime


async def create_document(
    db: AsyncDatabase,
    storage: Storage,
    user_id: str,
    filename: str,
    raw: bytes,
) -> tuple[DocumentResponse, bool]:
    """Validate an upload, store its bytes, and record it.

    Returns ``(document, created)``. ``created`` is ``True`` only when a new
    record was inserted, and ``False`` when this was a duplicate upload that
    returned the existing record. The caller uses this to decide whether to kick
    off ingestion: a duplicate must never be re-parsed or re-embedded — that
    wasted work (and duplicate vector chunks) is exactly what the content-hash
    dedup exists to prevent.

    Idempotent: re-uploading a file the user already has returns the existing
    record instead of creating a second one.
    """
    # Cheapest check first — no point sniffing bytes we're about to reject.
    if len(raw) > settings.max_upload_size_bytes:
        raise FileTooLargeError(len(raw), settings.max_upload_size_bytes)

    # Trust the bytes, never the client's Content-Type header. The filename only
    # breaks ties the bytes genuinely cannot — see _sniff_mime.
    mime = _sniff_mime(raw, filename)

    # Allow-list, so anything unrecognised fails closed.
    if mime not in ALLOWED_MIME_TYPES:
        raise UnsupportedFileTypeError(mime)

    # Unguarded on purpose: user_id arrives from the token, and get_current_user
    # has already rejected a claim that doesn't parse. Reaching here with a bad
    # one means the trust boundary was bypassed — a caller bug that should raise
    # loudly rather than be quietly absorbed.
    owner = ObjectId(user_id)

    # Blob BEFORE record: a crash here orphans a file (invisible, sweepable),
    # whereas a record written before its blob would 404 forever.
    # Note storage takes the *string* user_id — keys are strings.
    blob = await storage.put(user_id, raw)

    now = datetime.now(UTC)
    document = {
        "user_id": owner,
        "original_filename": filename,
        "mime_type": mime,
        "size_bytes": blob.size_bytes,
        "content_hash": blob.content_hash,
        "storage_key": blob.key,
        "status": DocumentStatus.pending.value,
        # Required by DOCUMENTS_VALIDATOR even when empty — omit it and Mongo
        # rejects the write.
        "error": None,
        # Ingestion hasn't run yet; the pipeline increments this when it claims
        # the document.
        "attempts": 0,
        "created_at": now,
        "updated_at": now,
    }

    try:
        result = await db[COLLECTION_NAME].insert_one(document)
    except DuplicateKeyError:
        # The unique (user_id, content_hash) index caught a re-upload. Return
        # what's already there — storage.put was a no-op, so nothing was
        # written twice. Catching beats checking first: two concurrent uploads
        # would both pass a check-then-insert and one would still fail here.
        existing = await db[COLLECTION_NAME].find_one(
            {"user_id": owner, "content_hash": blob.content_hash}
        )
        if existing is None:
            # Lost a race: the duplicate was removed between our insert failing
            # and this lookup. Re-raise the original error rather than crash on
            # None — retrying the insert would now succeed.
            raise
        logger.info(
            "duplicate upload ignored", extra={"document_id": str(existing["_id"])}
        )
        return _to_response(existing), False

    logger.info("document created", extra={"document_id": str(result.inserted_id)})
    return _to_response({**document, "_id": result.inserted_id}), True


async def get_document(
    db: AsyncDatabase, user_id: str, document_id: str
) -> DocumentResponse:
    """Fetch one of *this user's* documents.

    user_id is part of the query rather than an ownership check afterwards, so
    there is no window in which another tenant's document is in hand. A miss
    raises DocumentNotFoundError -> 404, never 403: a 403 would confirm the id
    exists, which is itself a leak.
    """
    # document_id comes from the URL, so it may be anything. user_id comes from
    # the token and get_current_user has already checked it parses.
    try:
        oid = ObjectId(document_id)
    except (InvalidId, TypeError) as exc:
        # A malformed id can never match a real document: that's a miss, not a
        # server error.
        raise DocumentNotFoundError(document_id) from exc

    doc = await db[COLLECTION_NAME].find_one({"_id": oid, "user_id": ObjectId(user_id)})
    if doc is None:
        raise DocumentNotFoundError(document_id)
    return _to_response(doc)


async def get_document_record(db: AsyncDatabase, document_id: str) -> dict | None:
    """Fetch the raw stored document by id, unscoped by user.

    Internal to the ingestion pipeline, which acts on a document it just created
    and needs fields (storage_key, mime_type) that DocumentResponse omits. Not for
    request handlers: those must use get_document, which enforces ownership. Keyed
    on ``_id`` alone for the same reason set_document_status is — no user request,
    no tenant boundary to cross.
    """
    return await db[COLLECTION_NAME].find_one({"_id": ObjectId(document_id)})


async def list_documents(
    db: AsyncDatabase, user_id: str, limit: int = 100
) -> list[DocumentResponse]:
    """Return a user's documents, newest first.

    Capped deliberately: to_list() with no length pulls the whole result set
    into memory. Real pagination can replace this later.
    """
    cursor = (
        db[COLLECTION_NAME].find({"user_id": ObjectId(user_id)}).sort("created_at", -1)
    )
    docs = await cursor.to_list(length=limit)
    return [_to_response(d) for d in docs]


async def claim_document_for_processing(db: AsyncDatabase, document_id: str) -> None:
    """Move a document to `processing` and count the attempt, atomically.

    Combines the status transition with an ``attempts`` increment so every run of
    the pipeline is counted exactly once, at the point it claims the work. The
    startup recovery sweep reads ``attempts`` to stop retrying a document that has
    already failed too many times. ``error`` is cleared here so a retry doesn't
    keep showing the previous failure's reason while it reprocesses.
    """
    await db[COLLECTION_NAME].update_one(
        {"_id": ObjectId(document_id)},
        {
            "$set": {
                "status": DocumentStatus.processing.value,
                "error": None,
                "updated_at": datetime.now(UTC),
            },
            "$inc": {"attempts": 1},
        },
    )


async def set_document_status(
    db: AsyncDatabase,
    document_id: str,
    status: DocumentStatus,
    error: str | None = None,
) -> None:
    """Move a document to a new pipeline status.

    Used by the ingestion pipeline to record progress: pending -> processing ->
    ready, or -> failed with the reason in ``error``. Keyed on ``_id`` alone (not
    ``user_id``): the caller is the pipeline acting on a document it already
    created, not a user request, so there is no tenant boundary to enforce here.

    ``error`` is written every time — cleared to ``None`` on the success path —
    so a document that failed once and is later reprocessed doesn't keep a stale
    error, and the field always matches the status.
    """
    await db[COLLECTION_NAME].update_one(
        {"_id": ObjectId(document_id)},
        {
            "$set": {
                "status": status.value,
                "error": error,
                "updated_at": datetime.now(UTC),
            }
        },
    )
