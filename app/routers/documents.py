import logging

from fastapi import (
    APIRouter,
    BackgroundTasks,
    HTTPException,
    Request,
    UploadFile,
    status,
)

from app.core.config import get_settings
from app.core.rate_limit import limiter
from app.dependencies import (
    DbDependency,
    EmbedderDependency,
    StorageDependency,
    UserDependency,
)
from app.schemas.document import DocumentResponse
from app.services.document_service import (
    DocumentNotFoundError,
    FileTooLargeError,
    UnsupportedFileTypeError,
    create_document,
    get_document,
    list_documents,
)
from app.services.ingestion_service import process_document

logger = logging.getLogger(__name__)
settings = get_settings()

router = APIRouter(prefix="/documents", tags=["Documents"])


@router.post(
    "/upload", response_model=DocumentResponse, status_code=status.HTTP_202_ACCEPTED
)
@limiter.limit("10/minute")
async def upload_document(
    request: Request,
    file: UploadFile,
    background_tasks: BackgroundTasks,
    db: DbDependency,
    storage: StorageDependency,
    embedder: EmbedderDependency,
    current_user: UserDependency,
) -> DocumentResponse:
    """Upload a file for the authenticated user and start ingesting it.

    Returns 202 Accepted with the document in `pending`: the response comes back
    immediately, and the slow work (parse -> chunk -> embed) runs in a background
    task that advances the status to `ready` (or `failed`). Poll GET
    /documents/{id} to follow it.

    Re-uploading identical content — even under a different filename — returns
    the existing record rather than creating a duplicate, and does *not* start a
    second ingestion: the service keys on the SHA-256 of the bytes, not the name
    (see create_document).
    """
    if not file.filename:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "A filename is required.")

    # Cheap pre-check: reject on the declared Content-Length before reading the
    # body into memory. The service still re-checks len(raw) as the authority —
    # file.size can be absent or lie.
    if file.size is not None and file.size > settings.max_upload_size_bytes:
        raise HTTPException(
            status.HTTP_413_CONTENT_TOO_LARGE,
            f"File exceeds the {settings.max_upload_size_bytes}-byte limit.",
        )

    raw = await file.read()

    try:
        document, created = await create_document(
            db, storage, current_user["id"], file.filename, raw
        )
    except FileTooLargeError as exc:
        raise HTTPException(
            status.HTTP_413_CONTENT_TOO_LARGE, str(exc)
        ) from exc
    except UnsupportedFileTypeError as exc:
        raise HTTPException(
            status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, str(exc)
        ) from exc

    if created:
        # Only a genuinely new document gets ingested. A duplicate already has (or
        # is having) its chunks built; re-running would waste work and pollute the
        # vector index. add_task runs process_document *after* the response is
        # sent, so the client isn't blocked on parsing/embedding.
        background_tasks.add_task(
            process_document, db, storage, embedder, document.id
        )

    logger.info(
        "document uploaded",
        extra={
            "document_id": document.id,
            "user_id": current_user["id"],
            "newly_created": created,
        },
    )
    return document


@router.get("", response_model=list[DocumentResponse])
async def list_user_documents(
    db: DbDependency, current_user: UserDependency
) -> list[DocumentResponse]:
    """List the authenticated user's documents, newest first."""
    return await list_documents(db, current_user["id"])


@router.get("/{document_id}", response_model=DocumentResponse)
async def read_document(
    document_id: str, db: DbDependency, current_user: UserDependency
) -> DocumentResponse:
    """Fetch one of the authenticated user's documents by id."""
    try:
        return await get_document(db, current_user["id"], document_id)
    except DocumentNotFoundError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(exc)) from exc
