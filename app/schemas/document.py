"""Document request/response schemas.

Upload is ``multipart/form-data`` (a raw file), so there is no request body model
here — the router takes FastAPI's ``UploadFile`` directly. Only responses are
modelled.
"""

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel


class DocumentStatus(StrEnum):
    """Where a document is in the ingestion pipeline.

        pending -> processing -> ready
                              -> failed

    ``pending`` is set at upload; the pipeline moves it forward from there.
    Using an enum (as with ``UserRole``) validates the value automatically —
    a plain ``str`` would happily accept a typo.
    """

    pending = "pending"
    processing = "processing"
    ready = "ready"
    failed = "failed"


class DocumentResponse(BaseModel):
    """A document as returned by the API.

    Note there are no defaults: every field is read from the stored document, so
    a default would let a bug that failed to set a value report a plausible-
    looking one instead of failing loudly.
    """

    # In MongoDB the primary key is the ObjectId `_id`, exposed here as a string.
    id: str
    original_filename: str
    # Sniffed from the file's magic bytes at upload — not the client's
    # Content-Type header, which is caller-supplied and can lie.
    mime_type: str
    size_bytes: int
    status: DocumentStatus
    # Only populated when status is `failed`; carries why ingestion stopped.
    error: str | None = None
    created_at: datetime
