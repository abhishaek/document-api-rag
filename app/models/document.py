"""Document database model (the MongoDB equivalent of a SQLAlchemy table).

MongoDB is schemaless, so unlike SQLAlchemy there is no table-creating class.
Instead we define the document shape here in three complementary ways:

1. ``DocumentDocument`` — a Pydantic model documenting/validating the shape in
   application code (the closest analog to a ``Document(Base)`` class).
2. ``DOCUMENTS_INDEXES`` — indexes, including the ``unique=True`` constraints
   that in SQLAlchemy lived on the columns.
3. ``DOCUMENTS_VALIDATOR`` — a MongoDB ``$jsonSchema`` validator that enforces
   the shape at the *database* level (the real equivalent of a table schema).

These are applied at startup by ``app.db.mongodb``.

A record here holds **metadata only**. The uploaded file's bytes live in blob
storage (see ``app.services.storage_service``) and are reached via
``storage_key``, keeping this collection small and fast to query.
"""

from datetime import datetime

from bson import ObjectId
from pydantic import BaseModel, ConfigDict, Field

from app.schemas.document import DocumentStatus

COLLECTION_NAME = "documents"

# An index spec consumed by ``create_index(field, **options)``: ``field`` is a
# plain field name for a single-field index, or a list of (field, direction)
# pairs for a compound one.
IndexSpec = tuple[str | list[tuple[str, int]], dict]


class DocumentDocument(BaseModel):
    """Shape of a document in the ``documents`` collection.

    (The doubled name follows the ``<Entity>Document`` convention — here the
    entity happens to be a document itself.)
    """

    # ObjectId is not a Pydantic-native type, so it must be allowed explicitly.
    model_config = ConfigDict(populate_by_name=True, arbitrary_types_allowed=True)

    # This record's own primary key, assigned by Mongo on insert.
    id: ObjectId | None = Field(default=None, alias="_id")
    # The owner: a foreign key to ``users._id``. Distinct from ``id`` above —
    # ``_id`` identifies *this document*, ``user_id`` identifies *who uploaded
    # it*. Every read filters on this; it is what keeps tenants separated.
    user_id: ObjectId
    # What the user called the file. The blob on disk is named by its hash, so
    # this is the only place the original name survives.
    original_filename: str
    # Sniffed from the file's magic bytes at upload — not the client's
    # Content-Type header, which is caller-supplied and can lie.
    mime_type: str
    size_bytes: int
    # SHA-256 of the file's bytes. Combined with user_id it forms storage_key,
    # and the unique index below uses it to make re-uploads idempotent.
    content_hash: str
    # Where the bytes live: "{user_id}/{content_hash}".
    storage_key: str
    status: DocumentStatus
    # Only populated when status is `failed`; carries why ingestion stopped.
    error: str | None = None
    # How many times ingestion has been attempted. Incremented each time the
    # pipeline claims the document (moves it to `processing`). The startup
    # recovery sweep uses it to stop retrying a document that keeps failing.
    attempts: int = 0
    created_at: datetime
    # Bumped on every status transition.
    updated_at: datetime


# Column-level `unique=True` in SQLAlchemy becomes unique indexes here.
# (field_name, options) pairs consumed by create_index.
DOCUMENTS_INDEXES: list[IndexSpec] = [
    # Every read is scoped to the owner, so this is the workhorse index.
    ("user_id", {}),
    # Re-uploading a file the user already has must not create a second record:
    # it would be parsed, chunked, and embedded again — paying twice and, worse,
    # putting duplicate chunks in the vector index, where they crowd out
    # genuinely distinct results at retrieval time. The unique index lets the
    # database enforce that, so create_document can catch DuplicateKeyError and
    # return the existing record rather than racing a check-then-insert.
    ([("user_id", 1), ("content_hash", 1)], {"unique": True}),
]

# The database-level schema. MongoDB validates every insert/update against this,
# rejecting documents that don't match — the true equivalent of a table schema.
DOCUMENTS_VALIDATOR: dict = {
    "$jsonSchema": {
        "bsonType": "object",
        "required": [
            "user_id",
            "original_filename",
            "mime_type",
            "size_bytes",
            "content_hash",
            "storage_key",
            "status",
            "error",
            "attempts",
            "created_at",
            "updated_at",
        ],
        "properties": {
            "user_id": {"bsonType": "objectId"},
            "original_filename": {"bsonType": "string"},
            "mime_type": {"bsonType": "string"},
            # PyMongo encodes a Python int as int32 when it fits and int64 when
            # it doesn't. Today's 25 MiB cap always fits, but accepting both
            # means raising the cap later can't start rejecting writes here.
            "size_bytes": {"bsonType": ["int", "long"]},
            "content_hash": {"bsonType": "string"},
            "storage_key": {"bsonType": "string"},
            "status": {"enum": [status.value for status in DocumentStatus]},
            # Required but nullable: the field is always written, so every record
            # has the same shape and the validator can insist it be present.
            "error": {"bsonType": ["string", "null"]},
            "attempts": {"bsonType": ["int", "long"]},
            "created_at": {"bsonType": "date"},
            "updated_at": {"bsonType": "date"},
        },
    }
}
