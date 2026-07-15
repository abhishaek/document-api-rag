"""Refresh-token database model.

Refresh tokens are long-lived credentials, so unlike the JWT access token they
are *stateful*: we persist a record per issued token so it can be looked up,
rotated, and revoked. We store only the SHA-256 **hash** of the token — never
the raw value — so a database leak can't be replayed against the API (the same
reasoning as never storing plaintext passwords).

As with ``app.models.user`` the shape is defined three ways:

1. ``RefreshTokenDocument`` — the Pydantic shape used in application code.
2. ``REFRESH_TOKENS_INDEXES`` — a unique index on ``token_hash`` (fast lookup +
   no duplicates) and a **TTL index** on ``expires_at`` so MongoDB deletes
   expired tokens automatically (``expireAfterSeconds=0`` = "delete once the
   date in this field is in the past").
3. ``REFRESH_TOKENS_VALIDATOR`` — the database-level ``$jsonSchema`` validator.

These are applied at startup by ``app.db.mongodb``.
"""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

COLLECTION_NAME = "refresh_tokens"


class RefreshTokenDocument(BaseModel):
    """Shape of a document in the ``refresh_tokens`` collection."""

    model_config = ConfigDict(populate_by_name=True)

    id: str | None = Field(default=None, alias="_id")
    user_id: str
    token_hash: str
    expires_at: datetime
    created_at: datetime


# (field_name, options) pairs consumed by create_index.
REFRESH_TOKENS_INDEXES: list[tuple[str, dict]] = [
    ("token_hash", {"unique": True}),
    # TTL index: Mongo's background task purges a document once ``expires_at``
    # has passed. expireAfterSeconds=0 means "expire exactly at expires_at".
    ("expires_at", {"expireAfterSeconds": 0}),
]

REFRESH_TOKENS_VALIDATOR: dict = {
    "$jsonSchema": {
        "bsonType": "object",
        "required": ["user_id", "token_hash", "expires_at", "created_at"],
        "properties": {
            "user_id": {"bsonType": "string"},
            "token_hash": {"bsonType": "string"},
            "expires_at": {"bsonType": "date"},
            "created_at": {"bsonType": "date"},
        },
    }
}
