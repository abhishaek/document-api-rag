"""Revoked access-token denylist model.

Access tokens are stateless JWTs — normally there's no way to invalidate one
before it expires. To support "log out now" we keep a small denylist of the
``jti`` (JWT ID) of every access token that has been explicitly revoked. On each
authenticated request we check this list; a hit means the token is dead.

The TTL index on ``expires_at`` lets MongoDB drop each entry once the token it
refers to would have expired anyway, so the collection never grows unbounded —
we only need to remember a revoked token until its natural expiry.
"""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

COLLECTION_NAME = "revoked_tokens"


class RevokedTokenDocument(BaseModel):
    """Shape of a document in the ``revoked_tokens`` collection."""

    model_config = ConfigDict(populate_by_name=True)

    id: str | None = Field(default=None, alias="_id")
    jti: str
    expires_at: datetime


REVOKED_TOKENS_INDEXES: list[tuple[str, dict]] = [
    ("jti", {"unique": True}),
    # TTL: purge the entry once the revoked token has expired on its own.
    ("expires_at", {"expireAfterSeconds": 0}),
]

REVOKED_TOKENS_VALIDATOR: dict = {
    "$jsonSchema": {
        "bsonType": "object",
        "required": ["jti", "expires_at"],
        "properties": {
            "jti": {"bsonType": "string"},
            "expires_at": {"bsonType": "date"},
        },
    }
}
