"""User database model (the MongoDB equivalent of a SQLAlchemy table).

MongoDB is schemaless, so unlike SQLAlchemy there is no table-creating class.
Instead we define the document shape here in three complementary ways:

1. ``UserDocument`` — a Pydantic model documenting/validating the shape in
   application code (the closest analog to your ``User(Base)`` class).
2. ``USERS_INDEXES`` — indexes, including the ``unique=True`` constraints that
   in SQLAlchemy lived on the columns.
3. ``USERS_VALIDATOR`` — a MongoDB ``$jsonSchema`` validator that enforces the
   shape at the *database* level (the real equivalent of a table schema).

These are applied at startup by ``app.db.mongodb``.
"""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.auth import UserRole

COLLECTION_NAME = "users"


class UserDocument(BaseModel):
    """Shape of a document in the ``users`` collection."""

    # Mongo's primary key is ``_id`` (an ObjectId); we carry it as a string.
    model_config = ConfigDict(populate_by_name=True)

    id: str | None = Field(default=None, alias="_id")
    email: str
    username: str
    hashed_password: str
    is_active: bool = True
    role: UserRole = UserRole.user
    created_at: datetime


# Column-level `unique=True` in SQLAlchemy becomes unique indexes here.
# (field_name, options) pairs consumed by create_index.
USERS_INDEXES: list[tuple[str, dict]] = [
    ("email", {"unique": True}),
    ("username", {"unique": True}),
]

# The database-level schema. MongoDB validates every insert/update against this,
# rejecting documents that don't match — the true equivalent of a table schema.
USERS_VALIDATOR: dict = {
    "$jsonSchema": {
        "bsonType": "object",
        "required": [
            "email",
            "username",
            "hashed_password",
            "is_active",
            "role",
            "created_at",
        ],
        "properties": {
            "email": {"bsonType": "string"},
            "username": {"bsonType": "string"},
            "hashed_password": {"bsonType": "string"},
            "is_active": {"bsonType": "bool"},
            "role": {"enum": [role.value for role in UserRole]},
            "created_at": {"bsonType": "date"},
        },
    }
}
