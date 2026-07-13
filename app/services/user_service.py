"""User management: creating and (later) fetching/updating users.

This service owns the user lifecycle. Authentication concerns (verifying
credentials, issuing tokens) live in ``auth_service``. Both share the password
primitives in ``app.core.security``.
"""

import logging
from datetime import UTC, datetime

from pymongo.asynchronous.database import AsyncDatabase
from pymongo.errors import DuplicateKeyError

from app.core.security import hash_password
from app.models.user import COLLECTION_NAME
from app.schemas.auth import CreateUserRequest, UserResponse

logger = logging.getLogger(__name__)


class DuplicateUserError(Exception):
    """Raised when a user's email or username is already registered.

    ``field`` is the conflicting field ("email" or "username") when known.
    """

    def __init__(self, field: str | None = None) -> None:
        self.field = field
        super().__init__("Email or username already exists")


async def create_user(db: AsyncDatabase, payload: CreateUserRequest) -> UserResponse:
    """Create a new user with a hashed password.

    Uniqueness is enforced by the unique indexes on ``email`` and ``username``
    (see app.models.user). We attempt the insert and translate a duplicate-key
    error into a domain error — this is atomic and race-safe, unlike a
    check-then-insert.
    """
    document = {
        "email": payload.email.lower(),
        "username": payload.username,
        "hashed_password": hash_password(payload.password),
        "is_active": True,
        "role": payload.role.value,
        "created_at": datetime.now(UTC),
    }

    try:
        result = await db[COLLECTION_NAME].insert_one(document)
    except DuplicateKeyError as exc:
        # exc.details["keyPattern"] tells us which unique index was violated.
        field = next(iter((exc.details or {}).get("keyPattern", {})), None)
        logger.warning("duplicate registration attempt", extra={"field": field})
        raise DuplicateUserError(field) from exc

    logger.info("user registered", extra={"user_id": str(result.inserted_id)})
    return UserResponse(
        id=str(result.inserted_id),
        email=document["email"],
        username=document["username"],
        role=payload.role,
        is_active=True,
    )
