"""Authentication & authorization logic.

This service owns credential verification and the token lifecycle. It is being
built out — planned functions:

* ``authenticate_user(username, password, db)`` — verify credentials, return the
  user (uses ``verify_password`` from ``app.core.security``).
* ``create_access_token(...)`` — issue a short-lived JWT.
* ``create_refresh_token`` / ``verify_refresh_token`` / ``rotate_refresh_token``
  — manage long-lived refresh tokens in a ``refresh_tokens`` collection
  (store only the SHA-256 hash, plus a TTL index for automatic expiry).

User creation lives in ``app.services.user_service`` (user management, not auth).
"""

import hashlib
import secrets
from datetime import UTC, datetime, timedelta

import jwt
from pymongo.asynchronous.database import AsyncDatabase
from pymongo.errors import DuplicateKeyError

from app.core.config import get_settings
from app.core.security import hash_password, verify_password
from app.models.refresh_token import COLLECTION_NAME as REFRESH_COLLECTION
from app.models.revoked_token import COLLECTION_NAME as REVOKED_COLLECTION
from app.models.user import COLLECTION_NAME

settings = get_settings()

# Precomputed hash of a throwaway password. When a login names a user that does
# not exist we still run verify_password against this, so the response time
# matches the "wrong password for a real user" path — otherwise timing would
# leak which usernames are registered.
_DUMMY_HASH = hash_password("not-a-real-password")


def create_access_token(username: str, user_id: str, role: str) -> str:
    payload = {
        "sub": username,
        "id": user_id,
        "role": role,
        # A unique token id so this specific token can be revoked (see the
        # revoked_tokens denylist used on logout).
        "jti": secrets.token_urlsafe(16),
        "exp": datetime.now(UTC)
        + timedelta(minutes=settings.access_token_expire_minutes),
    }
    return jwt.encode(payload, settings.secret_key, algorithm=settings.jwt_algorithm)


def _hash_token(raw: str) -> str:
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


async def authenticate_user(
    username: str, password: str, db: AsyncDatabase
) -> dict | None:
    """Return the user document if credentials are valid and the account is
    active, otherwise ``None``."""
    user = await db[COLLECTION_NAME].find_one({"username": username, "is_active": True})
    if user is None:
        # Burn equivalent time so a missing user is indistinguishable by timing
        # from a wrong password on a real user.
        verify_password(password, _DUMMY_HASH)
        return None
    if not verify_password(password, user["hashed_password"]):
        return None
    return user


async def create_refresh_token(user_id: str, db: AsyncDatabase) -> str:
    """Issue a new refresh token: generate a random value, store only its hash
    (with an expiry), and return the raw token to the caller (the only place it
    ever exists in plaintext)."""
    raw = secrets.token_urlsafe(32)
    now = datetime.now(UTC)
    await db[REFRESH_COLLECTION].insert_one(
        {
            "user_id": user_id,
            "token_hash": _hash_token(raw),
            "expires_at": now + timedelta(days=settings.refresh_token_expire_days),
            "created_at": now,
        }
    )
    return raw


async def verify_refresh_token(raw_token: str, db: AsyncDatabase) -> dict | None:
    """Return the stored token document if the raw token matches a stored hash
    and has not expired, otherwise ``None``.

    We check ``expires_at > now`` explicitly rather than trusting the TTL index:
    the TTL purge runs only about once a minute, so an expired token may still
    be present for a short window.
    """
    return await db[REFRESH_COLLECTION].find_one(
        {
            "token_hash": _hash_token(raw_token),
            "expires_at": {"$gt": datetime.now(UTC)},
        }
    )


async def rotate_refresh_token(old_token: dict, db: AsyncDatabase) -> str:
    """Invalidate the used refresh token and issue a fresh one for the same user
    (rotation limits the damage if a token is ever leaked). ``old_token`` is a
    document as returned by ``verify_refresh_token``."""
    await db[REFRESH_COLLECTION].delete_one({"_id": old_token["_id"]})
    return await create_refresh_token(old_token["user_id"], db)


async def revoke_refresh_token(raw_token: str, db: AsyncDatabase) -> None:
    """Delete a refresh token by its raw value (used on logout). Matches on the
    stored hash and removes it regardless of expiry; a no-op if it isn't found."""
    await db[REFRESH_COLLECTION].delete_one({"token_hash": _hash_token(raw_token)})


async def revoke_access_token(jti: str, expires_at: datetime, db: AsyncDatabase) -> None:
    """Add an access token's ``jti`` to the denylist so it stops being accepted
    before its natural expiry. ``expires_at`` drives the TTL cleanup. Idempotent:
    revoking an already-revoked token is a no-op."""
    try:
        await db[REVOKED_COLLECTION].insert_one(
            {"jti": jti, "expires_at": expires_at}
        )
    except DuplicateKeyError:
        pass


async def is_access_token_revoked(jti: str, db: AsyncDatabase) -> bool:
    """Return True if this access token id is on the denylist."""
    return await db[REVOKED_COLLECTION].find_one({"jti": jti}) is not None