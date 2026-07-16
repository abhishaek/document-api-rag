"""Unit tests for token creation and credential verification in the auth service."""

from datetime import UTC, datetime, timedelta

import jwt
import pytest

from app.core.config import get_settings
from app.core.security import hash_password
from app.models.refresh_token import COLLECTION_NAME as REFRESH_COLLECTION
from app.models.revoked_token import COLLECTION_NAME as REVOKED_COLLECTION
from app.services.auth_service import (
    _hash_token,
    authenticate_user,
    create_access_token,
    create_refresh_token,
    is_access_token_revoked,
    revoke_access_token,
    revoke_refresh_token,
    rotate_refresh_token,
    verify_refresh_token,
)


def _decode(token: str) -> dict:
    settings = get_settings()
    return jwt.decode(token, settings.secret_key, algorithms=[settings.jwt_algorithm])


def test_create_access_token_embeds_claims():
    token = create_access_token("alice", "507f1f77bcf86cd799439011", "user")

    assert isinstance(token, str)
    claims = _decode(token)
    assert claims["sub"] == "alice"
    assert claims["id"] == "507f1f77bcf86cd799439011"
    assert claims["role"] == "user"


def test_create_access_token_expiry_is_in_the_future():
    settings = get_settings()
    token = create_access_token("bob", "abc123", "admin")

    claims = _decode(token)
    now = datetime.now(UTC).timestamp()
    # exp should be roughly access_token_expire_minutes ahead, and never in the past.
    assert claims["exp"] > now
    assert claims["exp"] <= now + settings.access_token_expire_minutes * 60 + 5


def test_create_access_token_is_verifiable_with_configured_secret():
    """A token signed with the app secret must fail verification under a wrong key."""
    settings = get_settings()
    token = create_access_token("carol", "id999", "user")

    # Correct key decodes; a different key raises.
    _decode(token)
    try:
        jwt.decode(token, "wrong-secret-" + "x" * 32, algorithms=[settings.jwt_algorithm])
    except jwt.InvalidSignatureError:
        pass
    else:  # pragma: no cover - defensive
        raise AssertionError("token verified under the wrong secret")


def _seed_user(fake_db, *, username="alice", password="s3cret", is_active=True):
    """Insert a user document into the fake ``users`` collection."""
    from app.models.user import COLLECTION_NAME

    fake_db[COLLECTION_NAME].docs.append(
        {
            "_id": "fakeid1",
            "email": f"{username}@example.com",
            "username": username,
            "hashed_password": hash_password(password),
            "is_active": is_active,
            "role": "user",
        }
    )


@pytest.mark.asyncio
async def test_authenticate_user_valid_credentials(fake_db):
    _seed_user(fake_db, username="alice", password="s3cret")

    user = await authenticate_user("alice", "s3cret", fake_db)

    assert user is not None
    assert user["username"] == "alice"


@pytest.mark.asyncio
async def test_authenticate_user_wrong_password(fake_db):
    _seed_user(fake_db, username="alice", password="s3cret")

    assert await authenticate_user("alice", "wrong", fake_db) is None


@pytest.mark.asyncio
async def test_authenticate_user_unknown_username(fake_db):
    assert await authenticate_user("nobody", "whatever", fake_db) is None


@pytest.mark.asyncio
async def test_authenticate_user_inactive_account_is_rejected(fake_db):
    _seed_user(fake_db, username="alice", password="s3cret", is_active=False)

    # Correct password, but the account is inactive → no match.
    assert await authenticate_user("alice", "s3cret", fake_db) is None


@pytest.mark.asyncio
async def test_create_refresh_token_stores_only_the_hash(fake_db):
    raw = await create_refresh_token("user123", fake_db)

    stored = fake_db[REFRESH_COLLECTION].docs
    assert len(stored) == 1
    # The raw token is never persisted — only its SHA-256 hash.
    assert stored[0]["token_hash"] == _hash_token(raw)
    assert stored[0]["token_hash"] != raw
    assert stored[0]["user_id"] == "user123"
    assert stored[0]["expires_at"] > datetime.now(UTC)


@pytest.mark.asyncio
async def test_verify_refresh_token_roundtrip(fake_db):
    raw = await create_refresh_token("user123", fake_db)

    found = await verify_refresh_token(raw, fake_db)
    assert found is not None
    assert found["user_id"] == "user123"

    # A token that was never issued does not verify.
    assert await verify_refresh_token("bogus-token", fake_db) is None


@pytest.mark.asyncio
async def test_verify_refresh_token_rejects_expired(fake_db):
    # Insert a token whose expiry is already in the past.
    raw = "expired-raw-token"
    fake_db[REFRESH_COLLECTION].docs.append(
        {
            "_id": "fakeid1",
            "user_id": "user123",
            "token_hash": _hash_token(raw),
            "expires_at": datetime.now(UTC) - timedelta(seconds=1),
            "created_at": datetime.now(UTC) - timedelta(days=1),
        }
    )

    assert await verify_refresh_token(raw, fake_db) is None


@pytest.mark.asyncio
async def test_rotate_refresh_token_invalidates_old_and_issues_new(fake_db):
    old_raw = await create_refresh_token("user123", fake_db)
    old_doc = await verify_refresh_token(old_raw, fake_db)

    new_raw = await rotate_refresh_token(old_doc, fake_db)

    assert new_raw != old_raw
    # Old token no longer verifies; new one does.
    assert await verify_refresh_token(old_raw, fake_db) is None
    assert await verify_refresh_token(new_raw, fake_db) is not None
    # Exactly one active token remains for the user.
    assert len(fake_db[REFRESH_COLLECTION].docs) == 1


@pytest.mark.asyncio
async def test_revoke_refresh_token_deletes_it(fake_db):
    raw = await create_refresh_token("user123", fake_db)

    await revoke_refresh_token(raw, fake_db)

    assert await verify_refresh_token(raw, fake_db) is None
    assert fake_db[REFRESH_COLLECTION].docs == []


@pytest.mark.asyncio
async def test_revoke_refresh_token_is_a_noop_for_an_unknown_token(fake_db):
    """Logging out with a token that was never issued must not raise."""
    await revoke_refresh_token("never-issued-token", fake_db)


@pytest.mark.asyncio
async def test_revoke_access_token_adds_jti_to_the_denylist(fake_db):
    expires_at = datetime.now(UTC) + timedelta(minutes=15)

    assert await is_access_token_revoked("jti-abc", fake_db) is False
    await revoke_access_token("jti-abc", expires_at, fake_db)
    assert await is_access_token_revoked("jti-abc", fake_db) is True


@pytest.mark.asyncio
async def test_revoke_access_token_is_idempotent(fake_db):
    """Revoking the same jti twice hits the unique index on revoked_tokens.jti;
    the DuplicateKeyError is swallowed so a double logout is not a 500."""
    expires_at = datetime.now(UTC) + timedelta(minutes=15)

    await revoke_access_token("jti-abc", expires_at, fake_db)
    await revoke_access_token("jti-abc", expires_at, fake_db)

    assert len(fake_db[REVOKED_COLLECTION].docs) == 1


@pytest.mark.asyncio
async def test_revoke_access_token_stores_the_tokens_own_expiry(fake_db):
    """expires_at drives the TTL cleanup, so the denylist entry must expire when
    the token would have expired anyway — not sooner, not never."""
    expires_at = datetime.now(UTC) + timedelta(minutes=15)

    await revoke_access_token("jti-abc", expires_at, fake_db)

    assert fake_db[REVOKED_COLLECTION].docs[0]["expires_at"] == expires_at
