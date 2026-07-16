"""Integration tests for app/db/mongodb.py against a real MongoDB server.

These cover what the in-memory fake in conftest.py cannot model: the
``$jsonSchema`` validators, real index creation (including the TTL indexes),
and the idempotency of ``_ensure_schema``. The fake emulates unique indexes in
Python; only a real server can tell us the *database* enforces them.

They run against a throwaway database that is dropped afterwards, and skip
entirely when no MongoDB is reachable, so the suite still passes without Docker.

Start a server with:  uv run poe db-up
"""

from datetime import UTC, datetime

import pytest
import pytest_asyncio
from pymongo import AsyncMongoClient
from pymongo.errors import DuplicateKeyError, WriteError

from app.core.config import get_settings
from app.db import mongodb
from app.models import refresh_token as refresh_token_model
from app.models import revoked_token as revoked_token_model
from app.models import user as user_model

TEST_DB_NAME = "document_rag_test"

# Reachability is probed once per session rather than per test: when no server is
# running, each probe costs the full serverSelectionTimeoutMS, and paying that
# once per test would add ~18s to a suite that is meant to skip quickly.
_reachable: bool | None = None


async def _mongo_is_reachable(uri: str) -> bool:
    global _reachable
    if _reachable is None:
        probe = AsyncMongoClient(uri, serverSelectionTimeoutMS=1500)
        try:
            await probe.admin.command("ping")
            _reachable = True
        except Exception:
            _reachable = False
        finally:
            await probe.close()
    return _reachable


@pytest_asyncio.fixture
async def mongo_db():
    """Connect to a real server against a throwaway database, apply the schema,
    then drop it and reset module state."""
    settings = get_settings()

    if not await _mongo_is_reachable(settings.mongodb_uri):
        pytest.skip(
            f"no MongoDB reachable at {settings.mongodb_uri} — try: uv run poe db-up"
        )

    await mongodb.connect_to_mongo(
        settings.model_copy(update={"mongodb_db_name": TEST_DB_NAME})
    )
    try:
        yield mongodb.get_database()
    finally:
        # Always drop the throwaway DB and clear module state: the readiness
        # tests assert /ready is 503 precisely because no client is connected,
        # so leaking a live client here would break them depending on order.
        if mongodb._state.client is not None:
            await mongodb._state.client.drop_database(TEST_DB_NAME)
        await mongodb.close_mongo_connection()


def _user_doc(**overrides) -> dict:
    doc = {
        "email": "alice@example.com",
        "username": "alice",
        "hashed_password": "not-a-real-bcrypt-hash",
        "is_active": True,
        "role": "user",
        "created_at": datetime.now(UTC),
    }
    doc.update(overrides)
    return doc


async def test_connect_creates_every_collection(mongo_db):
    names = await mongo_db.list_collection_names()

    assert user_model.COLLECTION_NAME in names
    assert refresh_token_model.COLLECTION_NAME in names
    assert revoked_token_model.COLLECTION_NAME in names


async def test_ensure_schema_is_idempotent(mongo_db):
    """It runs on every startup, so a second pass over existing collections must
    update the validator via collMod rather than fail on "already exists"."""
    await mongodb._ensure_schema()
    await mongodb._ensure_schema()

    assert user_model.COLLECTION_NAME in await mongo_db.list_collection_names()


async def test_ping_returns_true_when_connected(mongo_db):
    assert await mongodb.ping() is True


# --- users: unique indexes enforced by the database itself ---


async def test_users_email_index_rejects_a_duplicate(mongo_db):
    await mongo_db[user_model.COLLECTION_NAME].insert_one(_user_doc())

    with pytest.raises(DuplicateKeyError):
        await mongo_db[user_model.COLLECTION_NAME].insert_one(
            _user_doc(username="alice2")
        )


async def test_users_username_index_rejects_a_duplicate(mongo_db):
    await mongo_db[user_model.COLLECTION_NAME].insert_one(_user_doc())

    with pytest.raises(DuplicateKeyError):
        await mongo_db[user_model.COLLECTION_NAME].insert_one(
            _user_doc(email="other@example.com")
        )


async def test_username_index_is_case_sensitive(mongo_db):
    """Pins the deliberate design recorded in app/schemas/auth.py against the
    real database: the unique index compares raw bytes with no collation, so
    "Abhi" and "abhi" are distinct keys and both inserts succeed."""
    users = mongo_db[user_model.COLLECTION_NAME]
    await users.insert_one(_user_doc(username="Abhi", email="a@example.com"))
    await users.insert_one(_user_doc(username="abhi", email="b@example.com"))

    assert await users.count_documents({}) == 2


# --- users: $jsonSchema validator enforced by the database itself ---


async def test_users_validator_rejects_missing_required_fields(mongo_db):
    with pytest.raises(WriteError):
        await mongo_db[user_model.COLLECTION_NAME].insert_one(
            {"email": "alice@example.com"}
        )


async def test_users_validator_rejects_a_wrong_bson_type(mongo_db):
    with pytest.raises(WriteError):
        await mongo_db[user_model.COLLECTION_NAME].insert_one(_user_doc(is_active="yes"))


async def test_users_validator_rejects_an_unknown_role(mongo_db):
    """role is an enum in the validator, so the DB refuses values outside it even
    if application-level validation were bypassed."""
    with pytest.raises(WriteError):
        await mongo_db[user_model.COLLECTION_NAME].insert_one(
            _user_doc(role="superadmin")
        )


# --- token collections: TTL indexes ---


async def test_refresh_tokens_has_a_ttl_index_on_expires_at(mongo_db):
    info = await mongo_db[refresh_token_model.COLLECTION_NAME].index_information()

    assert info["expires_at_1"]["expireAfterSeconds"] == 0
    assert info["token_hash_1"]["unique"] is True


async def test_revoked_tokens_has_a_ttl_index_on_expires_at(mongo_db):
    info = await mongo_db[revoked_token_model.COLLECTION_NAME].index_information()

    assert info["expires_at_1"]["expireAfterSeconds"] == 0
    assert info["jti_1"]["unique"] is True


async def test_refresh_tokens_token_hash_index_rejects_a_duplicate(mongo_db):
    doc = {
        "user_id": "user123",
        "token_hash": "a" * 64,
        "expires_at": datetime.now(UTC),
        "created_at": datetime.now(UTC),
    }
    await mongo_db[refresh_token_model.COLLECTION_NAME].insert_one(doc)

    with pytest.raises(DuplicateKeyError):
        await mongo_db[refresh_token_model.COLLECTION_NAME].insert_one(
            {**doc, "user_id": "user456"}
        )
