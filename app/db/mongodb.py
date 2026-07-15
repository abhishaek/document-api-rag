"""MongoDB connection management.

One ``AsyncMongoClient`` is created per process at startup and shared across all
requests (the client maintains its own connection pool — never open a client per
request). The lifecycle is driven by the app's ``lifespan`` in ``app.main``:

    startup   -> connect_to_mongo(settings)
    shutdown  -> close_mongo_connection()

Route handlers get the database via the ``get_database`` dependency:

    from fastapi import Depends
    from pymongo.asynchronous.database import AsyncDatabase
    from app.db.mongodb import get_database

    @router.get("/things")
    async def list_things(db: AsyncDatabase = Depends(get_database)):
        return await db.things.find().to_list(length=100)
"""

import logging

from pymongo import AsyncMongoClient
from pymongo.asynchronous.database import AsyncDatabase

from app.core.config import Settings
from app.models import refresh_token as refresh_token_model
from app.models import revoked_token as revoked_token_model
from app.models import user as user_model

logger = logging.getLogger(__name__)


class _MongoState:
    """Holds the process-wide client and database handles."""

    client: AsyncMongoClient | None = None
    database: AsyncDatabase | None = None


_state = _MongoState()


async def connect_to_mongo(settings: Settings) -> None:
    """Open the client and verify connectivity with a ping. Called at startup."""
    _state.client = AsyncMongoClient(
        settings.mongodb_uri,
        # Fail fast instead of hanging if the server is unreachable.
        serverSelectionTimeoutMS=5000,
        appname=settings.app_name,
    )
    _state.database = _state.client[settings.mongodb_db_name]

    # Force an actual round-trip so startup fails loudly if Mongo is down.
    await _state.client.admin.command("ping")
    logger.info("connected to mongodb", extra={"db": settings.mongodb_db_name})

    await _ensure_schema()


async def _ensure_collection(
    db: AsyncDatabase,
    name: str,
    validator: dict,
    indexes: list[tuple[str, dict]],
) -> None:
    """Apply a collection's $jsonSchema validator and indexes. Creates the
    collection with the validator if missing, otherwise updates it via collMod.
    Idempotent."""
    existing = await db.list_collection_names()
    if name not in existing:
        await db.create_collection(name, validator=validator)
    else:
        await db.command("collMod", name, validator=validator)
    for field, options in indexes:
        await db[name].create_index(field, **options)


async def _ensure_schema() -> None:
    """Apply DB-level schema (validators + indexes) for every collection.
    Idempotent — safe to run on every startup."""
    db = _state.database
    assert db is not None

    await _ensure_collection(
        db, user_model.COLLECTION_NAME, user_model.USERS_VALIDATOR, user_model.USERS_INDEXES
    )
    await _ensure_collection(
        db,
        refresh_token_model.COLLECTION_NAME,
        refresh_token_model.REFRESH_TOKENS_VALIDATOR,
        refresh_token_model.REFRESH_TOKENS_INDEXES,
    )
    await _ensure_collection(
        db,
        revoked_token_model.COLLECTION_NAME,
        revoked_token_model.REVOKED_TOKENS_VALIDATOR,
        revoked_token_model.REVOKED_TOKENS_INDEXES,
    )

    logger.info("ensured mongodb schema and indexes")


async def close_mongo_connection() -> None:
    """Close the client. Called at shutdown."""
    if _state.client is not None:
        await _state.client.close()
        _state.client = None
        _state.database = None
        logger.info("closed mongodb connection")


def get_database() -> AsyncDatabase:
    """FastAPI dependency: return the shared database handle.

    Raises if called before startup completed (misconfiguration, not a runtime
    condition), so failures surface clearly during development.
    """
    if _state.database is None:
        raise RuntimeError("MongoDB is not initialized; did startup run?")
    return _state.database


async def ping() -> bool:
    """Return True if the database answers a ping. Used by the readiness probe."""
    if _state.client is None:
        return False
    try:
        await _state.client.admin.command("ping")
        return True
    except Exception:
        logger.warning("mongodb ping failed", exc_info=True)
        return False
