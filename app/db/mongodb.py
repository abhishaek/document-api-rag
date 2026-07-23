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

import asyncio
import logging

from pymongo import AsyncMongoClient
from pymongo.asynchronous.database import AsyncDatabase
from pymongo.operations import SearchIndexModel

from app.core.config import Settings, get_settings
from app.models import chunk as chunk_model
from app.models import document as document_model
from app.models import refresh_token as refresh_token_model
from app.models import revoked_token as revoked_token_model
from app.models import user as user_model

logger = logging.getLogger(__name__)

# One entry in a model's ``*_INDEXES`` list, as consumed by
# ``create_index(field, **options)``: ``field`` is a plain field name for a
# single-field index, or a list of (field, direction) pairs for a compound one.
IndexSpec = tuple[str | list[tuple[str, int]], dict]


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
    indexes: list[IndexSpec],
) -> None:
    """Apply a collection's $jsonSchema validator and indexes. Creates the
    collection with the validator if missing, otherwise updates it via collMod.

    Idempotent, but **additive** for indexes: an index that is no longer declared
    is not dropped. Removing one from a model therefore leaves it in place until
    it is dropped by hand — including in production.
    """
    existing = await db.list_collection_names()
    if name not in existing:
        await db.create_collection(name, validator=validator)
    else:
        await db.command("collMod", name, validator=validator)
    for field, options in indexes:
        await db[name].create_index(field, **options)


def _vector_index_matches(existing_def: dict | None, desired_def: dict) -> bool:
    """Whether an existing vector index definition already matches the desired one.

    Compared on the parts that actually change search behaviour: each field's
    role (``type``) and ``path``, plus the vector field's ``numDimensions`` and
    ``similarity``. Extra keys Atlas may echo back with its own defaults are
    ignored deliberately — otherwise a matching index would look "drifted" and be
    rebuilt on every startup. A change we *do* care about (a filter field added or
    removed, the dimension or similarity changed) alters this set and is caught.
    """

    def field_key(field: dict):
        base = (field.get("type"), field.get("path"))
        if field.get("type") == "vector":
            return base + (field.get("numDimensions"), field.get("similarity"))
        return base

    existing_fields = (existing_def or {}).get("fields", [])
    desired_fields = desired_def.get("fields", [])
    return {field_key(f) for f in existing_fields} == {
        field_key(f) for f in desired_fields
    }


async def _wait_until_search_index_absent(
    collection, name: str, *, attempts: int = 30, delay: float = 1.0
) -> None:
    """Poll until a just-dropped search index is gone, so it can be recreated.

    A drop is asynchronous on Atlas; recreating under the same name before the
    drop settles is rejected. Bounded so a drop that never completes can't hang
    startup — after the budget we fall through and let the recreate attempt (and
    fail into the best-effort handler) rather than block forever.
    """
    for _ in range(attempts):
        cursor = await collection.list_search_indexes()
        if name not in {idx["name"] async for idx in cursor}:
            return
        await asyncio.sleep(delay)


async def _ensure_vector_index(db: AsyncDatabase) -> None:
    """Create *or update* the chunks vector search index so it matches the code.

    Best-effort: the Atlas Search index API only exists on Atlas (Cloud or the
    Atlas Local Docker image). On a server without it this logs a warning and
    returns rather than failing startup — the app still ingests and stores chunks;
    only vector *search* is unavailable until the index exists.

    Idempotent and drift-correcting: if the index is missing it's created; if it
    exists but its definition no longer matches ``build_vector_index_definition``
    (e.g. a new ``filter`` field was added in code), it is dropped and recreated.
    This is why a definition change doesn't need a *manual* drop/recreate — an
    earlier version skipped purely by name, which left a stale index behind. Drop
    + recreate (rather than ``update_search_index``) is used because updating a
    ``vectorSearch`` index is rejected as a text index on some Atlas builds; the
    index is derived from the collection, so nothing is lost — search is only
    briefly unavailable while it rebuilds, exactly as on a first create. Note a
    vector index builds *asynchronously* on Atlas: it may take a short while after
    create before queries can use the new definition.
    """
    collection = db[chunk_model.COLLECTION_NAME]
    dimensions = get_settings().embedding_dimensions
    desired = chunk_model.build_vector_index_definition(dimensions)
    name = chunk_model.VECTOR_INDEX_NAME
    try:
        # list_search_indexes() is a coroutine returning the cursor — await it
        # first, then iterate the cursor.
        cursor = await collection.list_search_indexes()
        # Newer servers report the live definition as `latestDefinition`; older
        # ones as `definition`. Fall back so drift detection works on both.
        current = {
            idx["name"]: idx.get("latestDefinition") or idx.get("definition")
            async for idx in cursor
        }

        if name in current:
            if _vector_index_matches(current[name], desired):
                return
            # Definition drifted (code changed the fields). Drop it, then let the
            # create below rebuild it under the same name.
            await collection.drop_search_index(name)
            await _wait_until_search_index_absent(collection, name)
            logger.info(
                "dropped drifted chunks vector search index for rebuild",
                extra={"index": name},
            )

        await collection.create_search_index(
            SearchIndexModel(definition=desired, name=name, type="vectorSearch")
        )
        logger.info(
            "created chunks vector search index",
            extra={"index": name, "dimensions": dimensions},
        )
    except Exception as exc:
        # Never fail startup over this. Catch broadly: a server without Atlas
        # Search may reject anywhere from the command to the cursor, surfacing
        # different error types across driver/server versions.
        logger.warning(
            "skipping chunks vector search index (server may not support Atlas "
            "Vector Search); vector search stays unavailable until it exists",
            exc_info=exc,
        )


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

    await _ensure_collection(
        db,
        document_model.COLLECTION_NAME,
        document_model.DOCUMENTS_VALIDATOR,
        document_model.DOCUMENTS_INDEXES,
    )

    await _ensure_collection(
        db,
        chunk_model.COLLECTION_NAME,
        chunk_model.CHUNKS_VALIDATOR,
        chunk_model.CHUNKS_INDEXES,
    )
    # The chunks collection must exist before its search index can be created.
    await _ensure_vector_index(db)

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
