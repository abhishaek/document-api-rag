"""Tests for chunk persistence (app/services/chunk_storage_service.py).

The contract is *incremental sync* with **position-independent** ids: a re-ingest
embeds only new/changed chunks, leaves unchanged chunks (and their embeddings) in
place — merely re-indexing ones that moved — and deletes stale ones. So
reprocessing never piles up stale chunks and never re-pays for embeddings it
already has, even after an in-place edit that shifts positions. These run against
the in-memory fake DB from conftest, whose ``FakeEmbedder`` records exactly which
texts it was asked to embed.
"""

from bson import ObjectId

from app.models.chunk import COLLECTION_NAME as CHUNKS
from app.models.chunk import build_chunk_id
from app.services.chunk_storage_service import sync_document_chunks

DOC_ID = ObjectId("507f1f77bcf86cd799439011")
USER_ID = ObjectId("507f191e810c19729de860ea")


async def _stored(fake_db):
    return await fake_db[CHUNKS].find({}).to_list(length=100)


def _counts(result):
    """(total, embedded, reused, reindexed, deleted) — compact for assertions."""
    return (result.total, result.embedded, result.reused, result.reindexed, result.deleted)


async def test_stores_chunks_with_index_and_owner(fake_db, fake_embedder):
    result = await sync_document_chunks(
        fake_db, DOC_ID, USER_ID, ["first", "second"], fake_embedder
    )

    assert _counts(result) == (2, 2, 0, 0, 0)
    stored = sorted(await _stored(fake_db), key=lambda c: c["chunk_index"])
    assert [c["chunk_index"] for c in stored] == [0, 1]
    assert [c["text"] for c in stored] == ["first", "second"]
    assert all("embedding" in c and "created_at" in c for c in stored)
    assert all(c["document_id"] == DOC_ID and c["user_id"] == USER_ID for c in stored)


async def test_reingest_removes_chunks_that_are_gone(fake_db, fake_embedder):
    """A second ingest with fewer chunks deletes the ones no longer present."""
    await sync_document_chunks(fake_db, DOC_ID, USER_ID, ["old one", "old two"], fake_embedder)

    result = await sync_document_chunks(fake_db, DOC_ID, USER_ID, ["old one"], fake_embedder)

    assert (result.total, result.deleted) == (1, 1)
    assert {c["text"] for c in await _stored(fake_db)} == {"old one"}


async def test_identical_reingest_reuses_everything_and_embeds_nothing(fake_db, fake_embedder):
    """The whole point: re-ingesting an unchanged document re-embeds NOTHING and
    leaves the stored chunks (and their ids) exactly as they were."""
    await sync_document_chunks(fake_db, DOC_ID, USER_ID, ["alpha", "beta"], fake_embedder)
    first = {c["_id"] for c in await _stored(fake_db)}
    fake_embedder.calls.clear()

    result = await sync_document_chunks(fake_db, DOC_ID, USER_ID, ["alpha", "beta"], fake_embedder)

    assert _counts(result) == (2, 0, 2, 0, 0)
    assert fake_embedder.calls == []  # no embedding call on the second run
    assert {c["_id"] for c in await _stored(fake_db)} == first


async def test_in_place_edit_reembeds_only_changed_text(fake_db, fake_embedder):
    """A paragraph inserted at the top shifts every following chunk down one
    position. Because ids ignore position, the shifted chunks keep their ids and
    embeddings — only the genuinely new chunk is embedded; the rest are re-indexed."""
    await sync_document_chunks(fake_db, DOC_ID, USER_ID, ["body one", "body two"], fake_embedder)
    # Snapshot values (not live dict refs — the fake mutates rows in place on update).
    before = {
        c["text"]: {
            "_id": c["_id"],
            "embedding": list(c["embedding"]),
            "chunk_index": c["chunk_index"],
        }
        for c in await _stored(fake_db)
    }
    fake_embedder.calls.clear()

    result = await sync_document_chunks(
        fake_db, DOC_ID, USER_ID, ["NEW intro", "body one", "body two"], fake_embedder
    )

    assert _counts(result) == (3, 1, 2, 2, 0)
    # Only the new text was sent to the embedder.
    assert fake_embedder.calls == [["NEW intro"]]
    stored = sorted(await _stored(fake_db), key=lambda c: c["chunk_index"])
    assert [(c["chunk_index"], c["text"]) for c in stored] == [
        (0, "NEW intro"),
        (1, "body one"),
        (2, "body two"),
    ]
    # The shifted chunks are the same rows: same id, same embedding, new index.
    for text in ("body one", "body two"):
        now = next(c for c in stored if c["text"] == text)
        assert now["_id"] == before[text]["_id"]
        assert now["embedding"] == before[text]["embedding"]
        assert now["chunk_index"] != before[text]["chunk_index"]


async def test_partial_change_embeds_only_the_changed_chunk(fake_db, fake_embedder):
    """One chunk edited in place: only it is re-embedded; the unchanged one keeps
    its stored embedding, and the old version is deleted."""
    await sync_document_chunks(fake_db, DOC_ID, USER_ID, ["keep me", "change me"], fake_embedder)
    kept_before = next(c for c in await _stored(fake_db) if c["text"] == "keep me")
    fake_embedder.calls.clear()

    result = await sync_document_chunks(
        fake_db, DOC_ID, USER_ID, ["keep me", "changed now"], fake_embedder
    )

    assert _counts(result) == (2, 1, 1, 0, 1)
    assert fake_embedder.calls == [["changed now"]]
    stored = {c["text"]: c for c in await _stored(fake_db)}
    assert set(stored) == {"keep me", "changed now"}
    assert stored["keep me"]["_id"] == kept_before["_id"]
    assert stored["keep me"]["embedding"] == kept_before["embedding"]


async def test_sync_is_scoped_to_the_document(fake_db, fake_embedder):
    """Re-ingesting one document doesn't touch another document's chunks."""
    other_doc = ObjectId()
    await sync_document_chunks(fake_db, other_doc, USER_ID, ["keep me"], fake_embedder)

    await sync_document_chunks(fake_db, DOC_ID, USER_ID, ["mine"], fake_embedder)

    assert {c["text"] for c in await _stored(fake_db)} == {"keep me", "mine"}


async def test_empty_chunks_clears_and_stores_nothing(fake_db, fake_embedder):
    await sync_document_chunks(fake_db, DOC_ID, USER_ID, ["old"], fake_embedder)

    result = await sync_document_chunks(fake_db, DOC_ID, USER_ID, [], fake_embedder)

    assert (result.total, result.deleted) == (0, 1)
    assert await _stored(fake_db) == []


async def test_chunk_ids_are_deterministic_and_content_addressed(fake_db, fake_embedder):
    """Stored ``_id``s match build_chunk_id — the content-addressed key, not a
    random ObjectId — so they can be reproduced and referenced."""
    await sync_document_chunks(fake_db, DOC_ID, USER_ID, ["first", "second"], fake_embedder)

    stored = {c["chunk_index"]: c["_id"] for c in await _stored(fake_db)}
    assert stored[0] == build_chunk_id(USER_ID, DOC_ID, "first", 0)
    assert stored[1] == build_chunk_id(USER_ID, DOC_ID, "second", 0)


async def test_duplicate_text_chunks_get_distinct_ids(fake_db, fake_embedder):
    """Two chunks with identical text in one document must not collapse to one id
    — the occurrence counter keeps them distinct, so neither is lost."""
    await sync_document_chunks(
        fake_db, DOC_ID, USER_ID, ["same text", "same text"], fake_embedder
    )

    stored = await _stored(fake_db)
    assert len(stored) == 2
    assert len({c["_id"] for c in stored}) == 2
    # The two ids are exactly the first- and second-occurrence keys.
    assert {c["_id"] for c in stored} == {
        build_chunk_id(USER_ID, DOC_ID, "same text", 0),
        build_chunk_id(USER_ID, DOC_ID, "same text", 1),
    }


def test_build_chunk_id_is_position_independent_and_namespaced():
    """The id changes when user, document, text, or occurrence changes — but NOT
    with position; equal inputs always give equal ids."""
    base = build_chunk_id(USER_ID, DOC_ID, "text", 0)

    assert base == build_chunk_id(USER_ID, DOC_ID, "text", 0)  # deterministic
    assert base != build_chunk_id(ObjectId(), DOC_ID, "text", 0)  # user
    assert base != build_chunk_id(USER_ID, ObjectId(), "text", 0)  # document
    assert base != build_chunk_id(USER_ID, DOC_ID, "other", 0)  # text
    assert base != build_chunk_id(USER_ID, DOC_ID, "text", 1)  # occurrence
    # A hex SHA-256 digest.
    assert len(base) == 64 and all(ch in "0123456789abcdef" for ch in base)
