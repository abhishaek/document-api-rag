"""Tests for chunk persistence (app/services/chunk_storage_service.py).

The contract is *replace, not append*: a document's chunks are overwritten on each
ingest so reprocessing never leaves stale chunks behind. These run against the
in-memory fake DB from conftest.
"""

import pytest
from bson import ObjectId

from app.models.chunk import COLLECTION_NAME as CHUNKS
from app.services.chunk_storage_service import replace_document_chunks

DOC_ID = ObjectId("507f1f77bcf86cd799439011")
USER_ID = ObjectId("507f191e810c19729de860ea")


async def _stored(fake_db):
    return await fake_db[CHUNKS].find({}).to_list(length=100)


async def test_stores_chunks_with_index_and_owner(fake_db):
    count = await replace_document_chunks(
        fake_db, DOC_ID, USER_ID, ["first", "second"], [[1.0, 2.0], [3.0, 4.0]]
    )

    assert count == 2
    stored = sorted(await _stored(fake_db), key=lambda c: c["chunk_index"])
    assert [c["chunk_index"] for c in stored] == [0, 1]
    assert [c["text"] for c in stored] == ["first", "second"]
    assert stored[0]["embedding"] == [1.0, 2.0]
    assert all(c["document_id"] == DOC_ID and c["user_id"] == USER_ID for c in stored)


async def test_reingest_replaces_previous_chunks(fake_db):
    """A second ingest of the same document overwrites the first set — no pile-up
    of stale chunks."""
    await replace_document_chunks(fake_db, DOC_ID, USER_ID, ["old"], [[1.0]])
    await replace_document_chunks(
        fake_db, DOC_ID, USER_ID, ["new one", "new two"], [[2.0], [3.0]]
    )

    stored = await _stored(fake_db)
    assert len(stored) == 2
    assert {c["text"] for c in stored} == {"new one", "new two"}


async def test_replace_is_scoped_to_the_document(fake_db):
    """Re-ingesting one document doesn't touch another document's chunks."""
    other_doc = ObjectId()
    await replace_document_chunks(fake_db, other_doc, USER_ID, ["keep me"], [[9.0]])

    await replace_document_chunks(fake_db, DOC_ID, USER_ID, ["mine"], [[1.0]])

    stored = await _stored(fake_db)
    assert {c["text"] for c in stored} == {"keep me", "mine"}


async def test_empty_chunks_clears_and_stores_nothing(fake_db):
    await replace_document_chunks(fake_db, DOC_ID, USER_ID, ["old"], [[1.0]])

    count = await replace_document_chunks(fake_db, DOC_ID, USER_ID, [], [])

    assert count == 0
    assert await _stored(fake_db) == []


async def test_length_mismatch_raises(fake_db):
    with pytest.raises(ValueError, match="same length"):
        await replace_document_chunks(fake_db, DOC_ID, USER_ID, ["a", "b"], [[1.0]])
