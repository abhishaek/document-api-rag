"""Verify the filesystem blob-storage backend.

These tests exercise ``FilesystemStorage`` against a real temporary directory
rather than a fake: the behaviours that matter here (atomic writes, content
addressing, path containment) are properties of the filesystem itself, so
mocking it out would test nothing. ``tmp_path`` gives each test its own root.
"""

import hashlib
from pathlib import Path

import pytest

from app.services.storage_service import (
    BlobNotFoundError,
    FilesystemStorage,
    InvalidKeyError,
    build_key,
)

USER_ID = "507f1f77bcf86cd799439011"
OTHER_USER_ID = "507f191e810c19729de860ea"
RAW = b"%PDF-1.4 fake pdf bytes"


@pytest.fixture
def storage(tmp_path: Path) -> FilesystemStorage:
    """A storage backend rooted in a per-test temp directory."""
    return FilesystemStorage(root=tmp_path / "storage")


async def test_put_returns_content_addressed_reference(storage: FilesystemStorage):
    """put() reports the digest, size, and key the caller records in Mongo."""
    blob = await storage.put(USER_ID, RAW)

    assert blob.content_hash == hashlib.sha256(RAW).hexdigest()
    assert blob.size_bytes == len(RAW)
    assert blob.key == build_key(USER_ID, blob.content_hash)


async def test_put_then_get_roundtrips_bytes(storage: FilesystemStorage):
    """Stored bytes come back byte-for-byte identical."""
    blob = await storage.put(USER_ID, RAW)

    assert await storage.get(blob.key) == RAW


async def test_put_creates_parent_directories(tmp_path: Path):
    """A storage root that doesn't exist yet is created on first write, so a
    fresh checkout works without a provisioning step."""
    storage = FilesystemStorage(root=tmp_path / "does" / "not" / "exist")

    blob = await storage.put(USER_ID, RAW)

    assert await storage.get(blob.key) == RAW


async def test_put_leaves_no_temp_files(storage: FilesystemStorage, tmp_path: Path):
    """The write-to-temp-then-rename path cleans up after itself; a stray temp
    file would be picked up by any future key listing (e.g. an orphan sweeper)."""
    await storage.put(USER_ID, RAW)

    user_dir = tmp_path / "storage" / USER_ID
    assert [p.name for p in user_dir.iterdir()] == [hashlib.sha256(RAW).hexdigest()]


async def test_put_is_idempotent_for_identical_bytes(
    storage: FilesystemStorage, tmp_path: Path
):
    """Re-uploading the same file yields the same key and stores one blob.

    This is what content addressing buys: dedup falls out of the key scheme, with
    no check-then-write race.
    """
    first = await storage.put(USER_ID, RAW)
    second = await storage.put(USER_ID, RAW)

    assert first.key == second.key
    assert len(list((tmp_path / "storage" / USER_ID).iterdir())) == 1


async def test_same_bytes_are_isolated_per_user(storage: FilesystemStorage):
    """The user_id prefix separates tenants: identical content uploaded by two
    users produces two distinct keys, so neither can reach the other's blob."""
    mine = await storage.put(USER_ID, RAW)
    theirs = await storage.put(OTHER_USER_ID, RAW)

    assert mine.key != theirs.key
    assert mine.content_hash == theirs.content_hash


async def test_different_bytes_produce_different_keys(storage: FilesystemStorage):
    """Distinct content never collides on a key."""
    first = await storage.put(USER_ID, RAW)
    second = await storage.put(USER_ID, b"a different document entirely")

    assert first.key != second.key


async def test_get_missing_key_raises_blob_not_found(storage: FilesystemStorage):
    """A key with no blob is a domain error, not an OSError leaking through."""
    with pytest.raises(BlobNotFoundError):
        await storage.get(build_key(USER_ID, "0" * 64))


async def test_exists_reflects_stored_state(storage: FilesystemStorage):
    blob = await storage.put(USER_ID, RAW)
    assert await storage.exists(blob.key) is True

    await storage.delete(blob.key)
    assert await storage.exists(blob.key) is False


async def test_delete_removes_blob(storage: FilesystemStorage):
    blob = await storage.put(USER_ID, RAW)

    await storage.delete(blob.key)

    with pytest.raises(BlobNotFoundError):
        await storage.get(blob.key)


async def test_delete_is_idempotent(storage: FilesystemStorage):
    """Deleting an absent blob succeeds. Cleanup paths (failed ingestion, an
    orphan sweeper) shouldn't have to know whether the write ever landed."""
    blob = await storage.put(USER_ID, RAW)

    await storage.delete(blob.key)
    await storage.delete(blob.key)


@pytest.mark.parametrize(
    "key",
    [
        "../../../etc/passwd",
        f"{USER_ID}/../../../etc/passwd",
        "/etc/passwd",
        "..",
    ],
)
@pytest.mark.parametrize("operation", ["get", "delete", "exists"])
async def test_keys_cannot_escape_storage_root(
    storage: FilesystemStorage, key: str, operation: str
):
    """A key that resolves outside the root is refused on every read path.

    Keys are built from a token-derived user_id and a hex digest, so they aren't
    user-supplied — but they also arrive back from the database on every read,
    and a corrupted record must not become arbitrary filesystem access.

    This asserts ``InvalidKeyError`` specifically rather than its ``StorageError``
    base: a traversal that happens to point at a nonexistent path would raise
    ``BlobNotFoundError`` (also a ``StorageError``), so matching the base class
    would pass whether or not the guard exists.
    """
    with pytest.raises(InvalidKeyError):
        await getattr(storage, operation)(key)


async def test_put_after_deleting_underlying_file_rewrites_blob(
    storage: FilesystemStorage,
):
    """put() short-circuits when the blob already exists; once it's gone, the
    next put must actually write again rather than trust the digest."""
    blob = await storage.put(USER_ID, RAW)
    await storage.delete(blob.key)

    await storage.put(USER_ID, RAW)

    assert await storage.get(blob.key) == RAW
