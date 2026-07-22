"""Blob storage for uploaded files.

Uploaded bytes are kept *outside* MongoDB. The ``documents`` collection stores
only a reference (``storage_key`` + ``content_hash``), so the database holds
metadata rather than megabytes: the working set stays cacheable and backups stay
small. This is the standard split — object storage for blobs, the database for
records.

Originals are kept rather than discarded after parsing, because the ingestion
pipeline gets re-run: every change to chunk size, embedding model, or parsing
strategy means re-processing the corpus from source.

``Storage`` is the interface the rest of the app depends on. ``FilesystemStorage``
is the only implementation today; an S3-compatible one can be added without
touching callers.

Keys are content-addressed as ``{user_id}/{sha256-of-bytes}``:

* the digest deduplicates — the same file uploaded twice stores one blob, and
  re-upload is idempotent rather than duplicating storage;
* the ``user_id`` prefix keeps tenants separated in the key space.
"""

import asyncio
import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

logger = logging.getLogger(__name__)


class StorageError(Exception):
    """Raised when a blob operation fails."""


class BlobNotFoundError(StorageError):
    """Raised when a key has no stored blob."""

    def __init__(self, key: str) -> None:
        self.key = key
        super().__init__(f"No blob stored at key: {key}")


class InvalidKeyError(StorageError):
    """Raised when a key does not address a location inside the storage root.

    Distinct from ``BlobNotFoundError`` on purpose: "this key is not allowed" and
    "nothing is stored here" are different failures, and collapsing them would
    let a containment breach masquerade as a routine miss.
    """

    def __init__(self, key: str) -> None:
        self.key = key
        super().__init__(f"Key escapes storage root: {key}")


@dataclass(frozen=True)
class StoredBlob:
    """Result of storing bytes: what the caller records in MongoDB."""

    key: str
    content_hash: str
    size_bytes: int


def _content_hash(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def build_key(user_id: str, content_hash: str) -> str:
    """Build the content-addressed key for a user's blob."""
    return f"{user_id}/{content_hash}"


class Storage(Protocol):
    """Blob storage interface.

    Implementations must be safe to call from async request handlers — file and
    network I/O belongs off the event loop.
    """

    async def put(self, user_id: str, raw: bytes) -> StoredBlob:
        """Store ``raw`` for ``user_id`` and return its reference.

        Idempotent: storing identical bytes twice yields the same key and does
        not duplicate storage.
        """
        ...

    async def get(self, key: str) -> bytes:
        """Return the blob at ``key``. Raises ``BlobNotFoundError`` if absent."""
        ...

    async def delete(self, key: str) -> None:
        """Remove the blob at ``key``. Succeeds if it is already gone."""
        ...

    async def exists(self, key: str) -> bool:
        """Whether a blob is stored at ``key``."""
        ...


class FilesystemStorage:
    """``Storage`` backed by a local directory tree.

    Blobs land at ``{root}/{user_id}/{sha256}``. Suitable for development and
    single-node deployments; swap in an S3-compatible backend when storage needs
    to outlive or scale beyond one host.
    """

    def __init__(self, root: Path) -> None:
        self._root = root.resolve()

    def _resolve(self, key: str) -> Path:
        """Map a key to an absolute path, refusing anything outside the root.

        Keys are built from a token-derived user_id and a hex digest, so they are
        not user-supplied — but a key also arrives back from the database on
        every read, and a path that escapes the root would turn a bad record into
        arbitrary filesystem access. Resolving and re-checking is cheap.
        """
        path = (self._root / key).resolve()
        if not path.is_relative_to(self._root):
            raise InvalidKeyError(key)
        return path

    def _put_sync(self, path: Path, raw: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        # Write to a temp file in the same directory, then rename. os.replace is
        # atomic within a filesystem, so a crash mid-write can't leave a
        # half-written blob at a key whose digest promises complete content.
        tmp = path.with_name(f".{path.name}.tmp")
        try:
            tmp.write_bytes(raw)
            tmp.replace(path)
        finally:
            tmp.unlink(missing_ok=True)

    async def put(self, user_id: str, raw: bytes) -> StoredBlob:
        content_hash = _content_hash(raw)
        key = build_key(user_id, content_hash)
        path = self._resolve(key)

        if path.exists():
            # Same bytes already stored — the digest guarantees the content
            # matches, so there is nothing to write.
            logger.debug("blob already stored", extra={"key": key})
            return StoredBlob(key=key, content_hash=content_hash, size_bytes=len(raw))

        try:
            await asyncio.to_thread(self._put_sync, path, raw)
        except OSError as exc:
            raise StorageError(f"Failed to store blob at {key}: {exc}") from exc

        logger.info("blob stored", extra={"key": key, "size_bytes": len(raw)})
        return StoredBlob(key=key, content_hash=content_hash, size_bytes=len(raw))

    async def get(self, key: str) -> bytes:
        path = self._resolve(key)
        try:
            return await asyncio.to_thread(path.read_bytes)
        except FileNotFoundError as exc:
            raise BlobNotFoundError(key) from exc
        except OSError as exc:
            raise StorageError(f"Failed to read blob at {key}: {exc}") from exc

    async def delete(self, key: str) -> None:
        path = self._resolve(key)
        try:
            await asyncio.to_thread(lambda: path.unlink(missing_ok=True))
        except OSError as exc:
            raise StorageError(f"Failed to delete blob at {key}: {exc}") from exc
        logger.info("blob deleted", extra={"key": key})

    async def exists(self, key: str) -> bool:
        path = self._resolve(key)
        return await asyncio.to_thread(path.exists)
