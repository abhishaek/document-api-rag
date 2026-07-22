"""Tests for the documents router (app/routers/documents.py).

These drive the router end-to-end through the real service and a real
FilesystemStorage rooted in a temp dir; only MongoDB is faked (via conftest).
That keeps the behaviour that matters here honest — most importantly the
content-hash dedup: the router's promise is that re-uploading identical bytes,
even under a new filename, returns the existing record rather than a duplicate.

Auth is real too: routes sit behind UserDependency, so requests carry a token
signed with the app secret, exactly as test_dependencies does.
"""

from datetime import UTC, datetime, timedelta

import jwt
import pytest
import pytest_asyncio

from app.core.config import get_settings
from app.dependencies import get_embedder, get_storage
from app.main import app
from app.services.storage_service import FilesystemStorage
from tests.conftest import FakeEmbedder

USER_A = "507f1f77bcf86cd799439011"
USER_B = "507f191e810c19729de860ea"

UPLOAD_URL = "/v1/documents/upload"
LIST_URL = "/v1/documents"

# libmagic reads these from the bytes. Plain text is on the allow-list; a GIF
# (image/gif) is not, so it exercises the 415 path.
TEXT_BYTES = b"This is a plain text document used for the upload tests.\n"
GIF_BYTES = (
    b"GIF89a\x01\x00\x01\x00\x80\x00\x00\x00\x00\x00\xff\xff\xff!\xf9\x04\x01"
    b"\x00\x00\x00\x00,\x00\x00\x00\x00\x01\x00\x01\x00\x00\x02\x02D\x01\x00;"
)


def _token(user_id: str) -> str:
    """Sign a valid access token for user_id with the app secret."""
    settings = get_settings()
    payload = {
        "sub": "tester",
        "id": user_id,
        "role": "user",
        "jti": f"jti-{user_id}",
        "exp": datetime.now(UTC) + timedelta(minutes=5),
    }
    return jwt.encode(payload, settings.secret_key, algorithm=settings.jwt_algorithm)


def _auth(user_id: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {_token(user_id)}"}


async def _upload(client, user_id: str, filename: str, content: bytes):
    return await client.post(
        UPLOAD_URL,
        files={"file": (filename, content, "application/octet-stream")},
        headers=_auth(user_id),
    )


@pytest_asyncio.fixture
async def doc_client(client, tmp_path):
    """The conftest client, plus a real filesystem storage backend in a temp dir
    and a fake embedder, so the background ingestion an upload triggers writes to
    disposable storage and never calls the Voyage API."""
    app.dependency_overrides[get_storage] = lambda: FilesystemStorage(
        root=tmp_path / "storage"
    )
    app.dependency_overrides[get_embedder] = FakeEmbedder
    yield client
    # client's own teardown clears every override.


async def test_upload_returns_202_with_pending_document(doc_client):
    response = await _upload(doc_client, USER_A, "notes.txt", TEXT_BYTES)

    assert response.status_code == 202
    body = response.json()
    assert body["original_filename"] == "notes.txt"
    assert body["mime_type"] == "text/plain"
    assert body["size_bytes"] == len(TEXT_BYTES)
    # The response is built before the background task runs, so the client always
    # sees `pending` here and polls GET /{id} for the terminal state.
    assert body["status"] == "pending"
    assert body["error"] is None
    assert body["id"]


async def test_upload_ingests_document_to_ready(doc_client):
    """End-to-end: the background task triggered by upload parses the file and
    advances it to `ready`. (Under ASGITransport the task completes before the
    upload call returns, so the follow-up GET is deterministic.)"""
    upload = await _upload(doc_client, USER_A, "notes.txt", TEXT_BYTES)
    document_id = upload.json()["id"]

    fetched = await doc_client.get(f"{LIST_URL}/{document_id}", headers=_auth(USER_A))

    assert fetched.status_code == 200
    assert fetched.json()["status"] == "ready"


async def test_reupload_identical_bytes_returns_same_document(doc_client):
    """The whole point of hashing: uploading the same file twice is idempotent."""
    first = await _upload(doc_client, USER_A, "notes.txt", TEXT_BYTES)
    second = await _upload(doc_client, USER_A, "notes.txt", TEXT_BYTES)

    assert first.status_code == 202
    assert second.status_code == 202
    assert first.json()["id"] == second.json()["id"]


async def test_reupload_same_content_new_filename_returns_same_document(doc_client):
    """Renaming a file must not defeat dedup: the hash is over the bytes, not the
    name, so the second upload returns the first record."""
    first = await _upload(doc_client, USER_A, "notes.txt", TEXT_BYTES)
    renamed = await _upload(doc_client, USER_A, "renamed-copy.txt", TEXT_BYTES)

    assert renamed.status_code == 202
    assert renamed.json()["id"] == first.json()["id"]
    # The stored record keeps the name it was first uploaded under.
    assert renamed.json()["original_filename"] == "notes.txt"


async def test_same_content_from_two_users_are_distinct_records(doc_client):
    """Dedup is per-user: identical bytes uploaded by different users are two
    separate documents, one per tenant."""
    mine = await _upload(doc_client, USER_A, "notes.txt", TEXT_BYTES)
    theirs = await _upload(doc_client, USER_B, "notes.txt", TEXT_BYTES)

    assert mine.json()["id"] != theirs.json()["id"]


async def test_upload_unsupported_type_returns_415(doc_client):
    response = await _upload(doc_client, USER_A, "pic.gif", GIF_BYTES)

    assert response.status_code == 415


async def test_upload_oversized_returns_413(doc_client, monkeypatch):
    """A file past the size cap is rejected. The cap is shrunk here rather than
    sending 25 MiB; both the router pre-check and the service check answer 413."""
    settings = get_settings()
    monkeypatch.setattr(settings, "max_upload_size_bytes", 8)

    response = await _upload(doc_client, USER_A, "notes.txt", TEXT_BYTES)

    assert response.status_code == 413


async def test_list_returns_user_documents_newest_first(doc_client):
    await _upload(doc_client, USER_A, "first.txt", b"first document body\n")
    await _upload(doc_client, USER_A, "second.txt", b"second document body\n")

    response = await doc_client.get(LIST_URL, headers=_auth(USER_A))

    assert response.status_code == 200
    docs = response.json()
    assert {d["original_filename"] for d in docs} == {"first.txt", "second.txt"}
    created = [d["created_at"] for d in docs]
    assert created == sorted(created, reverse=True)


async def test_list_is_scoped_to_the_caller(doc_client):
    """One user's list never includes another user's documents."""
    await _upload(doc_client, USER_A, "mine.txt", b"belongs to A\n")

    response = await doc_client.get(LIST_URL, headers=_auth(USER_B))

    assert response.status_code == 200
    assert response.json() == []


async def test_get_document_returns_the_document(doc_client):
    uploaded = await _upload(doc_client, USER_A, "notes.txt", TEXT_BYTES)
    document_id = uploaded.json()["id"]

    response = await doc_client.get(
        f"{LIST_URL}/{document_id}", headers=_auth(USER_A)
    )

    assert response.status_code == 200
    assert response.json()["id"] == document_id


async def test_get_other_users_document_returns_404(doc_client):
    """Fetching a document you don't own is a 404, not a 403 — a 403 would
    confirm the id exists."""
    uploaded = await _upload(doc_client, USER_A, "notes.txt", TEXT_BYTES)
    document_id = uploaded.json()["id"]

    response = await doc_client.get(
        f"{LIST_URL}/{document_id}", headers=_auth(USER_B)
    )

    assert response.status_code == 404


async def test_get_missing_document_returns_404(doc_client):
    response = await doc_client.get(
        f"{LIST_URL}/507f1f77bcf86cd799439099", headers=_auth(USER_A)
    )

    assert response.status_code == 404


@pytest.mark.parametrize(
    ("method", "url"),
    [("post", UPLOAD_URL), ("get", LIST_URL), ("get", f"{LIST_URL}/whatever")],
)
async def test_routes_require_authentication(doc_client, method, url):
    response = await getattr(doc_client, method)(url)

    assert response.status_code == 401
