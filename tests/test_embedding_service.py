"""Tests for the embedding service (app/services/embedding_service.py).

The Voyage API is never called: a fake client is injected so the tests cover our
logic — batching under the count/token caps, the fixed input_type/dimension
arguments, and the count-mismatch guard — not the vendor's.
"""

import pytest

from app.services import embedding_service
from app.services.embedding_service import (
    EmbeddingError,
    VoyageEmbedder,
    _iter_batches,
)


class _FakeResult:
    def __init__(self, embeddings):
        self.embeddings = embeddings


class _FakeVoyageClient:
    """Records each embed() call and returns one vector per input text."""

    def __init__(self, dim=4):
        self.dim = dim
        self.calls = []

    def embed(self, texts, *, model, input_type, output_dimension):
        self.calls.append(
            {
                "texts": list(texts),
                "model": model,
                "input_type": input_type,
                "output_dimension": output_dimension,
            }
        )
        return _FakeResult([[0.1] * output_dimension for _ in texts])


def _embedder_with(client, *, model="voyage-4-large", dim=4) -> VoyageEmbedder:
    embedder = VoyageEmbedder(api_key="unused", model=model, dimensions=dim)
    embedder._client = client  # inject the fake, so no real client is built
    return embedder


async def test_embeds_each_text_with_document_input_type():
    client = _FakeVoyageClient()
    embedder = _embedder_with(client, dim=4)

    vectors = await embedder.embed_documents(["first chunk", "second chunk"])

    assert len(vectors) == 2
    assert all(len(v) == 4 for v in vectors)
    # The vendor call used the document input type and configured model/dimension.
    assert client.calls[0]["input_type"] == "document"
    assert client.calls[0]["model"] == "voyage-4-large"
    assert client.calls[0]["output_dimension"] == 4


def test_client_is_built_with_retries_and_timeout(monkeypatch):
    """The retry/timeout config must reach the Voyage client — that's what makes
    transient failures self-heal instead of failing the ingest."""
    import voyageai

    captured = {}

    class _CapturingClient:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(voyageai, "Client", _CapturingClient)

    embedder = VoyageEmbedder(
        api_key="k", model="m", dimensions=4, max_retries=5, timeout=12.0
    )
    embedder._get_client()

    assert captured["max_retries"] == 5
    assert captured["timeout"] == 12.0
    assert captured["api_key"] == "k"


async def test_embed_query_uses_query_input_type():
    """A query must be embedded with input_type="query" so it lands in the same
    space as the "document"-embedded chunks it will be matched against."""
    client = _FakeVoyageClient()
    embedder = _embedder_with(client, dim=4)

    vector = await embedder.embed_query("what is the refund policy?")

    assert len(vector) == 4
    assert client.calls[0]["input_type"] == "query"
    assert client.calls[0]["model"] == "voyage-4-large"
    assert client.calls[0]["output_dimension"] == 4


async def test_embed_query_wraps_backend_exception():
    class _BoomClient(_FakeVoyageClient):
        def embed(self, texts, **kwargs):
            raise RuntimeError("429 rate limit")

    embedder = _embedder_with(_BoomClient())

    with pytest.raises(EmbeddingError, match="Voyage query embedding failed"):
        await embedder.embed_query("anything")


async def test_empty_input_makes_no_api_call():
    client = _FakeVoyageClient()
    embedder = _embedder_with(client)

    assert await embedder.embed_documents([]) == []
    assert client.calls == []


async def test_texts_are_split_into_batches(monkeypatch):
    """Inputs over the per-request count cap go out as multiple embed() calls, and
    all vectors come back in order."""
    monkeypatch.setattr(embedding_service, "_MAX_BATCH_TEXTS", 2)
    client = _FakeVoyageClient()
    embedder = _embedder_with(client)

    vectors = await embedder.embed_documents(["a", "b", "c", "d", "e"])

    assert len(vectors) == 5
    # 5 texts, cap of 2 -> batches of 2, 2, 1.
    assert [len(c["texts"]) for c in client.calls] == [2, 2, 1]


async def test_wrong_embedding_count_raises():
    """A backend that returns the wrong number of vectors must fail loudly, not
    silently misalign chunks and embeddings."""

    class _ShortClient(_FakeVoyageClient):
        def embed(self, texts, **kwargs):
            return _FakeResult([[0.0] * self.dim])  # one vector, ignoring input len

    embedder = _embedder_with(_ShortClient())

    with pytest.raises(EmbeddingError):
        await embedder.embed_documents(["a", "b", "c"])


async def test_backend_exception_is_wrapped():
    class _BoomClient(_FakeVoyageClient):
        def embed(self, texts, **kwargs):
            raise RuntimeError("401 unauthorized")

    embedder = _embedder_with(_BoomClient())

    with pytest.raises(EmbeddingError, match="Voyage embedding failed"):
        await embedder.embed_documents(["a"])


def test_iter_batches_respects_token_cap(monkeypatch):
    """Batching also splits on the token estimate, not just the count."""
    monkeypatch.setattr(embedding_service, "_MAX_BATCH_TEXTS", 1000)
    monkeypatch.setattr(embedding_service, "_MAX_BATCH_TOKENS", 10)
    monkeypatch.setattr(embedding_service, "_CHARS_PER_TOKEN", 1)  # 1 char = 1 token

    # Each text is 6 "tokens"; two fit under a 10-token cap only one at a time...
    batches = _iter_batches(["aaaaaa", "bbbbbb", "cccccc"])

    assert [len(b) for b in batches] == [1, 1, 1]
