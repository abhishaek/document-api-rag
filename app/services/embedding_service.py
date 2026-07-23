"""Turn chunk text into embedding vectors via Voyage.

An ``Embedder`` is the interface the pipeline depends on (the same Protocol seam
as ``Storage`` and ``Parser``); ``VoyageEmbedder`` is the only implementation
today. Swapping models within the Voyage-4 series is a config change, and a
different provider is one new class — callers depend on the Protocol, not the
vendor SDK.

Embeddings are generated with ``input_type="document"``. Voyage embeds documents
and queries into the *same* space but optimises each differently, so retrieval
(iteration 2) must embed the query with ``input_type="query"`` to match — that is
why the input type is fixed here rather than left to the caller.

Requests are batched to stay under Voyage's two per-request limits — a count cap
and a token cap — whichever binds first. The token count is estimated (chars/4)
rather than tokenised exactly: an estimate with margin is cheap and only risks a
slightly smaller batch, whereas a miscount that *exceeds* the real limit would
fail the whole request.

Transient failures (rate limits, network hiccups, timeouts, 5xx) are retried by
the SDK with backoff — configured via ``max_retries``/``timeout`` — so a momentary
blip self-heals rather than failing the ingest and stranding the document. Only
after retries are exhausted, or on a *permanent* error (bad key, invalid request,
which the SDK never retries), does the failure propagate as ``EmbeddingError``.
"""

import asyncio
import logging
from typing import Protocol

logger = logging.getLogger(__name__)

# Voyage caps a request at 1000 texts; large models also cap total tokens at
# ~120K. We stay comfortably under both — the token budget carries margin because
# the per-text token count is estimated, not exact.
_MAX_BATCH_TEXTS = 1000
_MAX_BATCH_TOKENS = 100_000
# Rough chars-per-token for English prose. Deliberately low (real ratio is ~4)
# so the estimate over-counts tokens and batches stay on the safe side.
_CHARS_PER_TOKEN = 4


class EmbeddingError(Exception):
    """Raised when the embedding backend fails to produce vectors."""


class Embedder(Protocol):
    """Produces one embedding vector per input text, in order."""

    async def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    async def embed_query(self, text: str) -> list[float]: ...


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // _CHARS_PER_TOKEN)


def _iter_batches(texts: list[str]) -> list[list[str]]:
    """Group texts into batches under both the count and token caps.

    A single text over the token cap still goes out alone — Voyage truncates an
    over-long input rather than rejecting it, so the request still succeeds.
    """
    batches: list[list[str]] = []
    current: list[str] = []
    current_tokens = 0
    for text in texts:
        tokens = _estimate_tokens(text)
        would_overflow = current and (
            len(current) >= _MAX_BATCH_TEXTS
            or current_tokens + tokens > _MAX_BATCH_TOKENS
        )
        if would_overflow:
            batches.append(current)
            current = []
            current_tokens = 0
        current.append(text)
        current_tokens += tokens
    if current:
        batches.append(current)
    return batches


class VoyageEmbedder:
    """``Embedder`` backed by the Voyage API."""

    def __init__(
        self,
        api_key: str | None,
        model: str,
        dimensions: int,
        max_retries: int = 0,
        timeout: float | None = None,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._dimensions = dimensions
        self._max_retries = max_retries
        self._timeout = timeout
        self._client = None  # built lazily so a missing key doesn't fail startup

    def _get_client(self):
        if self._client is None:
            # Imported lazily so the vendor SDK isn't pulled in at import time
            # (and so tests that inject a fake client never construct a real one).
            import voyageai

            # max_retries makes the SDK retry *transient* failures (rate limits,
            # network, timeouts, 5xx) with backoff; it does not retry permanent
            # ones (auth, invalid request). This is what keeps a momentary blip
            # from failing an ingest without any user involvement.
            self._client = voyageai.Client(
                api_key=self._api_key,
                max_retries=self._max_retries,
                timeout=self._timeout,
            )
        return self._client

    def _embed_batch_sync(self, batch: list[str]) -> list[list[float]]:
        result = self._get_client().embed(
            batch,
            model=self._model,
            input_type="document",
            output_dimension=self._dimensions,
        )
        return result.embeddings

    def _embed_query_sync(self, text: str) -> list[float]:
        # input_type="query" pairs with the "document" type used at ingest:
        # Voyage embeds both into the same space but optimises each side
        # differently, so a query must be embedded as a query to match its chunks.
        result = self._get_client().embed(
            [text],
            model=self._model,
            input_type="query",
            output_dimension=self._dimensions,
        )
        return result.embeddings[0]

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        embeddings: list[list[float]] = []
        try:
            for batch in _iter_batches(texts):
                # The SDK call is synchronous and network-bound — run it off the
                # event loop, like the filesystem and parsing work.
                embeddings.extend(
                    await asyncio.to_thread(self._embed_batch_sync, batch)
                )
        except Exception as exc:
            # Normalise auth/rate-limit/network failures to one domain error the
            # pipeline records as a failed document.
            raise EmbeddingError(f"Voyage embedding failed: {exc}") from exc

        if len(embeddings) != len(texts):
            # A backend that returns the wrong count would silently misalign
            # chunks and vectors — fail loudly instead.
            raise EmbeddingError(
                f"Expected {len(texts)} embeddings, got {len(embeddings)}"
            )
        return embeddings

    async def embed_query(self, text: str) -> list[float]:
        try:
            # Network-bound sync SDK call, off the event loop like embed_documents.
            return await asyncio.to_thread(self._embed_query_sync, text)
        except Exception as exc:
            # Same normalisation as embed_documents: auth/rate-limit/network
            # failures surface as one domain error the caller can map to a 5xx.
            raise EmbeddingError(f"Voyage query embedding failed: {exc}") from exc
