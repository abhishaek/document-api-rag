"""Search request/response schemas.

Search is a JSON POST (not multipart like upload), so unlike ``document.py`` it
*does* model a request body. The query text and the desired result count come in
as JSON; the response is a ranked list of chunks with their similarity scores.
"""

from typing import Literal

from bson import ObjectId
from pydantic import BaseModel, Field, field_validator


class SearchRequest(BaseModel):
    """A vector-search query over the caller's own chunks.

    ``limit`` is optional: omitted, the service falls back to
    ``settings.search_default_limit``. It's bounded here at the HTTP boundary
    (1..search-max) so a bad value is a 422 about the request, not something the
    service has to defend against — but the value is *clamped* again in the
    service too, since the service is also callable from code that skips this
    validation.

    ``document_id`` is optional: given, the search is restricted to that one
    document; omitted, it spans all of the caller's documents. It's validated as
    an ObjectId here so a malformed id is a 422 about the request rather than
    surfacing as an error deeper in the service.
    """

    query: str = Field(min_length=1, description="Natural-language search text.")
    limit: int | None = Field(
        default=None,
        ge=1,
        description="How many chunks to return. Defaults to the server's setting.",
    )
    document_id: str | None = Field(
        default=None,
        description="Restrict the search to a single document. Omit to search all.",
    )
    mode: Literal["vector", "text", "hybrid"] | None = Field(
        default=None,
        description=(
            "Retrieval strategy: 'vector' (dense/semantic), 'text' (BM25/lexical), "
            "or 'hybrid' (both, fused with RRF). Defaults to the server's setting."
        ),
    )

    @field_validator("document_id")
    @classmethod
    def _valid_object_id(cls, value: str | None) -> str | None:
        """A malformed document_id can never match a real document — reject it at
        the boundary (422) instead of letting ObjectId(...) raise in the service."""
        if value is not None and not ObjectId.is_valid(value):
            raise ValueError("document_id must be a valid ObjectId")
        return value


class SearchResult(BaseModel):
    """One matched chunk, with how well it matched.

    ``score`` is the ranking score, and **what it means depends on the mode**:

    * ``vector`` — Atlas's ``vectorSearchScore``: cosine similarity, roughly [0, 1].
    * ``text`` — Atlas's ``searchScore``: raw BM25, unbounded and comparable only
      within one query's results.
    * ``hybrid`` — the fused RRF score, ``sum of weight / (k + rank)`` over the two
      legs. With the default k=60 these are small numbers (~0.016 for a top hit)
      and are meaningful only as an *ordering* — they are not a similarity.

    ``vector_score`` and ``text_score`` are populated in hybrid mode only: each
    leg's own native score, carried through for inspection. They take no part in
    the ranking (that is the point of fusing on rank), but they show which leg
    found a chunk — a null ``text_score`` means only the dense leg retrieved it,
    and vice versa. In single-leg modes they stay null, because the one score
    there is already ``score``.

    ``document_id`` and ``chunk_index`` let a caller trace a result back to its
    source document and position (the groundwork the citation work in a later
    phase builds on).
    """

    id: str
    document_id: str
    chunk_index: int
    text: str
    score: float
    vector_score: float | None = None
    text_score: float | None = None


class SearchResponse(BaseModel):
    """The ranked results for one query, best match first.

    ``mode`` echoes the strategy that actually ran — the request's, or the
    server default when the request didn't name one. Without it a caller relying
    on the default can't tell how to read ``score`` (see ``SearchResult``), and an
    eval comparing strategies can't label its runs.
    """

    query: str
    mode: str
    results: list[SearchResult]
