"""Search request/response schemas.

Search is a JSON POST (not multipart like upload), so unlike ``document.py`` it
*does* model a request body. The query text and the desired result count come in
as JSON; the response is a ranked list of chunks with their similarity scores.
"""

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

    ``score`` is Atlas's ``vectorSearchScore`` for cosine similarity — higher is
    a closer match, in roughly [0, 1]. ``document_id`` and ``chunk_index`` let a
    caller trace a result back to its source document and position (the
    groundwork the citation work in a later phase builds on).
    """

    id: str
    document_id: str
    chunk_index: int
    text: str
    score: float


class SearchResponse(BaseModel):
    """The ranked results for one query, most similar first."""

    query: str
    results: list[SearchResult]
