import logging

from fastapi import APIRouter, Request

from app.core.rate_limit import limiter
from app.dependencies import DbDependency, EmbedderDependency, UserDependency
from app.schemas.search import SearchRequest, SearchResponse
from app.services.retrieval_service import search

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/search", tags=["Search"])


@router.post("", response_model=SearchResponse)
@limiter.limit("30/minute")
async def search_chunks(
    request: Request,
    payload: SearchRequest,
    db: DbDependency,
    embedder: EmbedderDependency,
    current_user: UserDependency,
) -> SearchResponse:
    """Vector-search the authenticated user's own chunks.

    The query is embedded and matched (cosine) against the user's chunks via
    Atlas ``$vectorSearch``; results come back most-similar-first with their
    scores. Scoped to the caller by the vector index's ``user_id`` filter, so a
    query never reaches another tenant's documents. Pass a ``document_id`` to
    restrict the search to a single document; omit it to search all of them.

    No LLM here — this returns the raw retrieved chunks. Turning them into a
    cited answer is a later endpoint (``POST /v1/ask``).
    """
    results = await search(
        db,
        embedder,
        current_user["id"],
        payload.query,
        payload.limit,
        payload.document_id,
    )
    return SearchResponse(query=payload.query, results=results)
