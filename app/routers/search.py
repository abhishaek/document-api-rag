import logging

from fastapi import APIRouter, Request

from app.core.rate_limit import limiter
from app.dependencies import DbDependency, EmbedderDependency, UserDependency
from app.schemas.search import SearchRequest, SearchResponse
from app.services.retrieval_service import resolve_mode, search

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
    """Search the authenticated user's own chunks.

    Three strategies, selected by ``mode`` (default: the server's setting):

    * ``vector`` — the query is embedded and matched by cosine similarity against
      the user's chunks via Atlas ``$vectorSearch``. Semantic; handles paraphrase.
    * ``text`` — BM25 over the chunk text via Atlas ``$search``. Lexical; handles
      exact identifiers, error codes and acronyms that embeddings blur away.
    * ``hybrid`` — both of the above, fused with Reciprocal Rank Fusion.

    Results come back best-first with their scores; how to read ``score`` depends
    on the mode, which is why the response echoes it back (see
    ``app.schemas.search.SearchResult``). Every mode is scoped to the caller
    *inside* the search, so a query never reaches another tenant's documents. Pass
    a ``document_id`` to restrict the search to a single document; omit it to
    search all of them.

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
        payload.mode,
    )
    return SearchResponse(
        query=payload.query, mode=resolve_mode(payload.mode), results=results
    )
