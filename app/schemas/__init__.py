"""Pydantic request/response schemas.

Re-exported here so callers can use ``from app.schemas import HealthResponse``
instead of reaching into each submodule. Add new schema modules and re-export
their public models below.
"""

from app.schemas.auth import (
    CreateUserRequest,
    RefreshRequest,
    Token,
    UserResponse,
    UserRole,
)
from app.schemas.common import ResponseMetadata
from app.schemas.document import DocumentResponse, DocumentStatus
from app.schemas.health import HealthResponse, ReadinessResponse
from app.schemas.search import SearchRequest, SearchResponse, SearchResult

__all__ = [
    "HealthResponse",
    "ReadinessResponse",
    "CreateUserRequest",
    "UserResponse",
    "UserRole",
    "RefreshRequest",
    "Token",
    "ResponseMetadata",
    "DocumentResponse",
    "DocumentStatus",
    "SearchRequest",
    "SearchResponse",
    "SearchResult",
]
