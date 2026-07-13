"""Custom ASGI middleware."""

import uuid

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

from app.logging_config import request_id_ctx

REQUEST_ID_HEADER = "X-Request-ID"


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Assigns a correlation id to every request.

    * Reuses an incoming ``X-Request-ID`` header if the caller (or an upstream
      gateway) provided one; otherwise generates a fresh id.
    * Stores it in a context var so every log line emitted while handling the
      request is tagged with it.
    * Echoes it back in the response header so clients can report it in bug
      reports and it can be traced end-to-end.
    """

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        request_id = request.headers.get(REQUEST_ID_HEADER) or uuid.uuid4().hex
        token = request_id_ctx.set(request_id)
        try:
            response = await call_next(request)
        finally:
            request_id_ctx.reset(token)
        response.headers[REQUEST_ID_HEADER] = request_id
        return response
