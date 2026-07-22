import time
import uuid
from datetime import UTC, datetime

from pydantic import BaseModel, Field


class ResponseMetadata(BaseModel):
    """Standard envelope metadata returned on every response so clients get a
    consistent, traceable shape across endpoints. Endpoint-specific metadata
    models extend this with extra fields."""

    request_id: str = Field(description="Unique id for this request; use it to correlate logs.")
    latency_ms: float = Field(description="Server-side processing time in milliseconds.")
    timestamp: datetime = Field(description="UTC time the response was produced.")

    @classmethod
    def build(cls, request_id: str, started: float, **extra):
        """Construct metadata from the request id and the handler's start time
        (a time.perf_counter() value). `extra` carries any subclass fields."""
        return cls(
            request_id=request_id,
            latency_ms=round((time.perf_counter() - started) * 1000, 2),
            timestamp=datetime.now(UTC),
            **extra,
        )


def new_request() -> tuple[str, float]:
    """Stamp the start of a handler: returns (request_id, start_time). Pass both
    to ResponseMetadata.build() (or a subclass) when forming the response."""
    return str(uuid.uuid4()), time.perf_counter()
