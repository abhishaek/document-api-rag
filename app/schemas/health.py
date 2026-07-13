from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    app: str
    environment: str


class ReadinessResponse(BaseModel):
    ready: bool
    checks: dict[str, str]
