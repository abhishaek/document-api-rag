"""Application configuration.

Settings are loaded from environment variables (and a local .env file in
development) via pydantic-settings. This keeps secrets out of the codebase and
follows 12-factor config principles.
"""

from functools import lru_cache
from typing import Annotated

from pydantic import field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


class Settings(BaseSettings):
    """Typed application settings, validated at startup."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- App metadata ---
    app_name: str = "document-api-rag"
    environment: str = "development"
    debug: bool = False

    # --- API server ---
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # --- MongoDB ---
    # For Atlas Local (Docker) use directConnection; for Atlas Cloud use the
    # mongodb+srv://... URI from your cluster.
    mongodb_uri: str = "mongodb://localhost:27018/?directConnection=true"
    mongodb_db_name: str = "document_rag"

    # --- Auth / JWT ---
    # SECRET_KEY is required from the environment (no default) so a real secret
    # is never committed. Generate one with:  openssl rand -hex 32
    secret_key: str
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 15
    refresh_token_expire_days: int = 7

    # --- Logging ---
    # LOG_LEVEL: DEBUG | INFO | WARNING | ERROR | CRITICAL
    # LOG_JSON: true in production (structured logs), false for readable dev logs.
    log_level: str = "INFO"
    log_json: bool = False

    # --- CORS ---
    # Origins allowed to call the API from a browser. Set via a comma-separated
    # env var, e.g. CORS_ORIGINS=http://localhost:3000,https://app.example.com
    # NoDecode stops pydantic-settings from JSON-parsing the env value so our
    # validator below can split the comma-separated string itself.
    cors_origins: Annotated[list[str], NoDecode] = ["http://localhost:3000"]

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _split_cors_origins(cls, value: object) -> object:
        """Accept a comma-separated string from the environment and turn it
        into a list. A JSON list or an actual list is passed through untouched."""
        if isinstance(value, str) and not value.strip().startswith("["):
            return [origin.strip() for origin in value.split(",") if origin.strip()]
        return value


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance (built once per process)."""
    return Settings()
