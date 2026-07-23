"""Application configuration.

Settings are loaded from environment variables (and a local .env file in
development) via pydantic-settings. This keeps secrets out of the codebase and
follows 12-factor config principles.
"""

from functools import lru_cache
from pathlib import Path
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

    # --- Document storage ---
    # Where uploaded files are kept. Blobs live outside MongoDB: documents in the
    # DB hold only a reference (storage_key), keeping the working set small and
    # backups fast. STORAGE_DIR is the filesystem backend's root; swapping in an
    # S3-compatible backend later is a config + one-class change (see
    # app.services.storage_service).
    storage_dir: Path = Path("./var/storage")
    # Rejected before any bytes are written. 25 MiB.
    max_upload_size_bytes: int = 25 * 1024 * 1024

    # --- Embedding (Voyage) ---
    # VOYAGE_API_KEY is optional so the app boots without it; ingestion then
    # reaches `failed` at the embedding step rather than crashing at startup. Get
    # a key at https://dashboard.voyageai.com. voyage-4-large defaults to 1024
    # dimensions, which the chunks vector index is built against — changing the
    # model or dimensions means rebuilding that index (see app.models.chunk).
    voyage_api_key: str | None = None
    embedding_model: str = "voyage-4-large"
    embedding_dimensions: int = 1024
    # Transient-failure handling for the Voyage API. The SDK retries rate-limit,
    # network, timeout, and 5xx errors with backoff when max_retries > 0 (its
    # default is 0 — no retries, so a single blip fails the whole ingest);
    # permanent errors (bad key, invalid request) are never retried. The timeout
    # caps one request so a hung call can't stall ingestion indefinitely.
    embedding_max_retries: int = 3
    embedding_timeout_seconds: float = 30.0

    # --- Retrieval (vector search) ---
    # How many chunks a search returns by default, and the hard ceiling a caller
    # may ask for. The ceiling stops a client from requesting an unbounded result
    # set (each chunk carries its text, so the response size is real).
    search_default_limit: int = 5
    search_max_limit: int = 20
    # $vectorSearch explores `numCandidates` graph nodes before returning `limit`
    # results — the HNSW recall/latency knob. More candidates = better recall,
    # slightly more work. We derive it as limit * this multiplier (a common rule
    # of thumb is 10-20x), capped by Atlas's 10,000 ceiling in the service.
    search_num_candidates_multiplier: int = 15

    # --- Ingestion recovery ---
    # On startup the app re-runs documents left `failed` or stuck `processing`
    # (e.g. a crash mid-ingest, or a config fix like adding the API key), so a
    # user never has to re-upload. Bounded so a genuinely broken document isn't
    # retried forever: once a document has been attempted this many times, the
    # sweep leaves it `failed`.
    ingestion_max_attempts: int = 3

    # --- Chunking ---
    # Characters per chunk and overlap between consecutive chunks. Exposed as
    # config because they're a primary lever on retrieval quality — tunable per
    # corpus without a code change. The chunker requires overlap < size; the
    # validator below enforces that at startup rather than at ingestion time.
    chunk_size: int = 2000
    chunk_overlap: int = 200

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

    @field_validator("chunk_overlap")
    @classmethod
    def _overlap_smaller_than_size(cls, value: int, info) -> int:
        """Fail fast if overlap >= size: the chunker can't advance its window and
        would raise on the first document. Catch it at startup instead."""
        # chunk_size is defined before chunk_overlap, so it's already in info.data.
        size = info.data.get("chunk_size")
        if size is not None and value >= size:
            raise ValueError(
                f"CHUNK_OVERLAP ({value}) must be smaller than CHUNK_SIZE ({size})"
            )
        return value

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
