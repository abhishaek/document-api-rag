"""Verify application settings load from the environment as expected.

These tests construct ``Settings`` directly (rather than the cached
``get_settings()``) so each case controls its own environment in isolation.
``_env_file=None`` skips the on-disk ``.env`` so tests don't depend on local
developer files.
"""

from pathlib import Path

import pytest
from pydantic import ValidationError

from app.core.config import Settings


def test_jwt_defaults(monkeypatch: pytest.MonkeyPatch):
    """Non-secret JWT knobs have sensible defaults when unset in the env."""
    monkeypatch.setenv("SECRET_KEY", "test-secret")
    for var in (
        "JWT_ALGORITHM",
        "ACCESS_TOKEN_EXPIRE_MINUTES",
        "REFRESH_TOKEN_EXPIRE_DAYS",
    ):
        monkeypatch.delenv(var, raising=False)

    settings = Settings(_env_file=None)

    assert settings.jwt_algorithm == "HS256"
    assert settings.access_token_expire_minutes == 15
    assert settings.refresh_token_expire_days == 7


def test_jwt_settings_read_from_environment(monkeypatch: pytest.MonkeyPatch):
    """Env vars override the defaults and are coerced to the declared types."""
    monkeypatch.setenv("SECRET_KEY", "super-secret-value")
    monkeypatch.setenv("JWT_ALGORITHM", "HS512")
    monkeypatch.setenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30")
    monkeypatch.setenv("REFRESH_TOKEN_EXPIRE_DAYS", "14")

    settings = Settings(_env_file=None)

    assert settings.secret_key == "super-secret-value"
    assert settings.jwt_algorithm == "HS512"
    assert settings.access_token_expire_minutes == 30
    assert settings.refresh_token_expire_days == 14


def test_secret_key_is_required(monkeypatch: pytest.MonkeyPatch):
    """With no SECRET_KEY in the environment, settings fail fast at startup."""
    monkeypatch.delenv("SECRET_KEY", raising=False)

    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_storage_defaults(monkeypatch: pytest.MonkeyPatch):
    """Storage knobs have working defaults, so the app boots without them set."""
    monkeypatch.setenv("SECRET_KEY", "test-secret")
    for var in ("STORAGE_DIR", "MAX_UPLOAD_SIZE_BYTES"):
        monkeypatch.delenv(var, raising=False)

    settings = Settings(_env_file=None)

    assert settings.storage_dir == Path("./var/storage")
    assert settings.max_upload_size_bytes == 25 * 1024 * 1024


def test_storage_settings_read_from_environment(monkeypatch: pytest.MonkeyPatch):
    """STORAGE_DIR is coerced to a Path and the size cap to an int."""
    monkeypatch.setenv("SECRET_KEY", "test-secret")
    monkeypatch.setenv("STORAGE_DIR", "/srv/blobs")
    monkeypatch.setenv("MAX_UPLOAD_SIZE_BYTES", "1048576")

    settings = Settings(_env_file=None)

    assert settings.storage_dir == Path("/srv/blobs")
    assert settings.max_upload_size_bytes == 1048576


def test_chunking_defaults_and_override(monkeypatch: pytest.MonkeyPatch):
    """Chunk size/overlap default sensibly and read from the environment."""
    monkeypatch.setenv("SECRET_KEY", "test-secret")
    for var in ("CHUNK_SIZE", "CHUNK_OVERLAP"):
        monkeypatch.delenv(var, raising=False)
    assert Settings(_env_file=None).chunk_size == 2000
    assert Settings(_env_file=None).chunk_overlap == 200

    monkeypatch.setenv("CHUNK_SIZE", "1500")
    monkeypatch.setenv("CHUNK_OVERLAP", "150")
    settings = Settings(_env_file=None)
    assert settings.chunk_size == 1500
    assert settings.chunk_overlap == 150


def test_chunk_overlap_must_be_smaller_than_size(monkeypatch: pytest.MonkeyPatch):
    """A misconfigured overlap >= size fails fast at startup, not mid-ingestion."""
    monkeypatch.setenv("SECRET_KEY", "test-secret")
    monkeypatch.setenv("CHUNK_SIZE", "500")
    monkeypatch.setenv("CHUNK_OVERLAP", "500")

    with pytest.raises(ValidationError):
        Settings(_env_file=None)
