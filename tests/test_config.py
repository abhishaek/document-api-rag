"""Verify application settings load from the environment as expected.

These tests construct ``Settings`` directly (rather than the cached
``get_settings()``) so each case controls its own environment in isolation.
``_env_file=None`` skips the on-disk ``.env`` so tests don't depend on local
developer files.
"""

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
