"""Unit tests for app/core/security.py (bcrypt password primitives).

Both services depend on these: user_service hashes at registration and
auth_service verifies at login, so a regression here breaks all of auth.
"""

from app.core.security import hash_password, verify_password


def test_hash_password_does_not_return_the_plaintext():
    hashed = hash_password("supersecret")

    assert hashed != "supersecret"
    assert "supersecret" not in hashed


def test_verify_password_accepts_the_correct_password():
    assert verify_password("supersecret", hash_password("supersecret")) is True


def test_verify_password_rejects_a_wrong_password():
    assert verify_password("wrong-password", hash_password("supersecret")) is False


def test_hash_password_uses_a_random_salt():
    """Two hashes of the same password must differ, so that identical passwords
    are not identifiable by comparing stored hashes."""
    assert hash_password("same-password") != hash_password("same-password")
