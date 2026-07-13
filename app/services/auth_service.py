"""Authentication & authorization logic.

This service owns credential verification and the token lifecycle. It is being
built out — planned functions:

* ``authenticate_user(username, password, db)`` — verify credentials, return the
  user (uses ``verify_password`` from ``app.core.security``).
* ``create_access_token(...)`` — issue a short-lived JWT.
* ``create_refresh_token`` / ``verify_refresh_token`` / ``rotate_refresh_token``
  — manage long-lived refresh tokens in a ``refresh_tokens`` collection
  (store only the SHA-256 hash, plus a TTL index for automatic expiry).

User creation lives in ``app.services.user_service`` (user management, not auth).
"""
