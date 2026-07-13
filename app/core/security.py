"""Password hashing primitives.

Stateless, dependency-free security helpers. They belong here — not in a
service — because both sides of auth use them: the user service hashes at
registration, the auth service verifies at login. Keeping them in a shared
module means neither service has to depend on the other for a primitive.
"""

import bcrypt


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(plain: str, hashed: str) -> bool:
    return bcrypt.checkpw(plain.encode("utf-8"), hashed.encode("utf-8"))
