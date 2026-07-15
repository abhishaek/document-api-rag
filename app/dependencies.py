"""Shared FastAPI dependencies, exposed as reusable type aliases.

A dependency alias bundles a type together with its ``Depends(...)`` so route
handlers can annotate a parameter with just the alias:

    from app.dependencies import DbDependency

    @router.post("/register")
    async def register_user(payload: CreateUserRequest, db: DbDependency):
        await db.users.insert_one(...)

FastAPI resolves ``get_database`` for each request and injects the shared
database handle (the connection pool opened once at startup — see app.db.mongodb).
"""

import logging
from typing import Annotated

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from pymongo.asynchronous.database import AsyncDatabase

from app.core.config import get_settings
from app.db.mongodb import get_database
from app.services.auth_service import is_access_token_revoked

logger = logging.getLogger(__name__)
settings = get_settings()

DbDependency = Annotated[AsyncDatabase, Depends(get_database)]

# Extracts the bearer token from the Authorization header. tokenUrl points at
# the login route so Swagger's "Authorize" button knows where to get a token.
oauth2_bearer = OAuth2PasswordBearer(tokenUrl="/v1/auth/login")


async def get_current_user(
    token: Annotated[str, Depends(oauth2_bearer)], db: DbDependency
) -> dict:
    """Decode and validate the access token, returning its identity claims.

    Raises 401 if the token is expired, malformed, missing required claims, or
    has been revoked (logged out).
    """
    try:
        payload = jwt.decode(
            token, settings.secret_key, algorithms=[settings.jwt_algorithm]
        )
    except jwt.ExpiredSignatureError:
        logger.warning("Expired token used")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired. Please log in again.",
        ) from None
    except jwt.InvalidTokenError:
        logger.warning("Invalid token: signature or format error")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials.",
        ) from None

    username: str | None = payload.get("sub")
    user_id: str | None = payload.get("id")
    role: str | None = payload.get("role")
    jti: str | None = payload.get("jti")
    exp: int | None = payload.get("exp")
    if username is None or user_id is None:
        logger.warning("Token missing required claims: sub or id absent")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials.",
        )
    if jti and await is_access_token_revoked(jti, db):
        logger.warning("Revoked token used: jti=%s", jti)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has been revoked. Please log in again.",
        )
    return {"username": username, "id": user_id, "role": role, "jti": jti, "exp": exp}


UserDependency = Annotated[dict, Depends(get_current_user)]
