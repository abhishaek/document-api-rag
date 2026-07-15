import logging
from datetime import UTC, datetime
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.security import OAuth2PasswordRequestForm

from app.core.rate_limit import limiter
from app.dependencies import DbDependency, UserDependency
from app.schemas import CreateUserRequest, RefreshRequest, Token, UserResponse
from app.services import user_service
from app.services.auth_service import (
    authenticate_user,
    create_access_token,
    create_refresh_token,
    revoke_access_token,
    revoke_refresh_token,
    rotate_refresh_token,
    verify_refresh_token,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/auth", tags=["Auth"])


# user registration in the application
@router.post(
    "/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED
)
@limiter.limit("3/minute")
async def register_user(
    request: Request, payload: CreateUserRequest, db: DbDependency
) -> UserResponse:
    """Register a new user. Rate limited to 3 requests/minute per client IP.

    Returns 400 if the email or username is already taken.
    """
    try:
        user = await user_service.create_user(db, payload)
    except user_service.DuplicateUserError as exc:
        logger.warning(
            "Duplicate registration attempt: email=%s username=%s",
            payload.email,
            payload.username,
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)
        ) from exc

    logger.info("User registered: username=%s role=%s", user.username, user.role)
    return user


# user logs in the application
@router.post("/login", response_model=Token, status_code=status.HTTP_200_OK)
@limiter.limit("5/minute")
async def user_login(
    request: Request,
    db: DbDependency,
    form_data: Annotated[OAuth2PasswordRequestForm, Depends()],
):
    user = await authenticate_user(form_data.username, form_data.password, db)
    if not user:
        logger.warning("Failed login attempt: username=%s", form_data.username)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password",
        )

    user_id = str(user["_id"])
    access_token = create_access_token(user["username"], user_id, user["role"])
    refresh_token = await create_refresh_token(user_id, db)
    logger.info("User logged in: username=%s", user["username"])

    return {
        "access_token": access_token,
        "token_type": "bearer",
        "refresh_token": refresh_token,
    }


# Refresh token logic
@router.post("/refresh", response_model=Token, status_code=status.HTTP_200_OK)
@limiter.limit("10/minute")
async def refresh(request: Request, payload: RefreshRequest, db: DbDependency):
    token_row = await verify_refresh_token(payload.refresh_token, db)
    if not token_row:
        logger.warning("Invalid or expired refresh token used")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired refresh token.",
        )
    user = await user_service.get_user_by_id(db, token_row["user_id"])
    if not user:
        logger.warning(
            "Refresh token references missing user_id=%s", token_row["user_id"]
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found.",
        )
    if not user.get("is_active", False):
        # Account was deactivated after this token was issued. Drop the token so
        # it can't be used again, and refuse to mint a new one.
        logger.warning("Refresh attempt by inactive user: username=%s", user["username"])
        await revoke_refresh_token(payload.refresh_token, db)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Account is inactive.",
        )
    new_refresh = await rotate_refresh_token(token_row, db)
    new_access = create_access_token(user["username"], str(user["_id"]), user["role"])
    logger.info("Token refreshed: username=%s", user["username"])
    return {
        "access_token": new_access,
        "token_type": "bearer",
        "refresh_token": new_refresh,
    }


# Logging out the user
@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
async def logout(
    payload: RefreshRequest, db: DbDependency, current_user: UserDependency
):
    # Drop the refresh token so it can't be exchanged again...
    await revoke_refresh_token(payload.refresh_token, db)
    # ...and denylist the current access token so it stops working immediately,
    # rather than staying valid until it naturally expires.
    jti = current_user.get("jti")
    exp = current_user.get("exp")
    if jti and exp:
        await revoke_access_token(jti, datetime.fromtimestamp(exp, tz=UTC), db)
    logger.info("User logged out: username=%s", current_user["username"])