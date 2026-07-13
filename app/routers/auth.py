import logging

from fastapi import APIRouter, HTTPException, Request, status

from app.core.rate_limit import limiter
from app.dependencies import DbDependency
from app.schemas import CreateUserRequest, UserResponse
from app.services import user_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/auth", tags=["Auth"])


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
