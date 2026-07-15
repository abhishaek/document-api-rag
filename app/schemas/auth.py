"""Auth request/response schemas."""

from enum import StrEnum

from pydantic import BaseModel, EmailStr, Field


class UserRole(StrEnum):
    """Allowed user roles. Using an enum validates the value automatically."""

    admin = "admin"
    user = "user"


class CreateUserRequest(BaseModel):
    email: EmailStr
    username: str = Field(min_length=3, max_length=50)
    password: str = Field(min_length=8, max_length=128)
    role: UserRole = UserRole.user


class UserResponse(BaseModel):
    # In MongoDB the primary key is the ObjectId `_id`, exposed here as a string.
    id: str
    email: EmailStr
    username: str
    role: UserRole
    is_active: bool


class Token(BaseModel):
    access_token: str
    token_type: str
    refresh_token: str


class RefreshRequest(BaseModel):
    refresh_token: str