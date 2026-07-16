"""Unit tests for app/services/user_service.py.

create_user's happy path is already covered end-to-end through the register
route in test_auth.py. These cover what the route tests cannot reach: the
conflicting-field detail on DuplicateUserError (which the 400 response does not
expose), and get_user_by_id's handling of ids that are not valid ObjectIds.
"""

import pytest
from bson import ObjectId

from app.schemas.auth import CreateUserRequest
from app.services.user_service import DuplicateUserError, create_user, get_user_by_id


def _request(**overrides) -> CreateUserRequest:
    data = {
        "email": "alice@example.com",
        "username": "alice",
        "password": "supersecret",
    }
    data.update(overrides)
    return CreateUserRequest(**data)


async def test_get_user_by_id_returns_the_user(fake_db):
    created = await create_user(fake_db, _request())

    found = await get_user_by_id(fake_db, created.id)

    assert found is not None
    assert found["username"] == "alice"


async def test_get_user_by_id_returns_none_for_unknown_id(fake_db):
    assert await get_user_by_id(fake_db, str(ObjectId())) is None


async def test_get_user_by_id_returns_none_for_malformed_id(fake_db):
    """A string that is not a valid ObjectId can never match a real user, so it
    returns None rather than letting InvalidId escape as a 500."""
    assert await get_user_by_id(fake_db, "not-an-objectid") is None


async def test_duplicate_email_error_reports_the_conflicting_field(fake_db):
    await create_user(fake_db, _request())

    with pytest.raises(DuplicateUserError) as exc_info:
        await create_user(fake_db, _request(username="alice2"))

    assert exc_info.value.field == "email"


async def test_duplicate_username_error_reports_the_conflicting_field(fake_db):
    await create_user(fake_db, _request())

    with pytest.raises(DuplicateUserError) as exc_info:
        await create_user(fake_db, _request(email="other@example.com"))

    assert exc_info.value.field == "username"
