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

from typing import Annotated

from fastapi import Depends
from pymongo.asynchronous.database import AsyncDatabase

from app.db.mongodb import get_database

DbDependency = Annotated[AsyncDatabase, Depends(get_database)]
