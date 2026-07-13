"""Rate limiting via slowapi.

The ``limiter`` is shared app-wide. Endpoints opt in with ``@limiter.limit(...)``
and must accept a ``request: Request`` parameter (slowapi reads the client IP
from it). The limiter is registered on the app and its exception handler wired
up in ``app.main``.

Note: the default in-memory store is per-process. For multiple workers/instances
in production, point slowapi at a shared store (e.g. Redis) via ``storage_uri``.
"""

from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
