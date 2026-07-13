"""Centralized logging configuration.

Call :func:`configure_logging` once at startup (from ``app.main``). It produces:

* human-readable console logs in development (``LOG_JSON=false``)
* single-line JSON logs in production (``LOG_JSON=true``) — the format log
  aggregators like Datadog, CloudWatch, and the ELK stack expect.

Every record emitted while handling a request carries a ``request_id`` so you
can trace all logs for a single request. That id is populated by
``RequestContextMiddleware`` (see ``app.middleware``) via the context var below.

Usage in any module:
    import logging
    logger = logging.getLogger(__name__)
    logger.info("document ingested", extra={"document_id": doc_id})
"""

import json
import logging
import logging.config
import sys
from contextvars import ContextVar
from datetime import UTC, datetime

from app.core.config import Settings

# The current request's correlation id. "-" outside of a request (e.g. startup).
request_id_ctx: ContextVar[str] = ContextVar("request_id", default="-")

# LogRecord attributes that are part of the standard library; anything else on a
# record is treated as a caller-supplied `extra` and included in JSON output.
_STANDARD_ATTRS = {
    "name", "msg", "args", "levelname", "levelno", "pathname", "filename",
    "module", "exc_info", "exc_text", "stack_info", "lineno", "funcName",
    "created", "msecs", "relativeCreated", "thread", "threadName",
    "processName", "process", "taskName", "request_id", "message",
    "color_message",  # uvicorn adds this ANSI-colored variant; not for JSON.
}


class RequestIdFilter(logging.Filter):
    """Attaches the current request_id to every log record."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_ctx.get()
        return True


class JsonFormatter(logging.Formatter):
    """Renders a log record as a single JSON line."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "request_id": getattr(record, "request_id", "-"),
        }
        # Merge any caller-supplied extras (logger.info(..., extra={...})).
        for key, value in record.__dict__.items():
            if key not in _STANDARD_ATTRS:
                payload[key] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging(settings: Settings) -> None:
    """Apply logging configuration for the whole process."""
    formatter = "json" if settings.log_json else "console"

    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "filters": {
                "request_id": {"()": RequestIdFilter},
            },
            "formatters": {
                "json": {"()": JsonFormatter},
                "console": {
                    "format": (
                        "%(asctime)s | %(levelname)-8s | %(name)s "
                        "| [%(request_id)s] %(message)s"
                    ),
                    "datefmt": "%Y-%m-%d %H:%M:%S",
                },
            },
            "handlers": {
                "default": {
                    "class": "logging.StreamHandler",
                    "stream": sys.stdout,
                    "formatter": formatter,
                    "filters": ["request_id"],
                },
            },
            # Root catches our app.* loggers and any third-party library.
            "root": {"level": settings.log_level, "handlers": ["default"]},
            "loggers": {
                # Route uvicorn's loggers through our handler so every line shares
                # one format. propagate=False stops duplicate log lines.
                name: {
                    "handlers": ["default"],
                    "level": settings.log_level,
                    "propagate": False,
                }
                for name in ("uvicorn", "uvicorn.error", "uvicorn.access")
            },
        }
    )
