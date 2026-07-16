"""Unit tests for app/logging_config.py.

The JSON formatter is what production log aggregators consume, so its output
contract matters: one line, parseable, carrying the request id and any caller
extras. These build LogRecords directly rather than going through the logging
config, so nothing here depends on global logger state.
"""

import json
import logging
import sys

from app.logging_config import JsonFormatter, RequestIdFilter, request_id_ctx


def _record(**overrides) -> logging.LogRecord:
    defaults = {
        "name": "app.test",
        "level": logging.INFO,
        "pathname": __file__,
        "lineno": 10,
        "msg": "hello",
        "args": (),
        "exc_info": None,
    }
    defaults.update(overrides)
    return logging.LogRecord(**defaults)


def test_json_formatter_emits_a_single_line_of_json():
    output = JsonFormatter().format(_record())

    assert "\n" not in output
    payload = json.loads(output)
    assert payload["message"] == "hello"
    assert payload["level"] == "INFO"
    assert payload["logger"] == "app.test"
    assert "timestamp" in payload


def test_json_formatter_interpolates_message_args():
    output = JsonFormatter().format(_record(msg="user %s logged in", args=("alice",)))

    assert json.loads(output)["message"] == "user alice logged in"


def test_json_formatter_includes_caller_supplied_extras():
    """logger.info(..., extra={"document_id": x}) must survive into the JSON."""
    record = _record()
    record.document_id = "abc123"

    payload = json.loads(JsonFormatter().format(record))

    assert payload["document_id"] == "abc123"


def test_json_formatter_omits_standard_logrecord_attributes():
    """Only the message, our fields, and caller extras belong in the output —
    not the whole LogRecord."""
    payload = json.loads(JsonFormatter().format(_record()))

    for noisy in ("pathname", "lineno", "args", "msg", "levelno", "thread"):
        assert noisy not in payload


def test_json_formatter_renders_exceptions():
    try:
        raise ValueError("boom")
    except ValueError:
        record = _record(exc_info=sys.exc_info())

    payload = json.loads(JsonFormatter().format(record))

    assert "ValueError: boom" in payload["exception"]


def test_json_formatter_falls_back_to_dash_without_a_request_id():
    payload = json.loads(JsonFormatter().format(_record()))

    assert payload["request_id"] == "-"


def test_json_formatter_serializes_non_json_extras():
    """default=str keeps a non-serializable extra from raising inside logging."""
    record = _record()
    record.when = object()

    payload = json.loads(JsonFormatter().format(record))

    assert isinstance(payload["when"], str)


def test_request_id_filter_stamps_the_current_request_id():
    token = request_id_ctx.set("req-42")
    try:
        record = _record()

        assert RequestIdFilter().filter(record) is True
        assert record.request_id == "req-42"
    finally:
        request_id_ctx.reset(token)


def test_request_id_filter_uses_dash_outside_a_request():
    """Startup and shutdown logs have no request context."""
    record = _record()

    RequestIdFilter().filter(record)

    assert record.request_id == "-"
