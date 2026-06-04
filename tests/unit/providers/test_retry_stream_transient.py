# -*- coding: utf-8 -*-
"""Tests for retry classification of Anthropic streaming-phase transient
errors (overloaded_error / api_error), which carry ``status_code=200``
because the error surfaces inside an already-established HTTP 200 stream.
"""
from __future__ import annotations

from qwenpaw.providers.retry_chat_model import (
    _is_anthropic_stream_transient,
    _is_retryable,
)


class FakeAPIStatusError(Exception):
    """Mimics ``anthropic.APIStatusError``: stringifies to its body dict
    and exposes ``body`` + ``status_code`` attributes.
    """

    def __init__(self, body: dict, status_code: int = 200):
        super().__init__(str(body))
        self.body = body
        self.status_code = status_code


def _overloaded_body() -> dict:
    return {
        "type": "error",
        "error": {"details": None, "type": "overloaded_error",
                  "message": "Overloaded"},
    }


def _api_error_body() -> dict:
    return {
        "type": "error",
        "error": {"details": None, "type": "api_error",
                  "message": "Internal server error"},
    }


def test_overloaded_stream_error_is_retryable() -> None:
    exc = FakeAPIStatusError(_overloaded_body(), status_code=200)
    assert _is_anthropic_stream_transient(exc)
    assert _is_retryable(exc)


def test_api_error_stream_error_is_retryable() -> None:
    exc = FakeAPIStatusError(_api_error_body(), status_code=200)
    assert _is_anthropic_stream_transient(exc)
    assert _is_retryable(exc)


def test_string_only_overloaded_is_retryable() -> None:
    # No structured body (e.g. body parsed as str) — fall back to the
    # stringified form.
    class StrOnly(Exception):
        status_code = 200

    exc = StrOnly(
        "{'type': 'error', 'error': {'type': 'overloaded_error'}}",
    )
    assert _is_anthropic_stream_transient(exc)
    assert _is_retryable(exc)


def test_client_error_not_retryable() -> None:
    # A genuine 400 client error must NOT be treated as a stream transient.
    exc = FakeAPIStatusError(
        {"type": "error",
         "error": {"type": "invalid_request_error", "message": "bad"}},
        status_code=400,
    )
    assert not _is_anthropic_stream_transient(exc)
    assert not _is_retryable(exc)


def test_plain_exception_not_retryable() -> None:
    assert not _is_anthropic_stream_transient(ValueError("nope"))
    assert not _is_retryable(ValueError("nope"))


def test_status_code_path_still_works() -> None:
    # A real 529 (non-streaming) still retries via the status-code check.
    class Plain529(Exception):
        status_code = 529

    assert _is_retryable(Plain529("overloaded"))
