# -*- coding: utf-8 -*-
"""Tests for the Anthropic ``stop_reason="refusal"`` surfacing path.

Fable 5 / Mythos-class models run a streaming safety classifier that can
hard-stop a response with ``stop_reason="refusal"`` and zero content.
The HTTP call succeeds, so without special handling the agent loop
treats the empty response as a normal completion and the channel goes
silent (observed live: ~/.copaw/logs/claude_model_fallback.jsonl).

Locks down:
* ``_peek_stream_for_cache`` raises ``ModelRefusalException`` when the
  stream ends with refusal and produced no text/tool_use.
* Partial text (or tool_use) before the refusal suppresses the raise —
  partial content must not be discarded.
* Known stop reasons pass through untouched.
* ``_resp_has_visible_content`` (non-streaming gate) classifies block
  shapes correctly.
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

import qwenpaw.providers.anthropic_provider as ap
from qwenpaw.exceptions import ModelRefusalException
from qwenpaw.providers.anthropic_provider import (
    _peek_stream_for_cache,
    _resp_has_visible_content,
)


@pytest.fixture(autouse=True)
def _no_anomaly_file_writes(monkeypatch, tmp_path):
    """Redirect the anomaly JSONL away from the real ~/.copaw path."""
    monkeypatch.setattr(
        ap,
        "_FALLBACK_LOG_PATH",
        str(tmp_path / "fallback.jsonl"),
    )


def _ev_message_start(
    model: str = "claude-fable-5",
    rid: str = "msg_test",
) -> Any:
    return SimpleNamespace(
        type="message_start",
        message=SimpleNamespace(model=model, id=rid, usage=None),
    )


def _ev_text_block_start() -> Any:
    return SimpleNamespace(
        type="content_block_start",
        index=0,
        content_block=SimpleNamespace(type="text"),
    )


def _ev_tool_use_block_start() -> Any:
    return SimpleNamespace(
        type="content_block_start",
        index=0,
        content_block=SimpleNamespace(type="tool_use", id="tu_1", name="t"),
    )


def _ev_text_delta(text: str) -> Any:
    return SimpleNamespace(
        type="content_block_delta",
        index=0,
        delta=SimpleNamespace(text=text, partial_json=None),
    )


def _ev_message_delta(stop_reason: str) -> Any:
    return SimpleNamespace(
        type="message_delta",
        delta=SimpleNamespace(stop_reason=stop_reason),
        usage=None,
    )


def _ev_message_stop() -> Any:
    return SimpleNamespace(type="message_stop")


async def _stream(events: list[Any]):
    for e in events:
        yield e


async def _drain(gen) -> list[Any]:
    return [e async for e in gen]


# ---------------------------------------------------------------- #
# Streaming path                                                    #
# ---------------------------------------------------------------- #


async def test_refusal_without_content_raises() -> None:
    gen = _peek_stream_for_cache(
        _stream(
            [
                _ev_message_start(rid="msg_refused"),
                _ev_message_delta("refusal"),
                _ev_message_stop(),
            ],
        ),
        None,
        "claude-fable-5",
    )
    with pytest.raises(ModelRefusalException) as exc_info:
        await _drain(gen)
    details = exc_info.value.details
    assert details["model_name"] == "claude-fable-5"
    assert details["response_id"] == "msg_refused"


async def test_refusal_after_partial_text_does_not_raise() -> None:
    events = [
        _ev_message_start(),
        _ev_text_block_start(),
        _ev_text_delta("partial answer"),
        _ev_message_delta("refusal"),
        _ev_message_stop(),
    ]
    out = await _drain(_peek_stream_for_cache(_stream(events), None, "m"))
    assert len(out) == len(events)


async def test_refusal_after_tool_use_does_not_raise() -> None:
    events = [
        _ev_message_start(),
        _ev_tool_use_block_start(),
        _ev_message_delta("refusal"),
        _ev_message_stop(),
    ]
    out = await _drain(_peek_stream_for_cache(_stream(events), None, "m"))
    assert len(out) == len(events)


async def test_end_turn_passes_through() -> None:
    events = [
        _ev_message_start(),
        _ev_text_block_start(),
        _ev_text_delta("hello"),
        _ev_message_delta("end_turn"),
        _ev_message_stop(),
    ]
    out = await _drain(_peek_stream_for_cache(_stream(events), None, "m"))
    assert len(out) == len(events)


async def test_other_unusual_stop_reason_does_not_raise() -> None:
    """Only ``refusal`` raises; other anomalies are log-only."""
    events = [
        _ev_message_start(),
        _ev_message_delta("some_future_reason"),
        _ev_message_stop(),
    ]
    out = await _drain(_peek_stream_for_cache(_stream(events), None, "m"))
    assert len(out) == len(events)


async def test_refusal_logs_anomaly_before_raising(tmp_path) -> None:
    """The JSONL diagnostic record must still be written on raise."""
    calls: list[tuple] = []

    def _capture(*args: Any) -> None:
        calls.append(args)

    orig = ap._log_model_anomaly
    ap._log_model_anomaly = _capture
    try:
        gen = _peek_stream_for_cache(
            _stream([_ev_message_start(), _ev_message_delta("refusal")]),
            None,
            "claude-fable-5",
        )
        with pytest.raises(ModelRefusalException):
            await _drain(gen)
    finally:
        ap._log_model_anomaly = orig
    assert any(c[2] == "refusal" for c in calls)


# ---------------------------------------------------------------- #
# Non-streaming content gate                                        #
# ---------------------------------------------------------------- #


def test_visible_content_empty_list_is_false() -> None:
    assert not _resp_has_visible_content(SimpleNamespace(content=[]))


def test_visible_content_none_is_false() -> None:
    assert not _resp_has_visible_content(SimpleNamespace(content=None))


def test_visible_content_whitespace_text_is_false() -> None:
    resp = SimpleNamespace(
        content=[SimpleNamespace(type="text", text="  \n")],
    )
    assert not _resp_has_visible_content(resp)


def test_visible_content_text_is_true() -> None:
    resp = SimpleNamespace(
        content=[SimpleNamespace(type="text", text="hi")],
    )
    assert _resp_has_visible_content(resp)


def test_visible_content_tool_use_is_true() -> None:
    resp = SimpleNamespace(
        content=[SimpleNamespace(type="tool_use", id="tu", name="t")],
    )
    assert _resp_has_visible_content(resp)


def test_visible_content_thinking_only_is_false() -> None:
    """Thinking blocks are not user-visible — refusal should still raise."""
    resp = SimpleNamespace(
        content=[SimpleNamespace(type="thinking", thinking="...")],
    )
    assert not _resp_has_visible_content(resp)
