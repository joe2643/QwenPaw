# -*- coding: utf-8 -*-
"""Stream-parser regression tests for z.ai ``model_context_window_exceeded``.

The provider streams a final empty delta with ``finish_reason=
"model_context_window_exceeded"`` instead of raising an HTTP error.  Upstream
agentscope's parser treats that as a normal end-of-stream and yields nothing,
which previously made the agent silently terminate.  The compat wrapper now
surfaces that path as ``ModelContextLengthExceededException``.
"""
from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace
from typing import Any

import pytest

from qwenpaw.exceptions import ModelContextLengthExceededException
from qwenpaw.providers.openai_chat_model_compat import (
    OpenAIChatModelCompat,
    _has_actionable_content,
)


class _Harness(OpenAIChatModelCompat):
    async def parse(self, stream: Any) -> list[Any]:
        out = []
        async for r in self._parse_openai_stream_response(
            datetime.now(),
            stream,
        ):
            out.append(r)
        return out


class _FakeStream:
    def __init__(self, items: list[Any]) -> None:
        self._items = items
        self._iter = None

    async def __aenter__(self) -> "_FakeStream":
        self._iter = iter(self._items)
        return self

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        return False

    def __aiter__(self) -> "_FakeStream":
        return self

    async def __anext__(self) -> Any:
        assert self._iter is not None
        try:
            return next(self._iter)
        except StopIteration as exc:
            raise StopAsyncIteration from exc


def _delta_chunk(
    *,
    content: str | None = None,
    tool_calls: list[Any] | None = None,
    finish_reason: str | None = None,
) -> Any:
    delta = SimpleNamespace(
        reasoning_content=None,
        content=content,
        tool_calls=tool_calls or [],
    )
    choice = SimpleNamespace(delta=delta, finish_reason=finish_reason)
    return SimpleNamespace(usage=None, choices=[choice])


def _model() -> _Harness:
    return _Harness("glm-5v-turbo", api_key="sk-test", stream=True)


async def test_raises_on_model_context_window_exceeded() -> None:
    """Z.AI's exact wire: empty delta + special finish_reason → raise."""
    stream = _FakeStream(
        [
            _delta_chunk(
                content="",
                finish_reason="model_context_window_exceeded",
            ),
        ],
    )
    with pytest.raises(ModelContextLengthExceededException) as ei:
        await _model().parse(stream)
    assert "glm-5v-turbo" in str(ei.value)


async def test_raises_on_context_length_exceeded_alias() -> None:
    """Alternate spelling some OpenAI-compat providers emit."""
    stream = _FakeStream(
        [_delta_chunk(finish_reason="context_length_exceeded")],
    )
    with pytest.raises(ModelContextLengthExceededException):
        await _model().parse(stream)


async def test_length_finish_reason_with_empty_content_raises() -> None:
    """``length`` + no body is treated as overflow (z.ai legacy code path)."""
    stream = _FakeStream(
        [
            _delta_chunk(content="", finish_reason="length"),
        ],
    )
    with pytest.raises(ModelContextLengthExceededException):
        await _model().parse(stream)


async def test_length_finish_reason_with_real_content_does_not_raise() -> None:
    """Ordinary max_tokens truncation must still yield, not raise."""
    stream = _FakeStream(
        [
            _delta_chunk(content="hello "),
            _delta_chunk(content="world", finish_reason="length"),
        ],
    )
    responses = await _model().parse(stream)
    assert responses
    # Last response must contain the accumulated text.
    last = responses[-1]
    text_blocks = [b for b in last.content if b.get("type") == "text"]
    assert text_blocks and "world" in text_blocks[-1]["text"]


async def test_normal_stop_does_not_raise() -> None:
    """Happy-path text reply must not be turned into a context error."""
    stream = _FakeStream(
        [
            _delta_chunk(content="hi"),
            _delta_chunk(content="!", finish_reason="stop"),
        ],
    )
    responses = await _model().parse(stream)
    assert responses
    text_blocks = [
        b
        for r in responses
        for b in r.content
        if b.get("type") == "text"
    ]
    assert text_blocks and text_blocks[-1]["text"].endswith("!")


async def test_tool_call_stream_does_not_raise_on_length() -> None:
    """A stream that yielded a tool_use must not be re-classified as overflow
    even if the provider tags the final chunk with ``length``."""
    tool_call = SimpleNamespace(
        index=0,
        id="call_x",
        function=SimpleNamespace(name="ping", arguments='{"x":1}'),
    )
    stream = _FakeStream(
        [
            _delta_chunk(tool_calls=[tool_call]),
            _delta_chunk(finish_reason="length"),
        ],
    )
    responses = await _model().parse(stream)
    tool_blocks = [
        b
        for r in responses
        for b in r.content
        if b.get("type") == "tool_use"
    ]
    assert tool_blocks


def test_has_actionable_content_recognizes_text_and_tool_use() -> None:
    assert _has_actionable_content([{"type": "text", "text": "hi"}])
    assert _has_actionable_content(
        [{"type": "tool_use", "id": "1", "name": "x", "input": {}}],
    )
    # Whitespace-only text doesn't count.
    assert not _has_actionable_content([{"type": "text", "text": "   "}])
    # Thinking-only doesn't count (z.ai's overflow path can leak a short
    # reasoning_content prefix before the empty terminal chunk).
    assert not _has_actionable_content(
        [{"type": "thinking", "thinking": "pondering..."}],
    )
    assert not _has_actionable_content([])
    assert not _has_actionable_content(None)
