# -*- coding: utf-8 -*-
"""Tests for ``QwenPawAgent._build_context_exceeded_reply``.

When ``OpenAIChatModelCompat._parse_openai_stream_response`` raises
``ModelContextLengthExceededException`` (because z.ai signalled
``finish_reason="model_context_window_exceeded"`` on an empty terminal
chunk), the agent must surface a user-visible reply instead of letting
the error propagate and the channel see silence.

Locks down:
* TextBlock output with the configured language's text.
* Empty assistant placeholder is mutated in place (no duplicate empty
  entry left in memory).
* Fallback path appends a fresh assistant Msg when no placeholder exists.
* ``print(msg, last=True)`` is always called so streaming channels emit
  the message as their final chunk.
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from agentscope.message import Msg, TextBlock

from qwenpaw.agents.react_agent import QwenPawAgent
from qwenpaw.exceptions import ModelContextLengthExceededException


class _FakeMemory:
    def __init__(self, content: list[Any]) -> None:
        # Mirror the shape react_agent reads — list of (msg, marks) pairs.
        self.content = content
        self.added: list[Msg] = []

    async def add(self, msg: Msg) -> None:
        self.added.append(msg)
        self.content.append((msg, None))


def _make_fake_self(
    *,
    language: str = "en",
    memory_content: list[Any] | None = None,
) -> Any:
    print_calls: list[tuple[Msg, bool]] = []

    async def fake_print(msg: Msg, last: bool = False, **_: Any) -> None:
        print_calls.append((msg, last))

    fake = SimpleNamespace(
        _language=language,
        _CONTEXT_EXCEEDED_TEXT=QwenPawAgent._CONTEXT_EXCEEDED_TEXT,
        name="test-agent",
        memory=_FakeMemory(list(memory_content or [])),
        print=fake_print,
        print_calls=print_calls,
    )
    return fake


def _empty_assistant_msg() -> Msg:
    return Msg("test-agent", [], "assistant")


async def _invoke(fake: Any) -> Msg:
    exc = ModelContextLengthExceededException(
        model_name="glm-5v-turbo",
        details={"finish_reason": "model_context_window_exceeded"},
    )
    return await QwenPawAgent._build_context_exceeded_reply(fake, exc)


async def test_replaces_empty_assistant_placeholder_in_memory() -> None:
    placeholder = _empty_assistant_msg()
    fake = _make_fake_self(memory_content=[(placeholder, None)])

    msg = await _invoke(fake)

    # Returns the SAME object (mutated in place) — so the parent's
    # _reasoning finally already wired this into memory, no duplicate.
    assert msg is placeholder
    text_blocks = msg.get_content_blocks("text")
    assert len(text_blocks) == 1
    assert "context window" in text_blocks[0]["text"].lower()
    # Nothing extra appended to memory.
    assert fake.memory.added == []


async def test_falls_back_to_append_when_no_placeholder() -> None:
    fake = _make_fake_self(memory_content=[])

    msg = await _invoke(fake)

    assert msg.role == "assistant"
    text_blocks = msg.get_content_blocks("text")
    assert text_blocks and "context window" in text_blocks[0]["text"].lower()
    # Added to memory exactly once.
    assert fake.memory.added == [msg]


async def test_language_zh_yields_chinese_text() -> None:
    fake = _make_fake_self(language="zh", memory_content=[])
    msg = await _invoke(fake)
    text = msg.get_content_blocks("text")[0]["text"]
    assert "上下文" in text
    assert "/new" in text and "/compact" in text


async def test_language_ru_yields_russian_text() -> None:
    fake = _make_fake_self(language="ru", memory_content=[])
    msg = await _invoke(fake)
    text = msg.get_content_blocks("text")[0]["text"]
    assert "контекст" in text.lower()


async def test_unknown_language_falls_back_to_english() -> None:
    fake = _make_fake_self(language="ja", memory_content=[])
    msg = await _invoke(fake)
    text = msg.get_content_blocks("text")[0]["text"]
    assert "context window" in text.lower()


async def test_calls_print_with_last_true() -> None:
    fake = _make_fake_self(memory_content=[])
    msg = await _invoke(fake)
    assert fake.print_calls == [(msg, True)]


async def test_reply_has_no_tool_use_blocks() -> None:
    """ReAct loop exits via the ``not has_tool_use`` branch — make sure
    we don't accidentally smuggle a tool_use through this path."""
    fake = _make_fake_self(memory_content=[])
    msg = await _invoke(fake)
    assert msg.get_content_blocks("tool_use") == []


async def test_does_not_replace_assistant_msg_with_real_content() -> None:
    """If the parent's _reasoning yielded partial real content before the
    overflow chunk, we MUST NOT clobber it — append a fresh msg instead."""
    partial = Msg(
        "test-agent",
        [TextBlock(type="text", text="partial answer so far")],
        "assistant",
    )
    fake = _make_fake_self(memory_content=[(partial, None)])

    msg = await _invoke(fake)

    # Partial msg is untouched.
    assert partial.get_content_blocks("text")[0]["text"] == (
        "partial answer so far"
    )
    # Reply is a different (newly added) msg.
    assert msg is not partial
    assert fake.memory.added == [msg]


async def test_memory_failure_does_not_raise() -> None:
    """A broken memory.add must NOT prevent surfacing the reply."""

    class _BrokenMemory(_FakeMemory):
        async def add(self, msg: Msg) -> None:  # type: ignore[override]
            raise RuntimeError("memory backend down")

    async def fake_print(msg: Msg, last: bool = False, **_: Any) -> None:
        pass

    fake = SimpleNamespace(
        _language="en",
        _CONTEXT_EXCEEDED_TEXT=QwenPawAgent._CONTEXT_EXCEEDED_TEXT,
        name="test-agent",
        memory=_BrokenMemory([]),
        print=fake_print,
    )
    msg = await QwenPawAgent._build_context_exceeded_reply(
        fake,
        ModelContextLengthExceededException("m", details={}),
    )
    assert msg.get_content_blocks("text")[0]["text"]


async def test_print_failure_does_not_raise() -> None:
    """A broken print must NOT prevent returning the reply."""

    async def broken_print(msg: Msg, last: bool = False, **_: Any) -> None:
        raise RuntimeError("channel hung up")

    fake = SimpleNamespace(
        _language="en",
        _CONTEXT_EXCEEDED_TEXT=QwenPawAgent._CONTEXT_EXCEEDED_TEXT,
        name="test-agent",
        memory=_FakeMemory([]),
        print=broken_print,
    )
    msg = await QwenPawAgent._build_context_exceeded_reply(
        fake,
        ModelContextLengthExceededException("m", details={}),
    )
    assert msg.get_content_blocks("text")[0]["text"]
