# -*- coding: utf-8 -*-
"""Tests for ``QwenPawAgent._build_refusal_reply``.

When the Anthropic provider raises ``ModelRefusalException`` (Fable 5
streaming safety classifier ends the response with
``stop_reason="refusal"`` and no content), the agent must surface a
user-visible notice instead of ending the turn silently.

The placeholder/print mechanics are shared with the context-exceeded
path via ``_surface_notice_reply`` and locked down in
``test_react_agent_context_exceeded.py`` — here we cover the
refusal-specific text selection and wiring.
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from agentscope.message import Msg

from qwenpaw.agents.react_agent import QwenPawAgent
from qwenpaw.exceptions import ModelRefusalException


class _FakeMemory:
    def __init__(self, content: list[Any]) -> None:
        self.content = content
        self.added: list[Msg] = []

    async def add(self, msg: Msg) -> None:
        self.added.append(msg)
        self.content.append((msg, None))


def _make_fake_self(*, language: str = "en") -> Any:
    print_calls: list[tuple[Msg, bool]] = []

    async def fake_print(msg: Msg, last: bool = False, **_: Any) -> None:
        print_calls.append((msg, last))

    fake = SimpleNamespace(
        _language=language,
        _REFUSAL_TEXT=QwenPawAgent._REFUSAL_TEXT,
        name="test-agent",
        memory=_FakeMemory([]),
        print=fake_print,
        print_calls=print_calls,
    )
    fake._surface_notice_reply = (
        lambda text: QwenPawAgent._surface_notice_reply(fake, text)
    )
    return fake


async def _invoke(fake: Any) -> Msg:
    exc = ModelRefusalException(
        "claude-fable-5",
        response_id="msg_refused",
    )
    return await QwenPawAgent._build_refusal_reply(fake, exc)


async def test_english_notice_mentions_refusal_and_commands() -> None:
    fake = _make_fake_self()
    msg = await _invoke(fake)
    text = msg.get_content_blocks("text")[0]["text"]
    assert "refusal" in text
    assert "/compact" in text and "/model" in text


async def test_zh_notice_selected_for_chinese_agents() -> None:
    fake = _make_fake_self(language="zh")
    msg = await _invoke(fake)
    text = msg.get_content_blocks("text")[0]["text"]
    assert "安全分類器" in text


async def test_regioned_language_code_matches_base_entry() -> None:
    """'zh-CN' must select the zh text, not fall back to English."""
    fake = _make_fake_self(language="zh-CN")
    msg = await _invoke(fake)
    text = msg.get_content_blocks("text")[0]["text"]
    assert "安全分類器" in text


async def test_unknown_language_falls_back_to_english() -> None:
    fake = _make_fake_self(language="ja")
    msg = await _invoke(fake)
    text = msg.get_content_blocks("text")[0]["text"]
    assert "safety classifier" in text


async def test_notice_is_printed_as_final_chunk() -> None:
    fake = _make_fake_self()
    msg = await _invoke(fake)
    assert fake.print_calls == [(msg, True)]


async def test_reply_has_no_tool_use_blocks() -> None:
    fake = _make_fake_self()
    msg = await _invoke(fake)
    assert msg.get_content_blocks("tool_use") == []


# ---------------------------------------------------------------- #
# Outer _reasoning coverage                                         #
# ---------------------------------------------------------------- #


async def test_outer_reasoning_surfaces_refusal_from_any_call_site() -> None:
    """The refusal catch lives in the OUTER ``_reasoning`` wrapper so it
    also covers the media-retry calls inside
    ``_reasoning_with_media_fallback`` — a refusal there must surface
    the notice, not the error-tombstone path."""
    sentinel = Msg("test-agent", "notice", "assistant")
    failures: list[Exception] = []

    async def drain() -> list:
        return []

    async def inner_raises(tool_choice=None) -> Msg:
        # Stands in for a refusal escaping ANY of the inner call sites
        # (first call or media retries).
        raise ModelRefusalException("claude-fable-5")

    async def build_reply(exc: ModelRefusalException) -> Msg:
        return sentinel

    async def record_failure(e: Exception, tool_choice) -> None:
        failures.append(e)

    fake = SimpleNamespace(
        _drain_pending_steer_messages=drain,
        _reasoning_with_media_fallback=inner_raises,
        _build_refusal_reply=build_reply,
        _record_reasoning_failure=record_failure,
    )

    # __wrapped__ strips agentscope's hook dispatcher, which demands a
    # full agent instance with hook registries.
    msg = await QwenPawAgent._reasoning.__wrapped__(fake)

    assert msg is sentinel
    # The refusal must NOT be recorded as a reasoning failure tombstone.
    assert failures == []
