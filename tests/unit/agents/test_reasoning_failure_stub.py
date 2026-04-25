# -*- coding: utf-8 -*-
"""Unit tests for the reasoning-failure tombstone path.

When ``QwenPawAgent._reasoning`` raises after retries are exhausted,
``_record_reasoning_failure`` writes an audit entry to memory so the
next user turn has context about what failed.  Three flavours:

1. Handshake / first-chunk error — parent's ``finally`` already wrote
   an empty assistant Msg.  We replace its content in place.
2. Mid-stream error after partial yield — assistant Msg holds chunks
   1..N.  We append a system note instead of overwriting.
3. No assistant placeholder at all (e.g. tool-guard short-circuit).
   Append a fresh system note.

These map directly to the three production failure modes captured in
``/tmp/qwenpaw_query_error_*.json`` dumps when Codex returns 503.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from agentscope.message import Msg, TextBlock, ToolUseBlock


class _FakeMemory:
    """Minimal in-memory store shaped like agentscope's working memory."""

    def __init__(self):
        self._content: list[tuple[Msg, list[str]]] = []

    @property
    def content(self):
        return self._content

    async def add(self, msg, marks=None):
        if msg is None:
            return
        self._content.append((msg, marks or []))


def _make_stub_recorder():
    """Build a callable that runs ``_record_reasoning_failure`` against
    a stand-in object exposing only the surface area the helper touches.

    Avoids constructing the full ``QwenPawAgent`` (whose ``__init__``
    pulls in agentscope-runtime, channel manager, plugin registry,
    etc.) — we just need the method's behaviour against a fake memory.
    """
    from qwenpaw.agents.react_agent import QwenPawAgent

    fake_self = SimpleNamespace(memory=_FakeMemory())
    return (
        fake_self,
        QwenPawAgent._record_reasoning_failure.__get__(
            fake_self,
            type(fake_self),
        ),
    )


@pytest.mark.asyncio
async def test_replaces_empty_assistant_placeholder():
    """Handshake error path — assistant Msg with empty content list
    must be rewritten to carry the error stub, not appended after."""
    obj, record = _make_stub_recorder()
    placeholder = Msg("assistant", [], "assistant")
    await obj.memory.add(placeholder)

    err = RuntimeError("Codex upstream HTTP 503")
    await record(err, "auto")

    # Still exactly one entry — placeholder rewritten in place.
    assert len(obj.memory.content) == 1
    last_msg, _ = obj.memory.content[0]
    assert last_msg.role == "assistant"
    blocks = last_msg.get_content_blocks("text")
    assert blocks, "assistant content should now carry the error stub"
    text = blocks[0].get("text", "")
    assert "RuntimeError" in text
    assert "503" in text
    assert "tool_choice=auto" in text


@pytest.mark.asyncio
async def test_appends_system_note_when_assistant_has_partial_text():
    """Mid-stream failure — assistant Msg has real text content.
    We must NOT overwrite it; append a system note instead."""
    obj, record = _make_stub_recorder()
    partial = Msg(
        "assistant",
        [TextBlock(type="text", text="I was about to call the tool when")],
        "assistant",
    )
    await obj.memory.add(partial)

    err = RuntimeError("Codex upstream HTTP 503")
    await record(err, "auto")

    # Partial preserved + system note appended.
    assert len(obj.memory.content) == 2
    preserved, _ = obj.memory.content[0]
    note, _ = obj.memory.content[1]
    assert preserved is partial
    assert preserved.get_content_blocks("text")[0]["text"].startswith(
        "I was about to call",
    )
    assert note.role == "system"
    assert "RuntimeError" in note.get_content_blocks("text")[0]["text"]


@pytest.mark.asyncio
async def test_appends_system_note_when_assistant_has_partial_tool_use():
    """Mid-stream failure — assistant Msg has tool_use blocks (no text
    yet).  Don't overwrite; append.  Without this test, a future
    ``has_text only`` heuristic could regress to overwriting tool calls."""
    obj, record = _make_stub_recorder()
    partial = Msg(
        "assistant",
        [
            ToolUseBlock(
                type="tool_use",
                id="call_1",
                name="search",
                input={"q": "foo"},
            ),
        ],
        "assistant",
    )
    await obj.memory.add(partial)

    err = RuntimeError("connection reset")
    await record(err, None)

    assert len(obj.memory.content) == 2
    preserved, _ = obj.memory.content[0]
    assert preserved.get_content_blocks("tool_use")[0]["name"] == "search"


@pytest.mark.asyncio
async def test_appends_system_note_when_no_prior_assistant():
    """Tool-guard short-circuit path — no model call ever happened, so
    no placeholder.  Append fresh system note."""
    obj, record = _make_stub_recorder()
    user_msg = Msg("user", [TextBlock(type="text", text="hi")], "user")
    await obj.memory.add(user_msg)

    err = ValueError("guarded")
    await record(err, "auto")

    assert len(obj.memory.content) == 2
    note, _ = obj.memory.content[1]
    assert note.role == "system"
    assert "ValueError" in note.get_content_blocks("text")[0]["text"]


@pytest.mark.asyncio
async def test_swallows_internal_helper_errors():
    """The tombstone helper must NEVER raise — masking the original
    exception would be a bigger regression than losing the audit
    record.  Simulate by wiring a memory that explodes on ``add``."""

    class _BrokenMemory:
        @property
        def content(self):
            return []

        async def add(self, msg, marks=None):
            raise RuntimeError("memory write failed")

    from qwenpaw.agents.react_agent import QwenPawAgent

    fake_self = SimpleNamespace(memory=_BrokenMemory())
    record = QwenPawAgent._record_reasoning_failure.__get__(
        fake_self,
        type(fake_self),
    )

    # Must not raise — the caller is already handling another exception.
    await record(RuntimeError("upstream 503"), "auto")


@pytest.mark.asyncio
async def test_truncates_long_error_messages():
    """Codex error bodies can be multi-KB.  The stub must clip so
    memory entries don't bloat the next prompt."""
    obj, record = _make_stub_recorder()
    long_body = "x" * 5000
    err = RuntimeError(long_body)

    await record(err, "auto")

    text = obj.memory.content[-1][0].get_content_blocks("text")[0]["text"]
    # 300-char clip from helper + surrounding annotation framing
    assert len(text) < 600, f"stub text length {len(text)} not clipped"
