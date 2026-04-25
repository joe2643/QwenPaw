# -*- coding: utf-8 -*-
"""Verify that ``agentscope_msg_to_message`` promotes a tool-using
turn's preamble text from ``MESSAGE`` → ``REASONING`` so channels
can suppress it as thinking — but ONLY when the active provider is
``codex-oauth``.  Claude and Qwen-family agents keep their existing
behaviour because their preambles are intentional polite acks
that users want to see."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from agentscope.message import Msg
from agentscope_runtime.engine.schemas.agent_schemas import MessageType

from qwenpaw.app.runner.utils import agentscope_msg_to_message


def _types(messages):
    return [m.type for m in messages]


@pytest.fixture
def as_codex_oauth():
    """Pretend the active agent's provider is ``codex-oauth`` —
    the gate checked by the promotion logic."""
    with patch(
        "qwenpaw.app.runner.utils._active_provider_is_codex_oauth",
        return_value=True,
    ):
        yield


@pytest.fixture
def as_other_provider():
    """Pretend the active agent's provider is something else
    (Claude, Qwen, …); promotion must NOT fire."""
    with patch(
        "qwenpaw.app.runner.utils._active_provider_is_codex_oauth",
        return_value=False,
    ):
        yield


# ---------------------------------------------------------------- #
# Codex OAuth path — promotion must fire                           #
# ---------------------------------------------------------------- #


def test_text_only_turn_stays_message_codex(as_codex_oauth):
    msg = Msg(
        role="assistant",
        name="Friday",
        content=[
            {"type": "text", "text": "Hello, how can I help?"},
        ],
    )
    out = agentscope_msg_to_message(msg)
    assert _types(out) == [MessageType.MESSAGE]


def test_text_then_tool_use_promotes_text_to_reasoning(as_codex_oauth):
    """The exact pattern that leaked in production: gpt-5.5 emits
    a scratch-style preamble (``"Need view_video maybe..."``)
    immediately followed by the tool call.  The text must surface
    as REASONING so channels drop it, while Console UI still
    receives the SSE event upstream."""
    msg = Msg(
        role="assistant",
        name="Friday",
        content=[
            {"type": "text", "text": "Need view_video maybe returns note?"},
            {
                "type": "tool_use",
                "id": "call_1",
                "name": "view_video",
                "input": {"video_path": "/tmp/v.mp4"},
            },
        ],
    )
    out = agentscope_msg_to_message(msg)
    # Two events: REASONING (the preamble) and PLUGIN_CALL (the tool).
    assert _types(out) == [MessageType.REASONING, MessageType.PLUGIN_CALL]
    # Preamble content is preserved verbatim — Console UI needs the
    # original text for its rendering.
    assert out[0].content[0].text == "Need view_video maybe returns note?"


def test_text_after_tool_use_stays_message(as_codex_oauth):
    """Text *after* a tool_use in the same Msg is final-reply text
    (the model wrapping up after the tool result), so it keeps
    MESSAGE and reaches the channel."""
    msg = Msg(
        role="assistant",
        name="Friday",
        content=[
            {"type": "text", "text": "Calling the tool..."},
            {
                "type": "tool_use",
                "id": "call_1",
                "name": "view_video",
                "input": {},
            },
            {"type": "text", "text": "Here is the answer."},
        ],
    )
    out = agentscope_msg_to_message(msg)
    # Three events: REASONING (preamble), PLUGIN_CALL, MESSAGE (final).
    assert _types(out) == [
        MessageType.REASONING,
        MessageType.PLUGIN_CALL,
        MessageType.MESSAGE,
    ]
    assert out[2].content[0].text == "Here is the answer."


def test_string_content_stays_message(as_codex_oauth):
    """The plain-string fast path doesn't go through the tool-use
    pre-scan and must keep its existing ``MESSAGE`` classification.
    Otherwise every plain string reply from a non-tool-using model
    would silently disappear from channels."""
    msg = Msg(role="assistant", name="Friday", content="just a reply")
    out = agentscope_msg_to_message(msg)
    assert _types(out) == [MessageType.MESSAGE]


def test_thinking_block_still_reasoning(as_codex_oauth):
    """``thinking`` blocks (Claude's ``<think>``) keep their existing
    REASONING classification — the tool-use heuristic doesn't
    touch this branch."""
    msg = Msg(
        role="assistant",
        name="Friday",
        content=[
            {"type": "thinking", "thinking": "Let me think..."},
        ],
    )
    out = agentscope_msg_to_message(msg)
    assert _types(out) == [MessageType.REASONING]


# ---------------------------------------------------------------- #
# Non-Codex providers — promotion MUST NOT fire                    #
# ---------------------------------------------------------------- #


def test_claude_preamble_kept_as_message(as_other_provider):
    """Claude's polite "I'll check that" before a tool_use is the
    intentional UX users want.  Promotion must be off for non-
    codex-oauth providers."""
    msg = Msg(
        role="assistant",
        name="Friday",
        content=[
            {"type": "text", "text": "I'll fetch the video for you."},
            {
                "type": "tool_use",
                "id": "call_1",
                "name": "view_video",
                "input": {},
            },
        ],
    )
    out = agentscope_msg_to_message(msg)
    # Preamble stays as MESSAGE — channels send it through.
    assert _types(out) == [MessageType.MESSAGE, MessageType.PLUGIN_CALL]
    assert out[0].content[0].text == "I'll fetch the video for you."


def test_qwen_preamble_kept_as_message(as_other_provider):
    """Same for Qwen-family providers — preamble reaches the channel."""
    msg = Msg(
        role="assistant",
        name="Friday",
        content=[
            {"type": "text", "text": "好，我搵下條 video。"},
            {
                "type": "tool_use",
                "id": "call_1",
                "name": "view_video",
                "input": {},
            },
        ],
    )
    out = agentscope_msg_to_message(msg)
    assert out[0].type == MessageType.MESSAGE
