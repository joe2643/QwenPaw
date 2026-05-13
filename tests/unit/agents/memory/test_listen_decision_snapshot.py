# -*- coding: utf-8 -*-
"""Tests for the v2.1 snapshot-based decision path.

The decision step now mirrors the action step's session-snapshot
pattern: a transient ReActAgent runs with the main agent's
``sys_prompt`` + ``name`` and a snapshot of the chat's persisted
memory.  Persona / past exchanges arrive through memory instead of a
text-rendered prompt slot, which was emitting ``agent_name="Default"``
when ``load_agent_config`` returned the fallback.

These tests verify the snapshot branch directly, without going through
the bare-agent fallback that legacy callers still hit.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from qwenpaw.agents.memory.listen import listen_responder
from qwenpaw.agents.memory.listen.listen_types import ListenConfig


class _FakeSessionService:
    def __init__(self, state: dict | None = None):
        self._state = state or {}

    async def get_session_state_dict(self, session_id, user_id, channel):
        return self._state


def _make_cfg(**overrides) -> ListenConfig:
    cfg = ListenConfig(
        enabled=True,
        interval_minutes=5,
        channel_name="whatsapp",
        chat_id="12345@g.us",
        chat_meta={"chat_jid": "12345@g.us"},
        agent_id="default",
        verbosity="normal",
        session_id="whatsapp:group:12345@g.us",
        user_id="group:12345@g.us",
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


async def test_ask_with_workspace_uses_main_agent_persona(monkeypatch):
    """When ``workspace.agent`` exists, the decision sub-agent must be
    built with main_agent.name and main_agent.sys_prompt — NOT with the
    load_agent_config fallback that gave us 'Default'."""

    main_agent = SimpleNamespace(
        name="夏慶",  # 夏慶 (Yukei) – verifying CJK passes through
        sys_prompt=(
            "You are 夏慶, a peer in this chat. You speak with warmth."
        ),
    )
    workspace = SimpleNamespace(
        agent=main_agent,
        runner=SimpleNamespace(session=_FakeSessionService()),
    )

    captured: dict[str, Any] = {}

    class _CapturingReActAgent:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        async def reply(self, msg):
            class _Resp:
                def get_text_content(self):
                    return "PASS"

            return _Resp()

    monkeypatch.setattr(
        listen_responder,
        "ReActAgent",
        _CapturingReActAgent,
    )
    # Avoid real model creation.
    monkeypatch.setattr(
        listen_responder,
        "create_model_and_formatter",
        lambda agent_id: (object(), object()),
    )

    raw = await listen_responder._ask_llm_to_chime_in(
        history_text="[alice]: hi",
        config=_make_cfg(),
        workspace=workspace,
    )
    assert raw == "PASS"
    # Main agent's name and sys_prompt reached the sub-agent.
    assert captured["name"] == "夏慶"
    assert "夏慶" in captured["sys_prompt"]
    # No tools on the decision agent.
    from agentscope.tool import Toolkit

    assert isinstance(captured["toolkit"], Toolkit)
    # Memory came from the snapshot (an InMemoryMemory, even if empty).
    from agentscope.memory import InMemoryMemory

    assert isinstance(captured["memory"], InMemoryMemory)
    assert captured["max_iters"] == 1


async def test_ask_without_workspace_uses_text_render_fallback(monkeypatch):
    """Tests that pass only history + config (no workspace kwarg) must
    still get a usable decision via the text-render fallback path.
    Critical because the legacy test fixtures hit this code path."""

    captured: dict[str, Any] = {}

    class _CapturingReActAgent:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        async def reply(self, msg):
            captured["user_msg_content"] = getattr(msg, "content", "")

            class _Resp:
                def get_text_content(self):
                    return "CHIME"

            return _Resp()

    monkeypatch.setattr(
        listen_responder,
        "ReActAgent",
        _CapturingReActAgent,
    )
    monkeypatch.setattr(
        listen_responder,
        "create_model_and_formatter",
        lambda agent_id: (object(), object()),
    )
    # load_agent_config will be called by the fallback; stub it.
    monkeypatch.setattr(
        listen_responder,
        "load_agent_config",
        lambda agent_id: SimpleNamespace(name="StubBot", language="en"),
    )

    raw = await listen_responder._ask_llm_to_chime_in(
        history_text="[alice]: lunch?",
        config=_make_cfg(),
        prior_conversation_text="[user]: hi\n[assistant]: hi there",
    )
    assert raw == "CHIME"
    # Fallback uses ListenDecider name + generic sys_prompt.
    assert captured["name"] == "ListenDecider"
    # No memory in fallback (decision step is bare).
    assert captured["memory"] is None
    # Persona block is synthesised into the user-turn content.
    user_content = captured.get("user_msg_content", "")
    assert "StubBot" in user_content
    assert "[user]: hi" in user_content
    assert "[+alice]: lunch?".lower() in user_content.lower() or \
        "[alice]: lunch?" in user_content


async def test_ask_with_workspace_but_no_main_agent_falls_back_gracefully(
    monkeypatch,
):
    """When ``workspace.agent`` is None (e.g. workspace under construction),
    the snapshot path uses a minimal persona-only sys_prompt so the
    sub-agent still has identity context."""

    workspace = SimpleNamespace(
        agent=None,
        runner=SimpleNamespace(session=_FakeSessionService()),
    )

    captured: dict[str, Any] = {}

    class _CapturingReActAgent:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        async def reply(self, msg):
            class _Resp:
                def get_text_content(self):
                    return "PASS"

            return _Resp()

    monkeypatch.setattr(
        listen_responder,
        "ReActAgent",
        _CapturingReActAgent,
    )
    monkeypatch.setattr(
        listen_responder,
        "create_model_and_formatter",
        lambda agent_id: (object(), object()),
    )

    raw = await listen_responder._ask_llm_to_chime_in(
        history_text="[alice]: hi",
        config=_make_cfg(),
        workspace=workspace,
    )
    assert raw == "PASS"
    assert captured["name"] == "Assistant"
    # Falls back to a minimal sys_prompt rather than empty string.
    assert "peer in a group chat" in captured["sys_prompt"].lower()


pytestmark = pytest.mark.asyncio
