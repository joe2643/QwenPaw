# -*- coding: utf-8 -*-
"""Regression: run_command_path must publish request ContextVars.

Before this fix, ``run_command_path`` skipped the
``set_current_agent_id`` / ``set_current_session_id`` /
``set_current_channel_meta`` calls that ``AgentRunner.stream_query``
runs on the non-command path.  Slash commands therefore saw stale
(or ``None``) ContextVars and ``/listen`` could not snapshot the
originating WhatsApp group, surfacing the user-facing "per-chat
command — use it from inside a group/DM" rejection in real groups.
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from agentscope.message import Msg

from qwenpaw.app.agent_context import (
    get_current_agent_id,
    get_current_channel_meta,
    get_current_session_id,
)


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class _StubAgentContext:
    """Minimal context_manager.get_agent_context() target.

    ``run_command_path`` calls ``load_state_dict`` on this object, but
    the conversation command handler never reads back from it for the
    cases we care about.  Keep it dumb.
    """

    def load_state_dict(self, state, strict: bool = False) -> None:
        pass

    def state_dict(self) -> dict[str, Any]:
        return {}


class _StubContextManager:
    def __init__(self):
        self._ctx = _StubAgentContext()

    def get_agent_context(self):
        return self._ctx


class _StubSession:
    async def get_session_state_dict(self, *, session_id, user_id, channel):
        return {}

    async def update_session_state(self, **kwargs):
        pass


class _StubRunner:
    agent_name = "default"
    agent_id = "default"

    def __init__(self):
        self.session = _StubSession()
        self.memory_manager = SimpleNamespace(agent_id="default")
        self.context_manager = _StubContextManager()
        # _is_control_command checks ``runner._workspace`` only for control
        # commands; our test query is /listen (a conversation command).
        self._workspace = None
        self._manager = None


class _CapturingHandler:
    """Stand-in CommandHandler that records ContextVars at call time."""

    def __init__(self, *args, **kwargs):
        pass

    async def handle_conversation_command(self, query: str) -> Msg:
        recorded["agent_id"] = get_current_agent_id()
        recorded["session_id"] = get_current_session_id()
        recorded["channel_meta"] = get_current_channel_meta()
        return Msg(name="default", role="assistant", content=[])


recorded: dict[str, Any] = {}


@pytest.fixture(autouse=True)
def _reset_recorded():
    recorded.clear()
    yield
    recorded.clear()


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


async def test_run_command_path_publishes_request_contextvars(monkeypatch):
    """A WhatsApp slash command must reach CommandHandler with the
    request's channel_meta visible via ContextVars."""
    from qwenpaw.app.runner import command_dispatch

    monkeypatch.setattr(
        command_dispatch,
        "CommandHandler",
        _CapturingHandler,
    )

    request = SimpleNamespace(
        session_id="sess-wa-1",
        user_id="+85251159218",
        channel="whatsapp",
        channel_meta={
            "platform": "whatsapp",
            "chat_jid": "12345@g.us",
            "sender_phone": "+85251159218",
            "is_group": True,
        },
    )
    user_msg = Msg(name="user", role="user", content="/listen on")

    out = []
    async for item in command_dispatch.run_command_path(
        request,
        [user_msg],
        _StubRunner(),
    ):
        out.append(item)

    assert out, "Expected the dispatcher to yield a response message"
    assert recorded["agent_id"] == "default"
    assert recorded["session_id"] == "sess-wa-1"
    meta = recorded["channel_meta"]
    assert isinstance(meta, dict)
    assert meta["platform"] == "whatsapp"
    assert meta["chat_jid"] == "12345@g.us"
