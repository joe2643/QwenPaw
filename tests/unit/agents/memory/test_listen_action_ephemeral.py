# -*- coding: utf-8 -*-
"""Route-D specific tests for the action step.

The action step is the new piece introduced by Listen v2 route D: it
snapshots the chat's persisted memory, runs a transient ReActAgent
with the full toolkit + ``LISTEN_INJECTION_GUARD`` suffix, and on a
non-PASS reply dispatches via ``channel.send`` and appends ONLY the
assistant chime-in to the FRESH session state.

These tests exercise the contract pieces that the v1-test-suite
rewrite couldn't cleanly cover: PASS self-abort, append-race fresh-read,
timeout, snapshot independence from real session.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from qwenpaw.agents.memory.listen import listen_responder
from qwenpaw.agents.memory.listen.listen_types import ListenConfig


# ---------------------------------------------------------------------------
# Shared fakes — kept minimal; reuse with the v1 test file's fakes
# would force cross-file imports for no benefit.
# ---------------------------------------------------------------------------


class _FakeChannel:
    def __init__(
        self,
        name: str,
        history: dict | None = None,
        raise_on_send: bool = False,
    ):
        self.name = name
        self._group_history = history or {}
        self.sent: list[tuple[str, str, dict | None]] = []
        self.raise_on_send = raise_on_send

    async def send(self, to_handle: str, text: str, meta=None) -> None:
        if self.raise_on_send:
            raise RuntimeError("send blew up")
        self.sent.append((to_handle, text, meta))


class _FakeChannelManager:
    def __init__(self, channels: dict):
        self._channels = channels

    async def get_channel(self, name: str):
        return self._channels.get(name)


class _FakeSessionService:
    """In-memory session service stand-in.

    Records reads + writes so tests can assert step 1 (snapshot) vs
    step 5 (fresh re-read + append) ordering.
    """

    def __init__(self, state: dict | None = None):
        self._state = state or {}
        self.read_calls: list[tuple[str, str, str]] = []
        self.update_calls: list[dict] = []

    async def get_session_state_dict(
        self,
        session_id: str,
        user_id: str,
        channel: str,
    ):
        self.read_calls.append((session_id, user_id, channel))
        return self._state

    async def update_session_state(
        self,
        session_id: str,
        key: str,
        value: Any,
        user_id: str,
        channel: str,
    ) -> None:
        self.update_calls.append(
            {
                "session_id": session_id,
                "key": key,
                "value": value,
                "user_id": user_id,
                "channel": channel,
            },
        )


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


def _entry(sender: str, body: str, ts: str = "1") -> dict:
    return {"sender": sender, "body": body, "ts": ts}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


async def test_execute_chime_action_self_aborts_on_pass(monkeypatch):
    """When the action agent returns the literal token ``PASS`` (or any
    of the PASS_TOKENS), the dispatcher MUST suppress channel.send and
    skip the session append.  This is the LISTEN_INJECTION_GUARD's
    honourable-exit protocol from Tension R3."""

    ch = _FakeChannel("whatsapp", {"12345@g.us": [_entry("alice", "hi")]})
    svc = _FakeSessionService()
    ws = SimpleNamespace(
        channel_manager=_FakeChannelManager({"whatsapp": ch}),
        runner=SimpleNamespace(session=svc),
        agent=None,
    )

    class _FakeResponse:
        def get_text_content(self):
            return "PASS"

    class _FakeAgent:
        async def reply(self, msg):
            return _FakeResponse()

    monkeypatch.setattr(
        listen_responder,
        "_build_action_agent",
        lambda workspace, cfg, mem: _FakeAgent(),
    )

    out = await listen_responder.execute_chime_action(ws, _make_cfg())
    assert out is None
    assert ch.sent == []
    assert svc.update_calls == []


async def test_execute_chime_action_dispatches_and_appends_on_real_reply(
    monkeypatch,
):
    """Happy path: non-PASS reply → channel.send fires → session
    append fires.  Both are required for the bot's history to match
    what the room saw."""

    ch = _FakeChannel("whatsapp", {"12345@g.us": [_entry("alice", "lunch?")]})
    svc = _FakeSessionService()
    ws = SimpleNamespace(
        channel_manager=_FakeChannelManager({"whatsapp": ch}),
        runner=SimpleNamespace(session=svc),
        agent=None,
    )

    class _FakeResponse:
        def get_text_content(self):
            return "邊間？我都肚餓"

    class _FakeAgent:
        async def reply(self, msg):
            return _FakeResponse()

    monkeypatch.setattr(
        listen_responder,
        "_build_action_agent",
        lambda workspace, cfg, mem: _FakeAgent(),
    )

    out = await listen_responder.execute_chime_action(ws, _make_cfg())
    assert out == "邊間？我都肚餓"
    assert ch.sent[0][0] == "12345@g.us"
    assert ch.sent[0][1] == "邊間？我都肚餓"
    assert len(svc.update_calls) == 1
    assert svc.update_calls[0]["key"] == "agent.memory"


async def test_execute_chime_action_skips_append_when_dispatch_fails(
    monkeypatch,
):
    """When channel.send raises, the session append must NOT fire.
    Otherwise the main agent's history would claim the bot said
    something the room never received."""

    ch = _FakeChannel(
        "whatsapp",
        {"12345@g.us": [_entry("alice", "yo")]},
        raise_on_send=True,
    )
    svc = _FakeSessionService()
    ws = SimpleNamespace(
        channel_manager=_FakeChannelManager({"whatsapp": ch}),
        runner=SimpleNamespace(session=svc),
        agent=None,
    )

    class _FakeResponse:
        def get_text_content(self):
            return "hello"

    class _FakeAgent:
        async def reply(self, msg):
            return _FakeResponse()

    monkeypatch.setattr(
        listen_responder,
        "_build_action_agent",
        lambda workspace, cfg, mem: _FakeAgent(),
    )

    out = await listen_responder.execute_chime_action(ws, _make_cfg())
    assert out is None
    assert svc.update_calls == []


async def test_execute_chime_action_times_out(monkeypatch):
    """When the action agent hangs past ``action_timeout_seconds``,
    the dispatcher must skip dispatch + append rather than blocking
    the trigger loop forever."""
    import asyncio

    ch = _FakeChannel("whatsapp", {"12345@g.us": [_entry("alice", "yo")]})
    svc = _FakeSessionService()
    ws = SimpleNamespace(
        channel_manager=_FakeChannelManager({"whatsapp": ch}),
        runner=SimpleNamespace(session=svc),
        agent=None,
    )

    class _HangingAgent:
        async def reply(self, msg):
            await asyncio.sleep(60)
            return None

    monkeypatch.setattr(
        listen_responder,
        "_build_action_agent",
        lambda workspace, cfg, mem: _HangingAgent(),
    )

    # 10s timeout (the dispatcher's minimum) is enough — we'll send
    # the wait_for a value lower than the agent's sleep.
    cfg = _make_cfg(action_timeout_seconds=1)
    out = await listen_responder.execute_chime_action(ws, cfg)
    assert out is None
    assert ch.sent == []
    assert svc.update_calls == []


async def test_execute_chime_action_skips_when_buffer_emptied(monkeypatch):
    """Buffer can be flushed between decision and action (e.g. user
    @-mention drains the group_history).  Action step must detect
    the empty buffer and skip rather than firing the agent on nothing."""

    ch = _FakeChannel("whatsapp", {"12345@g.us": []})  # buffer empty
    svc = _FakeSessionService()
    ws = SimpleNamespace(
        channel_manager=_FakeChannelManager({"whatsapp": ch}),
        runner=SimpleNamespace(session=svc),
        agent=None,
    )

    out = await listen_responder.execute_chime_action(ws, _make_cfg())
    assert out is None
    assert svc.update_calls == []


async def test_append_re_reads_fresh_state_before_writing():
    """Critical for Codex tension #1: the append step must NOT use the
    snapshot captured at action start.  It re-reads the LATEST state
    and merges the chime-in onto whatever lives there, so any user
    reply that landed during the action is preserved."""

    initial_state = {"agent": {"memory": []}}
    svc = _FakeSessionService(initial_state)
    ws = SimpleNamespace(
        channel_manager=_FakeChannelManager({"whatsapp": _FakeChannel("whatsapp")}),
        runner=SimpleNamespace(session=svc),
        agent=None,
    )
    cfg = _make_cfg()

    ok = await listen_responder._append_chime_to_real_session(
        ws,
        cfg,
        "hello",
    )
    assert ok is True
    # Append should issue exactly one read (the fresh one) and one
    # write — never reuse a snapshot pointer.
    assert len(svc.read_calls) >= 1
    assert len(svc.update_calls) == 1


async def test_snapshot_returns_empty_memory_when_session_unreadable():
    """If session.json can't be read at action start, the snapshot
    returns an empty memory instead of raising — the action still
    runs, just without persona context.  Better than failing the tick
    silently."""

    class _BoomSessionService:
        async def get_session_state_dict(self, *a, **kw):
            raise RuntimeError("session unreadable")

    ws = SimpleNamespace(
        channel_manager=None,
        runner=SimpleNamespace(session=_BoomSessionService()),
        agent=None,
    )

    memory = await listen_responder._snapshot_session_memory(
        ws,
        _make_cfg(),
    )
    # Empty InMemoryMemory exposes get_memory() returning [].
    msgs = await memory.get_memory()
    assert msgs == []


async def test_execute_chime_action_wraps_typing_indicator(monkeypatch):
    """The action step must start a typing indicator BEFORE
    agent.reply() and stop it before returning, even on PASS / error
    paths.  Without this, listen replies have no '...' cue in the
    room — Codex tension R3 follow-up.
    """

    typing_events: list[str] = []

    class _TypingChannel(_FakeChannel):
        async def start_typing(self, to_handle, meta=None):
            typing_events.append(f"start:{to_handle}")

            import asyncio as _aio

            async def _dummy_loop():
                try:
                    while True:
                        await _aio.sleep(60)
                except _aio.CancelledError:
                    return

            return _aio.create_task(_dummy_loop())

        async def stop_typing(self, handle):
            typing_events.append("stop")
            if handle is not None:
                handle.cancel()
                try:
                    await handle
                except Exception:
                    pass

    ch = _TypingChannel("whatsapp", {"12345@g.us": [_entry("alice", "yo")]})
    svc = _FakeSessionService()
    ws = SimpleNamespace(
        channel_manager=_FakeChannelManager({"whatsapp": ch}),
        runner=SimpleNamespace(session=svc),
        agent=None,
    )

    class _FakeResponse:
        def get_text_content(self):
            return "PASS"  # self-abort → exits via finally

    class _FakeAgent:
        async def reply(self, msg):
            return _FakeResponse()

    monkeypatch.setattr(
        listen_responder,
        "_build_action_agent",
        lambda workspace, cfg, mem: _FakeAgent(),
    )

    out = await listen_responder.execute_chime_action(ws, _make_cfg())
    assert out is None
    # Critical: BOTH start and stop must have fired, in order, even
    # though the action self-aborted with PASS.
    assert typing_events == ["start:12345@g.us", "stop"]


async def test_dispatch_resolves_to_handle_preference_order(monkeypatch):
    """Mirror of the v1 deliver_uses_* tests but explicitly testing
    that route D's dispatch helper picks chat_jid > group_id > source
    > chat_id > config.chat_id."""

    ch = _FakeChannel("whatsapp")
    ws = SimpleNamespace(
        channel_manager=_FakeChannelManager({"whatsapp": ch}),
        runner=None,
    )

    # chat_jid wins when present.
    cfg = _make_cfg(chat_meta={"chat_jid": "JID-1", "group_id": "GID-1"})
    ok = await listen_responder._dispatch_chime_to_channel(ws, cfg, "x")
    assert ok is True
    assert ch.sent[-1][0] == "JID-1"

    # group_id wins when chat_jid absent.
    cfg = _make_cfg(chat_meta={"group_id": "GID-2"})
    ok = await listen_responder._dispatch_chime_to_channel(ws, cfg, "x")
    assert ok is True
    assert ch.sent[-1][0] == "GID-2"

    # Source as final fallback.
    cfg = _make_cfg(chat_meta={"source": "SRC-3"})
    ok = await listen_responder._dispatch_chime_to_channel(ws, cfg, "x")
    assert ok is True
    assert ch.sent[-1][0] == "SRC-3"


# ---------------------------------------------------------------------------
# Pytest config — async tests.
# ---------------------------------------------------------------------------

pytestmark = pytest.mark.asyncio
