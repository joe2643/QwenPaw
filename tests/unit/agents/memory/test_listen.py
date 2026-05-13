# -*- coding: utf-8 -*-
"""Tests for the per-chat /listen mode.

Listen mode is a per-(channel, chat) timer that fires every N minutes,
hands the channel's ``_group_history`` buffer to a small LLM call, and
delivers any non-PASS reply back to the same chat.

These tests cover:

* The buffer formatter (truncation, malformed entries, multi-line bodies).
* The PASS-token detector (various LLM "stay silent" outputs).
* enable / disable lifecycle: config storage, idempotency, cancellation.
* generate_listen_reply: empty buffer, unchanged buffer, PASS, real reply.
* deliver_listen_reply: to_handle selection, channel.send call, error swallow.
* _extract_chat_target for the command handler's chat-key derivation.

The tests run with a stub LLM call — we monkeypatch
``_ask_llm_to_chime_in`` so no real model is invoked.
"""
from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any

import pytest

from qwenpaw.agents.memory.listen import listen_responder, listen_trigger
from qwenpaw.agents.memory.listen.listen_responder import (
    _format_buffer,
    _is_pass_response,
    deliver_listen_reply,
    generate_listen_reply,
)
from qwenpaw.agents.memory.listen.listen_trigger import (
    _extract_chat_target,
    disable_listen_for_chat,
    enable_listen_for_chat,
    listen_configs,
    listen_key,
    listen_tasks,
)
from qwenpaw.agents.memory.listen.listen_types import ListenConfig


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_listen_state():
    """Listen state is module-level; reset between tests."""
    for t in list(listen_tasks.values()):
        try:
            t.cancel()
        except Exception:
            pass
    listen_tasks.clear()
    listen_configs.clear()
    yield
    for t in list(listen_tasks.values()):
        try:
            t.cancel()
        except Exception:
            pass
    listen_tasks.clear()
    listen_configs.clear()


class _FakeChannel:
    def __init__(
        self,
        name: str = "whatsapp",
        group_history: dict[str, list[dict[str, Any]]] | None = None,
    ):
        self.channel = name
        self._group_history = group_history if group_history is not None else {}
        self.sent: list[tuple[str, str, dict[str, Any] | None]] = []
        self.raise_on_send = False

    async def send(
        self,
        to_handle: str,
        text: str,
        meta: dict[str, Any] | None = None,
    ) -> None:
        if self.raise_on_send:
            raise RuntimeError("channel down")
        self.sent.append((to_handle, text, meta))


class _FakeChannelManager:
    def __init__(self, channels: dict[str, Any]):
        self._channels = channels

    async def get_channel(self, name: str) -> Any:
        return self._channels.get(name)


def _workspace(channels: dict[str, Any] | None) -> Any:
    return SimpleNamespace(channel_manager=_FakeChannelManager(channels or {}))


def _entry(sender: str, body: str, ts: str = "1") -> dict[str, Any]:
    return {"sender": sender, "body": body, "ts": ts, "media": []}


# ---------------------------------------------------------------------------
# A. _format_buffer + _is_pass_response
# ---------------------------------------------------------------------------


def test_format_buffer_renders_sender_body_lines():
    buffer = [
        _entry("+Alice", "lunch?"),
        _entry("+Bob", "yeah I'm down"),
    ]
    out = _format_buffer(buffer)
    assert out.splitlines() == [
        "[+Alice]: lunch?",
        "[+Bob]: yeah I'm down",
    ]


def test_format_buffer_collapses_multiline_body():
    out = _format_buffer([_entry("+1", "line1\nline2\nline3")])
    body_line = out.splitlines()[0]
    assert "line1 line2 line3" in body_line
    assert "\n" not in body_line


def test_format_buffer_drops_malformed_entries():
    buffer = ["not-a-dict", _entry("+1", "hi"), 42]  # type: ignore[list-item]
    out = _format_buffer(buffer)
    assert out.splitlines() == ["[+1]: hi"]


def test_format_buffer_truncates_max_entries():
    """Only the last N entries should render, oldest dropped."""
    from qwenpaw.agents.memory.listen.listen_responder import (
        _LISTEN_MAX_ENTRIES,
    )

    over = _LISTEN_MAX_ENTRIES + 5
    buffer = [_entry(f"+{i}", f"msg-{i}") for i in range(over)]
    out = _format_buffer(buffer)
    lines = out.splitlines()
    assert len(lines) == _LISTEN_MAX_ENTRIES
    assert lines[0] != "[+0]: msg-0"  # earliest must be dropped
    assert lines[-1] == f"[+{over - 1}]: msg-{over - 1}"


def test_format_buffer_respects_char_cap():
    """Bodies are clamped to 400 chars each + total cap enforced."""
    long_body = "x" * 500  # will be truncated to 400 by formatter
    buffer = [_entry("+1", long_body), _entry("+2", "short")]
    out = _format_buffer(buffer)
    # The 400-char clamp applied per-entry; long line should be 400+overhead.
    first_line = out.splitlines()[0]
    assert len(first_line) < 500


@pytest.mark.parametrize(
    "text",
    [
        "",
        " ",
        "PASS",
        "pass",
        "pass.",
        "(PASS)",
        "[pass]",
        "skip",
        "No reply",
        "  PASS  ",
        "`PASS`",
        '"pass"',
    ],
)
def test_is_pass_response_recognizes_silence_tokens(text: str):
    assert _is_pass_response(text) is True


@pytest.mark.parametrize(
    "text",
    [
        "lol same",
        "我都係咁諗",
        "PASS the salt",  # contains PASS as a word; must still send
        "Yeah, I think so too.",
    ],
)
def test_is_pass_response_lets_real_replies_through(text: str):
    assert _is_pass_response(text) is False


# ---------------------------------------------------------------------------
# B. _extract_chat_target
# ---------------------------------------------------------------------------


def test_extract_chat_target_whatsapp_group():
    channel, chat_id = _extract_chat_target(
        {"platform": "whatsapp", "chat_jid": "12345@g.us"},
    )
    assert channel == "whatsapp"
    assert chat_id == "12345@g.us"


def test_extract_chat_target_signal_group():
    channel, chat_id = _extract_chat_target(
        {"platform": "signal", "group_id": "AbCdEf="},
    )
    assert channel == "signal"
    assert chat_id == "AbCdEf="


def test_extract_chat_target_falls_back_to_channel_key():
    channel, chat_id = _extract_chat_target(
        {"channel": "telegram", "chat_id": "tg-1"},
    )
    assert channel == "telegram"
    assert chat_id == "tg-1"


def test_extract_chat_target_empty_when_missing():
    assert _extract_chat_target(None) == ("", "")
    assert _extract_chat_target({}) == ("", "")
    # Channel without chat_id falls all the way through — no fallback
    # session_id either, so we get the empty signal.
    assert _extract_chat_target({"platform": "whatsapp"}) == ("", "")


# ---------------------------------------------------------------------------
# B.1. session_id fallback for console-UI-invoked /listen
# ---------------------------------------------------------------------------


def test_extract_chat_target_falls_back_to_session_id_whatsapp_group():
    """Console UI sitting on a WhatsApp group session POSTs /listen
    without channel_meta; the session_id alone must be enough."""
    channel, chat_id = _extract_chat_target(
        None,
        session_id="whatsapp:group:120363421135228220@g.us",
    )
    assert channel == "whatsapp"
    assert chat_id == "120363421135228220@g.us"


def test_extract_chat_target_falls_back_to_session_id_whatsapp_dm():
    channel, chat_id = _extract_chat_target(
        {},
        session_id="whatsapp:85251159218@s.whatsapp.net",
    )
    assert channel == "whatsapp"
    assert chat_id == "85251159218@s.whatsapp.net"


def test_extract_chat_target_falls_back_to_session_id_signal_group():
    channel, chat_id = _extract_chat_target(
        None,
        session_id="signal:group:AbCdEf=",
    )
    assert channel == "signal"
    assert chat_id == "AbCdEf="


def test_extract_chat_target_rejects_console_session_id():
    """``proactive_mode:default`` and console-only session ids must NOT
    be silently accepted as listen targets — listen has no buffer to
    read on those channels."""
    assert _extract_chat_target(None, "proactive_mode:default") == ("", "")
    assert _extract_chat_target(None, "console:some-id") == ("", "")
    assert _extract_chat_target(None, "default") == ("", "")


def test_extract_chat_target_prefers_channel_meta_over_session_id():
    """When both sources are present and disagree, real channel_meta
    wins — it's the more authoritative reflection of where the request
    actually originated."""
    channel, chat_id = _extract_chat_target(
        {"platform": "whatsapp", "chat_jid": "real@g.us"},
        session_id="signal:group:imposter=",
    )
    assert channel == "whatsapp"
    assert chat_id == "real@g.us"


# ---------------------------------------------------------------------------
# C. enable / disable lifecycle
# ---------------------------------------------------------------------------


async def test_enable_creates_config_and_task():
    result = enable_listen_for_chat(
        "whatsapp",
        "12345@g.us",
        interval_minutes=5,
        chat_meta={"chat_jid": "12345@g.us", "platform": "whatsapp"},
        agent_id="default",
    )
    key = listen_key("whatsapp", "12345@g.us")
    assert key in result
    assert key in listen_configs
    cfg = listen_configs[key]
    assert cfg.enabled is True
    assert cfg.interval_minutes == 5
    assert cfg.channel_name == "whatsapp"
    assert cfg.chat_id == "12345@g.us"
    assert cfg.agent_id == "default"
    assert cfg.chat_meta == {"chat_jid": "12345@g.us", "platform": "whatsapp"}
    assert key in listen_tasks


async def test_enable_is_idempotent_on_repeat_call():
    enable_listen_for_chat(
        "whatsapp",
        "12345@g.us",
        interval_minutes=5,
        chat_meta={"chat_jid": "12345@g.us"},
        agent_id="default",
    )
    first_task = listen_tasks[listen_key("whatsapp", "12345@g.us")]

    enable_listen_for_chat(
        "whatsapp",
        "12345@g.us",
        interval_minutes=10,
        chat_meta={"chat_jid": "12345@g.us"},
        agent_id="default",
    )
    cfg = listen_configs[listen_key("whatsapp", "12345@g.us")]
    assert cfg.interval_minutes == 10
    # Same task — no duplicate spawned.
    assert listen_tasks[listen_key("whatsapp", "12345@g.us")] is first_task


async def test_enable_rejects_zero_interval():
    with pytest.raises(ValueError):
        enable_listen_for_chat(
            "whatsapp",
            "12345@g.us",
            interval_minutes=0,
            chat_meta={},
        )


async def test_enable_rejects_missing_chat_id():
    with pytest.raises(ValueError):
        enable_listen_for_chat("whatsapp", "", interval_minutes=5, chat_meta={})


async def test_disable_cancels_task_and_removes_config():
    enable_listen_for_chat(
        "whatsapp",
        "12345@g.us",
        interval_minutes=5,
        chat_meta={"chat_jid": "12345@g.us"},
        agent_id="default",
    )
    key = listen_key("whatsapp", "12345@g.us")
    task = listen_tasks[key]

    out = disable_listen_for_chat("whatsapp", "12345@g.us")
    assert "disabled" in out.lower()
    assert key not in listen_configs
    assert key not in listen_tasks
    # Yield control once so the cancellation propagates.
    import asyncio

    await asyncio.sleep(0)
    assert task.cancelled() or task.done()


async def test_disable_when_inactive_is_idempotent():
    out = disable_listen_for_chat("whatsapp", "never-enabled")
    assert "not active" in out.lower()


# ---------------------------------------------------------------------------
# D. generate_listen_reply
# ---------------------------------------------------------------------------


def _make_cfg(**overrides: Any) -> ListenConfig:
    base = dict(
        enabled=True,
        interval_minutes=5,
        channel_name="whatsapp",
        chat_id="12345@g.us",
        chat_meta={"chat_jid": "12345@g.us", "platform": "whatsapp"},
        agent_id="default",
    )
    base.update(overrides)
    return ListenConfig(**base)


async def test_generate_listen_reply_returns_none_when_buffer_empty(
    monkeypatch,
    caplog,
    _propagating_listen_logger,
):
    import logging as _logging

    cfg = _make_cfg()
    ws = _workspace({"whatsapp": _FakeChannel("whatsapp", {})})

    called: list[str] = []

    async def fake_ask(history_text, config, prior_conversation_text="", **kwargs):
        called.append(history_text)
        return "should never be called"

    monkeypatch.setattr(listen_responder, "_ask_llm_to_chime_in", fake_ask)

    with caplog.at_level(_logging.INFO):
        out = await generate_listen_reply(ws, cfg)
    assert out is None
    assert called == []  # buffer empty ⇒ skip LLM
    # Empty-buffer skip must be visible in INFO logs so ops can tell
    # the difference between "task isn't firing" and "task fires but
    # has nothing to react to".
    assert "buffer empty" in caplog.text


async def test_generate_listen_reply_skips_when_buffer_unchanged(monkeypatch):
    cfg = _make_cfg()
    cfg.last_seen_ts = "1000"
    ch = _FakeChannel(
        "whatsapp",
        {"12345@g.us": [_entry("+1", "hi", ts="1000")]},
    )
    ws = _workspace({"whatsapp": ch})

    async def fake_ask(history_text, config, prior_conversation_text="", **kwargs):
        return "this should not run"

    monkeypatch.setattr(listen_responder, "_ask_llm_to_chime_in", fake_ask)

    out = await generate_listen_reply(ws, cfg)
    assert out is None


async def test_generate_listen_reply_returns_chime_when_llm_says_chime(monkeypatch):
    """Route D: decision step returns binary CHIME/PASS; the v1
    ``generate_listen_reply`` shim now returns the literal ``"CHIME"``
    or ``None``.  The actual chime text is generated separately by
    ``execute_chime_action`` (covered elsewhere)."""
    cfg = _make_cfg()
    ch = _FakeChannel(
        "whatsapp",
        {"12345@g.us": [_entry("+Alice", "lunch?", ts="1001")]},
    )
    ws = _workspace({"whatsapp": ch})

    async def fake_ask(history_text, config, prior_conversation_text="", **kwargs):
        assert "[+Alice]: lunch?" in history_text
        return "CHIME"

    monkeypatch.setattr(listen_responder, "_ask_llm_to_chime_in", fake_ask)

    out = await generate_listen_reply(ws, cfg)
    assert out == "CHIME"
    # Tracker updated so the next tick won't re-ask on the same content.
    assert cfg.last_seen_ts == "1001"


async def test_generate_listen_reply_returns_none_on_pass(monkeypatch):
    cfg = _make_cfg()
    ch = _FakeChannel(
        "whatsapp",
        {"12345@g.us": [_entry("+Alice", "anyway, dispute time", ts="2000")]},
    )
    ws = _workspace({"whatsapp": ch})

    async def fake_ask(history_text, config, prior_conversation_text="", **kwargs):
        return "PASS"

    monkeypatch.setattr(listen_responder, "_ask_llm_to_chime_in", fake_ask)
    out = await generate_listen_reply(ws, cfg)
    assert out is None
    # last_seen_ts STILL bumped — otherwise the next tick would re-ask on the
    # same buffer and waste tokens.
    assert cfg.last_seen_ts == "2000"


async def test_generate_listen_reply_swallows_llm_error(monkeypatch):
    cfg = _make_cfg()
    ch = _FakeChannel(
        "whatsapp",
        {"12345@g.us": [_entry("+Alice", "hi", ts="3000")]},
    )
    ws = _workspace({"whatsapp": ch})

    async def fake_ask(history_text, config, prior_conversation_text="", **kwargs):
        raise RuntimeError("LLM down")

    monkeypatch.setattr(listen_responder, "_ask_llm_to_chime_in", fake_ask)
    out = await generate_listen_reply(ws, cfg)
    assert out is None  # error swallowed, loop continues


async def test_generate_listen_reply_returns_none_when_channel_missing(monkeypatch):
    cfg = _make_cfg()
    ws = _workspace({})  # no whatsapp channel registered

    async def fake_ask(history_text, config, prior_conversation_text="", **kwargs):
        return "ignored"

    monkeypatch.setattr(listen_responder, "_ask_llm_to_chime_in", fake_ask)
    assert await generate_listen_reply(ws, cfg) is None


# ---------------------------------------------------------------------------
# E. Channel dispatch — route D moves dispatch INSIDE execute_chime_action;
# the v1 ``deliver_listen_reply`` is a no-op shim now.  These tests
# target the internal ``_dispatch_chime_to_channel`` which is what the
# action step actually calls.
# ---------------------------------------------------------------------------


async def test_dispatch_uses_chat_jid_from_meta():
    from qwenpaw.agents.memory.listen.listen_responder import (
        _dispatch_chime_to_channel,
    )

    cfg = _make_cfg(chat_meta={"chat_jid": "12345@g.us", "platform": "whatsapp"})
    ch = _FakeChannel("whatsapp")
    ws = _workspace({"whatsapp": ch})

    ok = await _dispatch_chime_to_channel(ws, cfg, "邊間？")
    assert ok is True
    assert ch.sent == [
        (
            "12345@g.us",
            "邊間？",
            {"chat_jid": "12345@g.us", "platform": "whatsapp"},
        ),
    ]


async def test_dispatch_uses_group_id_for_signal():
    from qwenpaw.agents.memory.listen.listen_responder import (
        _dispatch_chime_to_channel,
    )

    cfg = _make_cfg(
        channel_name="signal",
        chat_id="AbCdEf=",
        chat_meta={"group_id": "AbCdEf=", "platform": "signal"},
    )
    ch = _FakeChannel("signal")
    ws = _workspace({"signal": ch})

    ok = await _dispatch_chime_to_channel(ws, cfg, "hi")
    assert ok is True
    assert ch.sent[0][0] == "AbCdEf="


async def test_dispatch_falls_back_to_chat_id_when_meta_empty():
    from qwenpaw.agents.memory.listen.listen_responder import (
        _dispatch_chime_to_channel,
    )

    cfg = _make_cfg(chat_meta={})
    ch = _FakeChannel("whatsapp")
    ws = _workspace({"whatsapp": ch})

    ok = await _dispatch_chime_to_channel(ws, cfg, "x")
    assert ok is True
    assert ch.sent[0][0] == "12345@g.us"


async def test_dispatch_returns_false_when_channel_missing():
    from qwenpaw.agents.memory.listen.listen_responder import (
        _dispatch_chime_to_channel,
    )

    cfg = _make_cfg()
    ws = _workspace({})
    assert await _dispatch_chime_to_channel(ws, cfg, "x") is False


async def test_dispatch_swallows_send_errors():
    from qwenpaw.agents.memory.listen.listen_responder import (
        _dispatch_chime_to_channel,
    )

    cfg = _make_cfg()
    ch = _FakeChannel("whatsapp")
    ch.raise_on_send = True
    ws = _workspace({"whatsapp": ch})

    assert await _dispatch_chime_to_channel(ws, cfg, "x") is False


async def test_dispatch_returns_false_when_no_to_handle():
    from qwenpaw.agents.memory.listen.listen_responder import (
        _dispatch_chime_to_channel,
    )

    cfg = _make_cfg(chat_id="", chat_meta={})
    ws = _workspace({"whatsapp": _FakeChannel("whatsapp")})
    assert await _dispatch_chime_to_channel(ws, cfg, "x") is False


async def test_deliver_listen_reply_v1_shim_returns_false():
    """The v1 ``deliver_listen_reply`` is intentionally a no-op so any
    in-flight code path that still imports it doesn't accidentally
    bypass the route-D dispatch-inside-action-step contract."""
    cfg = _make_cfg()
    ws = _workspace({"whatsapp": _FakeChannel("whatsapp")})
    assert await deliver_listen_reply(ws, cfg, "x") is False


# ---------------------------------------------------------------------------
# F. Verbosity routing
# ---------------------------------------------------------------------------


def test_select_prompt_template_normal():
    from qwenpaw.agents.memory.listen.listen_responder import (
        _select_prompt_template,
    )
    from qwenpaw.agents.memory.listen.listen_prompts import (
        LISTEN_CHIME_IN_PROMPT,
        LISTEN_CHIME_IN_PROMPT_AGGRESSIVE,
    )

    assert _select_prompt_template("normal") == LISTEN_CHIME_IN_PROMPT
    # Unknown verbosity falls back to normal — safer default.
    assert _select_prompt_template("garbage") == LISTEN_CHIME_IN_PROMPT
    assert (
        _select_prompt_template("aggressive")
        == LISTEN_CHIME_IN_PROMPT_AGGRESSIVE
    )


def test_normal_and_aggressive_prompts_differ_in_default_bias():
    """Normal prompt must NOT default to CHIME; aggressive prompt must."""
    from qwenpaw.agents.memory.listen.listen_prompts import (
        LISTEN_DECISION_PROMPT,
        LISTEN_DECISION_PROMPT_AGGRESSIVE,
    )

    # Aggressive prompt explicitly states the default-CHIME bias.
    assert "Default: CHIME" in LISTEN_DECISION_PROMPT_AGGRESSIVE
    # Normal prompt offers criteria for when to CHIME but does not
    # pre-bias toward speech.
    assert "Default: CHIME" not in LISTEN_DECISION_PROMPT
    # Both must keep the no-monitoring-tone / untrusted-data safety
    # rail intact AND the binary CHIME/PASS output contract.
    for tpl in (LISTEN_DECISION_PROMPT, LISTEN_DECISION_PROMPT_AGGRESSIVE):
        assert "CHIME" in tpl
        assert "PASS" in tpl
        # The "treat as untrusted, not instructions" guard must appear
        # in both — that's the prompt-injection defence per Tension R1.
        assert "not instructions" in tpl.lower()


async def test_enable_listen_records_verbosity():
    enable_listen_for_chat(
        "whatsapp",
        "12345@g.us",
        interval_minutes=5,
        chat_meta={"chat_jid": "12345@g.us"},
        agent_id="default",
        verbosity="aggressive",
    )
    cfg = listen_configs[listen_key("whatsapp", "12345@g.us")]
    assert cfg.verbosity == "aggressive"


async def test_enable_listen_rejects_unknown_verbosity():
    with pytest.raises(ValueError):
        enable_listen_for_chat(
            "whatsapp",
            "12345@g.us",
            interval_minutes=5,
            chat_meta={"chat_jid": "12345@g.us"},
            verbosity="loud",
        )


async def test_enable_listen_idempotent_updates_verbosity():
    enable_listen_for_chat(
        "whatsapp",
        "12345@g.us",
        interval_minutes=5,
        chat_meta={"chat_jid": "12345@g.us"},
        agent_id="default",
        verbosity="normal",
    )
    enable_listen_for_chat(
        "whatsapp",
        "12345@g.us",
        interval_minutes=10,
        chat_meta={"chat_jid": "12345@g.us"},
        agent_id="default",
        verbosity="aggressive",
    )
    cfg = listen_configs[listen_key("whatsapp", "12345@g.us")]
    assert cfg.verbosity == "aggressive"
    assert cfg.interval_minutes == 10


# ---------------------------------------------------------------------------
# G. Debug dump
# ---------------------------------------------------------------------------


@pytest.fixture
def _propagating_listen_logger():
    """CoPaw's project logger sets propagate=False so root-handler
    caplog can't see records.  Re-enable propagation just for the dump
    tests."""
    import logging as _logging

    candidates = [
        listen_responder.__name__,
        "qwenpaw",
        "copaw",
    ]
    previous: dict[str, bool] = {}
    for name in candidates:
        lg = _logging.getLogger(name)
        previous[name] = lg.propagate
        lg.propagate = True
    yield
    for name, prev in previous.items():
        _logging.getLogger(name).propagate = prev


def test_dump_off_by_default(monkeypatch, caplog, _propagating_listen_logger):
    import logging as _logging

    monkeypatch.delenv("COPAW_LISTEN_DUMP", raising=False)
    cfg = _make_cfg()
    with caplog.at_level(_logging.INFO):
        listen_responder._maybe_dump_listen_prompt(cfg, "prompt-x", "raw-y")
    assert "listen_dump" not in caplog.text


def test_dump_on_logs_prompt_and_raw(
    monkeypatch,
    caplog,
    _propagating_listen_logger,
):
    import logging as _logging

    monkeypatch.setenv("COPAW_LISTEN_DUMP", "1")
    cfg = _make_cfg(verbosity="aggressive")
    with caplog.at_level(_logging.INFO):
        listen_responder._maybe_dump_listen_prompt(
            cfg,
            "MY-PROMPT-BODY",
            "MY-RAW-RESPONSE",
        )
    assert "listen_dump" in caplog.text
    assert "verbosity=aggressive" in caplog.text
    assert "MY-PROMPT-BODY" in caplog.text
    assert "MY-RAW-RESPONSE" in caplog.text


def test_dump_truncates_long_payloads(
    monkeypatch,
    caplog,
    _propagating_listen_logger,
):
    import logging as _logging

    monkeypatch.setenv("COPAW_LISTEN_DUMP", "1")
    cfg = _make_cfg()
    with caplog.at_level(_logging.INFO):
        listen_responder._maybe_dump_listen_prompt(
            cfg,
            "x" * 5000,
            "y" * 5000,
        )
    assert "[truncated]" in caplog.text


# ---------------------------------------------------------------------------
# H. Session-history injection (persisted agent memory of past
#    @-mention exchanges)
# ---------------------------------------------------------------------------


class _FakeMsg:
    """Minimal stand-in for agentscope.Msg with the attrs the loader reads."""

    def __init__(self, role: str, text: str | None):
        self.role = role
        self.content = (
            [{"type": "text", "text": text}] if text is not None else []
        )


class _FakeSessionService:
    """Stand-in for ``workspace.runner.session``.

    Stores a single canned state dict (or raises).  Captures the
    arguments the loader passes so the test can assert routing.
    """

    def __init__(
        self,
        state: dict | None = None,
        raise_on_get: bool = False,
    ):
        self._state = state
        self.raise_on_get = raise_on_get
        self.last_call: dict[str, str] = {}

    async def get_session_state_dict(
        self,
        session_id: str,
        user_id: str,
        channel: str,
    ) -> dict | None:
        self.last_call = {
            "session_id": session_id,
            "user_id": user_id,
            "channel": channel,
        }
        if self.raise_on_get:
            raise RuntimeError("session service down")
        return self._state


def _ws_with_session(state: dict | None, **session_kwargs) -> Any:
    """Build a workspace that exposes the right runner/session shape."""
    svc = _FakeSessionService(state, **session_kwargs)
    runner = SimpleNamespace(session=svc)
    cm = _FakeChannelManager({"whatsapp": _FakeChannel("whatsapp")})
    return SimpleNamespace(channel_manager=cm, runner=runner), svc


async def test_load_session_history_renders_user_assistant_text(monkeypatch):
    """Memory entries with text blocks render as ``[role]: text`` lines
    in oldest-first order so the prompt is naturally readable."""
    state = {"agent": {"memory": "OPAQUE-STATE-DOESNT-MATTER"}}
    ws, svc = _ws_with_session(state)

    def fake_load_state_dict(self, mem_state, strict=False):
        self._messages = [
            _FakeMsg("user", "remember when we talked about pizza?"),
            _FakeMsg("assistant", "yeah, the new place on main street"),
            _FakeMsg("system", "internal tool note — should NOT appear"),
        ]

    async def fake_get_memory(self):
        return getattr(self, "_messages", [])

    from agentscope.memory import InMemoryMemory

    monkeypatch.setattr(InMemoryMemory, "load_state_dict", fake_load_state_dict)
    monkeypatch.setattr(InMemoryMemory, "get_memory", fake_get_memory)

    cfg = _make_cfg(session_id="whatsapp:group:12345@g.us")
    out = await listen_responder._load_session_history(ws, cfg)

    lines = out.splitlines()
    assert "[user]: remember when we talked about pizza?" in lines
    assert "[assistant]: yeah, the new place on main street" in lines
    # System / non-text entries are dropped.
    assert all("internal tool note" not in ln for ln in lines)
    # Routed correctly to the captured chat.
    assert svc.last_call["session_id"] == "whatsapp:group:12345@g.us"
    assert svc.last_call["channel"] == "whatsapp"


async def test_load_session_history_defaults_user_id_to_session_id():
    """When config doesn't capture an explicit user_id, fall back to
    session_id (WhatsApp/Signal persist state that way)."""
    ws, svc = _ws_with_session({"agent": {"memory": {}}})

    cfg = _make_cfg(
        session_id="whatsapp:group:12345@g.us",
        user_id="",
    )
    await listen_responder._load_session_history(ws, cfg)
    assert svc.last_call["user_id"] == "whatsapp:group:12345@g.us"


async def test_load_session_history_uses_explicit_user_id_when_provided():
    ws, svc = _ws_with_session({"agent": {"memory": {}}})

    cfg = _make_cfg(
        session_id="whatsapp:85251159218@s.whatsapp.net",
        user_id="+85251159218",
    )
    await listen_responder._load_session_history(ws, cfg)
    assert svc.last_call["user_id"] == "+85251159218"


async def test_load_session_history_returns_empty_when_no_session_id():
    ws, _svc = _ws_with_session({"agent": {"memory": {}}})
    cfg = _make_cfg(session_id="")
    out = await listen_responder._load_session_history(ws, cfg)
    assert out == ""


async def test_load_session_history_returns_empty_when_no_runner():
    """A workspace without a runner / session service must NOT crash."""
    ws = SimpleNamespace(
        channel_manager=_FakeChannelManager({}),
        runner=None,
    )
    cfg = _make_cfg(session_id="whatsapp:group:12345@g.us")
    assert await listen_responder._load_session_history(ws, cfg) == ""


async def test_load_session_history_swallows_session_errors():
    ws, _svc = _ws_with_session(
        {"agent": {"memory": {}}},
        raise_on_get=True,
    )
    cfg = _make_cfg(session_id="whatsapp:group:12345@g.us")
    out = await listen_responder._load_session_history(ws, cfg)
    assert out == ""


async def test_load_session_history_respects_char_cap(monkeypatch):
    """When session memory is huge, only the most-recent entries that
    fit under ``_LISTEN_PRIOR_MAX_CHARS`` are rendered — oldest get
    dropped first, but the kept subset is emitted in oldest-first order."""
    state = {"agent": {"memory": "STATE"}}
    ws, _svc = _ws_with_session(state)

    long_body = "x" * 300
    fake_messages = [
        _FakeMsg("user", f"msg-{i}: {long_body}") for i in range(30)
    ]

    def fake_load_state_dict(self, mem_state, strict=False):
        self._messages = fake_messages

    async def fake_get_memory(self):
        return getattr(self, "_messages", [])

    from agentscope.memory import InMemoryMemory

    monkeypatch.setattr(
        InMemoryMemory,
        "load_state_dict",
        fake_load_state_dict,
    )
    monkeypatch.setattr(InMemoryMemory, "get_memory", fake_get_memory)

    cfg = _make_cfg(session_id="whatsapp:group:12345@g.us")
    out = await listen_responder._load_session_history(ws, cfg)
    # Total length stays under the cap (plus a little overhead for line
    # joins / role prefix).
    assert len(out) <= listen_responder._LISTEN_PRIOR_MAX_CHARS + 200
    # The earliest message must be dropped, the latest must be kept.
    assert "msg-0:" not in out
    assert "msg-29:" in out


async def test_generate_listen_reply_passes_prior_conversation_to_llm(
    monkeypatch,
):
    """v2.1: the decision step receives a ``workspace`` reference so it
    can build the snapshot-memory sub-agent.  prior_conversation_text
    is no longer populated by the caller — persona / past exchanges
    arrive through snapshot memory instead.  We assert here that the
    workspace kwarg reaches the LLM call and the chatter buffer is
    rendered correctly into history_text."""
    state = {"agent": {"memory": "STATE"}}
    svc = _FakeSessionService(state)
    runner = SimpleNamespace(session=svc)
    ch = _FakeChannel(
        "whatsapp",
        {"12345@g.us": [_entry("+Alice", "anyone up for lunch?", ts="999")]},
    )
    ws = SimpleNamespace(
        channel_manager=_FakeChannelManager({"whatsapp": ch}),
        runner=runner,
    )

    def fake_load_state_dict(self, mem_state, strict=False):
        self._messages = [
            _FakeMsg("user", "what's a good pho place?"),
            _FakeMsg("assistant", "Pho Hoa, two blocks east."),
        ]

    async def fake_get_memory(self):
        return getattr(self, "_messages", [])

    from agentscope.memory import InMemoryMemory

    monkeypatch.setattr(
        InMemoryMemory,
        "load_state_dict",
        fake_load_state_dict,
    )
    monkeypatch.setattr(InMemoryMemory, "get_memory", fake_get_memory)

    captured: dict[str, str] = {}

    async def fake_ask(
        history_text,
        config,
        prior_conversation_text="",
        **kwargs,
    ):
        captured["history"] = history_text
        captured["prior"] = prior_conversation_text
        captured["workspace_passed"] = "workspace" in kwargs
        captured["workspace"] = kwargs.get("workspace")
        return "CHIME"

    monkeypatch.setattr(listen_responder, "_ask_llm_to_chime_in", fake_ask)

    cfg = _make_cfg(session_id="whatsapp:group:12345@g.us")
    out = await listen_responder.generate_listen_reply(ws, cfg)

    # Route D: ``generate_listen_reply`` is a decision-only shim that
    # returns the binary ``"CHIME"`` / ``None`` token.
    assert out == "CHIME"
    assert "[+Alice]: anyone up for lunch?" in captured["history"]
    # v2.1: workspace handle reaches the LLM call so the snapshot
    # decision path can build a sub-agent with the chat's real memory.
    assert captured["workspace_passed"] is True
    assert captured["workspace"] is ws


async def test_generate_listen_reply_works_without_session_memory(monkeypatch):
    """Missing / empty session memory must not stop the chime-in path —
    prior-conversation slot just renders empty."""
    svc = _FakeSessionService(None)  # session service returns None state
    runner = SimpleNamespace(session=svc)
    ch = _FakeChannel(
        "whatsapp",
        {"12345@g.us": [_entry("+Alice", "lunch?", ts="999")]},
    )
    ws = SimpleNamespace(
        channel_manager=_FakeChannelManager({"whatsapp": ch}),
        runner=runner,
    )

    captured: dict[str, str] = {}

    async def fake_ask(
        history_text,
        config,
        prior_conversation_text="",
        **kwargs,
    ):
        captured["prior"] = prior_conversation_text
        return "CHIME"

    monkeypatch.setattr(listen_responder, "_ask_llm_to_chime_in", fake_ask)

    cfg = _make_cfg(session_id="whatsapp:group:12345@g.us")
    out = await listen_responder.generate_listen_reply(ws, cfg)
    # Route D: binary decision token return.
    assert out == "CHIME"
    assert captured["prior"] == ""


async def test_enable_listen_records_session_id_and_user_id():
    enable_listen_for_chat(
        "whatsapp",
        "12345@g.us",
        interval_minutes=5,
        chat_meta={"chat_jid": "12345@g.us"},
        agent_id="default",
        session_id="whatsapp:group:12345@g.us",
        user_id="+85251159218",
    )
    cfg = listen_configs[listen_key("whatsapp", "12345@g.us")]
    assert cfg.session_id == "whatsapp:group:12345@g.us"
    assert cfg.user_id == "+85251159218"


# ---------------------------------------------------------------------------
# I. _append_listen_reply_to_session
# ---------------------------------------------------------------------------


class _RecordingSessionService(_FakeSessionService):
    """Records ``update_session_state`` payloads for assertions."""

    def __init__(self, state: dict | None = None, **kw):
        super().__init__(state=state, **kw)
        self.updates: list[dict] = []
        self.raise_on_update = False

    async def update_session_state(
        self,
        *,
        session_id: str,
        key: str,
        value: Any,
        user_id: str,
        channel: str,
    ) -> None:
        if self.raise_on_update:
            raise RuntimeError("session write down")
        self.updates.append(
            {
                "session_id": session_id,
                "key": key,
                "value": value,
                "user_id": user_id,
                "channel": channel,
            },
        )


def _ws_with_recording_session(
    state: dict | None,
    **session_kwargs,
) -> tuple[Any, _RecordingSessionService]:
    svc = _RecordingSessionService(state, **session_kwargs)
    runner = SimpleNamespace(session=svc)
    cm = _FakeChannelManager({"whatsapp": _FakeChannel("whatsapp")})
    return SimpleNamespace(channel_manager=cm, runner=runner), svc


async def test_append_writes_tagged_assistant_msg_to_session(monkeypatch):
    ws, svc = _ws_with_recording_session({"agent": {"memory": {}}})

    # Stub InMemoryMemory so we don't depend on agentscope internals.
    captured_messages: list[Any] = []
    saved_state: dict[str, Any] = {}

    class _StubMemory:
        def __init__(self):
            self._msgs: list[Any] = []

        def load_state_dict(self, state, strict: bool = False) -> None:
            self._msgs.extend(state.get("msgs", []))

        async def add(self, msg) -> None:
            self._msgs.append(msg)
            captured_messages.append(msg)

        def state_dict(self) -> dict:
            saved_state["msgs"] = list(self._msgs)
            return {"msgs": list(self._msgs)}

    monkeypatch.setattr(
        "agentscope.memory.InMemoryMemory",
        _StubMemory,
    )

    cfg = _make_cfg(
        session_id="whatsapp:group:12345@g.us",
        user_id="group:12345@g.us",
    )
    ok = await listen_responder._append_listen_reply_to_session(
        ws,
        cfg,
        "邊間？",
    )

    assert ok is True
    # Tagged msg landed in memory.
    assert len(captured_messages) == 1
    appended = captured_messages[0]
    assert appended.role == "assistant"
    assert appended.name == "listen"
    body = appended.content[0]["text"]
    assert body.startswith(
        listen_responder._LISTEN_REPLY_MEMORY_PREFIX,
    )
    assert body.endswith("邊間？")
    # Session was written under the right key + ids.
    assert len(svc.updates) == 1
    upd = svc.updates[0]
    assert upd["session_id"] == "whatsapp:group:12345@g.us"
    assert upd["user_id"] == "group:12345@g.us"
    assert upd["key"] == "agent.memory"
    assert upd["channel"] == "whatsapp"


async def test_append_swallows_session_read_errors():
    ws, svc = _ws_with_recording_session(None, raise_on_get=True)
    cfg = _make_cfg(session_id="whatsapp:group:12345@g.us")
    assert (
        await listen_responder._append_listen_reply_to_session(
            ws,
            cfg,
            "hi",
        )
        is False
    )
    assert svc.updates == []  # no write attempted


async def test_append_swallows_session_write_errors(monkeypatch):
    ws, svc = _ws_with_recording_session({"agent": {"memory": {}}})
    svc.raise_on_update = True

    class _StubMemory:
        async def add(self, msg):
            pass

        def load_state_dict(self, state, strict: bool = False):
            pass

        def state_dict(self):
            return {}

    monkeypatch.setattr("agentscope.memory.InMemoryMemory", _StubMemory)

    cfg = _make_cfg(session_id="whatsapp:group:12345@g.us")
    assert (
        await listen_responder._append_listen_reply_to_session(
            ws,
            cfg,
            "hi",
        )
        is False
    )


async def test_append_no_op_when_session_id_missing():
    ws, svc = _ws_with_recording_session({"agent": {"memory": {}}})
    cfg = _make_cfg(session_id="")
    assert (
        await listen_responder._append_listen_reply_to_session(ws, cfg, "x")
        is False
    )
    assert svc.updates == []


async def test_append_no_op_when_workspace_lacks_runner():
    ws = SimpleNamespace(channel_manager=_FakeChannelManager({}), runner=None)
    cfg = _make_cfg(session_id="whatsapp:group:12345@g.us")
    assert (
        await listen_responder._append_listen_reply_to_session(ws, cfg, "x")
        is False
    )


async def test_append_no_op_when_reply_text_empty():
    ws, svc = _ws_with_recording_session({"agent": {"memory": {}}})
    cfg = _make_cfg(session_id="whatsapp:group:12345@g.us")
    assert (
        await listen_responder._append_listen_reply_to_session(ws, cfg, "")
        is False
    )
    assert svc.updates == []


# ---------------------------------------------------------------------------
# J. _fire_once orchestration: append fires only after successful deliver
# ---------------------------------------------------------------------------


async def test_fire_once_runs_action_only_when_decision_chimes(monkeypatch):
    """Route D orchestration: decision → action.  When decision says
    PASS, the action step must NOT run (cost win + no token waste)."""
    from qwenpaw.agents.memory.listen import listen_trigger
    from qwenpaw.agents.memory.listen import listen_responder

    action_calls: list[str] = []

    async def fake_should(workspace, cfg):
        return False  # PASS

    async def fake_action(workspace, cfg):
        action_calls.append(cfg.chat_id)
        return "should-not-run"

    async def fake_is_chat_busy(workspace, chat_id, **kwargs):
        return False

    monkeypatch.setattr(listen_responder, "should_chime_in", fake_should)
    monkeypatch.setattr(listen_responder, "execute_chime_action", fake_action)
    monkeypatch.setattr(
        "qwenpaw.agents.memory.proactive.proactive_utils.is_chat_busy",
        fake_is_chat_busy,
    )

    cfg = _make_cfg(session_id="whatsapp:group:12345@g.us")
    await listen_trigger._fire_once(SimpleNamespace(), cfg)

    assert action_calls == []
    assert cfg.last_chime_ts is None


async def test_fire_once_records_last_chime_ts_when_action_dispatched(
    monkeypatch,
):
    """When the action step actually dispatches a chime, the trigger
    loop records ``last_chime_ts`` so ``min_chime_gap_seconds`` can
    throttle bursts on the next tick."""
    from qwenpaw.agents.memory.listen import listen_trigger
    from qwenpaw.agents.memory.listen import listen_responder

    async def fake_should(workspace, cfg):
        return True

    async def fake_action(workspace, cfg):
        return "hello"  # non-PASS, treated as dispatched

    async def fake_is_chat_busy(workspace, chat_id, **kwargs):
        return False

    monkeypatch.setattr(listen_responder, "should_chime_in", fake_should)
    monkeypatch.setattr(listen_responder, "execute_chime_action", fake_action)
    monkeypatch.setattr(
        "qwenpaw.agents.memory.proactive.proactive_utils.is_chat_busy",
        fake_is_chat_busy,
    )

    cfg = _make_cfg(session_id="whatsapp:group:12345@g.us")
    assert cfg.last_chime_ts is None
    await listen_trigger._fire_once(SimpleNamespace(), cfg)
    assert cfg.last_chime_ts is not None


async def test_fire_once_skips_chime_ts_when_action_self_aborts(monkeypatch):
    """When the action agent returns PASS (or empty), ``last_chime_ts``
    must stay unset — otherwise the throttle would suppress future
    ticks even though we never actually chimed in."""
    from qwenpaw.agents.memory.listen import listen_trigger
    from qwenpaw.agents.memory.listen import listen_responder

    async def fake_should(workspace, cfg):
        return True

    async def fake_action(workspace, cfg):
        return None  # action self-aborted

    async def fake_is_chat_busy(workspace, chat_id, **kwargs):
        return False

    monkeypatch.setattr(listen_responder, "should_chime_in", fake_should)
    monkeypatch.setattr(listen_responder, "execute_chime_action", fake_action)
    monkeypatch.setattr(
        "qwenpaw.agents.memory.proactive.proactive_utils.is_chat_busy",
        fake_is_chat_busy,
    )

    cfg = _make_cfg(session_id="whatsapp:group:12345@g.us")
    await listen_trigger._fire_once(SimpleNamespace(), cfg)
    assert cfg.last_chime_ts is None


async def test_fire_once_respects_per_chat_busy_gate(monkeypatch):
    """When ``is_chat_busy`` reports the chat has an active user task,
    neither the decision nor action step should fire."""
    from qwenpaw.agents.memory.listen import listen_trigger
    from qwenpaw.agents.memory.listen import listen_responder

    calls: list[str] = []

    async def fake_should(workspace, cfg):
        calls.append("should")
        return True

    async def fake_action(workspace, cfg):
        calls.append("action")
        return "hello"

    async def fake_is_chat_busy(workspace, chat_id, **kwargs):
        return True  # chat is busy

    monkeypatch.setattr(listen_responder, "should_chime_in", fake_should)
    monkeypatch.setattr(listen_responder, "execute_chime_action", fake_action)
    monkeypatch.setattr(
        "qwenpaw.agents.memory.proactive.proactive_utils.is_chat_busy",
        fake_is_chat_busy,
    )

    cfg = _make_cfg(session_id="whatsapp:group:12345@g.us")
    await listen_trigger._fire_once(SimpleNamespace(), cfg)
    assert calls == []
    assert cfg.last_chime_ts is None


async def test_fire_once_respects_min_chime_gap(monkeypatch):
    """When the previous chime was inside ``min_chime_gap_seconds``,
    the throttle must short-circuit before the decision LLM call."""
    from datetime import datetime, timedelta, timezone
    from qwenpaw.agents.memory.listen import listen_trigger
    from qwenpaw.agents.memory.listen import listen_responder

    calls: list[str] = []

    async def fake_should(workspace, cfg):
        calls.append("should")
        return True

    async def fake_action(workspace, cfg):
        calls.append("action")
        return "hello"

    async def fake_is_chat_busy(workspace, chat_id, **kwargs):
        return False

    monkeypatch.setattr(listen_responder, "should_chime_in", fake_should)
    monkeypatch.setattr(listen_responder, "execute_chime_action", fake_action)
    monkeypatch.setattr(
        "qwenpaw.agents.memory.proactive.proactive_utils.is_chat_busy",
        fake_is_chat_busy,
    )

    cfg = _make_cfg(session_id="whatsapp:group:12345@g.us")
    # 60 seconds ago — well inside the default 300s gap.
    cfg.last_chime_ts = datetime.now(timezone.utc) - timedelta(seconds=60)
    cfg.min_chime_gap_seconds = 300

    await listen_trigger._fire_once(SimpleNamespace(), cfg)
    assert calls == []


# ---------------------------------------------------------------------------
# K. /listen command captures correct user_id for groups vs DMs
# ---------------------------------------------------------------------------


def test_prompt_only_takes_history_slot_in_v2_1():
    """v2.1: decision prompts dropped the {agent_name} / {prior_conversation}
    / {channel_name} / {language} slots because persona / past
    exchanges now live in the sub-agent's sys_prompt + snapshot memory.
    Only {history} remains."""
    from qwenpaw.agents.memory.listen.listen_prompts import (
        LISTEN_CHIME_IN_PROMPT,
    )

    rendered = LISTEN_CHIME_IN_PROMPT.format(history="[+Alice]: hi")
    assert "[+Alice]: hi" in rendered
    # Sanity: no unrendered slots left.
    assert "{" not in rendered or "{history}" not in rendered
