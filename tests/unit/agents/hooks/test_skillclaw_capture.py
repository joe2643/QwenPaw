# -*- coding: utf-8 -*-
"""Unit tests for the SkillClaw capture hook.

Validates two guarantees:
1. The on-disk schema matches what ``skillclaw/api_server.py`` writes
   (one JSONL record per turn: ``{session_id, turn, timestamp, messages}``)
2. Typed ``Msg`` content (text / tool_use / tool_result / thinking /
   image) flattens to a non-lossy textual representation that the
   evolve pipeline can reason over.
"""
from __future__ import annotations

import json
import pytest

from qwenpaw.agents.hooks.skillclaw_capture import (
    SkillClawCaptureHook,
    _msg_to_openai_dict,
)


class _FakeMemory:
    def __init__(self, messages):
        self._messages = messages

    async def get_memory(self):
        return self._messages


class _FakeAgent:
    def __init__(self, messages):
        self.memory = _FakeMemory(messages)


class _FakeMsg:
    """Minimal agentscope Msg stand-in — just role + content."""

    def __init__(self, role, content):
        self.role = role
        self.content = content


@pytest.mark.asyncio
async def test_writes_one_record_per_invocation(tmp_path):
    hook = SkillClawCaptureHook(
        records_dir=tmp_path,
        session_id="test-session",
    )
    agent = _FakeAgent(
        messages=[
            _FakeMsg("system", "You are helpful."),
            _FakeMsg("user", "Hi"),
        ],
    )
    await hook(agent, {})
    await hook(agent, {})

    records_file = tmp_path / "conversations.jsonl"
    assert records_file.exists()
    lines = records_file.read_text().strip().splitlines()
    assert len(lines) == 2

    r1 = json.loads(lines[0])
    r2 = json.loads(lines[1])
    assert r1["session_id"] == "test-session"
    assert r1["turn"] == 1
    assert r2["turn"] == 2
    # Schema contract the evolve_server summarizer relies on
    assert set(r1.keys()) == {"session_id", "turn", "timestamp", "messages"}
    assert r1["messages"] == [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hi"},
    ]


@pytest.mark.asyncio
async def test_session_id_prefix_applied(tmp_path):
    hook = SkillClawCaptureHook(
        records_dir=tmp_path,
        session_id="sig-12345",
        session_id_prefix="copaw-default--",
    )
    await hook(_FakeAgent(messages=[_FakeMsg("user", "hi")]), {})
    line = (tmp_path / "conversations.jsonl").read_text().strip()
    assert json.loads(line)["session_id"] == "copaw-default--sig-12345"


def test_msg_to_openai_dict_str_content():
    m = _FakeMsg("user", "Hello")
    assert _msg_to_openai_dict(m) == {"role": "user", "content": "Hello"}


def test_msg_to_openai_dict_mixed_block_list():
    # Simulates an assistant turn with reasoning + a tool call + text.
    m = _FakeMsg(
        "assistant",
        [
            {"type": "thinking", "thinking": "let me think"},
            {"type": "text", "text": "I'll search first."},
            {
                "type": "tool_use",
                "name": "web_search",
                "input": {"q": "weather"},
            },
        ],
    )
    out = _msg_to_openai_dict(m)
    assert out["role"] == "assistant"
    # Order preserved, tool_call serialised readably
    assert "[thinking: let me think]" in out["content"]
    assert "I'll search first." in out["content"]
    assert '[tool_call: web_search({"q": "weather"})]' in out["content"]


def test_msg_to_openai_dict_tool_result_nested_blocks():
    m = _FakeMsg(
        "tool",
        [
            {
                "type": "tool_result",
                "output": [
                    {"type": "text", "text": "sunny, 24C"},
                ],
            },
        ],
    )
    out = _msg_to_openai_dict(m)
    assert out == {"role": "tool", "content": "[tool_result: sunny, 24C]"}


def test_msg_to_openai_dict_multimodal_placeholder():
    m = _FakeMsg(
        "user",
        [
            {"type": "text", "text": "What's this?"},
            {"type": "image", "source": {"type": "base64", "data": "..."}},
        ],
    )
    out = _msg_to_openai_dict(m)
    assert "What's this?" in out["content"]
    assert "[image: base64]" in out["content"]


@pytest.mark.asyncio
async def test_append_survives_extraction_error(tmp_path, caplog):
    """Hook must never break the agent loop, even if memory throws."""

    class _BrokenMemory:
        async def get_memory(self):
            raise RuntimeError("simulated memory failure")

    class _BrokenAgent:
        memory = _BrokenMemory()

    hook = SkillClawCaptureHook(
        records_dir=tmp_path,
        session_id="err-session",
    )
    # Should not raise
    result = await hook(_BrokenAgent(), {})
    assert result is None
    # No file written (error happened before append)
    assert not (tmp_path / "conversations.jsonl").exists()


# --------------------------------------------------------------------- #
# HTTP mode                                                              #
# --------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_http_mode_posts_to_ingest_endpoint(tmp_path, monkeypatch):
    """HTTP mode should POST the record body and skip file write
    when the server returns 2xx."""
    captured: dict = {}

    class _MockResponse:
        def __init__(self, status_code=200, text=""):
            self.status_code = status_code
            self.text = text

    class _MockClient:
        def __init__(self, *args, **kwargs):
            pass

        async def post(self, url, json=None, headers=None):
            captured["url"] = url
            captured["json"] = json
            captured["headers"] = headers
            return _MockResponse(status_code=200)

    monkeypatch.setattr(
        "qwenpaw.agents.hooks.skillclaw_capture.httpx.AsyncClient",
        _MockClient,
    )
    monkeypatch.setattr(
        "qwenpaw.agents.hooks.skillclaw_capture.httpx.Timeout",
        lambda *a, **k: None,
    )

    hook = SkillClawCaptureHook(
        records_dir=tmp_path,
        session_id="http-test",
        mode="http",
        ingest_url="http://localhost:8787/v1/sessions/ingest",
        ingest_api_key="secret",
    )
    await hook(_FakeAgent(messages=[_FakeMsg("user", "hi")]), {})

    assert (
        captured["url"]
        == "http://localhost:8787/v1/sessions/ingest"
    )
    assert captured["json"]["session_id"] == "http-test"
    assert captured["json"]["turn"] == 1
    assert captured["headers"]["Authorization"] == "Bearer secret"
    # File NOT written — http succeeded
    assert not (tmp_path / "conversations.jsonl").exists()


@pytest.mark.asyncio
async def test_http_mode_falls_back_to_file_on_5xx(tmp_path, monkeypatch):
    """If the ingest endpoint errors, the hook must fall back to
    file write so turns aren't silently dropped while SkillClaw is
    down."""

    class _MockResponse:
        def __init__(self):
            self.status_code = 503
            self.text = "service unavailable"

    class _MockClient:
        def __init__(self, *a, **k): pass

        async def post(self, url, json=None, headers=None):
            return _MockResponse()

    monkeypatch.setattr(
        "qwenpaw.agents.hooks.skillclaw_capture.httpx.AsyncClient",
        _MockClient,
    )
    monkeypatch.setattr(
        "qwenpaw.agents.hooks.skillclaw_capture.httpx.Timeout",
        lambda *a, **k: None,
    )

    hook = SkillClawCaptureHook(
        records_dir=tmp_path,
        session_id="fallback-test",
        mode="http",
        ingest_url="http://localhost:8787/v1/sessions/ingest",
    )
    await hook(_FakeAgent(messages=[_FakeMsg("user", "hi")]), {})

    # File DID get written — fallback path engaged
    line = (tmp_path / "conversations.jsonl").read_text().strip()
    rec = json.loads(line)
    assert rec["session_id"] == "fallback-test"
    assert rec["turn"] == 1


@pytest.mark.asyncio
async def test_http_mode_falls_back_on_connection_error(tmp_path, monkeypatch):
    """Network-level failure (DNS / refused) must also fall back."""
    import httpx

    class _MockClient:
        def __init__(self, *a, **k): pass

        async def post(self, url, json=None, headers=None):
            raise httpx.ConnectError("connection refused")

    monkeypatch.setattr(
        "qwenpaw.agents.hooks.skillclaw_capture.httpx.AsyncClient",
        _MockClient,
    )
    monkeypatch.setattr(
        "qwenpaw.agents.hooks.skillclaw_capture.httpx.Timeout",
        lambda *a, **k: None,
    )

    hook = SkillClawCaptureHook(
        records_dir=tmp_path,
        session_id="net-err-test",
        mode="http",
        ingest_url="http://nonexistent:1/v1/sessions/ingest",
    )
    await hook(_FakeAgent(messages=[_FakeMsg("user", "hi")]), {})

    line = (tmp_path / "conversations.jsonl").read_text().strip()
    assert json.loads(line)["session_id"] == "net-err-test"
