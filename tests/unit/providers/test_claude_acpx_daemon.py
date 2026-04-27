# -*- coding: utf-8 -*-
"""Unit tests for the claude-acpx daemon — Lane B.

Strategy: replace the ``cmd_builder`` with a tuple that points at a
small inline Python script so :func:`asyncio.create_subprocess_exec`
runs predictably without needing a real ``acpx`` binary on PATH or
network access.  The script prints synthetic ACP JSON-RPC lines on
stdout, optionally consumes JSON-RPC reply envelopes from stdin, and
exits.  This exercises both the request-router (Claude → us) and
the line-pass-through (notifications + final response) paths.
"""

from __future__ import annotations

import asyncio
import json
import sys
from typing import Any

import pytest

from qwenpaw.providers import claude_acpx_metrics
from qwenpaw.providers.claude_acpx_daemon import (
    AcpxDaemon,
    AcpxDaemonError,
    _payload_text,
)


# ----------------------------------------------------------------- #
# Helpers — fake "acpx" subprocess via inline python
# ----------------------------------------------------------------- #


def _python_cmd_builder(script: str) -> Any:
    """Return a ``cmd_builder`` callable for AcpxDaemon that runs the
    given inline Python script.  Mirrors the real
    :func:`acpx_translate.stateful_acpx_cmd` shape: globals go into
    the argv head, ``--ttl`` lives in the global slot, the daemon
    then appends prompt as a trailing positional.  argv layout:

      argv[0] = python (interpreter)
      argv[1] = "-c"
      argv[2] = script source
      argv[3] = session_name
      argv[4..5] = "--ttl", "<n>" (when ttl_seconds given)
      argv[-1] = prompt text
    """

    def builder(
        session_name: str,
        *,
        ttl_seconds: int | None = None,
        cwd: str | None = None,  # noqa: ARG001 — unused in tests but
        # accepted for signature compat with the real builder.
    ) -> tuple[str, ...]:
        args: list[str] = [sys.executable, "-c", script, session_name]
        if ttl_seconds is not None:
            args += ["--ttl", str(ttl_seconds)]
        return tuple(args)

    return builder


# Script: emits a session/update + final response and exits.  Sleeps
# briefly so the test can observe the streaming nature.
_BASIC_SCRIPT = r"""
import json, sys, time
sys.stdout.write(json.dumps({
    "jsonrpc": "2.0",
    "method": "session/update",
    "params": {
        "sessionId": "sess_x",
        "update": {
            "sessionUpdate": "agent_message_chunk",
            "content": {"type": "text", "text": "hello"},
        },
    },
}) + "\n")
sys.stdout.flush()
sys.stdout.write(json.dumps({
    "jsonrpc": "2.0",
    "id": "1",
    "result": {"stopReason": "end_turn"},
}) + "\n")
sys.stdout.flush()
"""


# Script: echoes argv-supplied prompt back through the JSON-RPC stream
# so the test can verify the prompt reached acpx via positional args.
_ECHO_PROMPT_SCRIPT = r"""
import sys, json
# argv[-1] is the prompt; daemon also appends --ttl <n> before that.
prompt = sys.argv[-1]
sys.stdout.write(json.dumps({
    "jsonrpc": "2.0",
    "method": "session/update",
    "params": {
        "sessionId": "sess_x",
        "update": {
            "sessionUpdate": "agent_message_chunk",
            "content": {"type": "text", "text": prompt},
        },
    },
}) + "\n")
sys.stdout.flush()
sys.stdout.write(json.dumps({
    "jsonrpc": "2.0", "id": "1",
    "result": {"stopReason": "end_turn"},
}) + "\n")
sys.stdout.flush()
"""


# Script: emits a Claude→client request, expects a JSON-RPC reply on
# stdin, then emits the final response.  Used to verify the daemon's
# bidirectional routing.  Prompt arrives via argv (matching the
# updated daemon contract); stdin is reserved for ACP reply traffic.
_REQUEST_SCRIPT = r"""
import json, sys
sys.stdout.write(json.dumps({
    "jsonrpc": "2.0",
    "id": "req_1",
    "method": "fs/read_text_file",
    "params": {"sessionId": "sess_x", "path": "/tmp/x"},
}) + "\n")
sys.stdout.flush()
# Read one reply line — this should be the daemon's JSON-RPC reply.
reply_line = sys.stdin.readline()
# Echo it as a session/update so the test can assert on it.
sys.stdout.write(json.dumps({
    "jsonrpc": "2.0",
    "method": "session/update",
    "params": {
        "sessionId": "sess_x",
        "update": {
            "sessionUpdate": "agent_message_chunk",
            "content": {"type": "text", "text": reply_line.strip()},
        },
    },
}) + "\n")
sys.stdout.flush()
sys.stdout.write(json.dumps({
    "jsonrpc": "2.0", "id": "1",
    "result": {"stopReason": "end_turn"},
}) + "\n")
sys.stdout.flush()
"""


_BAD_RC_SCRIPT = r"""
import sys
sys.stderr.write("boom\n")
sys.exit(7)
"""


_STALL_SCRIPT = r"""
import time
# Sleep longer than the test timeout.
time.sleep(60)
"""


# ----------------------------------------------------------------- #
# Fixtures
# ----------------------------------------------------------------- #


@pytest.fixture(autouse=True)
def _reset_metrics_and_singleton() -> None:
    """Each test starts with clean metrics + no daemon singleton."""
    claude_acpx_metrics.reset_for_test()
    AcpxDaemon.reset_singleton_for_test()
    yield
    AcpxDaemon.reset_singleton_for_test()


# ----------------------------------------------------------------- #
# _payload_text
# ----------------------------------------------------------------- #


class TestPayloadText:
    def test_text_blocks_joined(self) -> None:
        out = _payload_text([
            {"type": "text", "text": "hello"},
            {"type": "text", "text": "world"},
        ])
        assert out == "hello\n\nworld"

    def test_empty_blocks_yield_placeholder(self) -> None:
        # ACP doesn't accept zero-byte stdin; we replace with sentinel.
        assert _payload_text([]) == "(empty prompt)"
        assert _payload_text([{"type": "text", "text": ""}]) == "(empty prompt)"

    def test_image_block_collapses_to_placeholder(self) -> None:
        out = _payload_text([
            {"type": "text", "text": "describe"},
            {"type": "image", "mimeType": "image/png", "data": "AAAA"},
        ])
        # Image folds via _content_text → "[image attached]"
        assert "describe" in out
        assert "[image attached]" in out


# ----------------------------------------------------------------- #
# get_or_spawn — singleton
# ----------------------------------------------------------------- #


class TestSingleton:
    def test_get_or_spawn_returns_same_instance(self) -> None:
        a = AcpxDaemon.get_or_spawn()
        b = AcpxDaemon.get_or_spawn()
        assert a is b

    def test_reset_for_test_clears_singleton(self) -> None:
        a = AcpxDaemon.get_or_spawn()
        AcpxDaemon.reset_singleton_for_test()
        b = AcpxDaemon.get_or_spawn()
        assert a is not b


# ----------------------------------------------------------------- #
# submit_turn — basic streaming
# ----------------------------------------------------------------- #


class TestSubmitTurnBasic:
    @pytest.mark.asyncio
    async def test_yields_session_update_and_final(self) -> None:
        daemon = AcpxDaemon(
            auto_ensure_session=False,
            cmd_builder=_python_cmd_builder(_BASIC_SCRIPT),
        )
        lines = []
        async for raw in daemon.submit_turn(
            session_name="copaw-test",
            prompt_blocks=[{"type": "text", "text": "hi"}],
            is_seed=True,
        ):
            lines.append(raw.strip())
        # We expect both messages to come through unaltered.
        assert len(lines) == 2
        msg0 = json.loads(lines[0])
        assert msg0["method"] == "session/update"
        msg1 = json.loads(lines[1])
        assert msg1["result"]["stopReason"] == "end_turn"

    @pytest.mark.asyncio
    async def test_prompt_blocks_reach_subprocess_stdin(self) -> None:
        daemon = AcpxDaemon(
            auto_ensure_session=False,
            cmd_builder=_python_cmd_builder(_ECHO_PROMPT_SCRIPT),
        )
        lines = []
        async for raw in daemon.submit_turn(
            session_name="copaw-test",
            prompt_blocks=[{"type": "text", "text": "PING"}],
            is_seed=False,
        ):
            lines.append(raw.strip())
        # The first line is a session/update echoing what we sent.
        msg = json.loads(lines[0])
        text = msg["params"]["update"]["content"]["text"]
        assert "PING" in text

    @pytest.mark.asyncio
    async def test_empty_blocks_become_empty_prompt_sentinel(self) -> None:
        daemon = AcpxDaemon(
            auto_ensure_session=False,
            cmd_builder=_python_cmd_builder(_ECHO_PROMPT_SCRIPT),
        )
        lines = []
        async for raw in daemon.submit_turn(
            session_name="copaw-test",
            prompt_blocks=[],
            is_seed=True,
        ):
            lines.append(raw.strip())
        msg = json.loads(lines[0])
        assert "(empty prompt)" in msg["params"]["update"]["content"]["text"]


# ----------------------------------------------------------------- #
# submit_turn — request routing
# ----------------------------------------------------------------- #


class TestSubmitTurnRouting:
    @pytest.mark.asyncio
    async def test_request_dispatched_and_reply_written(self) -> None:
        captured: list[dict] = []

        async def handler(params: dict) -> dict:
            captured.append(params)
            return {"content": "FAKE_FILE_CONTENT"}

        daemon = AcpxDaemon(
            auto_ensure_session=False,
            cmd_builder=_python_cmd_builder(_REQUEST_SCRIPT),
        )
        daemon.set_handler("fs/read_text_file", handler)

        lines = []
        async for raw in daemon.submit_turn(
            session_name="copaw-test",
            prompt_blocks=[{"type": "text", "text": "hi"}],
            is_seed=True,
        ):
            lines.append(raw.strip())
        # Handler ran with the params from the request.
        assert captured and captured[0]["path"] == "/tmp/x"
        # The yielded lines exclude the Claude→client request itself.
        # We should see exactly one session/update (the echo) and the
        # final response.
        assert len(lines) == 2
        echo = json.loads(lines[0])
        echo_text = echo["params"]["update"]["content"]["text"]
        # The echo carries the JSON the daemon wrote back to stdin.
        echo_obj = json.loads(echo_text)
        assert echo_obj["id"] == "req_1"
        assert echo_obj["result"]["content"] == "FAKE_FILE_CONTENT"

    @pytest.mark.asyncio
    async def test_unknown_method_returns_method_not_found(self) -> None:
        # No handler registered for fs/read_text_file: daemon should
        # write back an error envelope automatically.
        daemon = AcpxDaemon(
            auto_ensure_session=False,
            cmd_builder=_python_cmd_builder(_REQUEST_SCRIPT),
        )
        lines = []
        async for raw in daemon.submit_turn(
            session_name="copaw-test",
            prompt_blocks=[{"type": "text", "text": "hi"}],
            is_seed=True,
        ):
            lines.append(raw.strip())
        echo = json.loads(lines[0])
        echo_text = echo["params"]["update"]["content"]["text"]
        echo_obj = json.loads(echo_text)
        assert echo_obj["id"] == "req_1"
        assert "error" in echo_obj
        assert echo_obj["error"]["code"] == -32601

    @pytest.mark.asyncio
    async def test_handler_exception_yields_jsonrpc_error(self) -> None:
        async def boom(_params: dict) -> dict:
            raise RuntimeError("kaboom")

        daemon = AcpxDaemon(
            auto_ensure_session=False,
            cmd_builder=_python_cmd_builder(_REQUEST_SCRIPT),
        )
        daemon.set_handler("fs/read_text_file", boom)
        lines = []
        async for raw in daemon.submit_turn(
            session_name="copaw-test",
            prompt_blocks=[{"type": "text", "text": "hi"}],
            is_seed=True,
        ):
            lines.append(raw.strip())
        echo_obj = json.loads(
            json.loads(lines[0])["params"]["update"]["content"]["text"]
        )
        assert echo_obj["error"]["code"] == -32000
        assert "kaboom" in echo_obj["error"]["message"]

    def test_set_handler_and_has_handler(self) -> None:
        daemon = AcpxDaemon()
        assert not daemon.has_handler("fs/read_text_file")

        async def h(_p: dict) -> dict:
            return {}

        daemon.set_handler("fs/read_text_file", h)
        assert daemon.has_handler("fs/read_text_file")


# ----------------------------------------------------------------- #
# submit_turn — error paths
# ----------------------------------------------------------------- #


class TestSubmitTurnErrors:
    @pytest.mark.asyncio
    async def test_after_shutdown_raises(self) -> None:
        daemon = AcpxDaemon(
            auto_ensure_session=False,
            cmd_builder=_python_cmd_builder(_BASIC_SCRIPT),
        )
        await daemon.shutdown()
        with pytest.raises(AcpxDaemonError, match="shut down"):
            async for _ in daemon.submit_turn(
                session_name="x",
                prompt_blocks=[{"type": "text", "text": "hi"}],
                is_seed=True,
            ):
                pass

    @pytest.mark.asyncio
    async def test_missing_binary_raises_helpful_error(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Pretend npx isn't on PATH.
        monkeypatch.setattr(
            "qwenpaw.providers.claude_acpx_daemon._binary_available",
            lambda: False,
        )
        daemon = AcpxDaemon()
        with pytest.raises(AcpxDaemonError, match="acpx binary not found"):
            async for _ in daemon.submit_turn(
                session_name="x",
                prompt_blocks=[{"type": "text", "text": "hi"}],
                is_seed=True,
            ):
                pass

    @pytest.mark.asyncio
    async def test_subprocess_timeout_raises(self) -> None:
        daemon = AcpxDaemon(
            auto_ensure_session=False,
            cmd_builder=_python_cmd_builder(_STALL_SCRIPT),
            turn_timeout_seconds=0.5,
        )
        with pytest.raises(AcpxDaemonError, match="stalled past"):
            async for _ in daemon.submit_turn(
                session_name="x",
                prompt_blocks=[{"type": "text", "text": "hi"}],
                is_seed=True,
            ):
                pass
        # Error counter incremented.
        assert claude_acpx_metrics.snapshot()["error"] >= 1

    @pytest.mark.asyncio
    async def test_nonzero_rc_logged_and_metric_recorded(self) -> None:
        daemon = AcpxDaemon(
            auto_ensure_session=False,
            cmd_builder=_python_cmd_builder(_BAD_RC_SCRIPT),
        )
        # Subprocess exits before producing JSON-RPC; the loop just
        # sees EOF.  We don't raise here — the translator (Lane A)
        # would notice no final response.  But _reap should record
        # error.
        async for _ in daemon.submit_turn(
            session_name="x",
            prompt_blocks=[{"type": "text", "text": "hi"}],
            is_seed=True,
        ):
            pass
        assert claude_acpx_metrics.snapshot()["error"] >= 1


# ----------------------------------------------------------------- #
# shutdown
# ----------------------------------------------------------------- #


class TestShutdown:
    @pytest.mark.asyncio
    async def test_shutdown_kills_inflight(self) -> None:
        daemon = AcpxDaemon(
            auto_ensure_session=False,
            cmd_builder=_python_cmd_builder(_STALL_SCRIPT),
            turn_timeout_seconds=10,
        )
        # Start a turn but don't drain it; we want it parked.
        gen = daemon.submit_turn(
            session_name="x",
            prompt_blocks=[{"type": "text", "text": "hi"}],
            is_seed=True,
        )
        # Pump once so the spawn happens.
        consumer = asyncio.create_task(_drain_silently(gen))
        # Give the spawn a chance to enter _stream_lines.
        await asyncio.sleep(0.1)
        # Daemon has at least one inflight process.
        assert len(daemon._inflight) == 1
        await daemon.shutdown()
        # Inflight cleared.
        assert daemon._inflight == set()
        # Consumer finishes (subprocess killed → EOF on stdout).
        await asyncio.wait_for(consumer, timeout=5)

    @pytest.mark.asyncio
    async def test_shutdown_idempotent(self) -> None:
        daemon = AcpxDaemon()
        await daemon.shutdown()
        await daemon.shutdown()  # no error.


async def _drain_silently(gen: Any) -> None:
    try:
        async for _ in gen:
            pass
    except AcpxDaemonError:
        return
    except Exception:  # noqa: BLE001
        return


# ----------------------------------------------------------------- #
# teardown / run_set_config
# ----------------------------------------------------------------- #


class TestTeardownAndSetConfig:
    @pytest.mark.asyncio
    async def test_teardown_no_op_when_binary_missing(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(
            "qwenpaw.providers.claude_acpx_daemon._binary_available",
            lambda: False,
        )
        daemon = AcpxDaemon()
        # Should not raise.
        await daemon.teardown("copaw-x")

    @pytest.mark.asyncio
    async def test_teardown_after_shutdown_is_noop(self) -> None:
        daemon = AcpxDaemon()
        await daemon.shutdown()
        await daemon.teardown("copaw-x")

    @pytest.mark.asyncio
    async def test_teardown_runs_cli_and_records_metric(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Replace asyncio.create_subprocess_exec with a fake that
        # returns a process whose communicate() exits zero.
        captured_cmds: list[tuple[str, ...]] = []

        async def fake_exec(*cmd: str, **_: Any) -> Any:
            captured_cmds.append(cmd)

            class _FakeProc:
                returncode = 0

                async def communicate(self) -> tuple[bytes, bytes]:
                    return (b"", b"")

            return _FakeProc()

        monkeypatch.setattr(
            "qwenpaw.providers.claude_acpx_daemon.asyncio."
            "create_subprocess_exec",
            fake_exec,
        )
        before = claude_acpx_metrics.snapshot()["tear_down"]
        daemon = AcpxDaemon()
        await daemon.teardown("copaw-x")
        after = claude_acpx_metrics.snapshot()["tear_down"]
        assert after == before + 1
        # Verify cmd composition.
        assert captured_cmds, "expected at least one teardown spawn"
        assert "claude" in captured_cmds[0]
        assert "sessions" in captured_cmds[0]
        assert "close" in captured_cmds[0]
        assert "copaw-x" in captured_cmds[0]

    @pytest.mark.asyncio
    async def test_run_set_config_increments_metric(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        captured_cmds: list[tuple[str, ...]] = []

        async def fake_exec(*cmd: str, **_: Any) -> Any:
            captured_cmds.append(cmd)

            class _FakeProc:
                returncode = 0

                async def communicate(self) -> tuple[bytes, bytes]:
                    return (b"", b"")

            return _FakeProc()

        monkeypatch.setattr(
            "qwenpaw.providers.claude_acpx_daemon.asyncio."
            "create_subprocess_exec",
            fake_exec,
        )
        daemon = AcpxDaemon(auto_ensure_session=False)
        before = claude_acpx_metrics.snapshot()["effort_set"]
        await daemon.run_set_config("copaw-x", "thinking", "high")
        after = claude_acpx_metrics.snapshot()["effort_set"]
        assert after == before + 1
        # Verify args contain key/value
        cmd = captured_cmds[0]
        assert "set" in cmd
        assert "thinking" in cmd
        assert "high" in cmd

    @pytest.mark.asyncio
    async def test_run_set_config_nonzero_rc_raises(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        async def fake_exec(*_cmd: str, **_: Any) -> Any:
            class _FakeProc:
                returncode = 3

                async def communicate(self) -> tuple[bytes, bytes]:
                    return (b"", b"upstream broke")

            return _FakeProc()

        monkeypatch.setattr(
            "qwenpaw.providers.claude_acpx_daemon.asyncio."
            "create_subprocess_exec",
            fake_exec,
        )
        daemon = AcpxDaemon(auto_ensure_session=False)
        with pytest.raises(AcpxDaemonError, match="acpx claude set"):
            await daemon.run_set_config("copaw-x", "thinking", "high")

    @pytest.mark.asyncio
    async def test_run_set_config_after_shutdown_raises(self) -> None:
        daemon = AcpxDaemon()
        await daemon.shutdown()
        with pytest.raises(AcpxDaemonError, match="shut down"):
            await daemon.run_set_config("copaw-x", "k", "v")
