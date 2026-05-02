# -*- coding: utf-8 -*-
"""Unit tests for the claude-acpx ACP method handlers — Lane B.

The handlers form the bridge between Claude Code's ACP requests
(``fs/*`` + ``terminal/*`` + ``session/request_permission``) and
CoPaw's filesystem / process subsystems, gated through the same
guardian engine that protects direct CoPaw tool calls.

Tests use a fake guard engine factory + tmp_path so we can pin the
behaviour without touching real config or the global engine.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import Any

import pytest

from qwenpaw.providers.claude_acpx_handlers import (
    AcpxFsHandlers,
    AcpxHandlerError,
    AcpxPermissionHandler,
    AcpxTerminalHandlers,
    _build_env,
    _join_argv_for_guard,
    _slice_lines,
    register_handlers,
)


# ----------------------------------------------------------------- #
# Fake guard engine helpers
# ----------------------------------------------------------------- #


@dataclass
class _FakeFinding:
    title: str = "denied"
    severity: str = "HIGH"


@dataclass
class _FakeGuardResult:
    findings: list[_FakeFinding] = field(default_factory=list)

    @property
    def is_safe(self) -> bool:
        return not self.findings


class _FakeGuardEngine:
    """Stand-in for ToolGuardEngine.  Records calls; returns either
    a denying or allowing result based on configured rule set.
    """

    def __init__(
        self,
        *,
        deny_tools: set[str] | None = None,
        return_none: bool = False,
    ) -> None:
        self.deny_tools = deny_tools or set()
        self.return_none = return_none
        self.calls: list[tuple[str, dict]] = []

    def guard(
        self,
        tool_name: str,
        params: dict[str, Any],
    ) -> _FakeGuardResult | None:
        self.calls.append((tool_name, dict(params)))
        if self.return_none:
            return None
        if tool_name in self.deny_tools:
            return _FakeGuardResult(findings=[_FakeFinding()])
        return _FakeGuardResult(findings=[])


def _factory(engine: _FakeGuardEngine | None) -> Any:
    return lambda: engine


# ----------------------------------------------------------------- #
# _slice_lines
# ----------------------------------------------------------------- #


class TestSliceLines:
    def test_no_window_returns_full_text(self) -> None:
        assert _slice_lines("a\nb\nc\n", line=None, limit=None) == "a\nb\nc\n"

    def test_line_offset_1_indexed(self) -> None:
        # ACP line is 1-indexed; line=2 should drop "a\n".
        assert _slice_lines("a\nb\nc\n", line=2, limit=None) == "b\nc\n"

    def test_limit_caps_lines(self) -> None:
        assert _slice_lines("a\nb\nc\nd\n", line=None, limit=2) == "a\nb\n"

    def test_line_and_limit_compose(self) -> None:
        assert _slice_lines("a\nb\nc\nd\n", line=2, limit=2) == "b\nc\n"

    def test_garbage_inputs_no_op(self) -> None:
        # line/limit of None or non-int both treated as "no window".
        assert _slice_lines("hi\n", line="oops", limit=-1) == "hi\n"


# ----------------------------------------------------------------- #
# _build_env
# ----------------------------------------------------------------- #


class TestBuildEnv:
    def test_empty_returns_allowlisted_parent_env(self) -> None:
        env = _build_env([])
        # PATH is in the allowlist and is virtually always set.
        assert "PATH" in env

    def test_acp_overrides_allowlisted_parent(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # PATH is allowlisted, so a parent value flows through and
        # the ACP entry overrides it.
        monkeypatch.setenv("PATH", "from_parent")
        env = _build_env([{"name": "PATH", "value": "from_acp"}])
        assert env["PATH"] == "from_acp"

    def test_acp_can_set_unlisted_var(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # ACP entries are layered on top, so the tool can still set
        # arbitrary names — we just don't inherit them from the parent.
        monkeypatch.setenv("FOO", "from_parent")
        env = _build_env([{"name": "FOO", "value": "from_acp"}])
        assert env["FOO"] == "from_acp"

    def test_malformed_entries_skipped(self) -> None:
        env = _build_env([
            "not-a-dict",
            {"name": "OK", "value": "yes"},
            {"name": 123, "value": "bad"},  # name not str
        ])
        assert env["OK"] == "yes"

    def test_parent_secrets_excluded(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The whole point of the allowlist: provider API keys held by
        the parent CoPaw process must NOT flow into terminal/create
        children.  A compromised or curious tool turn could otherwise
        read them via ``env`` / ``printenv`` and exfiltrate via
        stdout."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-leaky")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-leaky")
        monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "leaky")
        monkeypatch.setenv("DASHSCOPE_API_KEY", "leaky")

        env = _build_env([])

        assert "OPENAI_API_KEY" not in env
        assert "ANTHROPIC_API_KEY" not in env
        assert "AWS_SECRET_ACCESS_KEY" not in env
        assert "DASHSCOPE_API_KEY" not in env

    def test_locale_and_home_inherit(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Allowlist covers what tools actually need.  Without HOME the
        tool can't find user config; without LANG/LC_* it'll fall back
        to C locale and break unicode in node/python child processes."""
        monkeypatch.setenv("HOME", "/home/test")
        monkeypatch.setenv("LANG", "en_US.UTF-8")
        env = _build_env([])
        assert env["HOME"] == "/home/test"
        assert env["LANG"] == "en_US.UTF-8"

    def test_acp_env_denylist_drops_execution_hijack_hooks(self) -> None:
        """Codex 2026-04-28 review caught that the guardian inspects
        ``command``/``cwd`` only — ACP-supplied env entries flow to the
        child unguarded.  ``BASH_ENV`` / ``NODE_OPTIONS`` /
        ``LD_PRELOAD`` etc. would let the model hijack any otherwise-
        allowed command's semantics.  Deny-list drops them; legitimate
        env entries still flow through."""
        env = _build_env([
            {"name": "BASH_ENV", "value": "/tmp/evil.sh"},
            {"name": "NODE_OPTIONS", "value": "--require=/tmp/evil.js"},
            {"name": "LD_PRELOAD", "value": "/tmp/evil.so"},
            {"name": "DYLD_INSERT_LIBRARIES", "value": "/tmp/evil.dylib"},
            {"name": "PYTHONSTARTUP", "value": "/tmp/evil.py"},
            {"name": "RUBYOPT", "value": "-r/tmp/evil.rb"},
            {"name": "PERL5OPT", "value": "-Mevil"},
            {"name": "GIT_CONFIG_GLOBAL", "value": "/tmp/.evilgit"},
            {"name": "GIT_SSH_COMMAND", "value": "/tmp/evil-ssh"},
            # Legitimate variables — should pass through.
            {"name": "MY_TOOL_FLAG", "value": "yes"},
            {"name": "DEBUG", "value": "1"},
        ])
        # Deny-list entries must NOT be in the env.
        for hijack in (
            "BASH_ENV",
            "NODE_OPTIONS",
            "LD_PRELOAD",
            "DYLD_INSERT_LIBRARIES",
            "PYTHONSTARTUP",
            "RUBYOPT",
            "PERL5OPT",
            "GIT_CONFIG_GLOBAL",
            "GIT_SSH_COMMAND",
        ):
            assert hijack not in env, (
                f"deny-list breach: {hijack} flowed to child env"
            )
        # Legitimate entries flow through.
        assert env["MY_TOOL_FLAG"] == "yes"
        assert env["DEBUG"] == "1"

    def test_acp_env_denylist_does_not_block_other_overrides(self) -> None:
        """Deny-list is targeted — generic var overrides still work."""
        env = _build_env([
            {"name": "PATH", "value": "/custom/path"},
            {"name": "FOO", "value": "bar"},
        ])
        assert env["PATH"] == "/custom/path"
        assert env["FOO"] == "bar"


# ----------------------------------------------------------------- #
# _join_argv_for_guard
# ----------------------------------------------------------------- #


class TestJoinArgvForGuard:
    def test_simple_command(self) -> None:
        assert _join_argv_for_guard("ls", ["-la"]) == "ls -la"

    def test_quotes_arg_with_space(self) -> None:
        out = _join_argv_for_guard("echo", ["hello world"])
        assert "'hello world'" in out

    def test_quotes_command_with_special_chars(self) -> None:
        out = _join_argv_for_guard("/bin/echo", ["$PATH"])
        # shlex.quote escapes $ properly.
        assert "'$PATH'" in out


# ----------------------------------------------------------------- #
# AcpxFsHandlers.read_text_file
# ----------------------------------------------------------------- #


class TestFsRead:
    @pytest.mark.asyncio
    async def test_reads_file(self, tmp_path) -> None:
        f = tmp_path / "x.txt"
        f.write_text("Hello\nWorld\n", encoding="utf-8")
        engine = _FakeGuardEngine()
        h = AcpxFsHandlers(guard_engine_factory=_factory(engine))
        out = await h.read_text_file({"sessionId": "s", "path": str(f)})
        assert out["content"] == "Hello\nWorld\n"
        # Guard engine consulted with view_text_file shape.
        assert engine.calls[0][0] == "view_text_file"
        assert engine.calls[0][1]["file_path"] == str(f)

    @pytest.mark.asyncio
    async def test_line_and_limit_applied(self, tmp_path) -> None:
        f = tmp_path / "x.txt"
        f.write_text("a\nb\nc\nd\n", encoding="utf-8")
        h = AcpxFsHandlers(guard_engine_factory=_factory(_FakeGuardEngine()))
        out = await h.read_text_file({
            "sessionId": "s",
            "path": str(f),
            "line": 2,
            "limit": 2,
        })
        assert out["content"] == "b\nc\n"

    @pytest.mark.asyncio
    async def test_missing_path_raises_invalid_params(self) -> None:
        h = AcpxFsHandlers(guard_engine_factory=_factory(_FakeGuardEngine()))
        with pytest.raises(AcpxHandlerError) as ei:
            await h.read_text_file({"sessionId": "s"})
        assert ei.value.code == -32602

    @pytest.mark.asyncio
    async def test_empty_path_raises_invalid_params(self) -> None:
        h = AcpxFsHandlers(guard_engine_factory=_factory(_FakeGuardEngine()))
        with pytest.raises(AcpxHandlerError) as ei:
            await h.read_text_file({"sessionId": "s", "path": ""})
        assert ei.value.code == -32602

    @pytest.mark.asyncio
    async def test_nonexistent_file_raises_not_found(
        self,
        tmp_path,
    ) -> None:
        h = AcpxFsHandlers(guard_engine_factory=_factory(_FakeGuardEngine()))
        target = tmp_path / "nope.txt"
        with pytest.raises(AcpxHandlerError) as ei:
            await h.read_text_file({"sessionId": "s", "path": str(target)})
        assert ei.value.code == -32003

    @pytest.mark.asyncio
    async def test_guardian_deny_raises_handler_error(self, tmp_path) -> None:
        f = tmp_path / "secret.txt"
        f.write_text("oops", encoding="utf-8")
        engine = _FakeGuardEngine(deny_tools={"view_text_file"})
        h = AcpxFsHandlers(guard_engine_factory=_factory(engine))
        with pytest.raises(AcpxHandlerError) as ei:
            await h.read_text_file({"sessionId": "s", "path": str(f)})
        assert ei.value.code == -32001
        assert "denied" in str(ei.value).lower()

    @pytest.mark.asyncio
    async def test_guard_returning_none_treated_as_allow(
        self,
        tmp_path,
    ) -> None:
        f = tmp_path / "x.txt"
        f.write_text("ok", encoding="utf-8")
        # return_none=True means guard is disabled.
        engine = _FakeGuardEngine(return_none=True)
        h = AcpxFsHandlers(guard_engine_factory=_factory(engine))
        out = await h.read_text_file({"sessionId": "s", "path": str(f)})
        assert out["content"] == "ok"


# ----------------------------------------------------------------- #
# AcpxFsHandlers.write_text_file
# ----------------------------------------------------------------- #


class TestFsWrite:
    @pytest.mark.asyncio
    async def test_writes_file(self, tmp_path) -> None:
        f = tmp_path / "out.txt"
        h = AcpxFsHandlers(guard_engine_factory=_factory(_FakeGuardEngine()))
        out = await h.write_text_file({
            "sessionId": "s",
            "path": str(f),
            "content": "hello",
        })
        assert out == {}
        assert f.read_text(encoding="utf-8") == "hello"

    @pytest.mark.asyncio
    async def test_creates_parent_dirs(self, tmp_path) -> None:
        f = tmp_path / "deep" / "nested" / "out.txt"
        h = AcpxFsHandlers(guard_engine_factory=_factory(_FakeGuardEngine()))
        await h.write_text_file({
            "sessionId": "s",
            "path": str(f),
            "content": "x",
        })
        assert f.exists()

    @pytest.mark.asyncio
    async def test_missing_content_raises_invalid_params(
        self,
        tmp_path,
    ) -> None:
        h = AcpxFsHandlers(guard_engine_factory=_factory(_FakeGuardEngine()))
        with pytest.raises(AcpxHandlerError) as ei:
            await h.write_text_file({
                "sessionId": "s",
                "path": str(tmp_path / "x"),
            })
        assert ei.value.code == -32602

    @pytest.mark.asyncio
    async def test_guardian_denies_write(self, tmp_path) -> None:
        engine = _FakeGuardEngine(deny_tools={"write_text_file"})
        h = AcpxFsHandlers(guard_engine_factory=_factory(engine))
        with pytest.raises(AcpxHandlerError) as ei:
            await h.write_text_file({
                "sessionId": "s",
                "path": str(tmp_path / "x"),
                "content": "nope",
            })
        assert ei.value.code == -32001


# ----------------------------------------------------------------- #
# AcpxTerminalHandlers — full lifecycle
# ----------------------------------------------------------------- #


class TestTerminal:
    @pytest.mark.asyncio
    async def test_create_output_wait_release_lifecycle(self) -> None:
        engine = _FakeGuardEngine()
        h = AcpxTerminalHandlers(guard_engine_factory=_factory(engine))
        # Use the test interpreter so the test works on any OS.
        out = await h.create({
            "sessionId": "s",
            "command": sys.executable,
            "args": ["-c", "import sys; sys.stdout.write('hi'); sys.exit(0)"],
        })
        terminal_id = out["terminalId"]
        assert terminal_id.startswith("term_")
        # Wait_for_exit drains and returns.
        exit_status = await h.wait_for_exit({
            "sessionId": "s",
            "terminalId": terminal_id,
        })
        assert exit_status == {"exitCode": 0, "signal": None}
        # Output snapshot has the captured stdout.
        snap = await h.output({
            "sessionId": "s",
            "terminalId": terminal_id,
        })
        assert snap["output"] == "hi"
        assert snap["truncated"] is False
        assert snap["exitStatus"] == {"exitCode": 0, "signal": None}
        # Release succeeds and is idempotent.
        assert await h.release({
            "sessionId": "s",
            "terminalId": terminal_id,
        }) == {}
        assert await h.release({
            "sessionId": "s",
            "terminalId": terminal_id,
        }) == {}

    @pytest.mark.asyncio
    async def test_create_missing_command_raises(self) -> None:
        h = AcpxTerminalHandlers(guard_engine_factory=_factory(_FakeGuardEngine()))
        with pytest.raises(AcpxHandlerError) as ei:
            await h.create({"sessionId": "s"})
        assert ei.value.code == -32602

    @pytest.mark.asyncio
    async def test_create_args_must_be_list(self) -> None:
        h = AcpxTerminalHandlers(guard_engine_factory=_factory(_FakeGuardEngine()))
        with pytest.raises(AcpxHandlerError) as ei:
            await h.create({
                "sessionId": "s",
                "command": "echo",
                "args": "hi",  # type: ignore[arg-type]
            })
        assert ei.value.code == -32602

    @pytest.mark.asyncio
    async def test_create_unknown_command_raises_not_found(self) -> None:
        h = AcpxTerminalHandlers(guard_engine_factory=_factory(_FakeGuardEngine()))
        with pytest.raises(AcpxHandlerError) as ei:
            await h.create({
                "sessionId": "s",
                "command": "/no/such/binary/xyz1234",
                "args": [],
            })
        assert ei.value.code == -32003

    @pytest.mark.asyncio
    async def test_create_guarded_deny_raises(self) -> None:
        engine = _FakeGuardEngine(deny_tools={"execute_shell_command"})
        h = AcpxTerminalHandlers(guard_engine_factory=_factory(engine))
        with pytest.raises(AcpxHandlerError) as ei:
            await h.create({
                "sessionId": "s",
                "command": "rm",
                "args": ["-rf", "/"],
            })
        assert ei.value.code == -32001

    @pytest.mark.asyncio
    async def test_unknown_terminal_id_returns_error(self) -> None:
        h = AcpxTerminalHandlers(guard_engine_factory=_factory(_FakeGuardEngine()))
        with pytest.raises(AcpxHandlerError) as ei:
            await h.output({"sessionId": "s", "terminalId": "term_bogus"})
        assert ei.value.code == -32004

    @pytest.mark.asyncio
    async def test_output_before_exit_reports_no_status(self) -> None:
        h = AcpxTerminalHandlers(guard_engine_factory=_factory(_FakeGuardEngine()))
        out = await h.create({
            "sessionId": "s",
            "command": sys.executable,
            "args": [
                "-c",
                "import time; print('tick'); time.sleep(0.5);"
                " print('tock')",
            ],
        })
        snap = await h.output({
            "sessionId": "s",
            "terminalId": out["terminalId"],
        })
        # Process likely still running; exitStatus None.
        # (Slim chance the kernel finished the subprocess between
        # create and output; assert flexibly.)
        assert "output" in snap
        # Cleanup.
        await h.wait_for_exit({"sessionId": "s", "terminalId": out["terminalId"]})
        await h.release({"sessionId": "s", "terminalId": out["terminalId"]})

    @pytest.mark.asyncio
    async def test_output_truncates_when_byte_limit_hit(self) -> None:
        h = AcpxTerminalHandlers(guard_engine_factory=_factory(_FakeGuardEngine()))
        # Spawn a subprocess that prints more than the limit.
        out = await h.create({
            "sessionId": "s",
            "command": sys.executable,
            "args": [
                "-c",
                "import sys; sys.stdout.write('A' * 200); sys.exit(0)",
            ],
            "outputByteLimit": 50,
        })
        await h.wait_for_exit({"sessionId": "s", "terminalId": out["terminalId"]})
        snap = await h.output({
            "sessionId": "s",
            "terminalId": out["terminalId"],
        })
        assert snap["truncated"] is True
        assert len(snap["output"]) == 50
        await h.release({"sessionId": "s", "terminalId": out["terminalId"]})

    @pytest.mark.asyncio
    async def test_release_terminates_running_process(self) -> None:
        h = AcpxTerminalHandlers(guard_engine_factory=_factory(_FakeGuardEngine()))
        out = await h.create({
            "sessionId": "s",
            "command": sys.executable,
            "args": ["-c", "import time; time.sleep(30)"],
        })
        terminal_id = out["terminalId"]
        # Release while running.
        result = await h.release({
            "sessionId": "s",
            "terminalId": terminal_id,
        })
        assert result == {}
        # Subsequent release is no-op (idempotent).
        assert await h.release({
            "sessionId": "s",
            "terminalId": terminal_id,
        }) == {}


# ----------------------------------------------------------------- #
# AcpxPermissionHandler
# ----------------------------------------------------------------- #


class TestPermission:
    @pytest.mark.asyncio
    async def test_allow_option_chosen(self) -> None:
        h = AcpxPermissionHandler()
        out = await h.request_permission({
            "sessionId": "s",
            "toolCall": {},
            "options": [
                {"optionId": "deny", "name": "Reject", "kind": "reject_once"},
                {"optionId": "yes", "name": "Allow", "kind": "allow_once"},
            ],
        })
        assert out["outcome"]["outcome"] == "selected"
        assert out["outcome"]["optionId"] == "yes"

    @pytest.mark.asyncio
    async def test_no_options_yields_cancelled(self) -> None:
        h = AcpxPermissionHandler()
        out = await h.request_permission({
            "sessionId": "s",
            "toolCall": {},
            "options": [],
        })
        assert out["outcome"]["outcome"] == "cancelled"

    @pytest.mark.asyncio
    async def test_no_allow_falls_back_to_first(self) -> None:
        # No allow_* kind — pick first option as a "say yes to whatever
        # the agent put in front of us" fallback rather than blocking.
        h = AcpxPermissionHandler()
        out = await h.request_permission({
            "sessionId": "s",
            "toolCall": {},
            "options": [
                {"optionId": "first", "name": "First", "kind": "ask"},
                {"optionId": "second", "name": "Second", "kind": "ask"},
            ],
        })
        assert out["outcome"]["outcome"] == "selected"
        assert out["outcome"]["optionId"] == "first"

    @pytest.mark.asyncio
    async def test_options_must_be_list(self) -> None:
        h = AcpxPermissionHandler()
        out = await h.request_permission({
            "sessionId": "s",
            "options": "not-a-list",
        })
        assert out["outcome"]["outcome"] == "cancelled"


# ----------------------------------------------------------------- #
# register_handlers — wiring
# ----------------------------------------------------------------- #


class TestRegisterHandlers:
    def test_registers_all_acp_methods(self) -> None:
        from qwenpaw.providers.claude_acpx_daemon import AcpxDaemon

        AcpxDaemon.reset_singleton_for_test()
        daemon = AcpxDaemon()
        register_handlers(daemon)
        for method in (
            "fs/read_text_file",
            "fs/write_text_file",
            "terminal/create",
            "terminal/output",
            "terminal/wait_for_exit",
            "terminal/release",
            "session/request_permission",
        ):
            assert daemon.has_handler(method), f"missing handler: {method}"

    def test_register_is_idempotent(self) -> None:
        from qwenpaw.providers.claude_acpx_daemon import AcpxDaemon

        AcpxDaemon.reset_singleton_for_test()
        daemon = AcpxDaemon()
        register_handlers(daemon)
        register_handlers(daemon)  # no error.
        assert daemon.has_handler("fs/read_text_file")


# ----------------------------------------------------------------- #
# AcpxHandlerError shape
# ----------------------------------------------------------------- #


class TestHandlerError:
    def test_error_carries_code_and_message(self) -> None:
        e = AcpxHandlerError(code=-32001, message="denied")
        assert e.code == -32001
        assert "denied" in str(e)


# ----------------------------------------------------------------- #
# Trust-mode bypass (COPAW_ACPX_SKIP_GUARDIAN)
# ----------------------------------------------------------------- #


class TestTrustModeBypass:
    """When ``COPAW_ACPX_SKIP_GUARDIAN`` is set, fs/* and terminal/*
    requests skip the guardian engine entirely and log a WARNING."""

    @pytest.mark.asyncio
    async def test_trust_mode_bypasses_fs_write_guardian(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path,
    ) -> None:
        from qwenpaw.providers.claude_acpx_handlers import (
            AcpxFsHandlers,
        )

        # Guardian would normally deny ``write_text_file``.
        engine = _FakeGuardEngine(deny_tools={"write_text_file"})
        fs = AcpxFsHandlers(guard_engine_factory=_factory(engine))

        # Without trust mode → denied.
        target = tmp_path / "blocked.txt"
        with pytest.raises(AcpxHandlerError):
            await fs.write_text_file({
                "sessionId": "s",
                "path": str(target),
                "content": "x",
            })
        assert not target.exists()

        # With trust mode → allowed (engine NOT consulted).
        monkeypatch.setenv("COPAW_ACPX_SKIP_GUARDIAN", "1")
        bypassed = tmp_path / "allowed.txt"
        await fs.write_text_file({
            "sessionId": "s",
            "path": str(bypassed),
            "content": "y",
        })
        assert bypassed.read_text() == "y"
        # Engine wasn't called for the bypass turn (only the first
        # call was recorded; the second one was bypassed entirely).
        assert len(engine.calls) == 1

    def test_trust_mode_disabled_for_falsy_values(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from qwenpaw.providers.claude_acpx_handlers import (
            _acpx_trust_mode_enabled,
        )

        for falsy in ("", "0", "false", "no", "off", "  "):
            monkeypatch.setenv("COPAW_ACPX_SKIP_GUARDIAN", falsy)
            assert not _acpx_trust_mode_enabled(), (
                f"value {falsy!r} should NOT enable trust mode"
            )

    def test_trust_mode_enabled_for_truthy_values(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from qwenpaw.providers.claude_acpx_handlers import (
            _acpx_trust_mode_enabled,
        )

        for truthy in ("1", "true", "TRUE", "yes", "on", "y", "t"):
            monkeypatch.setenv("COPAW_ACPX_SKIP_GUARDIAN", truthy)
            assert _acpx_trust_mode_enabled(), (
                f"value {truthy!r} should enable trust mode"
            )
