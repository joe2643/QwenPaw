# -*- coding: utf-8 -*-
"""ACP client-side method handlers — Lane B.

When the user picks the ``claude-acpx`` provider, CoPaw runs in
**Hybrid mode** (per the design plan): Claude Code proposes tool
invocations via ``tool_call`` notifications, but CoPaw EXECUTES the
actual filesystem and terminal operations on the local box.  ACP
expresses that contract via client-side request methods that flow
in the OPPOSITE direction from prompts — Claude → us, not us →
Claude.

This module bridges those incoming ACP requests to:

* CoPaw's existing :mod:`qwenpaw.security.tool_guard` engine
  (file_path_tool_guardian + shell_evasion_guardian + rule_guardian),
  so the same allow/deny policy that gates direct CoPaw tool calls
  also gates Claude-Code-driven calls.
* The local filesystem and process subsystems for the actual work
  (read/write a file, spawn a terminal command).
* :class:`AcpxPermissionHandler` — auto-allow (v1).  Documented v2
  hook into a real permission UI.

Wire-up (called once per :class:`AcpxDaemon`):

.. code-block:: python

   register_handlers(daemon)  # see bottom of file

After that, daemon's ``_dispatch_request`` finds the right callable
by method name (``fs/read_text_file``, ``terminal/create``, ...).

ACP method shapes (from upstream schema 2026-04 snapshot, verified
during Lane B impl against agentclientprotocol.com/protocol/schema):

* ``fs/read_text_file``  params {sessionId, path, line?, limit?}
                         result {content: str}
* ``fs/write_text_file`` params {sessionId, path, content}
                         result {}
* ``terminal/create``    params {sessionId, command, args, env, cwd, outputByteLimit?}
                         result {terminalId: str}
* ``terminal/output``    params {sessionId, terminalId}
                         result {output: str, truncated: bool, exitStatus: object|null}
* ``terminal/wait_for_exit`` params {sessionId, terminalId}
                             result {exitCode: int|null, signal: str|null}
* ``terminal/release``   params {sessionId, terminalId}
                         result {}
* ``session/request_permission`` params {sessionId, toolCall, options}
                                 result {outcome: {outcome: "selected"|"cancelled", optionId?: str}}
"""

from __future__ import annotations

import asyncio
import logging
import os
import secrets
import signal
from pathlib import Path
from typing import Any, TYPE_CHECKING

from qwenpaw.providers.claude_acpx_daemon import _HandlerError

if TYPE_CHECKING:
    from qwenpaw.providers.claude_acpx_daemon import AcpxDaemon

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------- #
# JSON-RPC error codes
# ----------------------------------------------------------------- #
#
# Keep these wide of the JSON-RPC reserved range (-32768..-32000) so
# we don't clash with parser/transport errors that acpx itself might
# emit.  Range -32001..-32099 is reserved for "implementation-defined
# server errors" per JSON-RPC 2.0 §5.1.
_ERR_GUARDIAN_DENY: int = -32001
_ERR_INVALID_PARAMS: int = -32602  # standard JSON-RPC
_ERR_IO: int = -32002
_ERR_NOT_FOUND: int = -32003
_ERR_TERMINAL_UNKNOWN: int = -32004
_ERR_INTERNAL: int = -32000  # generic implementation-defined server error


# ----------------------------------------------------------------- #
# Shared error type (re-exported alias for handler call sites)
# ----------------------------------------------------------------- #


class AcpxHandlerError(_HandlerError):
    """Raise to surface a JSON-RPC error envelope back to acpx.

    The daemon catches this and forms a proper
    ``{"error": {code, message}}`` reply.  Handler bugs that escape
    as plain ``Exception`` get coerced to a generic ``-32000`` —
    raising :class:`AcpxHandlerError` is preferred when the failure
    is expected (deny, missing file, etc.).
    """


# ----------------------------------------------------------------- #
# Trust-mode bypass
# ----------------------------------------------------------------- #
#
# **DANGEROUS — use only in trusted single-user setups.** When the
# env var ``COPAW_ACPX_SKIP_GUARDIAN`` is truthy (``1``, ``true``,
# ``yes``, ``on``, case-insensitive), the guardian engine is
# bypassed for every ``fs/*`` and ``terminal/*`` ACP request from
# Claude Code. This trades the file_path / shell_evasion / rule
# guardian protections for ergonomic flow when Claude Code wants
# to iterate on real local work (writing scratch files under
# ``/tmp``, running helper scripts, etc.).
#
# Each bypass logs a WARNING with the tool name and short params so
# the audit trail still records what got through.

_ACPX_TRUST_ENV_VAR: str = "COPAW_ACPX_SKIP_GUARDIAN"
_TRUTHY = frozenset({"1", "true", "yes", "on", "y", "t"})


def _acpx_trust_mode_enabled() -> bool:
    """Return True if the trust-mode bypass is active.

    Read on every call rather than cached so the operator can flip
    the flag without restarting the service (fish: ``set -gx
    COPAW_ACPX_SKIP_GUARDIAN 1`` in the qwenpaw shell, then send a
    SIGHUP-equivalent or just toggle through the next turn).
    """
    raw = os.environ.get(_ACPX_TRUST_ENV_VAR, "")
    return raw.strip().lower() in _TRUTHY


_DEFAULT_TERMINAL_WAIT_SECONDS: float = 600.0


def _resolve_terminal_wait_timeout() -> float:
    """Read ``acpx_provider.terminal_wait_seconds`` from live config.

    ``load_config`` is mtime-cached, so calling this on every
    ``terminal/wait_for_exit`` is cheap.  Falls back to the module
    default on any load error so a corrupt config doesn't pin the
    handler forever.
    """
    try:
        from qwenpaw.config import load_config

        return float(load_config().acpx_provider.terminal_wait_seconds)
    except Exception:  # noqa: BLE001
        return _DEFAULT_TERMINAL_WAIT_SECONDS


def _short_repr(params: dict[str, Any], cap: int = 200) -> str:
    """Compress params dict to a one-liner for the audit log.

    Long ``content`` payloads (file writes) or long ``args`` lists
    (shell commands) get truncated mid-value rather than being
    spelled out in full.
    """
    try:
        text = repr(params)
    except Exception:  # noqa: BLE001
        return "<unrepr>"
    if len(text) <= cap:
        return text
    return text[:cap] + "...(truncated)"


# ----------------------------------------------------------------- #
# Filesystem handlers
# ----------------------------------------------------------------- #


class AcpxFsHandlers:
    """Handlers for ``fs/read_text_file`` and ``fs/write_text_file``.

    Routes both through CoPaw's
    :class:`~qwenpaw.security.tool_guard.engine.ToolGuardEngine` —
    using the file_path_tool_guardian (which enforces the
    ``security.file_guard.sensitive_files`` allowlist among other
    things) — before touching the filesystem.

    The guard call uses the same tool names as CoPaw's first-party
    tools (``view_text_file`` / ``write_text_file``), which keeps
    one set of guardian rules covering both code paths.
    """

    def __init__(
        self,
        *,
        guard_engine_factory: Any | None = None,
    ) -> None:
        # Lazy-import the engine so unit tests can substitute via
        # the factory without a guard import at module load.
        self._guard_engine_factory = guard_engine_factory

    async def read_text_file(self, params: dict[str, Any]) -> dict[str, Any]:
        """Read a UTF-8 text file from disk after guard check.

        Honours optional ``line`` (1-indexed first line) and
        ``limit`` (max lines) params per ACP spec.  Bytes that
        aren't valid UTF-8 surface as replacement chars rather than
        crashing — Claude Code's adapter expects a string back.
        """
        path = self._require_path(params)
        self._guard_or_deny(
            tool_name="view_text_file",
            tool_params={"file_path": path, "path": path},
        )
        try:
            content = await asyncio.to_thread(_read_text, path)
        except FileNotFoundError as e:
            raise AcpxHandlerError(
                code=_ERR_NOT_FOUND,
                message=f"file not found: {path}",
            ) from e
        except OSError as e:
            raise AcpxHandlerError(
                code=_ERR_IO,
                message=f"read error: {e}",
            ) from e

        # Optional line / limit windowing.  ACP spec: line is 1-based
        # ("the line number to start reading from"); limit caps lines.
        line = params.get("line")
        limit = params.get("limit")
        if line is not None or limit is not None:
            content = _slice_lines(content, line=line, limit=limit)

        return {"content": content}

    async def write_text_file(self, params: dict[str, Any]) -> dict[str, Any]:
        """Atomically replace ``path`` with ``content`` after guard
        check.  Creates parent dirs so Claude can create files in
        new subdirectories.  Returns empty result per ACP spec.
        """
        path = self._require_path(params)
        content = params.get("content")
        if not isinstance(content, str):
            raise AcpxHandlerError(
                code=_ERR_INVALID_PARAMS,
                message="fs/write_text_file: 'content' must be a string",
            )
        self._guard_or_deny(
            tool_name="write_text_file",
            tool_params={"file_path": path, "path": path, "content": content},
        )
        try:
            await asyncio.to_thread(_write_text, path, content)
        except OSError as e:
            raise AcpxHandlerError(
                code=_ERR_IO,
                message=f"write error: {e}",
            ) from e
        return {}

    # ----- Internal helpers ---------------------------------------- #

    @staticmethod
    def _require_path(params: dict[str, Any]) -> str:
        path = params.get("path")
        if not isinstance(path, str) or not path:
            raise AcpxHandlerError(
                code=_ERR_INVALID_PARAMS,
                message="missing or empty 'path' parameter",
            )
        return path

    def _guard_or_deny(
        self,
        *,
        tool_name: str,
        tool_params: dict[str, Any],
    ) -> None:
        """Run the guard engine; raise AcpxHandlerError on deny.

        Imports inside the function so a test that doesn't care about
        guarding can supply a no-op factory and avoid the engine's
        config / filesystem touch.

        Honours :func:`_acpx_trust_mode_enabled` — when the trust-mode
        env var is set, the guardian is bypassed entirely. Use ONLY
        in trusted single-user setups; this gives Claude Code
        unrestricted local fs/terminal access.
        """
        if _acpx_trust_mode_enabled():
            logger.warning(
                "acpx trust-mode bypass: %s allowed without guardian "
                "check (params=%s)",
                tool_name,
                _short_repr(tool_params),
            )
            return
        if self._guard_engine_factory is not None:
            engine = self._guard_engine_factory()
        else:
            from qwenpaw.security.tool_guard.engine import get_guard_engine

            engine = get_guard_engine()

        if engine is None:
            return  # guarding disabled at engine level
        result = engine.guard(tool_name, tool_params)
        if result is None:
            return  # guard disabled
        if result.is_safe:
            return
        # Compose a single-line denial reason from the highest-severity
        # finding.  Detail is in CoPaw logs already; ACP error message
        # is short for Claude Code's adapter UI.
        first = result.findings[0] if result.findings else None
        reason = first.title if first is not None else "guarded denied"
        raise AcpxHandlerError(
            code=_ERR_GUARDIAN_DENY,
            message=f"file_guardian denied {tool_name}: {reason}",
        )


# ----------------------------------------------------------------- #
# Terminal handlers
# ----------------------------------------------------------------- #


class _TerminalSession:
    """Per-terminal book-keeping: the asyncio subprocess + a string
    accumulator for stdout (capped at ``output_byte_limit``).
    """

    __slots__ = (
        "process",
        "stdout_buf",
        "stderr_buf",
        "byte_limit",
        "exit_code",
        "signal",
        "drain_task",
        "stderr_drain_task",
    )

    def __init__(
        self,
        process: asyncio.subprocess.Process,
        byte_limit: int,
    ) -> None:
        self.process = process
        self.stdout_buf = bytearray()
        self.stderr_buf = bytearray()
        self.byte_limit = byte_limit
        self.exit_code: int | None = None
        self.signal: str | None = None
        self.drain_task: asyncio.Task | None = None
        self.stderr_drain_task: asyncio.Task | None = None


class AcpxTerminalHandlers:
    """Handlers for the ``terminal/*`` ACP methods.

    Each ``terminal/create`` spawns a real subprocess; output is
    drained into an in-memory buffer (capped to the per-terminal
    ``outputByteLimit`` so a runaway producer can't OOM us) until
    ``terminal/release`` cleans up.  Mirrors the abstraction Zed's
    Agent Client implementation exposes; CoPaw's distinction is the
    pre-spawn guard check.
    """

    # 1 MiB default cap — generous for typical command output, small
    # enough that 100 concurrent terminals can't blow up RSS.
    _DEFAULT_OUTPUT_BYTE_LIMIT: int = 1 << 20

    def __init__(
        self,
        *,
        guard_engine_factory: Any | None = None,
        terminals: dict[str, _TerminalSession] | None = None,
    ) -> None:
        self._guard_engine_factory = guard_engine_factory
        self._terminals: dict[str, _TerminalSession] = terminals or {}
        self._lock = asyncio.Lock()

    async def create(self, params: dict[str, Any]) -> dict[str, Any]:
        """Spawn a subprocess and register a terminal id.  Returns
        ``{"terminalId": <str>}``.
        """
        command = params.get("command")
        if not isinstance(command, str) or not command.strip():
            raise AcpxHandlerError(
                code=_ERR_INVALID_PARAMS,
                message="terminal/create: missing or empty 'command'",
            )
        args_raw = params.get("args") or []
        if not isinstance(args_raw, list):
            raise AcpxHandlerError(
                code=_ERR_INVALID_PARAMS,
                message="terminal/create: 'args' must be a list",
            )
        args: list[str] = [str(a) for a in args_raw]

        cwd = params.get("cwd")
        if cwd is not None and not isinstance(cwd, str):
            raise AcpxHandlerError(
                code=_ERR_INVALID_PARAMS,
                message="terminal/create: 'cwd' must be a string or null",
            )

        env_raw = params.get("env") or []
        env_map = _build_env(env_raw)

        byte_limit = params.get("outputByteLimit")
        if not isinstance(byte_limit, int) or byte_limit <= 0:
            byte_limit = self._DEFAULT_OUTPUT_BYTE_LIMIT

        # Reconstruct the human-readable command for guarding.  ACP's
        # terminal/create is structurally close to ``argv``; CoPaw's
        # shell_evasion_guardian + file_path_tool_guardian both expect
        # the ``execute_shell_command`` shape with a single ``command``
        # string.  Build that shape conservatively (shell-escape args)
        # so the guards see the same surface they'd see for a direct
        # CoPaw shell tool.
        shell_command = _join_argv_for_guard(command, args)
        self._guard_or_deny(
            tool_name="execute_shell_command",
            tool_params={"command": shell_command, "cwd": cwd or ""},
        )

        try:
            proc = await asyncio.create_subprocess_exec(
                command,
                *args,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd or None,
                env=env_map,
                # New session so terminal/release can SIGKILL the
                # whole process group, not just the leader — tools
                # routinely spawn shells that fork children.
                start_new_session=True,
            )
        except FileNotFoundError as e:
            raise AcpxHandlerError(
                code=_ERR_NOT_FOUND,
                message=f"command not found: {command}",
            ) from e
        except OSError as e:
            raise AcpxHandlerError(
                code=_ERR_IO,
                message=f"spawn failed: {e}",
            ) from e

        terminal_id = _make_terminal_id()
        session = _TerminalSession(proc, byte_limit)

        # Drain stdout / stderr in the background so .output / .release
        # can read the accumulator at any time without a re-blocking read.
        session.drain_task = asyncio.create_task(
            _drain_stream(proc.stdout, session.stdout_buf, byte_limit),
        )
        session.stderr_drain_task = asyncio.create_task(
            _drain_stream(proc.stderr, session.stderr_buf, byte_limit),
        )

        async with self._lock:
            self._terminals[terminal_id] = session

        return {"terminalId": terminal_id}

    async def output(self, params: dict[str, Any]) -> dict[str, Any]:
        """Snapshot the accumulated stdout (and exit status if the
        subprocess has terminated).  Output stays in the buffer for
        subsequent calls — clients that want chunked retrieval can
        diff against what they previously read.
        """
        session = await self._require_terminal(params)
        rc = session.process.returncode
        truncated = len(session.stdout_buf) >= session.byte_limit

        out_text = bytes(session.stdout_buf).decode("utf-8", errors="replace")
        result: dict[str, Any] = {
            "output": out_text,
            "truncated": truncated,
        }
        if rc is None:
            result["exitStatus"] = None
        else:
            result["exitStatus"] = {
                "exitCode": rc if rc >= 0 else None,
                # POSIX convention: rc < 0 means killed by signal
                # ``-rc``.  ACP wants signal name as string, but Python
                # gives us the number; emit the numeric stringified.
                "signal": str(-rc) if rc < 0 else None,
            }
        return result

    async def wait_for_exit(self, params: dict[str, Any]) -> dict[str, Any]:
        """Block until the subprocess exits; return its exit status.

        Honours ``acpx_provider.terminal_wait_seconds`` from config.
        Default 600s; ``0`` means wait forever (legacy behaviour).
        Without this cap a runaway Bash spawned by Claude Code in
        bypassPermissions mode would pin the entire prompt subprocess
        until the daemon-level turn timeout fires (observed
        2026-05-02 — a tool stalled, the prompt hit 300s and was
        killed only after the user's whole turn was lost)."""
        session = await self._require_terminal(params)
        timeout = _resolve_terminal_wait_timeout()
        try:
            if timeout > 0:
                rc = await asyncio.wait_for(
                    session.process.wait(),
                    timeout=timeout,
                )
            else:
                rc = await session.process.wait()
        except asyncio.TimeoutError:
            # SIGTERM the process group; the next release call (or
            # this exception bubbling up) is responsible for cleanup.
            logger.warning(
                "terminal/wait_for_exit: process %s exceeded %.0fs; "
                "killing process group",
                session.process.pid,
                timeout,
            )
            try:
                os.killpg(
                    os.getpgid(session.process.pid),
                    signal.SIGTERM,
                )
            except (ProcessLookupError, PermissionError):
                pass
            try:
                rc = await asyncio.wait_for(
                    session.process.wait(),
                    timeout=5,
                )
            except asyncio.TimeoutError:
                try:
                    os.killpg(
                        os.getpgid(session.process.pid),
                        signal.SIGKILL,
                    )
                except (ProcessLookupError, PermissionError):
                    pass
                rc = await session.process.wait()
            raise AcpxHandlerError(
                code=_ERR_INTERNAL,
                message=(
                    f"terminal/wait_for_exit: process exceeded "
                    f"{timeout:.0f}s timeout (acpx_provider."
                    f"terminal_wait_seconds); killed"
                ),
            )
        # Make sure both drain tasks settle so subsequent .output
        # snapshots include the tail.
        for t in (session.drain_task, session.stderr_drain_task):
            if t is not None:
                try:
                    await asyncio.wait_for(t, timeout=2)
                except asyncio.TimeoutError:
                    t.cancel()
        if rc < 0:
            return {"exitCode": None, "signal": str(-rc)}
        return {"exitCode": rc, "signal": None}

    async def release(self, params: dict[str, Any]) -> dict[str, Any]:
        """Forget the terminal; if still running, send SIGTERM so the
        subprocess doesn't leak.  Idempotent."""
        terminal_id = params.get("terminalId")
        if not isinstance(terminal_id, str):
            raise AcpxHandlerError(
                code=_ERR_INVALID_PARAMS,
                message="terminal/release: missing 'terminalId'",
            )
        async with self._lock:
            session = self._terminals.pop(terminal_id, None)
        if session is None:
            return {}
        if session.process.returncode is None:
            # Terminate the entire process group: tool subprocesses
            # routinely fork (sh -c '...', npm scripts that spawn node,
            # etc.).  start_new_session=True at spawn time means we
            # can killpg the leader and reap the whole tree.
            _kill_terminal_pg(session.process, signal.SIGTERM)
            try:
                await asyncio.wait_for(session.process.wait(), timeout=5)
            except asyncio.TimeoutError:
                _kill_terminal_pg(session.process, signal.SIGKILL)
                await session.process.wait()
        for t in (session.drain_task, session.stderr_drain_task):
            if t is not None and not t.done():
                t.cancel()
        return {}

    # ----- Internal helpers ---------------------------------------- #

    async def _require_terminal(self, params: dict[str, Any]) -> _TerminalSession:
        terminal_id = params.get("terminalId")
        if not isinstance(terminal_id, str):
            raise AcpxHandlerError(
                code=_ERR_INVALID_PARAMS,
                message="missing 'terminalId'",
            )
        async with self._lock:
            session = self._terminals.get(terminal_id)
        if session is None:
            raise AcpxHandlerError(
                code=_ERR_TERMINAL_UNKNOWN,
                message=f"unknown terminal: {terminal_id}",
            )
        return session

    def _guard_or_deny(
        self,
        *,
        tool_name: str,
        tool_params: dict[str, Any],
    ) -> None:
        if _acpx_trust_mode_enabled():
            logger.warning(
                "acpx trust-mode bypass: %s allowed without guardian "
                "check (params=%s)",
                tool_name,
                _short_repr(tool_params),
            )
            return
        if self._guard_engine_factory is not None:
            engine = self._guard_engine_factory()
        else:
            from qwenpaw.security.tool_guard.engine import get_guard_engine

            engine = get_guard_engine()
        if engine is None:
            return
        result = engine.guard(tool_name, tool_params)
        if result is None:
            return
        if result.is_safe:
            return
        first = result.findings[0] if result.findings else None
        reason = first.title if first is not None else "guarded"
        raise AcpxHandlerError(
            code=_ERR_GUARDIAN_DENY,
            message=f"shell_guardian denied: {reason}",
        )


# ----------------------------------------------------------------- #
# Permission handler
# ----------------------------------------------------------------- #


class AcpxPermissionHandler:
    """Handle ``session/request_permission``.

    v1 strategy: auto-allow.  Rationale — the guardian engine has
    already gated every fs/* and terminal/* call by the time Claude
    might ask for permission; if the operation got this far the
    guardian decided it was safe.  Echoing "yes" in ACP avoids an
    unnecessary permission round-trip while preserving the audit
    log that fires inside the guardian.

    v2 (documented hook): swap in a real permission UI.  Easiest
    path is to push a CoPaw notification through the active channel
    (Console / WhatsApp / Signal) and block on the user's reply.
    Until then, choosing the first ``allow`` option keeps the agent
    moving.  Outright denials must come from the guardian, not from
    here.
    """

    async def request_permission(self, params: dict[str, Any]) -> dict[str, Any]:
        options = params.get("options") or []
        if not isinstance(options, list) or not options:
            # No options offered — ACP spec says we still must respond;
            # cancel is the safe default in that case.
            return {"outcome": {"outcome": "cancelled"}}

        # Find the first allow-flavoured option.  ACP convention is
        # ``kind ∈ {"allow_once", "allow_always", "reject_once",
        # "reject_always"}``; pick the first allow we see.
        chosen_id = None
        for opt in options:
            if not isinstance(opt, dict):
                continue
            kind = (opt.get("kind") or "").lower()
            if kind.startswith("allow"):
                chosen_id = opt.get("optionId")
                break
        if chosen_id is None:
            # No allow option present — fall back to first available.
            first = options[0]
            if isinstance(first, dict):
                chosen_id = first.get("optionId")

        if chosen_id is None:
            return {"outcome": {"outcome": "cancelled"}}
        return {"outcome": {"outcome": "selected", "optionId": chosen_id}}


# ----------------------------------------------------------------- #
# Wiring
# ----------------------------------------------------------------- #


_BYPASS_STARTUP_WARNED: bool = False


def register_handlers(daemon: "AcpxDaemon") -> None:
    """Wire handler instances into ``daemon`` so ACP requests from
    Claude get dispatched correctly.

    Idempotent: safe to call repeatedly.  Re-registration replaces
    earlier handlers, which is what tests want when they swap in
    a faked guard engine factory.
    """
    fs = AcpxFsHandlers()
    terminal = AcpxTerminalHandlers()
    permission = AcpxPermissionHandler()
    daemon.set_handler("fs/read_text_file", fs.read_text_file)
    daemon.set_handler("fs/write_text_file", fs.write_text_file)
    daemon.set_handler("terminal/create", terminal.create)
    daemon.set_handler("terminal/output", terminal.output)
    daemon.set_handler("terminal/wait_for_exit", terminal.wait_for_exit)
    daemon.set_handler("terminal/release", terminal.release)
    daemon.set_handler(
        "session/request_permission",
        permission.request_permission,
    )
    # Codex round-6 finding: surface the bypass at startup, not just
    # per-call.  One-time module-scoped flag so repeated handler
    # re-registration (test fixtures, daemon restarts) does not flood
    # the log.
    global _BYPASS_STARTUP_WARNED
    if _acpx_trust_mode_enabled() and not _BYPASS_STARTUP_WARNED:
        logger.warning(
            "acpx trust-mode bypass ENABLED at startup "
            "(COPAW_ACPX_SKIP_GUARDIAN=1): CoPaw ToolGuard short-"
            "circuited for fs/* and terminal/* AND Claude Code "
            "internal permission gate disabled via ACP set-mode "
            "bypassPermissions on every session.  USE ONLY in "
            "trusted single-user setups.",
        )
        _BYPASS_STARTUP_WARNED = True


# ----------------------------------------------------------------- #
# Module-level helpers
# ----------------------------------------------------------------- #


def _read_text(path: str) -> str:
    """Read a UTF-8 text file, replacement chars on decode error.
    Runs in a thread (called via :func:`asyncio.to_thread`).
    """
    return Path(path).read_text(encoding="utf-8", errors="replace")


def _write_text(path: str, content: str) -> None:
    """Write a file, creating parent dirs."""
    p = Path(path)
    if p.parent and not p.parent.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")


def _slice_lines(
    text: str,
    *,
    line: Any,
    limit: Any,
) -> str:
    """Apply ACP's ``line``/``limit`` windowing semantics.

    ACP says ``line`` is 1-indexed; we accept ``None`` (start of file)
    and any non-int as a no-op for that side.  ``limit`` similarly
    optional.
    """
    lines = text.splitlines(keepends=True)
    start = 0
    if isinstance(line, int) and line > 0:
        start = line - 1
    end: int | None = None
    if isinstance(limit, int) and limit > 0:
        end = start + limit
    return "".join(lines[start:end])


# Allowlist of parent-process env vars exposed to ``terminal/create``
# children.  Keep this minimal: anything Claude Code's spawned tool
# actually needs to function (PATH so binaries resolve, HOME for tool
# config dirs, USER/LOGNAME for whoami-style lookups, LANG/LC_*/TZ for
# locale, TERM/COLORTERM for tty-aware tools, TMPDIR for scratch
# files).  Crucially this excludes provider API keys — OPENAI_API_KEY,
# ANTHROPIC_API_KEY, AWS_*, GCP credentials etc. — that the parent
# CoPaw process holds and that a compromised or curious tool turn
# could otherwise exfiltrate via stdout.  ACP-specified env entries
# (per the env_raw list) are layered on top and can override anything
# in this baseline — except for the high-leverage names called out in
# :data:`_ENV_DENYLIST` below.
_ENV_ALLOWLIST: tuple[str, ...] = (
    "PATH",
    "HOME",
    "USER",
    "LOGNAME",
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "TERM",
    "COLORTERM",
    "TMPDIR",
    "TZ",
)


# Deny-list of ACP-supplied env names that would let the model hijack
# an otherwise-allowed command's semantics.  The terminal-create
# guardian inspects ``command``/``cwd`` only; an ACP entry like
# ``BASH_ENV=/tmp/x.sh`` or ``NODE_OPTIONS=--require=/tmp/x.js`` would
# slip code execution past that check.  Same goes for the dynamic-
# linker hooks (``LD_PRELOAD``, ``LD_LIBRARY_PATH``, ``DYLD_*``) which
# can override library resolution to inject behavior, and the various
# language-runtime ``*OPT`` / ``*PATH`` knobs that auto-execute or
# import on startup.  We log + drop these rather than raising so a
# benign pass-through with one bad entry still runs.
_ENV_DENYLIST: frozenset[str] = frozenset(
    {
        # POSIX shells.
        "BASH_ENV",
        "ENV",
        "PROMPT_COMMAND",
        # Dynamic linker overrides.
        "LD_PRELOAD",
        "LD_LIBRARY_PATH",
        "LD_AUDIT",
        # macOS dynamic linker overrides.
        "DYLD_INSERT_LIBRARIES",
        "DYLD_LIBRARY_PATH",
        "DYLD_FRAMEWORK_PATH",
        "DYLD_FALLBACK_LIBRARY_PATH",
        "DYLD_FALLBACK_FRAMEWORK_PATH",
        # Language runtime auto-execute / import hooks.
        "NODE_OPTIONS",
        "PYTHONSTARTUP",
        "PYTHONPATH",
        "RUBYOPT",
        "RUBYLIB",
        "PERL5OPT",
        "PERL5LIB",
        # Git config redirection (lets the model rewrite hooks/aliases).
        "GIT_CONFIG_GLOBAL",
        "GIT_CONFIG_SYSTEM",
        "GIT_CONFIG_COUNT",
        "GIT_SSH_COMMAND",
    },
)


def _kill_terminal_pg(
    proc: asyncio.subprocess.Process,
    sig: signal.Signals,
) -> None:
    """Send ``sig`` to the process group rooted at ``proc``.

    ``terminal/create`` spawns with ``start_new_session=True`` so the
    process is a session leader and ``killpg(pgid, sig)`` reaps both
    the leader and any forked workers in one call.  Falls back to a
    per-pid signal if the pgid is unreadable (process already gone or
    permission lost), or if ``killpg`` itself races with reap.
    """
    if proc.pid is None:
        return
    try:
        pgid = os.getpgid(proc.pid)
    except (ProcessLookupError, PermissionError):
        try:
            proc.send_signal(sig)
        except ProcessLookupError:
            pass
        return
    try:
        os.killpg(pgid, sig)
    except ProcessLookupError:
        # Group leader was reaped between getpgid and killpg.  Try a
        # per-pid signal as a best-effort fallback in case some grand-
        # child outlived the leader; ProcessLookupError there means
        # everything is already gone, which is the desired end state.
        try:
            proc.send_signal(sig)
        except ProcessLookupError:
            pass


def _build_env(env_raw: list[Any]) -> dict[str, str]:
    """ACP env is a list of ``{name, value}`` objects.

    Returns a minimal env: an allowlist projection of ``os.environ``
    (PATH, HOME, locale, terminal type, etc.) layered with the ACP-
    specified entries.  We deliberately do NOT inherit the full
    parent env — this process holds API keys (OPENAI_API_KEY,
    ANTHROPIC_API_KEY) that a tool-spawned subprocess has no business
    seeing.  ACP entries always win over the allowlist baseline,
    except for high-leverage names in :data:`_ENV_DENYLIST` (shell
    init hooks, dynamic-linker overrides, language-runtime auto-
    execute knobs) — those are dropped so the model can't hijack an
    otherwise-allowed command's semantics through env injection.
    """
    env_map: dict[str, str] = {}
    for key in _ENV_ALLOWLIST:
        val = os.environ.get(key)
        if val is not None:
            env_map[key] = val
    if not isinstance(env_raw, list):
        return env_map
    for item in env_raw:
        if not isinstance(item, dict):
            continue
        name = item.get("name")
        value = item.get("value")
        if not (isinstance(name, str) and isinstance(value, str)):
            continue
        if name in _ENV_DENYLIST:
            logger.warning(
                "acpx terminal/create: dropping ACP-supplied env %r "
                "(in deny-list — execution-hijack hook)",
                name,
            )
            continue
        env_map[name] = value
    return env_map


def _join_argv_for_guard(command: str, args: list[str]) -> str:
    """Best-effort reconstruction of a shell-style command string
    so :class:`ShellEvasionGuardian` sees something it can scan.

    We don't actually execute through a shell — :func:`create` uses
    :func:`asyncio.create_subprocess_exec` with explicit argv — so
    quoting issues here only affect what the guardian sees, not
    what runs.  Still, conservative quoting prevents the guardian
    from mis-parsing benign multi-word args as multi-tokens.
    """
    import shlex

    parts = [shlex.quote(command)]
    parts.extend(shlex.quote(a) for a in args)
    return " ".join(parts)


def _make_terminal_id() -> str:
    """Stable random-ish terminal ids.  Hex so they're URL-safe and
    short enough to fit in an ACP envelope without truncation.
    """
    return f"term_{secrets.token_hex(8)}"


async def _drain_stream(
    stream: asyncio.StreamReader | None,
    buf: bytearray,
    byte_limit: int,
) -> None:
    """Continuously read from ``stream`` into ``buf`` until EOF or
    the buffer hits ``byte_limit``.  Beyond the limit we keep reading
    (so the producer doesn't block on a full pipe) but discard the
    overflow — :meth:`AcpxTerminalHandlers.output` reports
    ``truncated=True`` when this happens.
    """
    if stream is None:
        return
    try:
        while True:
            chunk = await stream.read(4096)
            if not chunk:
                return
            remaining = byte_limit - len(buf)
            if remaining > 0:
                if len(chunk) <= remaining:
                    buf.extend(chunk)
                else:
                    buf.extend(chunk[:remaining])
            # Past byte_limit we silently drop to avoid backpressure
            # deadlocks from a producer that won't shut up.
    except (asyncio.CancelledError, BrokenPipeError, ConnectionResetError):
        return
    except Exception:  # noqa: BLE001 -- stream IO is noisy; protect the task.
        logger.debug("acpx terminal drain ended with exception", exc_info=True)
        return
