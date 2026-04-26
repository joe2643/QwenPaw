# -*- coding: utf-8 -*-
"""Claude Code (acpx) subprocess driver — Lane B.

Owns the per-turn ``acpx claude -s <name> --ttl <s>`` invocation that
Lane D's ``ClaudeAcpxChatModel`` wrapper feeds prompts into.  Multiplexes
across conversations by spawning a *separate* short-lived process for
each turn, while the underlying acpx queue owner (kept alive by
``--ttl``) persists Claude Code's ACP adapter between invocations so
Anthropic's prompt cache prefix stays warm.

Why per-turn spawn (not a sustained Python-managed daemon)
----------------------------------------------------------

Two paths considered:

A. **Per-turn spawn with TTL** (this file).  Each ``submit_turn``
   invocation does ``acpx claude -s <name> --ttl <ttl> --format json
   --json-strict``, writes the prompt to stdin, drains stdout JSON-
   RPC lines, dispatches client-side method requests through registered
   handlers, exits.  acpx's queue-owner process keeps the underlying
   ACP adapter alive ``--ttl`` seconds across calls — that's where
   the cache-warm property comes from, NOT from us holding the
   subprocess.

B. **Sustained Python-managed daemon** that all sessions multiplex
   through.  More plumbing (crash recovery, stdin/stdout race,
   per-session response routing).  Marginal latency win at best
   because acpx's process startup is ~50ms while LLM latency is
   measured in seconds.

We picked (A) because the contract Lane D imports is the same either
way: :meth:`AcpxDaemon.submit_turn` / :meth:`teardown` /
:meth:`run_set_config` / :meth:`shutdown`.  If real-world latency
profiling later shows fork/exec overhead matters, swap the
implementation behind this class without changing callers.  See the
plan's "Daemon strategy" section.

Hybrid mode bidirectional flow
------------------------------

ACP is bidirectional over the same stdin/stdout channel:

* ``session/update`` notifications stream from acpx → us.  The Lane A
  translator filters these into chat-completion deltas.
* The final ``session/prompt`` response (id-keyed ``result`` /
  ``error``) closes the turn.
* Mid-turn, **Claude can also send requests TO us** — these are
  ``{jsonrpc, id, method, params}`` lines for ``fs/read_text_file``,
  ``terminal/create``, ``session/request_permission``, etc.  Hybrid
  mode means CoPaw executes those (security-guarded) and writes the
  response back to acpx stdin.

The daemon transparently routes the latter into registered handler
callbacks via :func:`register_handlers` (see :mod:`claude_acpx_handlers`)
and only yields the former through to the translator.  Caller code
(Lane D) doesn't see Claude's requests at all.

acpx CLI surface (verified against 0.6.1, the version pinned in
:mod:`acpx_translate`):

* ``acpx claude -s <name> --format json --json-strict --ttl <s>`` —
  prompt is read from stdin (or args).  Stdout is newline-delimited
  ACP JSON-RPC.  ``--json-strict`` suppresses non-JSON stderr noise.
* ``acpx claude sessions close <name>`` — tear down a stateful session.
* ``acpx claude set -s <name> <key> <value>`` — push a session-scoped
  config option (e.g. thinking effort).
"""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
from typing import Any, AsyncIterator, Awaitable, Callable

from qwenpaw.providers import claude_acpx_metrics, acpx_translate

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------- #
# Tunables
# ----------------------------------------------------------------- #


# acpx queue-owner TTL.  10 minutes covers Anthropic's 5-minute cache
# TTL with comfortable margin for human turn cadence.  Keeping the
# queue owner alive between turns is THE WHOLE POINT of going stateful;
# a too-short TTL forces acpx to re-spawn the ACP adapter on every
# call, defeating the cache prefix.
_DEFAULT_TTL_SECONDS: int = 600

# Per-turn spawn timeout.  Real Claude Code calls can take 60+ seconds
# for thinking-heavy turns; cap at 5 minutes so a wedged subprocess
# doesn't pin the runner indefinitely.  Anything past this is treated
# as a daemon failure: kill, count restart, surface to caller.
_DEFAULT_TURN_TIMEOUT_SECONDS: float = 300.0


# ----------------------------------------------------------------- #
# Errors
# ----------------------------------------------------------------- #


class AcpxDaemonError(RuntimeError):
    """Raised when the underlying acpx subprocess fails to start,
    exits non-zero before the prompt completes, or otherwise wedges.
    Lane D's wrapper surfaces this to agentscope as a model error.
    """


# Handler signature.  Each registered handler is an async callable
# taking the request ``params`` dict and returning the response
# ``result`` dict — or raising to surface as a JSON-RPC error.
HandlerFn = Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]


# ----------------------------------------------------------------- #
# Daemon
# ----------------------------------------------------------------- #


class AcpxDaemon:
    """Per-turn acpx subprocess driver with bidirectional ACP routing.

    Process-singleton accessor :meth:`get_or_spawn`; the class itself
    is fine to instantiate directly in tests with a fake ``cmd_builder``.

    Path-A implementation: each :meth:`submit_turn` is independent —
    we spawn ``acpx claude -s <name> ...`` fresh, write the prompt,
    drain stdout while servicing any incoming Claude→client requests,
    exit.  ``--ttl`` keeps the underlying queue-owner warm across
    these spawns, which is where the cache-hit benefit comes from.
    """

    _GLOBAL: "AcpxDaemon | None" = None

    def __init__(
        self,
        *,
        ttl_seconds: int = _DEFAULT_TTL_SECONDS,
        turn_timeout_seconds: float = _DEFAULT_TURN_TIMEOUT_SECONDS,
        # Test seam: override how we materialise the argv tuple.
        # Production uses :func:`acpx_translate.stateful_acpx_cmd`.
        cmd_builder: Callable[[str], tuple[str, ...]] = (
            acpx_translate.stateful_acpx_cmd
        ),
    ) -> None:
        self._ttl_seconds = ttl_seconds
        self._turn_timeout = turn_timeout_seconds
        self._cmd_builder = cmd_builder
        # Method-name → handler.  Populated by
        # :func:`register_handlers` from claude_acpx_handlers.
        self._handlers: dict[str, HandlerFn] = {}
        # Set of in-flight processes so :meth:`shutdown` can kill them
        # rather than orphaning.
        self._inflight: set[asyncio.subprocess.Process] = set()
        self._closed = False

    # ----- Singleton accessor (Lane D entrypoint) ---------------- #

    @classmethod
    def get_or_spawn(cls) -> "AcpxDaemon":
        """Return the process-singleton instance, creating it lazily.

        "Spawn" is a misnomer in path A — we don't actually create
        a subprocess until the first :meth:`submit_turn` call.  The
        name matches the contract Lane D imports.
        """
        if cls._GLOBAL is None:
            cls._GLOBAL = cls()
        return cls._GLOBAL

    @classmethod
    def reset_singleton_for_test(cls) -> None:
        """Tests instantiate their own daemon; reset between cases."""
        cls._GLOBAL = None

    # ----- Handler registration ---------------------------------- #

    def set_handler(self, method: str, fn: HandlerFn) -> None:
        """Register a handler for a Claude→client ACP method.

        Called by :func:`register_handlers` in
        :mod:`claude_acpx_handlers`.  Re-registering replaces the
        prior handler.  Methods without a registered handler reply
        with an ACP "method not found" error.
        """
        self._handlers[method] = fn

    def has_handler(self, method: str) -> bool:
        return method in self._handlers

    # ----- Core: submit_turn ------------------------------------- #

    async def submit_turn(
        self,
        *,
        session_name: str,
        prompt_blocks: list[dict],
        is_seed: bool,  # noqa: ARG002 -- accepted for contract; per-turn spawn doesn't differentiate
    ) -> AsyncIterator[str]:
        """Push one ``session/prompt`` for ``session_name`` and yield
        ACP JSON-RPC lines from acpx stdout until the prompt
        terminates.  Lane D feeds the yielded lines into Lane A's
        :func:`acpx_translate.translate_acp_updates_to_chat_chunks`.

        Path A doesn't differentiate ``is_seed`` on the wire —
        ``acpx claude -s <name>`` auto-creates the session if it
        doesn't exist, and ships the stdin payload as the user
        prompt regardless.  We keep the parameter for forward-
        compatibility with a hypothetical sustained-daemon
        rewrite that would need to ``sessions/new`` first.

        Mid-turn, Claude→client requests (``fs/read_text_file``,
        ``terminal/create``, etc.) are intercepted, dispatched to
        registered handlers, and the response is written back to
        acpx's stdin.  Those requests are NOT yielded to the caller.

        Raises
        ------
        AcpxDaemonError
            If the binary is missing on PATH, the subprocess fails
            to start, or it exits before producing terminal output.
        """
        if self._closed:
            raise AcpxDaemonError("AcpxDaemon is shut down")
        if not _binary_available():
            raise AcpxDaemonError(
                "acpx binary not found on PATH (npx-shimmed install).  "
                "Run `npm i -g acpx` or ensure `npx` is available.",
            )

        text_payload = _payload_text(prompt_blocks)
        # The prompt rides as a positional CLI argument so stdin stays
        # free for ACP reply traffic (Claude→client requests like
        # ``fs/read_text_file`` are answered by writing JSON-RPC reply
        # envelopes back on the same stdin channel).  acpx accepts
        # ``[prompt...]`` positional args under ``acpx claude``.  Going
        # via ``-f -`` would need us to half-close stdin to signal EOF
        # of the prompt while keeping it open for replies, which the
        # Python asyncio.subprocess API doesn't expose without unsafe
        # tricks.  Argv route is simpler and matches the real CLI.
        cmd = (
            self._cmd_builder(session_name)
            + ("--ttl", str(self._ttl_seconds))
            + (text_payload,)
        )

        proc = await self._spawn(cmd)
        try:
            assert proc.stdin is not None
            # No stdin payload — prompt is in argv.  Stdin stays open
            # so handler replies (written by _dispatch_request) land
            # on the channel acpx is reading for ACP replies.
            async for line in self._stream_lines(proc):
                yield line
        finally:
            # Close stdin now (defensive — acpx is already done).
            try:
                if proc.stdin is not None and not proc.stdin.is_closing():
                    proc.stdin.close()
            except Exception:  # noqa: BLE001
                pass
            await self._reap(proc)

    # ----- Auxiliary ACP control surface -------------------------- #

    async def run_set_config(
        self,
        session_name: str,
        key: str,
        value: str,
    ) -> None:
        """Push ``acpx claude set -s <name> <key> <value>``.

        Used by Lane D for thinking-effort delta sync without
        re-shipping the prompt.  Returns when the subprocess exits.
        """
        if self._closed:
            raise AcpxDaemonError("AcpxDaemon is shut down")
        if not _binary_available():
            raise AcpxDaemonError("acpx binary not found on PATH")

        version = acpx_translate._PINNED_ACPX_VERSION
        cmd = (
            "npx",
            f"acpx@{version}",
            "claude",
            "set",
            "-s",
            session_name,
            key,
            value,
        )
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except FileNotFoundError as e:
            raise AcpxDaemonError(f"acpx spawn failed: {e}") from e

        try:
            _, stderr = await asyncio.wait_for(
                proc.communicate(),
                timeout=30,
            )
        except asyncio.TimeoutError as e:
            proc.kill()
            await proc.wait()
            raise AcpxDaemonError(
                f"acpx claude set timed out for {session_name}",
            ) from e

        if proc.returncode != 0:
            err_msg = (stderr or b"").decode("utf-8", errors="replace")
            raise AcpxDaemonError(
                f"acpx claude set {key}={value} failed (rc={proc.returncode}): "
                f"{err_msg[:500]}",
            )
        claude_acpx_metrics.record_effort_set()

    # ----- Tear-down / shutdown ----------------------------------- #

    async def teardown(self, session_name: str) -> None:
        """Close one session — wired into :class:`Registry`'s
        ``tear_down_cb`` slot.  Shells out to
        ``acpx claude sessions close <name>``.  Failures are logged
        and swallowed: the registry caller treats tear-down as best-
        effort, and acpx's own LRU eventually GCs orphan sessions on
        disk.
        """
        if self._closed:
            return
        if not _binary_available():
            logger.debug(
                "acpx not available; skipping teardown of %s",
                session_name,
            )
            return

        version = acpx_translate._PINNED_ACPX_VERSION
        cmd = (
            "npx",
            f"acpx@{version}",
            "claude",
            "sessions",
            "close",
            session_name,
        )
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except FileNotFoundError:
            logger.warning(
                "acpx teardown spawn failed (binary missing): %s",
                session_name,
            )
            return

        try:
            _, stderr = await asyncio.wait_for(
                proc.communicate(),
                timeout=15,
            )
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            logger.warning(
                "acpx claude sessions close timed out for %s",
                session_name,
            )
            return

        if proc.returncode != 0:
            err_msg = (stderr or b"").decode("utf-8", errors="replace")
            logger.warning(
                "acpx claude sessions close %s failed (rc=%s): %s",
                session_name,
                proc.returncode,
                err_msg[:300],
            )
        else:
            claude_acpx_metrics.record_tear_down()

    async def shutdown(self) -> None:
        """Stop the daemon — kill any in-flight subprocesses.  Called
        on agent stop / process exit.  Idempotent: subsequent calls
        and any further :meth:`submit_turn` raise.
        """
        self._closed = True
        # Snapshot because _reap mutates the set.
        for proc in list(self._inflight):
            if proc.returncode is None:
                try:
                    proc.kill()
                except ProcessLookupError:
                    pass
        for proc in list(self._inflight):
            try:
                await asyncio.wait_for(proc.wait(), timeout=5)
            except asyncio.TimeoutError:
                logger.warning(
                    "acpx subprocess %s did not exit after kill",
                    proc.pid,
                )
        self._inflight.clear()

    # ----- Internal helpers --------------------------------------- #

    async def _spawn(
        self,
        cmd: tuple[str, ...],
    ) -> asyncio.subprocess.Process:
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except FileNotFoundError as e:
            raise AcpxDaemonError(f"acpx spawn failed: {e}") from e
        self._inflight.add(proc)
        return proc

    async def _stream_lines(
        self,
        proc: asyncio.subprocess.Process,
    ) -> AsyncIterator[str]:
        """Yield decoded lines from ``proc.stdout`` until EOF or
        timeout, intercepting client-side requests for handler dispatch.

        For each line:
          - Try to decode as JSON.  If decode fails, pass through —
            Lane A's translator already tolerates malformed lines.
          - If the message has both ``method`` and ``id``, it's a
            Claude→client request.  Dispatch to the registered handler,
            write the JSON-RPC response back to acpx stdin.  Do NOT
            yield (the translator doesn't care about these).
          - Otherwise it's a notification or response — yield through.
        """
        assert proc.stdout is not None
        deadline = asyncio.get_event_loop().time() + self._turn_timeout
        while True:
            remaining = deadline - asyncio.get_event_loop().time()
            if remaining <= 0:
                claude_acpx_metrics.record_error()
                raise AcpxDaemonError(
                    f"acpx subprocess (pid={proc.pid}) exceeded "
                    f"{self._turn_timeout}s turn timeout",
                )
            try:
                raw = await asyncio.wait_for(
                    proc.stdout.readline(),
                    timeout=remaining,
                )
            except asyncio.TimeoutError as e:
                claude_acpx_metrics.record_error()
                raise AcpxDaemonError(
                    f"acpx subprocess (pid={proc.pid}) stalled past "
                    f"{self._turn_timeout}s",
                ) from e
            if not raw:
                # EOF.  Either acpx terminated cleanly (final
                # ``stopReason`` should already have been emitted and
                # consumed by the translator) or it died without
                # producing one.  Both cases: stop yielding; caller's
                # reap-loop checks rc.
                return

            line = raw.decode("utf-8", errors="replace")

            if not self._maybe_dispatch_request(proc, line):
                yield line

    def _maybe_dispatch_request(
        self,
        proc: asyncio.subprocess.Process,
        line: str,
    ) -> bool:
        """If ``line`` is a Claude→client JSON-RPC request, dispatch
        it on a background task and return True.  Otherwise return
        False so the caller yields the line as-is.

        Dispatch is fire-and-forget because:
          - Handler responses are written back to acpx stdin
            independently of the stdout reader.
          - Blocking the stdout drain on handler completion would
            risk deadlocking when a handler depends on more stdout
            (which doesn't happen today, but the constraint is cheap).
        """
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            return False
        if not isinstance(msg, dict):
            return False
        method = msg.get("method")
        msg_id = msg.get("id")
        if not isinstance(method, str) or msg_id is None:
            return False
        # Notifications (method-only, no id) — translator handles
        # ``session/update`` etc., not us.
        # Responses (id + result/error, no method) — translator final.
        # Only requests (method + id + ideally params) reach here.

        params = msg.get("params") or {}
        asyncio.create_task(
            self._dispatch_request(proc, method, msg_id, params),
        )
        return True

    async def _dispatch_request(
        self,
        proc: asyncio.subprocess.Process,
        method: str,
        msg_id: Any,
        params: dict[str, Any],
    ) -> None:
        """Run the registered handler for ``method`` and write the
        JSON-RPC response back to acpx stdin.  Errors map to JSON-RPC
        error envelopes.
        """
        handler = self._handlers.get(method)
        if handler is None:
            await self._reply_error(
                proc,
                msg_id,
                code=-32601,
                message=f"method not found: {method}",
            )
            return

        try:
            result = await handler(params)
        except _HandlerError as e:
            await self._reply_error(proc, msg_id, code=e.code, message=str(e))
            return
        except Exception as e:  # noqa: BLE001
            logger.exception(
                "acpx handler for %s raised; replying with -32000",
                method,
            )
            await self._reply_error(
                proc,
                msg_id,
                code=-32000,
                message=f"handler error: {e}",
            )
            return

        await self._reply_result(proc, msg_id, result)

    async def _reply_result(
        self,
        proc: asyncio.subprocess.Process,
        msg_id: Any,
        result: dict[str, Any],
    ) -> None:
        envelope = {"jsonrpc": "2.0", "id": msg_id, "result": result}
        await self._write_line(proc, envelope)

    async def _reply_error(
        self,
        proc: asyncio.subprocess.Process,
        msg_id: Any,
        *,
        code: int,
        message: str,
    ) -> None:
        envelope = {
            "jsonrpc": "2.0",
            "id": msg_id,
            "error": {"code": code, "message": message},
        }
        await self._write_line(proc, envelope)

    @staticmethod
    async def _write_line(
        proc: asyncio.subprocess.Process,
        obj: dict[str, Any],
    ) -> None:
        if proc.stdin is None or proc.stdin.is_closing():
            logger.debug("acpx stdin closed; dropping reply %s", obj.get("id"))
            return
        try:
            payload = json.dumps(obj, separators=(",", ":")).encode("utf-8") + b"\n"
            proc.stdin.write(payload)
            await proc.stdin.drain()
        except (BrokenPipeError, ConnectionResetError):
            logger.debug("acpx stdin broken; dropping reply %s", obj.get("id"))

    async def _reap(self, proc: asyncio.subprocess.Process) -> None:
        """Wait on ``proc`` and discard.  Logs non-zero rc as an
        error counter so ops can spot a failing acpx pinned version
        without parsing tracebacks.
        """
        self._inflight.discard(proc)
        if proc.returncode is None:
            try:
                await asyncio.wait_for(proc.wait(), timeout=5)
            except asyncio.TimeoutError:
                logger.warning(
                    "acpx subprocess (pid=%s) did not exit; killing",
                    proc.pid,
                )
                try:
                    proc.kill()
                except ProcessLookupError:
                    pass
                await proc.wait()

        if proc.returncode and proc.returncode != 0:
            stderr_bytes = b""
            if proc.stderr is not None:
                try:
                    stderr_bytes = await asyncio.wait_for(
                        proc.stderr.read(),
                        timeout=2,
                    )
                except asyncio.TimeoutError:
                    pass
            err_msg = stderr_bytes.decode("utf-8", errors="replace")
            logger.warning(
                "acpx subprocess (pid=%s) exited rc=%s: %s",
                proc.pid,
                proc.returncode,
                err_msg[:500],
            )
            claude_acpx_metrics.record_error()


# ----------------------------------------------------------------- #
# Module-level helpers
# ----------------------------------------------------------------- #


class _HandlerError(Exception):
    """Internal alias for the handler-side error type so the daemon
    doesn't import :mod:`claude_acpx_handlers` (avoiding circular
    imports).  Handlers raise their own :class:`AcpxHandlerError`
    which is a subclass of this.
    """

    code: int = -32000

    def __init__(self, *, code: int, message: str) -> None:
        super().__init__(message)
        self.code = code


def _binary_available() -> bool:
    """``npx`` is the gateway: even when ``acpx`` itself isn't
    globally installed, ``npx acpx@<version>`` resolves it via the npm
    registry on first call.  Test by probing ``npx`` only.
    """
    return shutil.which("npx") is not None


def _payload_text(prompt_blocks: list[dict]) -> str:
    """Collapse ContentBlock[] → single stdin payload.  Multimodal
    blocks fold to placeholders via :func:`acpx_translate._content_text`
    — same lossy path the legacy stateless driver uses.

    Empty payloads become ``"(empty prompt)"`` so acpx doesn't see a
    zero-byte stdin and emit a parse error.
    """
    text_payload = "\n\n".join(
        acpx_translate._content_text(b)
        for b in prompt_blocks
        if acpx_translate._content_text(b)
    )
    return text_payload or "(empty prompt)"
