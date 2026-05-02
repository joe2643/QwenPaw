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
  config option (e.g. effort).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import signal
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

# Threshold (bytes of UTF-8 encoded prompt) above which we route the
# prompt through ``acpx claude -f <path>`` instead of an argv string.
# System ARG_MAX is typically 2 MB on Linux but the budget is shared
# with environment variables AND the entire argv tuple — and on cold-
# mint/seed_full a long WhatsApp group history easily exceeds that
# combined budget, producing ``OSError: [Errno 7] Argument list too
# long`` (observed 2026-05-02). Routing through a tempfile sidesteps
# the kernel exec budget entirely. 64 KB keeps the fast argv path for
# the common ship_tail case (a turn or two) while routing every
# realistic seed_full through the file path with comfortable headroom.
_ARGV_PROMPT_THRESHOLD: int = 64 * 1024

# Per-line buffer cap re-exported from ``acpx_translate`` so both
# spawn sites (daemon-driven stateful + legacy stateless one-shot)
# share one source of truth.
from qwenpaw.providers.acpx_translate import (
    _STDOUT_LINE_LIMIT,  # noqa: F401  re-exported for back-compat
)


# ----------------------------------------------------------------- #
# Permission bypass (``COPAW_ACPX_SKIP_GUARDIAN``)
# ----------------------------------------------------------------- #
#
# When the user has opted in to ``COPAW_ACPX_SKIP_GUARDIAN=1``, two
# layers need to be neutralised:
#
# 1. **CoPaw's tool guard** (``claude_acpx_handlers``) — the existing
#    ``_acpx_trust_mode_enabled`` check short-circuits ``fs/*`` and
#    ``terminal/*`` ACP requests before they hit the guardian.
#
# 2. **Claude Code's *internal* permission flow** (this file).  Even
#    with our handlers wide open, the Claude Code SDK refuses Write/
#    Bash by default and never even sends ``session/request_permission``
#    over ACP — the deny happens entirely inside the SDK.  Symptom:
#    Claude narrates "Write/Bash got denied" in its reply text but no
#    ``request_permission``/``fs/write_text_file``/``terminal/create``
#    traffic appears in our logs.  The SDK gates this on the session's
#    permission *mode* — ``default`` asks (and the ask is suppressed
#    by claude-agent-acp for some tool families), ``bypassPermissions``
#    skips entirely.
#
# Fix: after ``acpx claude sessions ensure`` succeeds, run
# ``acpx claude set-mode -s <name> bypassPermissions`` so the on-disk
# session is in bypassPermissions mode for every subsequent prompt.
# The mode persists across queue-owner restarts because ``setMode`` in
# acpx defaults to ``sessionMode="persistent"``.
#
# Same gate as the handler-side trust mode so a single env var controls
# the whole bypass surface.
_ACPX_TRUST_ENV_VAR: str = "COPAW_ACPX_SKIP_GUARDIAN"
_ACPX_BYPASS_MODE_ID: str = "bypassPermissions"
_TRUTHY = frozenset({"1", "true", "yes", "on", "y", "t"})


def _acpx_trust_mode_enabled() -> bool:
    raw = os.environ.get(_ACPX_TRUST_ENV_VAR, "")
    return raw.strip().lower() in _TRUTHY


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
        # Production uses :func:`acpx_translate.stateful_acpx_cmd`
        # with ttl + cwd parameters.
        cmd_builder: Callable[..., tuple[str, ...]] = (
            acpx_translate.stateful_acpx_cmd
        ),
        # Test seam: production always runs ``acpx claude sessions
        # ensure --name <name>`` before the first prompt for a
        # session, but unit tests use script-based fake binaries that
        # don't model that surface.  Setting False makes submit_turn
        # skip the ensure round trip entirely.
        auto_ensure_session: bool = True,
    ) -> None:
        self._ttl_seconds = ttl_seconds
        self._turn_timeout = turn_timeout_seconds
        self._cmd_builder = cmd_builder
        self._auto_ensure_session = auto_ensure_session
        # Method-name → handler.  Populated by
        # :func:`register_handlers` from claude_acpx_handlers.
        self._handlers: dict[str, HandlerFn] = {}
        # Set of in-flight processes so :meth:`shutdown` can kill them
        # rather than orphaning.
        self._inflight: set[asyncio.subprocess.Process] = set()
        # Sessions we've already shelled out ``sessions ensure`` for
        # this process — acpx requires the named session to exist on
        # disk before ``-s <name>`` will route a prompt to it.  See
        # smoke test 2026-04-27.  Per-process cache; a daemon restart
        # re-runs ensure (idempotent on disk).
        self._ensured_sessions: set[str] = set()
        # Sessions for which we've already pushed
        # ``set-mode bypassPermissions`` this process — only populated
        # when ``COPAW_ACPX_SKIP_GUARDIAN`` is truthy.  acpx persists
        # the mode on disk so re-running on a daemon restart is
        # cheap-but-redundant; the per-process cache avoids the round
        # trip on the warm path.
        self._bypass_set_sessions: set[str] = set()
        self._ensure_lock = asyncio.Lock()
        self._closed = False
        # Strong refs for fire-and-forget handler-dispatch tasks.
        # Without this set the loop only weakrefs the task and the GC
        # can collect it mid-flight, leaving acpx waiting on a JSON-RPC
        # response that never arrives.  Tasks self-discard on done.
        self._pending_dispatches: set[asyncio.Task[Any]] = set()

    # ----- Singleton accessor (Lane D entrypoint) ---------------- #

    @classmethod
    def get_or_spawn(cls) -> "AcpxDaemon":
        """Return the process-singleton instance, creating it lazily.

        "Spawn" is a misnomer in path A — we don't actually create
        a subprocess until the first :meth:`submit_turn` call.  The
        name matches the contract Lane D imports.

        On first construction we also wire the ACP handler set
        (fs/read_text_file, fs/write_text_file, terminal/*,
        session/request_permission) so any tool-using turn sees the
        registered handler instead of a -32601 ``method not found``
        bounce.  Import is deferred to break the
        daemon ↔ handlers ↔ daemon cycle at module load time.
        """
        if cls._GLOBAL is None:
            inst = cls()
            from .claude_acpx_handlers import register_handlers

            register_handlers(inst)
            cls._GLOBAL = inst
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

        # acpx requires the session to exist on disk before ``-s
        # <name>`` works — otherwise the JSON-RPC reply is a
        # ``NO_SESSION`` error.  Idempotent shell-out, cached per
        # process so repeat calls in the same daemon don't pay the
        # round trip.  Unit tests opt out via ``auto_ensure_session=
        # False`` because their fake binaries don't model the ensure
        # subcommand.
        if self._auto_ensure_session:
            await self._ensure_session(session_name)

        text_payload = _payload_text(prompt_blocks)
        # Two prompt-delivery modes:
        #
        # 1. **Short prompt → argv positional.** The prompt rides as a
        #    positional CLI argument under ``acpx claude [prompt...]``
        #    so stdin stays free for ACP reply traffic (Claude→client
        #    requests like ``fs/read_text_file`` are answered by
        #    writing JSON-RPC reply envelopes back on the same stdin
        #    channel). Going via ``-f -`` would need us to half-close
        #    stdin to signal EOF of the prompt while keeping it open
        #    for replies, which the Python asyncio.subprocess API
        #    doesn't expose without unsafe tricks.
        #
        # 2. **Long prompt → ``-f <tempfile>``.** Linux ARG_MAX caps
        #    the combined size of argv + envp; cold-mint/seed_full
        #    routinely exceeds it for WhatsApp group histories
        #    (observed 2026-05-02 ``OSError: [Errno 7] Argument list
        #    too long``). A tempfile keeps argv tiny regardless of
        #    history size, while stdin stays free for ACP replies.
        #
        # ttl rides in the global-options slot via ``cmd_builder`` so
        # acpx's queue-owner inherits it; appending after the
        # subcommand triggers ``error: unknown option --ttl`` (smoke
        # test 2026-04-27).
        encoded_payload = text_payload.encode("utf-8")
        cmd_prefix = self._cmd_builder(
            session_name,
            ttl_seconds=self._ttl_seconds,
        )
        prompt_file: str | None = None
        if len(encoded_payload) > _ARGV_PROMPT_THRESHOLD:
            # Tempfile path: write payload to disk, pass ``-f <path>``.
            # We deliberately do NOT use ``-f -`` (stdin) because the
            # daemon needs stdin for client-side ACP request replies.
            import tempfile  # local import — only seed_full hits this

            fd, prompt_file = tempfile.mkstemp(
                prefix="qwenpaw-acpx-prompt-",
                suffix=".txt",
            )
            try:
                with os.fdopen(fd, "wb") as f:
                    f.write(encoded_payload)
            except Exception:
                # If write itself fails, clean up before re-raising —
                # otherwise we'd leak the tempfile.
                try:
                    os.unlink(prompt_file)
                except Exception:  # noqa: BLE001
                    pass
                raise
            cmd = cmd_prefix + ("-f", prompt_file)
            logger.info(
                "acpx submit_turn: prompt %d bytes > %d threshold; "
                "routed via tempfile %s",
                len(encoded_payload),
                _ARGV_PROMPT_THRESHOLD,
                prompt_file,
            )
        else:
            cmd = cmd_prefix + (text_payload,)

        # Outer try/finally ensures the tempfile is unlinked even if
        # ``_spawn`` itself raises (binary missing on PATH, exec
        # failure) — earlier the cleanup was only inside the
        # post-spawn try block, so a spawn-time exception leaked
        # the tempfile.
        proc: asyncio.subprocess.Process | None = None
        try:
            proc = await self._spawn(cmd)
            assert proc.stdin is not None
            # No stdin payload — prompt is either in argv or in the
            # tempfile passed via ``-f``.  Stdin stays open so handler
            # replies (written by _dispatch_request) land on the channel
            # acpx is reading for ACP replies.
            async for line in self._stream_lines(proc):
                yield line
        finally:
            if proc is not None:
                # Close stdin now (defensive — acpx is already done).
                try:
                    if (
                        proc.stdin is not None
                        and not proc.stdin.is_closing()
                    ):
                        proc.stdin.close()
                except Exception:  # noqa: BLE001
                    pass
                await self._reap(proc)
            if prompt_file is not None:
                try:
                    os.unlink(prompt_file)
                except FileNotFoundError:
                    pass
                except Exception as e:  # noqa: BLE001
                    logger.warning(
                        "acpx submit_turn: failed to unlink prompt "
                        "tempfile %s: %s",
                        prompt_file,
                        e,
                    )

    # ----- Auxiliary ACP control surface -------------------------- #

    async def run_set_config(
        self,
        session_name: str,
        key: str,
        value: str,
    ) -> None:
        """Push ``acpx claude set -s <name> <key> <value>``.

        Used by Lane D for effort delta sync without
        re-shipping the prompt.  Returns when the subprocess exits.

        Calls :meth:`_ensure_session` first because Lane D may invoke
        this before the first ``submit_turn`` for a fresh session —
        without ensure, acpx replies ``rc=4 NO_SESSION`` and the
        effort delta silently gets dropped (Lane D's wrapper logs a
        warning but proceeds with whatever effort acpx defaulted to,
        which is wrong when the user explicitly chose ``high``).
        Smoke test 2026-04-27 caught this.
        """
        if self._closed:
            raise AcpxDaemonError("AcpxDaemon is shut down")
        if not _binary_available():
            raise AcpxDaemonError("acpx binary not found on PATH")
        if self._auto_ensure_session:
            await self._ensure_session(session_name)

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
                start_new_session=True,
            )
        except FileNotFoundError as e:
            raise AcpxDaemonError(f"acpx spawn failed: {e}") from e

        try:
            _, stderr = await asyncio.wait_for(
                proc.communicate(),
                timeout=30,
            )
        except asyncio.TimeoutError as e:
            _kill_proc_tree(proc)
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
        # Drop from the ensured cache regardless of whether the close
        # call lands — re-mint with a fresh name still pays the
        # ``sessions ensure`` round trip but stays correct.
        self._ensured_sessions.discard(session_name)
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
                start_new_session=True,
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
            _kill_proc_tree(proc)
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
        """Stop the daemon — kill any in-flight subprocesses + cancel
        any in-flight ACP request dispatches.  Called on agent stop /
        process exit.  Idempotent: subsequent calls and any further
        :meth:`submit_turn` raise.
        """
        self._closed = True
        # Cancel handler-dispatch tasks first so a slow handler (e.g.
        # ``terminal/wait_for_exit``) doesn't outlive the subprocess
        # whose stdin it was supposed to reply to.  Without this they'd
        # be retained by ``_pending_dispatches`` forever — strong-ref
        # cleanup keeps the GC away but doesn't unblock a stuck await.
        pending = list(self._pending_dispatches)
        for task in pending:
            if not task.done():
                task.cancel()
        # Snapshot because _reap mutates the set.
        for proc in list(self._inflight):
            if proc.returncode is None:
                _kill_proc_tree(proc)
        for proc in list(self._inflight):
            try:
                await asyncio.wait_for(proc.wait(), timeout=5)
            except asyncio.TimeoutError:
                logger.warning(
                    "acpx subprocess %s did not exit after kill",
                    proc.pid,
                )
        # Drain cancelled dispatch tasks so they exit cleanly before
        # we return.  Use return_exceptions so a CancelledError from
        # any task doesn't propagate out of shutdown().
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        self._inflight.clear()

    # ----- Internal helpers --------------------------------------- #

    async def _ensure_session(self, session_name: str) -> None:
        """Idempotent ``acpx claude sessions ensure --name <name>``.

        Lazy-cached per-process: the second call for the same name
        is a no-op.  acpx itself is also idempotent on this command
        (returns ``(exists)`` instead of ``(created)`` when the
        session is already on disk), so re-running on a daemon
        restart is safe.

        Failures raise :class:`AcpxDaemonError`; the caller bubbles
        up rather than continuing to a NO_SESSION error from the
        prompt path.
        """
        if session_name in self._ensured_sessions:
            return
        async with self._ensure_lock:
            if session_name in self._ensured_sessions:
                return
            version = acpx_translate._PINNED_ACPX_VERSION
            cmd = (
                "npx",
                f"acpx@{version}",
                "claude",
                "sessions",
                "ensure",
                "--name",
                session_name,
            )
            try:
                proc = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdin=asyncio.subprocess.DEVNULL,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    start_new_session=True,
                )
            except FileNotFoundError as e:
                raise AcpxDaemonError(
                    f"acpx spawn failed for sessions ensure: {e}",
                ) from e
            try:
                _, stderr = await asyncio.wait_for(
                    proc.communicate(),
                    timeout=30,
                )
            except asyncio.TimeoutError as e:
                _kill_proc_tree(proc)
                await proc.wait()
                raise AcpxDaemonError(
                    f"acpx claude sessions ensure timed out for "
                    f"{session_name}",
                ) from e
            if proc.returncode != 0:
                err_msg = (stderr or b"").decode("utf-8", errors="replace")
                raise AcpxDaemonError(
                    f"acpx claude sessions ensure --name {session_name} "
                    f"failed (rc={proc.returncode}): {err_msg[:500]}",
                )
            self._ensured_sessions.add(session_name)
        # Outside the ensure lock — set-mode talks to a different
        # subprocess and uses its own lock-free per-process cache.
        if _acpx_trust_mode_enabled():
            await self._set_session_bypass_mode(session_name)

    async def _set_session_bypass_mode(self, session_name: str) -> None:
        """Push ``acpx claude set-mode -s <name> bypassPermissions``.

        Idempotent on disk (acpx overwrites any prior mode) and cached
        per process so we don't shell out on every turn.  Failure is
        logged at WARNING but does NOT block the prompt — the worst
        case is that Claude Code self-denies a tool call, which is
        the *current* behaviour without this fix.  We don't want to
        wedge a turn on a transient acpx hiccup just because the
        bypass flip didn't take.
        """
        if session_name in self._bypass_set_sessions:
            return
        version = acpx_translate._PINNED_ACPX_VERSION
        cmd = (
            "npx",
            f"acpx@{version}",
            "claude",
            "set-mode",
            "-s",
            session_name,
            _ACPX_BYPASS_MODE_ID,
        )
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                start_new_session=True,
            )
        except FileNotFoundError as e:
            logger.warning(
                "acpx set-mode bypassPermissions: spawn failed for "
                "session %s: %s",
                session_name,
                e,
            )
            return
        try:
            _, stderr = await asyncio.wait_for(
                proc.communicate(),
                timeout=30,
            )
        except asyncio.TimeoutError:
            _kill_proc_tree(proc)
            await proc.wait()
            logger.warning(
                "acpx set-mode bypassPermissions: timed out for "
                "session %s",
                session_name,
            )
            return
        if proc.returncode != 0:
            err_msg = (stderr or b"").decode("utf-8", errors="replace")
            logger.warning(
                "acpx set-mode bypassPermissions: rc=%s for session "
                "%s: %s",
                proc.returncode,
                session_name,
                err_msg[:500],
            )
            return
        # Mark cached so we don't re-shell on every turn.  Log at
        # WARNING (not INFO) so the bypass shows up in any prod scan
        # for "elevated permissions in effect" — codex round-6 finding.
        self._bypass_set_sessions.add(session_name)
        logger.warning(
            "acpx set-mode bypassPermissions: ACTIVE for session %s "
            "(COPAW_ACPX_SKIP_GUARDIAN=1) — Claude Code internal "
            "permission gate disabled for this session",
            session_name,
        )

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
                start_new_session=True,
                # Bump the ``StreamReader`` per-line buffer above the
                # asyncio default (64 KB) so a single big ACP JSON-RPC
                # message — typically a ``tool_call_update`` carrying
                # a large file Read result — does not blow up with
                # ``ValueError: Separator is not found, and chunk
                # exceed the limit``.  See ``_STDOUT_LINE_LIMIT``.
                limit=_STDOUT_LINE_LIMIT,
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
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self._turn_timeout
        while True:
            remaining = deadline - loop.time()
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
        # Strong-ref the dispatch task in _pending_dispatches; the loop
        # only weakrefs scheduled tasks per Python docs, and GCing a
        # mid-flight handler would silently drop acpx's request.
        task = asyncio.create_task(
            self._dispatch_request(proc, method, msg_id, params),
        )
        self._pending_dispatches.add(task)
        task.add_done_callback(self._pending_dispatches.discard)
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
                _kill_proc_tree(proc)
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


_NPX_AVAILABLE_CACHE: bool | None = None


def _kill_proc_tree(proc: asyncio.subprocess.Process) -> None:
    """SIGKILL the entire process group rooted at ``proc``.

    Spawns are made with ``start_new_session=True`` so the process is
    a session leader; ``killpg(pgid, SIGKILL)`` reaps both the npm/npx
    wrapper and the node/acpx grandchildren in one call.  A bare
    ``proc.kill()`` only kills the session-leader PID, leaving the
    actual workers as orphans holding stdio pipes.

    Falls back to ``proc.kill()`` if the process is already gone or
    we can't read its pgid.
    """
    if proc.pid is None:
        return
    try:
        pgid = os.getpgid(proc.pid)
    except (ProcessLookupError, PermissionError):
        # Already reaped, or we lost ownership — best-effort fall back.
        try:
            proc.kill()
        except ProcessLookupError:
            pass
        return
    try:
        os.killpg(pgid, signal.SIGKILL)
    except ProcessLookupError:
        # Group leader was reaped between getpgid and killpg.  Try a
        # per-pid kill as a fallback for any grandchild that outlived
        # the leader.
        try:
            proc.kill()
        except ProcessLookupError:
            pass


def _binary_available() -> bool:
    """``npx`` is the gateway: even when ``acpx`` itself isn't
    globally installed, ``npx acpx@<version>`` resolves it via the npm
    registry on first call.  Test by probing ``npx`` only.

    The result is cached for the process lifetime — ``npx`` doesn't
    appear/disappear at runtime, and ``shutil.which`` walks PATH on
    every call.  Tests that need to flip the answer can reset
    ``_NPX_AVAILABLE_CACHE`` directly.
    """
    global _NPX_AVAILABLE_CACHE  # noqa: PLW0603
    if _NPX_AVAILABLE_CACHE is None:
        _NPX_AVAILABLE_CACHE = shutil.which("npx") is not None
    return _NPX_AVAILABLE_CACHE


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
