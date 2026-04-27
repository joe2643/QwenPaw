# -*- coding: utf-8 -*-
"""Claude Code (acpx) chat-completions wrapper — Lane D wiring.

CoPaw provider that routes ``chat.completions.create`` calls through
``acpx`` — a CLI tool speaking the Agent Client Protocol (ACP) to the
Claude Code IDE-grade agent.  Stateful per-conversation sessions reuse
Claude Code's prompt cache so the cache prefix stays stable across
turns (vs the stateless Anthropic-direct path).

Hybrid tool-execution mode (decided 2026-04-26 in the design doc):

* Claude Code proposes tools via ACP ``tool_call`` notifications;
* CoPaw EXECUTES tools via ACP client-side ``fs/*`` + ``terminal/*``
  method handlers, routing through CoPaw's existing MCP / security
  guardian / tool dispatch stack;
* Permission flow integrates with CoPaw's existing security guardians;
* Tool results flow back to Claude via ACP ``tool_call_update`` with
  status=completed and content arrays — and ALSO surface in CoPaw's
  normal logging/UI path.

Wiring map:

* :func:`_wrapped_create` is installed by :meth:`_install_wrapper` and
  replaces the OpenAI SDK's HTTP path entirely.
* Per turn:
    1. Read ``agent_id`` / ``session_id`` from CoPaw ContextVars.
    2. Compute env_hash over (system, tools, cwd, perm, generate_kwargs).
    3. Call :meth:`Registry.plan_turn` for a ``seed_full`` or
       ``ship_tail`` ``ShipPlan``.
    4. Render prompt blocks via ``render_history_for_seed`` /
       ``extract_tail_from_history``.
    5. If thinking-effort changed since last turn, push ``acpx claude
       set effort <level>`` via :meth:`AcpxDaemon.run_set_config`.
    6. Acquire ``entry.lock`` and submit the prompt through
       :meth:`AcpxDaemon.submit_turn`; pipe stdout into Lane A's
       :func:`translate_acp_updates_to_chat_chunks`.
    7. On clean completion, :meth:`Registry.commit_turn` advances
       ``last_shipped_idx`` / ``last_msg_chain_hash`` and the lock
       releases.

Mirrors :class:`qwenpaw.providers.codex_oauth_model.CodexOAuthChatModel`
in shape — both subclass :class:`OpenAIChatModel` so agentscope's
existing response parsing keeps working unchanged.  Streaming returns
an :class:`_AcpxStreamAdapter`; non-streaming drains the same pipeline
into a single :class:`ChatCompletion`.

V1 limitations (2026-04-27)
---------------------------

These are accepted tradeoffs for v1 — surface to operators/users so
they don't run into them blind:

* **Daemon restart loses cache.**  The session registry lives only in
  memory; on qwenpaw daemon restart the prior on-disk acpx session
  is orphaned (visible via ``acpx claude sessions list``) and the
  next chat turn cold-mints a new one.  Acpx's own LRU eventually
  reaps the orphans.  Future work: persist registry state so a
  restart can re-attach.

* **Agent rename leaks slowly.**  The registry keys by ``agent_id``;
  if an agent is renamed (rather than replaced via env_hash change),
  the old session lingers until the registry's LRU (cap=200) evicts
  it.  Not active corruption — just slower cleanup than env_hash
  change paths.

* **Multimodal degraded.**  v1 collapses non-text prompt blocks
  (images, audio, resources) to plain-text placeholders before
  shipping to acpx.  Users sending images through the claude-acpx
  provider will silently get degraded context vs a direct Anthropic
  path.  Lane B's content-block translation is the v2 fix.

* **Cancellation-after-effort-set divergence (rare).**  In ``_open``,
  effort is pushed to acpx + recorded in the registry BEFORE
  ``submit_turn`` is called.  If a cancel fires between those two,
  registry believes the new effort is in place but the session was
  never used.  The next turn's effort-delta check then short-circuits
  — Claude inherits the (correct) effort it was set to, so no real
  divergence.  Listed for completeness.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, AsyncIterator

from agentscope.model import OpenAIChatModel
from openai.types.chat import ChatCompletion, ChatCompletionChunk

from . import claude_acpx_metrics
from .acpx_translate import (
    StreamState,
    collect_as_chat_completion,
    extract_tail_from_history,
    render_history_for_seed,
    translate_acp_updates_to_chat_chunks,
)
from .claude_acpx_daemon import AcpxDaemon, AcpxDaemonError
from .claude_acpx_session_registry import (
    AcpxSessionEntry,
    Registry,
    env_hash as compute_env_hash,
    get_registry,
)

logger = logging.getLogger(__name__)


# =========================================================================
# Helpers — env_hash inputs, effort detection, registry wiring
# =========================================================================


def _wire_registry_tear_down(registry: Registry, daemon: AcpxDaemon) -> None:
    """Bind the global registry's tear-down callback to the daemon's
    ``acpx claude sessions close`` shell-out, on first use.  Idempotent
    — the marker attribute prevents repeated re-binds when several
    :class:`ClaudeAcpxChatModel` instances co-exist (e.g. one per
    chat model in the catalogue).
    """
    if getattr(registry, "_tear_down_wired", False):
        return
    registry._tear_down = daemon.teardown  # type: ignore[attr-defined]
    registry._tear_down_wired = True  # type: ignore[attr-defined]


def _extract_system_prompt(messages: list[dict]) -> str:
    """Concatenate text from leading ``role=system`` messages.  Stops
    at the first non-system role — middle-of-history system messages
    are unusual and don't carry env-hash signal."""
    parts: list[str] = []
    for msg in messages:
        if msg.get("role") != "system":
            break
        content = msg.get("content")
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") in (
                    "text",
                    "input_text",
                ):
                    parts.append(str(block.get("text", "")))
    return "\n\n".join(p for p in parts if p)


def _extract_tool_names(tools: Any) -> list[str]:
    """Pull function names from chat-completions ``tools`` array.  Used
    only for env_hash; order is normalised by the registry."""
    if not isinstance(tools, list):
        return []
    out: list[str] = []
    for t in tools:
        if not isinstance(t, dict):
            continue
        fn = t.get("function") or {}
        name = fn.get("name")
        if name:
            out.append(str(name))
    return out


def _detect_effort(
    call_kwargs: dict,
    generate_kwargs: dict | None,
) -> str | None:
    """Return a stable string for the current thinking-effort, or None.
    Per-call kwargs win over constructor defaults.  Recognised shapes:

    * OpenAI-flavored: ``reasoning_effort="medium"`` /
      ``reasoning={"effort": "medium"}``.

    Anthropic-style ``thinking={"budget_tokens": N}`` is intentionally
    NOT mapped here — acpx's underlying ``effort`` config option
    accepts only ``{low, medium, high, xhigh, max}`` (smoke test
    2026-04-27 surfaced the rejection).  Callers wanting a custom
    budget should configure Claude Code's profile directly.

    The string we return is round-tripped through ``acpx claude set
    -s <name> effort <value>`` so the daemon's effort matches what we
    last recorded — see :meth:`Registry.update_effort`.  Unknown
    values are passed through unchanged so we surface acpx's own
    rejection message rather than masking it.
    """
    for src in (call_kwargs, generate_kwargs or {}):
        eff = src.get("reasoning_effort")
        if eff:
            return str(eff)
        reasoning = src.get("reasoning")
        if isinstance(reasoning, dict) and reasoning.get("effort"):
            return str(reasoning["effort"])
    return None


# =========================================================================
# Provider class
# =========================================================================


class ClaudeAcpxChatModel(OpenAIChatModel):
    """Claude Code (acpx) variant of :class:`OpenAIChatModel`.

    Subclasses agentscope's ``OpenAIChatModel`` so the downstream
    response-parsing path runs against the same SDK types the real
    OpenAI client returns.  ``_install_wrapper`` swaps
    ``client.chat.completions.create`` with our daemon-backed coroutine
    at init time; the SDK client itself is never used to make HTTP
    calls — its sole purpose is to satisfy agentscope's type
    expectations.
    """

    def __init__(
        self,
        *,
        model_name: str,
        stream: bool = True,
        generate_kwargs: dict[str, Any] | None = None,
        client_kwargs: dict[str, Any] | None = None,
        api_key: str | None = None,
        stream_tool_parsing: bool = False,
    ) -> None:
        # OpenAI SDK refuses to construct without an ``api_key`` (even
        # though our ``_install_wrapper`` redirects every request away
        # from its default base URL).  Seed with a harmless sentinel —
        # it never reaches the wire.  Mirrors
        # :class:`CodexOAuthChatModel` (codex_oauth_model.py:212-214).
        if not api_key:
            api_key = "claude-acpx-unused"
        super().__init__(
            model_name=model_name,
            stream=stream,
            api_key=api_key,
            stream_tool_parsing=stream_tool_parsing,
            client_kwargs=client_kwargs or {},
            generate_kwargs=generate_kwargs or {},
        )
        self._install_wrapper()

    def _install_wrapper(self) -> None:
        """Override ``self.client.chat.completions.create`` with the
        acpx daemon path.  The closure captures ``self`` so per-instance
        ``model_name`` and ``generate_kwargs`` are baked in; per-call
        kwargs override them on the way through."""

        async def _wrapped_create(**call_kwargs: Any) -> Any:
            messages = list(call_kwargs.get("messages") or [])
            tools = call_kwargs.get("tools")
            stream_requested = bool(call_kwargs.get("stream", False))

            # ContextVars are populated by AgentRunner.stream_query for
            # every agent-driven call.  We REQUIRE both — without a
            # stable conversation key, two unrelated callers would
            # collide on the same session and contaminate each other's
            # Claude state.  The registry has the same defensive check;
            # surfacing it here keeps the error attribution local
            # rather than blaming the registry for a missing context.
            from ..app.agent_context import (
                get_current_agent_id,
                get_current_session_id,
            )
            agent_id = get_current_agent_id()
            session_id = get_current_session_id()
            if not agent_id or not session_id:
                raise RuntimeError(
                    "ClaudeAcpxChatModel requires both agent_id and "
                    "session_id ContextVars to be set (got "
                    f"agent_id={agent_id!r}, session_id={session_id!r}). "
                    "Run inside an AgentRunner context that calls "
                    "set_current_agent_id / set_current_session_id.",
                )

            system_prompt = _extract_system_prompt(messages)
            tool_names = _extract_tool_names(tools)
            cwd = os.getcwd()
            permission_mode = "default"  # v1: not exposed yet

            env_hash_value = compute_env_hash(
                system_prompt=system_prompt,
                tool_names=tool_names,
                cwd=cwd,
                permission_mode=permission_mode,
                generate_kwargs=self.generate_kwargs,
            )

            registry = get_registry()
            daemon = AcpxDaemon.get_or_spawn()
            _wire_registry_tear_down(registry, daemon)

            plan = await registry.plan_turn(
                agent_id=agent_id,
                session_id=session_id,
                model=self.model_name,
                env_hash_value=env_hash_value,
                messages=messages,
            )

            if plan.mode == "seed_full":
                claude_acpx_metrics.record_seed_full()
                prompt_blocks = render_history_for_seed(messages)
            else:
                claude_acpx_metrics.record_ship_tail()
                prompt_blocks = extract_tail_from_history(
                    messages,
                    plan.from_idx,
                )

            # _detect_effort precedence: per-call kwargs win, then
            # constructor generate_kwargs.  This means an explicit
            # ``reasoning_effort=None`` in call_kwargs cannot clear a
            # generate_kwargs default — the call_kwargs path returns
            # None which falls through to generate_kwargs.  In practice
            # agentscope merges these two upstream so the limitation
            # never bites; documenting the asymmetry here so a future
            # caller adding a "clear effort" feature reads the existing
            # contract before adding a sentinel.
            effort = _detect_effort(call_kwargs, self.generate_kwargs)
            entry = plan.entry

            if stream_requested:
                # Streaming path: hand ownership of effort-set + lock
                # acquire + submit_turn to the adapter so we never
                # leak a lock if the consumer abandons the iterator
                # before opening it.  See codex review note [1].
                return _AcpxStreamAdapter(
                    registry=registry,
                    entry=entry,
                    messages=messages,
                    session_name=plan.session_name,
                    prompt_blocks=prompt_blocks,
                    is_seed=plan.mode == "seed_full",
                    daemon=daemon,
                    state=StreamState(model=self.model_name),
                    effort=effort,
                )

            # Non-stream: acquire lock, ship, commit, release inline.
            # The registry contract — "one ship cycle per entry at a
            # time" — is enforced upstream by AgentRunner per-session
            # serialisation; if that breaks down, plan_turn's snapshot
            # could go stale between return and acquire (see codex
            # review note [7]).  Single-tenant v1 ships under that
            # contract; a multi-worker promotion needs registry-side
            # lock-and-replan.
            await entry.lock.acquire()
            try:
                if effort and effort != entry.last_effort:
                    try:
                        await daemon.run_set_config(
                            plan.session_name,
                            "effort",
                            effort,
                        )
                        await registry.update_effort(entry, effort)
                    except AcpxDaemonError as e:
                        # Best-effort: Claude inherits its prior
                        # effort if set-config didn't take.
                        logger.warning(
                            "acpx set effort %s failed for %s: %s",
                            effort,
                            plan.session_name,
                            e,
                        )

                state = StreamState(model=self.model_name)
                line_iter = daemon.submit_turn(
                    session_name=plan.session_name,
                    prompt_blocks=prompt_blocks,
                    is_seed=plan.mode == "seed_full",
                )
                try:
                    body = await collect_as_chat_completion(line_iter, state)
                finally:
                    aclose = getattr(line_iter, "aclose", None)
                    if aclose is not None:
                        try:
                            await aclose()
                        except Exception as e:  # noqa: BLE001
                            logger.debug(
                                "acpx line_iter.aclose failed: %s",
                                e,
                            )
                # Advance to ``len(messages)`` only — the assistant
                # reply Claude just produced isn't yet in our
                # ``messages`` list (agentscope appends it after we
                # return).  Including +1 here would make
                # ``last_msg_chain_hash`` count messages we haven't
                # actually seen, triggering spurious drift on the
                # next turn (smoke test 2026-04-27 caught this).
                # Lane A's :func:`extract_tail_from_history` skips
                # any leading assistant/tool messages in the next
                # turn's tail, so the slight overshoot in tail span
                # is absorbed there.
                await registry.commit_turn(
                    entry,
                    new_shipped_idx=len(messages),
                    messages=messages,
                    effort=effort,
                )
                return ChatCompletion.model_validate(body)
            except Exception:
                claude_acpx_metrics.record_error()
                raise
            finally:
                if entry.lock.locked():
                    try:
                        entry.lock.release()
                    except RuntimeError:
                        pass

        self.client.chat.completions.create = _wrapped_create  # type: ignore[method-assign]


# =========================================================================
# Streaming adapter
# =========================================================================


class _AcpxStreamAdapter:
    """AsyncStream-compatible iterator returned for ``stream=True``.

    Owns four lifecycles:

    * the registry entry lock (acquired lazily on first __anext__);
    * the per-turn ``acpx claude set effort`` push (lazy, same);
    * the daemon's ``submit_turn`` async generator (raw ACP lines);
    * the ACP→chat-completion translator built on top of it.

    Lifecycle paths:

    * **Constructed but never iterated** — nothing acquired, nothing
      to clean.  __del__ no-ops.
    * **Iterated to natural end** — ``commit_turn`` runs, lock
      released.
    * **Iterated then exception / cancellation** — translator gets
      GeneratorExit, daemon's submit_turn finally runs, lock released
      without commit.
    * **Iterated then abandoned** (no aclose, no async-with) — best
      effort: __del__ schedules ``_finalize(commit=False)`` on the
      running loop if there is one.  If no loop is running we log a
      warning and rely on ``loop.shutdown_asyncgens`` to reap the
      inner async generator at process exit.

    Mirrors :class:`CodexOAuthAsyncStream` in surface so agentscope's
    ``_parse_openai_stream_completion_response`` consumes this without
    knowing it's not the real SDK type.
    """

    def __init__(
        self,
        *,
        registry: Registry,
        entry: AcpxSessionEntry,
        messages: list[dict],
        session_name: str,
        prompt_blocks: list[dict],
        is_seed: bool,
        daemon: AcpxDaemon,
        state: StreamState,
        effort: str | None,
    ) -> None:
        self._registry = registry
        self._entry = entry
        self._messages = messages
        self._session_name = session_name
        self._prompt_blocks = prompt_blocks
        self._is_seed = is_seed
        self._daemon = daemon
        self._state = state
        self._effort = effort
        self._line_iter: AsyncIterator[str] | None = None
        self._chunk_iter: AsyncIterator[dict] | None = None
        self._closed = False
        self._committed = False
        self._lock_held = False
        self._opened = False

    def __aiter__(self) -> "_AcpxStreamAdapter":
        return self

    async def _open(self) -> None:
        """Idempotent: acquire entry.lock, push effort delta if any,
        and start the daemon's submit_turn generator.

        Note the ``_closed`` re-check after acquiring the lock — a
        concurrent ``close()`` can mark the adapter closed while we
        are blocked on ``lock.acquire()``.  Without the re-check we
        would proceed to spawn submit_turn after close() promised the
        adapter was inert (codex review note [N1]).
        """
        if self._opened or self._closed:
            return
        self._opened = True

        await self._entry.lock.acquire()
        if self._closed:
            # close() ran while we were blocked on acquire.  Release
            # immediately and stay closed; __anext__ will surface
            # StopAsyncIteration.
            try:
                self._entry.lock.release()
            except RuntimeError:
                pass
            return
        self._lock_held = True

        # Effort delta sync — only push when changed.  Best-effort:
        # any failure (network, daemon, registry update) leaves Claude
        # on its prior effort, which is safer than aborting the turn.
        # Catch broadly so a non-AcpxDaemonError from update_effort
        # doesn't leave registry/daemon state inconsistent (codex
        # review note [N2]).
        if self._effort and self._effort != self._entry.last_effort:
            try:
                await self._daemon.run_set_config(
                    self._session_name,
                    "effort",
                    self._effort,
                )
                await self._registry.update_effort(
                    self._entry,
                    self._effort,
                )
            except Exception as e:  # noqa: BLE001
                logger.warning(
                    "acpx set thinking %s failed for %s: %s",
                    self._effort,
                    self._session_name,
                    e,
                )

        self._line_iter = self._daemon.submit_turn(
            session_name=self._session_name,
            prompt_blocks=self._prompt_blocks,
            is_seed=self._is_seed,
        )
        self._chunk_iter = translate_acp_updates_to_chat_chunks(
            self._line_iter,
            self._state,
        )

    async def __anext__(self) -> ChatCompletionChunk:
        if self._closed:
            raise StopAsyncIteration

        if self._chunk_iter is None:
            try:
                await self._open()
            except BaseException:
                # Open itself failed (e.g. daemon binary missing).
                # Release lock if we managed to acquire before the
                # failure; mark closed so subsequent __anext__ stops.
                await self._finalize(commit=False)
                raise

        try:
            chunk_dict = await self._chunk_iter.__anext__()
        except StopAsyncIteration:
            await self._finalize(commit=True)
            raise
        except BaseException:
            # Includes asyncio.CancelledError + GeneratorExit + plain
            # exceptions.  Drop the lock without committing so the
            # next plan_turn re-evaluates drift from a known-stable
            # snapshot.
            await self._finalize(commit=False)
            raise

        return ChatCompletionChunk.model_validate(chunk_dict)

    async def _finalize(self, *, commit: bool) -> None:
        if self._closed:
            return
        self._closed = True
        # Tear down translator first, then the underlying line reader.
        # aclose on a finished generator is a no-op; on an in-flight
        # one it raises GeneratorExit at the suspended yield, which
        # runs through submit_turn's finally (proc.stdin.close +
        # _reap), keeping the subprocess from leaking.
        for it in (self._chunk_iter, self._line_iter):
            if it is None:
                continue
            aclose = getattr(it, "aclose", None)
            if aclose is None:
                continue
            try:
                await aclose()
            except Exception as e:  # noqa: BLE001
                logger.debug("acpx stream cleanup: %s", e)
        if commit and not self._committed:
            try:
                await self._registry.commit_turn(
                    self._entry,
                    new_shipped_idx=len(self._messages),
                    messages=self._messages,
                    effort=self._effort,
                )
                self._committed = True
            except Exception as e:  # noqa: BLE001
                logger.warning("acpx registry commit_turn failed: %s", e)
        if self._lock_held and self._entry.lock.locked():
            try:
                self._entry.lock.release()
            except RuntimeError:
                # Lock was already released elsewhere — defensive,
                # shouldn't happen given _closed gate above but
                # safer than letting an exception escape finalize.
                pass
            self._lock_held = False

    async def close(self) -> None:
        """Idempotent close — safe to call multiple times."""
        await self._finalize(commit=False)

    async def __aenter__(self) -> "_AcpxStreamAdapter":
        return self

    async def __aexit__(self, *_exc: Any) -> None:
        # Don't auto-commit on exit — natural end-of-iter via __anext__
        # already did that.  __aexit__ runs after iteration and is the
        # SDK's "you're done with this stream" hook.  If iteration
        # raised, the exception bubbles up *first*, so by the time we
        # reach __aexit__ we're either already closed (commit=True
        # path) or recovering (commit=False path).
        await self._finalize(commit=False)

    def __del__(self) -> None:
        """Best-effort cleanup for adapters abandoned mid-iter without
        ``close()``/``__aexit__``.  We can't ``await`` from ``__del__``,
        so the strategy is:

        * If the asyncio loop is still running, schedule a finalize
          task — caveats apply (the loop may close before the task
          runs), but at least we avoid leaking the lock and the
          subprocess for the common abandon-without-close case.
        * Otherwise (loop closed / never started), log a warning.
          Python's ``loop.shutdown_asyncgens`` already aclose()s the
          inner ``submit_turn`` generator at loop close, which runs
          its own finally to reap the subprocess.  The lock is per-
          process state and cleans up with the loop teardown.

        Ref codex review note [2].
        """
        if self._closed:
            return
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.create_task(self._finalize(commit=False))
                return
        except RuntimeError:
            pass
        if self._lock_held or self._chunk_iter is not None:
            try:
                logger.warning(
                    "_AcpxStreamAdapter abandoned without close() for "
                    "session %s; relying on loop.shutdown_asyncgens to "
                    "reap subprocess.",
                    self._session_name,
                )
            except ValueError:
                # Interpreter shutdown — stderr is closed, the
                # logging handler raises ValueError("I/O operation
                # on closed file").  Swallow ONLY this case; a real
                # logging-config bug will still surface (codex
                # review note [N3]).
                pass
