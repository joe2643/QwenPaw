# -*- coding: utf-8 -*-
"""Process-singleton registry mapping (agent_id, session_id, model,
env_hash) → live acpx Claude Code session.

Lives between ``ClaudeAcpxChatModel`` (caller) and the acpx subprocess
daemon.  For each chat-completions call the wrapper asks the registry
"what session do I push this turn into, and ship full history or just
the tail?".  Registry consults its in-memory state, decides between
``seed_full`` (mint or reseed an acpx session) and ``ship_tail`` (only
push messages we haven't sent yet), and returns a :class:`ShipPlan`.

Why a registry at all
---------------------

acpx supports stateful per-session conversations: ``acpx claude -s
<name>`` keeps history server-side across invocations.  If we (a) keep
the same name across CoPaw turns of the same conversation and (b) only
push the new user message each time, Claude Code's underlying Anthropic
calls maintain a stable cache prefix, which is the whole point of
this provider.

Drift policy (codex C2/C6/C14 + my A2/A3)
-----------------------------------------

Naive "ship messages.last each turn" breaks the moment CoPaw's view of
history diverges from acpx's: user ``/clear``, agentscope memory
compaction, history edits.  We protect against those by:

1. **env_hash** keys the session.  System prompt, tool catalog, cwd,
   permission_mode, and generate_kwargs keys all feed it.  If any
   change between turns, we mint a new session — the old one stays on
   disk LRU-evictable but is no longer routed to.

2. **last_msg_chain_hash** is recomputed per call against
   ``messages[:last_shipped_idx]``.  Mismatch ⇒ tear down + reseed.

3. **shorter-or-equal length than last shipped** ⇒ user cleared or
   compacted ⇒ tear down + reseed.

Otherwise the new tail ships and ``last_shipped_idx`` advances.

Session name
------------

``copaw-{hostname[:8]}-{pid}-{agent[:8]}-{session[:12]}-{model_short}``.
Hostname + pid (codex C5): single-host multi-worker doesn't collide on
disk.  Multi-host horizontal scaling is out of scope for v1 (per plan
T3 decision); if revisited, promote registry to Redis so name
collisions are coordinated rather than papered over.

Tear-down
---------

LRU eviction calls a caller-supplied async ``tear_down_cb`` so registry
stays subprocess-agnostic.  Production wiring shells out
``acpx claude sessions close <name>`` (codex C12: don't leak disk
sessions).  Tests inject a no-op or a recorder.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import secrets
import socket
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Literal

logger = logging.getLogger(__name__)

# Cap chosen to comfortably hold the active conversations of a single
# CoPaw deployment without unbounded growth.  Each entry is a few
# bytes of metadata; cost of overshooting is ~0.  Cost of
# undershooting is needless reseed when a quiet conversation gets
# evicted then re-engaged.  200 ≈ "more than any single human's
# concurrent chat workload".
_DEFAULT_CAP: int = 200


# =========================================================================
# Hashing helpers — content-stable, format-stable
# =========================================================================


def _hash(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def _plain_text(content: Any) -> str:
    """Flatten chat content (str | list of blocks) to text for hashing.
    Strips image base64 payloads — base64 re-encoding by upstream
    formatters would otherwise spuriously trigger drift detection.
    Mirrors the lossy collapse in :func:`acpx_translate._content_text`
    but with even tighter "image present" markers (no URI to vary).
    """
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for item in content:
        if not isinstance(item, dict):
            continue
        t = item.get("type")
        if t in ("text", "input_text", "output_text"):
            parts.append(str(item.get("text", "")))
        elif t in ("image_url", "input_image", "image"):
            parts.append("[IMG]")
        elif t in ("audio", "input_audio"):
            parts.append("[AUD]")
    return "".join(parts)


def msg_signature(msg: dict) -> str:
    """Per-message content-stable signature.  Two messages that should
    be considered equivalent for drift purposes (text-equal modulo
    base64 / whitespace-stripped) hash to the same string.

    Includes role, plain text, and tool_call name+args.  Tool_call_id
    on assistant messages is NOT included — agentscope sometimes
    re-mints these on memory replay.
    """
    role = msg.get("role", "")
    parts = [role, _plain_text(msg.get("content"))]
    for tc in msg.get("tool_calls") or []:
        fn = tc.get("function") or {}
        parts.append(f"call|{fn.get('name','')}|{fn.get('arguments','')}")
    if role == "tool":
        # Tool result message: tool_call_id is part of the linkage;
        # include it because reseeding the wrong tool result would
        # be silently corrupting.
        parts.append(f"tcid|{msg.get('tool_call_id','')}")
    return "|".join(parts)


def history_hash(messages: list[dict], up_to_idx: int) -> str:
    """SHA-1 chain over msg_signatures of ``messages[:up_to_idx]``."""
    if up_to_idx <= 0:
        return _hash("")
    h = hashlib.sha1()
    for msg in messages[:up_to_idx]:
        h.update(msg_signature(msg).encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def env_hash(
    *,
    system_prompt: str,
    tool_names: list[str] | None,
    cwd: str,
    permission_mode: str,
    generate_kwargs: dict[str, Any] | None,
) -> str:
    """SHA-1 over the dimensions that change Claude session behavior
    independently of message history (codex C2/C6/C14).  Any of
    these flipping invalidates session reuse.
    """
    payload = {
        "sys": system_prompt,
        "tools": sorted(tool_names or []),
        "cwd": cwd,
        "perm": permission_mode,
        # Hash sorted keys + values; keys present-but-empty are still
        # signal (e.g. ``thinking={}`` ≠ no thinking at all).
        "gk": sorted((generate_kwargs or {}).items()),
    }
    return _hash(json.dumps(payload, sort_keys=True, default=str))


# =========================================================================
# Session naming
# =========================================================================


_HOSTNAME: str = socket.gethostname().split(".")[0][:8] or "local"


def _model_short(model: str) -> str:
    """Shorten claude model id for session-name suffix.
    ``claude-sonnet-4-6`` → ``s4.6``, ``claude-opus-4-7`` → ``o4.7``.
    Fallback: first 6 chars of model id.
    """
    m = model.lower()
    for fam, prefix in (("opus", "o"), ("sonnet", "s"), ("haiku", "h")):
        if fam in m:
            # Pull the first numeric token after the family name.
            tail = m.split(fam, 1)[1].lstrip("-")
            digits = "".join(c if c.isdigit() or c == "." else "-" for c in tail).strip("-")
            return f"{prefix}{digits.replace('-', '.')[:6]}"
    return m[:6]


def make_session_name(*, agent_id: str, session_id: str, model: str) -> str:
    """``copaw-{host}-{pid}-{agent}-{session}-{model}-{rnd}``.

    Random 4-char suffix uniqueifies each mint so reseed (after
    drift) produces a name that doesn't collide with the just-torn-
    down acpx session — important because acpx's ``sessions new
    --name X`` would otherwise either fail or, worse, silently
    re-attach to dirty state if tear-down hadn't propagated yet.
    """
    return (
        f"copaw-{_HOSTNAME}-{os.getpid()}-"
        f"{agent_id[:8] or 'na'}-"
        f"{session_id[:12] or 'na'}-"
        f"{_model_short(model)}-"
        f"{secrets.token_hex(2)}"
    )


# =========================================================================
# Registry state
# =========================================================================


@dataclass(frozen=True)
class SessionKey:
    agent_id: str
    session_id: str
    model: str
    env_hash: str


@dataclass
class AcpxSessionEntry:
    session_name: str
    last_msg_chain_hash: str = ""
    last_shipped_idx: int = 0
    last_effort: str | None = None
    last_used_at: float = field(default_factory=time.time)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


@dataclass
class ShipPlan:
    """Caller's marching orders for a single chat-completions turn."""

    session_name: str
    mode: Literal["seed_full", "ship_tail"]
    from_idx: int
    """For ``seed_full``: 0 (caller ships all messages).
    For ``ship_tail``: caller ships ``messages[from_idx:]``."""
    entry: AcpxSessionEntry
    """Caller mutates ``entry.last_shipped_idx`` /
    ``last_msg_chain_hash`` after a successful ship via
    :meth:`Registry.commit_turn` (under entry.lock)."""


TearDownCb = Callable[[str], Awaitable[None]]


class Registry:
    """Process-singleton acpx session registry.

    Thread-safe via a process-wide ``asyncio.Lock`` for the dict
    mutation and a per-entry lock for the ship/commit cycle.

    Not safe across processes — multi-worker / multi-host deployment
    is out of scope per plan T3 decision.  If horizontal scaling
    becomes a requirement, swap this for a Redis-backed implementation
    behind the same surface.
    """

    def __init__(
        self,
        *,
        cap: int = _DEFAULT_CAP,
        tear_down_cb: TearDownCb | None = None,
    ) -> None:
        self._entries: dict[SessionKey, AcpxSessionEntry] = {}
        self._global_lock = asyncio.Lock()
        self._cap = cap
        # Default no-op so unit tests don't need to wire this; production
        # wires ``acpx claude sessions close <name>``.
        self._tear_down: TearDownCb = tear_down_cb or _noop_tear_down

    async def plan_turn(
        self,
        *,
        agent_id: str,
        session_id: str,
        model: str,
        env_hash_value: str,
        messages: list[dict],
    ) -> ShipPlan:
        """Decide whether this turn should seed_full or ship_tail.
        Caller MUST acquire ``plan.entry.lock`` before invoking acpx
        and call :meth:`commit_turn` after success (or release without
        commit on failure).
        """
        if not agent_id or not session_id:
            # Defensive — A4 from eng review.  Without a stable key we
            # cannot do drift detection, and silent fallback to a
            # process-wide bucket would corrupt unrelated sessions.
            raise RuntimeError(
                "Acpx registry called without ContextVars set "
                f"(agent_id={agent_id!r}, session_id={session_id!r}). "
                "ClaudeAcpxChatModel wrapper must run inside an "
                "AgentRunner context.",
            )

        key = SessionKey(agent_id, session_id, model, env_hash_value)
        async with self._global_lock:
            entry = self._entries.get(key)

            if entry is None:
                # Cold lookup.  Mint and seed_full.  No tear-down here
                # because there's no prior session — but check for
                # stale entries on a different env_hash for the same
                # (agent, session, model) so user-facing reuse-after-
                # env-change logs cleanly.
                self._evict_stale_for_conversation_locked(
                    agent_id=agent_id,
                    session_id=session_id,
                    model=model,
                    keep_env_hash=env_hash_value,
                )
                entry = AcpxSessionEntry(
                    session_name=make_session_name(
                        agent_id=agent_id,
                        session_id=session_id,
                        model=model,
                    ),
                )
                self._entries[key] = entry
                await self._maybe_evict_lru_locked()
                entry.last_used_at = time.time()
                logger.info(
                    "acpx registry: mint %s (key=%s, env_hash=%s)",
                    entry.session_name,
                    f"{agent_id[:8]}/{session_id[:12]}/{model}",
                    env_hash_value[:8],
                )
                return ShipPlan(
                    session_name=entry.session_name,
                    mode="seed_full",
                    from_idx=0,
                    entry=entry,
                )

            entry.last_used_at = time.time()

        # Hot lookup — outside global_lock now.  Compute drift under
        # entry-level lock so concurrent commit doesn't race us.  We
        # do not actually take the entry lock here; caller does it
        # before invoking acpx.  But the read of last_shipped_idx /
        # last_msg_chain_hash is a snapshot — concurrent commits on
        # the SAME entry are forbidden by the runner contract (one
        # turn at a time per conversation), so a stale read here is
        # impossible in practice.

        n = len(messages)

        if n <= entry.last_shipped_idx:
            # Shorter or equal: history truncated (user /clear, or
            # agentscope compaction).  Tear down and reseed.
            return await self._reseed(key, entry)

        observed_hash = history_hash(messages, entry.last_shipped_idx)
        if observed_hash != entry.last_msg_chain_hash:
            return await self._reseed(key, entry)

        return ShipPlan(
            session_name=entry.session_name,
            mode="ship_tail",
            from_idx=entry.last_shipped_idx,
            entry=entry,
        )

    async def commit_turn(
        self,
        entry: AcpxSessionEntry,
        *,
        new_shipped_idx: int,
        messages: list[dict],
        effort: str | None = None,
    ) -> None:
        """Record what was successfully shipped.  Must be called under
        ``entry.lock`` by the caller.  ``messages`` is the full list
        seen this turn; we recompute the chain hash up to
        ``new_shipped_idx``.
        """
        entry.last_shipped_idx = new_shipped_idx
        entry.last_msg_chain_hash = history_hash(messages, new_shipped_idx)
        entry.last_used_at = time.time()
        if effort is not None:
            entry.last_effort = effort

    async def update_effort(
        self,
        entry: AcpxSessionEntry,
        effort: str,
    ) -> None:
        """Record a ``set thinking <level>`` push.  Independent of
        commit_turn so callers can update mid-flight."""
        entry.last_effort = effort
        entry.last_used_at = time.time()

    async def _reseed(
        self,
        key: SessionKey,
        old_entry: AcpxSessionEntry,
    ) -> ShipPlan:
        """Tear down the old acpx session, mint a fresh one for the
        same key, and return a seed_full ShipPlan."""
        try:
            await self._tear_down(old_entry.session_name)
        except Exception as e:  # noqa: BLE001
            # Tear-down failures aren't fatal — the old session gets
            # GC'd by acpx's own LRU eventually.  Log and continue.
            logger.warning(
                "acpx registry: tear_down(%s) failed: %s — proceeding "
                "with reseed regardless",
                old_entry.session_name,
                e,
            )

        fresh = AcpxSessionEntry(
            session_name=make_session_name(
                agent_id=key.agent_id,
                session_id=key.session_id,
                model=key.model,
            ),
        )
        async with self._global_lock:
            self._entries[key] = fresh
        logger.info(
            "acpx registry: reseed %s → %s (drift)",
            old_entry.session_name,
            fresh.session_name,
        )
        return ShipPlan(
            session_name=fresh.session_name,
            mode="seed_full",
            from_idx=0,
            entry=fresh,
        )

    def _evict_stale_for_conversation_locked(
        self,
        *,
        agent_id: str,
        session_id: str,
        model: str,
        keep_env_hash: str,
    ) -> None:
        """When env_hash changes for an existing conversation/model
        (system prompt edit, tool catalog change, cwd switch), the
        old entry is unreachable but still on disk via acpx.  Schedule
        tear-down so it doesn't pile up.

        Caller holds ``self._global_lock``.
        """
        stale_keys: list[SessionKey] = []
        for k in self._entries:
            if (
                k.agent_id == agent_id
                and k.session_id == session_id
                and k.model == model
                and k.env_hash != keep_env_hash
            ):
                stale_keys.append(k)
        for k in stale_keys:
            stale = self._entries.pop(k)
            # Schedule tear-down off the lock — fire and forget.
            asyncio.get_event_loop().create_task(
                self._tear_down_safe(stale.session_name),
            )

    async def _tear_down_safe(self, name: str) -> None:
        try:
            await self._tear_down(name)
        except Exception as e:  # noqa: BLE001
            logger.warning("acpx registry: deferred tear_down(%s) failed: %s", name, e)

    async def _maybe_evict_lru_locked(self) -> None:
        """Caller holds ``self._global_lock``.  Evict oldest entries
        until size ≤ cap, calling tear_down on each."""
        if len(self._entries) <= self._cap:
            return
        # Sort ascending by last_used_at; pop the oldest.
        sorted_items = sorted(
            self._entries.items(),
            key=lambda kv: kv[1].last_used_at,
        )
        excess = len(self._entries) - self._cap
        for key, evicted in sorted_items[:excess]:
            self._entries.pop(key, None)
            asyncio.get_event_loop().create_task(
                self._tear_down_safe(evicted.session_name),
            )

    # ----- Test/diagnostic helpers --------------------------------- #

    def __len__(self) -> int:  # convenience for tests
        return len(self._entries)

    def _entry_for_test(self, key: SessionKey) -> AcpxSessionEntry | None:
        return self._entries.get(key)


async def _noop_tear_down(name: str) -> None:  # noqa: ARG001
    return None


# Process-singleton accessor.  Lazy because tests want to instantiate
# their own Registry instance — module-level singleton avoids global
# state leakage between test cases.
_GLOBAL: Registry | None = None


def get_registry() -> Registry:
    global _GLOBAL  # noqa: PLW0603
    if _GLOBAL is None:
        _GLOBAL = Registry()
    return _GLOBAL


def set_registry_for_test(reg: Registry | None) -> None:
    """Tests inject a fresh Registry to avoid bleed across cases."""
    global _GLOBAL  # noqa: PLW0603
    _GLOBAL = reg
