# -*- coding: utf-8 -*-
"""Trigger logic for listen mode: per-chat timer + dispatcher."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple

from .listen_types import ListenConfig

logger = logging.getLogger(__name__)


# Module-level state.  Keyed by ``listen_key(channel, chat_id)``.  In-memory
# only — restarting copaw wipes these; the user must re-enable.
listen_configs: Dict[str, ListenConfig] = {}
listen_tasks: Dict[str, asyncio.Task] = {}

# Minimum tick granularity, regardless of how long ``interval_minutes`` is.
# Lets ``/listen off`` propagate within a reasonable window even when a
# big interval was set.
_MIN_TICK_SECONDS = 30


def listen_key(channel_name: str, chat_id: str) -> str:
    """Canonical key for the configs / tasks dicts."""
    return f"{channel_name}:{chat_id}"


def _now() -> datetime:
    return datetime.now(timezone.utc)


# Channels we recognise when falling back to session_id parsing.  The
# session_id pattern is ``{channel}:[group:]?{chat_id}`` (see e.g.
# ``WhatsAppChannel.resolve_session_id``); only channels we know expose
# ``_group_history`` are valid listen targets.
_LISTEN_SUPPORTED_CHANNELS = frozenset({"whatsapp", "signal"})


def _extract_chat_target(
    channel_meta: Dict[str, Any] | None,
    session_id: str = "",
) -> Tuple[str, str]:
    """Pull (channel_name, chat_id) for a listen target.

    Primary source: the active request's ``channel_meta`` (populated by
    inbound channel handlers like WhatsApp / Signal).

    Fallback: parse ``session_id`` — needed when ``/listen`` is invoked
    from a console UI that's pointing at a remote-channel session.  In
    that case the request hits ``/agent/process`` directly without
    going through the channel's inbound flow, so ``channel_meta`` is
    missing even though the session itself refers to a real group/DM.

    Returns ``("", "")`` when neither source yields a usable target,
    or the parsed channel is not one we support for listen mode.
    """
    if isinstance(channel_meta, dict):
        channel = str(
            channel_meta.get("platform") or channel_meta.get("channel") or "",
        )
        chat_id = str(
            channel_meta.get("chat_jid")
            or channel_meta.get("group_id")
            or channel_meta.get("source")
            or channel_meta.get("chat_id")
            or "",
        )
        if channel and chat_id:
            return channel, chat_id

    parsed = _parse_session_id(session_id)
    if parsed is not None:
        return parsed

    return "", ""


def _parse_session_id(session_id: str) -> Optional[Tuple[str, str]]:
    """Decode ``{channel}:[group:]?{chat_id}`` shaped session IDs.

    Returns ``None`` when the channel prefix isn't one we support for
    listen mode (so the caller can surface a useful error rather than
    silently enabling listen for, say, ``proactive_mode:default``).
    """
    if not session_id or ":" not in session_id:
        return None
    head, rest = session_id.split(":", 1)
    head = head.strip().lower()
    if head not in _LISTEN_SUPPORTED_CHANNELS:
        return None
    if not rest:
        return None
    if rest.startswith("group:"):
        rest = rest[len("group:"):]
    rest = rest.strip()
    if not rest:
        return None
    return head, rest


def enable_listen_for_chat(
    channel_name: str,
    chat_id: str,
    *,
    interval_minutes: int = 5,
    chat_meta: Optional[Dict[str, Any]] = None,
    agent_id: str = "",
    verbosity: str = "normal",
    session_id: str = "",
    user_id: str = "",
) -> str:
    """Enable listen mode for one chat.

    Idempotent on the (channel, chat_id) key — calling again just
    updates the interval, chat_meta, and verbosity without spawning a
    duplicate background task.
    """
    if not channel_name or not chat_id:
        raise ValueError(
            "listen: channel_name and chat_id are both required",
        )
    if interval_minutes < 1:
        raise ValueError("listen: interval_minutes must be >= 1")
    if verbosity not in ("normal", "aggressive"):
        raise ValueError(
            "listen: verbosity must be 'normal' or 'aggressive'",
        )

    key = listen_key(channel_name, chat_id)
    existing = listen_configs.get(key)
    if existing is not None:
        existing.enabled = True
        existing.interval_minutes = interval_minutes
        existing.chat_meta = dict(chat_meta or {})
        existing.verbosity = verbosity  # type: ignore[assignment]
        if agent_id:
            existing.agent_id = agent_id
        if session_id:
            existing.session_id = session_id
        if user_id:
            existing.user_id = user_id
        cfg = existing
    else:
        cfg = ListenConfig(
            enabled=True,
            interval_minutes=interval_minutes,
            channel_name=channel_name,
            chat_id=chat_id,
            chat_meta=dict(chat_meta or {}),
            agent_id=agent_id,
            verbosity=verbosity,  # type: ignore[arg-type]
            session_id=session_id,
            user_id=user_id,
            last_fire=None,
        )
        listen_configs[key] = cfg

    if key not in listen_tasks or listen_tasks[key].done():
        listen_tasks[key] = asyncio.create_task(_run_trigger_loop(key))

    logger.info(
        "listen: enabled %s interval=%dmin verbosity=%s agent=%s",
        key,
        interval_minutes,
        verbosity,
        agent_id or "<unknown>",
    )
    return (
        f"Listen mode enabled on {key} "
        f"(every {interval_minutes} min, verbosity={verbosity})."
    )


def disable_listen_for_chat(channel_name: str, chat_id: str) -> str:
    """Cancel listen for one chat.  Idempotent."""
    key = listen_key(channel_name, chat_id)
    cfg = listen_configs.pop(key, None)
    task = listen_tasks.pop(key, None)
    if cfg is not None:
        cfg.enabled = False
    if task is not None and not task.done():
        task.cancel()
    if cfg is None and task is None:
        return f"Listen mode was not active on {key}."
    logger.info("listen: disabled %s", key)
    return f"Listen mode disabled on {key}."


async def _run_trigger_loop(key: str) -> None:
    """Resilient wrapper so a crashing tick doesn't kill the loop."""
    try:
        await listen_trigger_loop(key)
    except asyncio.CancelledError:
        logger.info("listen: %s loop cancelled", key)
        raise
    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.exception("listen: %s loop crashed: %s", key, e)


async def listen_trigger_loop(key: str) -> None:
    """Background loop: every ``interval_minutes``, fire one chime-in tick.

    The loop exits cleanly when the config goes away (e.g. after
    ``/listen off``).  Each tick is wrapped so transient failures
    (network, channel disconnected, LLM hiccup) don't terminate the
    monitoring task.
    """
    # Resolve workspace lazily — at task startup the multi-agent manager
    # may not be registered yet on a fresh app boot.
    workspace = None
    while True:
        cfg = listen_configs.get(key)
        if cfg is None or not cfg.enabled:
            return

        sleep_seconds = max(_MIN_TICK_SECONDS, cfg.interval_minutes * 60)
        try:
            await asyncio.sleep(sleep_seconds)
        except asyncio.CancelledError:
            return

        cfg = listen_configs.get(key)
        if cfg is None or not cfg.enabled:
            return

        if workspace is None:
            workspace = await _resolve_workspace(cfg.agent_id)
            if workspace is None:
                continue  # try again next tick

        try:
            await _fire_once(workspace, cfg)
        except asyncio.CancelledError:
            return
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.exception(
                "listen: tick failed for %s: %s",
                key,
                e,
            )

        cfg.last_fire = _now()


async def _resolve_workspace(agent_id: str):
    """Look up the workspace via the process-scoped multi-agent manager."""
    try:
        from ....app.multi_agent_manager import (
            get_registered_multi_agent_manager,
        )
    except Exception:  # pylint: disable=broad-exception-caught
        return None
    mam = get_registered_multi_agent_manager()
    if mam is None:
        logger.warning(
            "listen: MultiAgentManager not yet registered — will retry",
        )
        return None
    try:
        return await mam.get_agent(agent_id)
    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.warning(
            "listen: get_agent(%r) failed: %s",
            agent_id,
            e,
        )
        return None


async def _fire_once(workspace, cfg: ListenConfig) -> None:
    """Run one listen tick end-to-end (route D).

    Pipeline:
      1. Per-chat busy gate via ``is_chat_busy``.  If THIS chat has a
         user @-mention in flight, skip — we'd just be talking over
         the user's own reply.
      2. ``min_chime_gap_seconds`` throttle.  If we chimed recently,
         skip even if the LLM would say CHIME now.
      3. Decision step: ``should_chime_in``.  Returns True iff the
         decision agent emitted CHIME.
      4. Action step: ``execute_chime_action``.  Reads a snapshot of
         the chat's persisted memory, runs a transient ReActAgent with
         the full toolkit + injection guard, dispatches via channel.send
         and appends the chime to the FRESH session state.

    Each tick is wrapped in the caller (``listen_trigger_loop``) so
    transient failures don't terminate the loop.
    """
    from ..proactive.proactive_utils import is_chat_busy
    from .listen_responder import execute_chime_action, should_chime_in

    # 1. Per-chat busy gate (Tension R2 = A).
    chat_busy = await is_chat_busy(
        workspace,
        cfg.chat_id,
        session_id=cfg.session_id,
        user_id=cfg.user_id,
        channel=cfg.channel_name,
    )
    if chat_busy:
        logger.info(
            "listen: chat busy for %s:%s — skipping tick",
            cfg.channel_name,
            cfg.chat_id,
        )
        return

    # 2. min_chime_gap_seconds throttle (Issue 9 = A).
    if cfg.last_chime_ts is not None:
        gap = (_now() - cfg.last_chime_ts).total_seconds()
        if gap < cfg.min_chime_gap_seconds:
            logger.info(
                "listen: within min_chime_gap (%.0fs < %ds) for %s:%s "
                "— skipping tick",
                gap,
                cfg.min_chime_gap_seconds,
                cfg.channel_name,
                cfg.chat_id,
            )
            return

    # 3. Decision step.
    will_chime = await should_chime_in(workspace, cfg)
    if not will_chime:
        return

    # 4. Action step.  Records last_chime_ts only when the action
    # actually dispatched (action may still self-abort with PASS).
    dispatched = await execute_chime_action(workspace, cfg)
    if dispatched:
        cfg.last_chime_ts = _now()
