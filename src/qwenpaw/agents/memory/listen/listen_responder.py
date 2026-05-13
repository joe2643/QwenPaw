# -*- coding: utf-8 -*-
"""Listen-mode responder (route D).

Two-step design:

1. ``should_chime_in`` — cheap LLM call that returns CHIME or PASS.  No
   tools, no memory.  Pays one round-trip per tick.

2. ``execute_chime_action`` — fires only when the decision is CHIME.
   Snapshots the chat's persisted memory, runs a transient ReActAgent
   with the full toolkit + ``LISTEN_INJECTION_GUARD`` suffix against
   the snapshot, and on a non-PASS response: sends the text via the
   channel and appends ONLY the assistant chime-in to the *fresh* real
   session memory.  Synthetic user-turns are never persisted.

The route-D pivot (vs the earlier share_session/stream_query plan) buys
us out of:

- A formatter wrapper that filters synthetic listen-trigger turns.
- A session-write race between listen action and concurrent user reply.
- A ``share_session_in_group`` config validation that doesn't actually
  exist on the channels listen supports.

Trade-off: the action step builds a fresh agent each fire (no KV-cache
hit on the full agent's prompt prefix).  Decision step still hits cache
because its prompt is short and prefix-stable.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import TYPE_CHECKING, Any, List, Optional

from agentscope.agent import ReActAgent
from agentscope.message import Msg

from ....config.config import load_agent_config
from ...model_factory import create_model_and_formatter
from .listen_prompts import (
    LISTEN_DECISION_PROMPT,
    LISTEN_DECISION_PROMPT_AGGRESSIVE,
    LISTEN_INJECTION_GUARD,
    render_action_buffer,
)
from .listen_types import ListenConfig

if TYPE_CHECKING:  # pragma: no cover
    from ....app.workspace import Workspace

logger = logging.getLogger(__name__)


# Hard caps on the rendered chatter excerpt — keeps the prompt focused
# and bounds cost.  These are intentionally tighter than proactive's
# 50K char budget because listen fires far more frequently.
_LISTEN_MAX_ENTRIES = 20
_LISTEN_MAX_CHARS = 3000

# Hard caps on the prior-conversation block (persisted agent memory of
# past @-mention exchanges).  Smaller than the chatter cap because we
# want NEW chatter to dominate the model's attention; prior conversation
# is anchoring context, not the primary signal.
_LISTEN_PRIOR_MAX_MESSAGES = 12
_LISTEN_PRIOR_MAX_CHARS = 2000

# Marker rendered when a context source has nothing to show.  Visible
# in the prompt so the model can tell "no signal" apart from "I forgot
# to populate this block".
_LISTEN_EMPTY_CONTEXT_MARKER = "(none)"

# Action-step max iterations.  Listen replies should be quick: a tool
# call or two at most, then text.  Cap tight to prevent the action
# agent from spiralling.
_LISTEN_ACTION_MAX_ITERS = 5

# What we treat as "the agent self-aborted with PASS".  Applies to BOTH
# the decision step output AND the action step's final text (per
# Tension R3: action step gets its own honourable-exit protocol).
_LISTEN_PASS_TOKENS = frozenset(
    {
        "",
        "pass",
        "pass.",
        "(pass)",
        "[pass]",
        "skip",
        "no",
        "none",
        "no reply",
        "no response",
    },
)

# Prefix stamped onto chime-in turns we append to the real session
# memory.  Lets the main agent (when next @-mentioned) tell its own
# past output apart from chime-ins delivered without an explicit user
# turn, so it doesn't repeat the same chime-in or get confused about
# turn order.
_LISTEN_REPLY_MEMORY_PREFIX = "[listen chime-in] "


# ---------------------------------------------------------------------------
# Helpers — context loading, buffer formatting, response classification.
# ---------------------------------------------------------------------------


async def _load_session_history(
    workspace: "Workspace",
    config: ListenConfig,
    *,
    max_messages: int = _LISTEN_PRIOR_MAX_MESSAGES,
    max_chars: int = _LISTEN_PRIOR_MAX_CHARS,
) -> str:
    """Read the chat's persisted agent memory and render it as
    ``[role]: text`` lines for the decision-step prompt.

    Returns an empty string on any failure (missing runner / unknown
    session / unreadable state file) so the caller can fall back to the
    buffer-only prompt without aborting the tick.

    Skips system / tool_use / tool_result blocks — only text from
    ``user`` / ``assistant`` turns goes in.  This is enough for persona
    anchoring without bloating the prompt with internal artefacts.
    """
    session_id = config.session_id or ""
    if not session_id:
        return ""
    user_id = config.user_id or session_id

    runner = getattr(workspace, "runner", None)
    session_svc = getattr(runner, "session", None) if runner else None
    if session_svc is None:
        return ""

    try:
        state = await session_svc.get_session_state_dict(
            session_id,
            user_id,
            config.channel_name,
        )
    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.debug(
            "listen: session state read failed for %s/%s/%s: %s",
            config.channel_name,
            user_id,
            session_id,
            e,
        )
        return ""
    if not state:
        return ""

    memory_state = state.get("agent", {}).get("memory")
    if not memory_state:
        return ""

    try:
        from agentscope.memory import InMemoryMemory

        memory = InMemoryMemory()
        memory.load_state_dict(memory_state, strict=False)
        messages = await memory.get_memory()
    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.debug(
            "listen: failed to materialise InMemoryMemory for "
            "%s/%s: %s",
            config.channel_name,
            session_id,
            e,
        )
        return ""

    # ``get_memory()`` returns oldest-first; iterate in reverse so we keep
    # the tail (newest) when we hit the char cap, then reverse the kept
    # subset back to oldest-first for prompt readability.
    selected: List[str] = []
    total = 0
    for msg in reversed(messages):
        role = getattr(msg, "role", "") or "?"
        if role == "system":
            continue
        content = getattr(msg, "content", None)
        text_parts: List[str] = []
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    txt = block.get("text") or ""
                    if txt.strip():
                        text_parts.append(txt.strip())
        elif isinstance(content, str) and content.strip():
            text_parts.append(content.strip())
        if not text_parts:
            continue
        body = " ".join(text_parts).replace("\n", " ")[:400]
        line = f"[{role}]: {body}"
        if total + len(line) > max_chars:
            break
        selected.append(line)
        total += len(line) + 1
        if len(selected) >= max_messages:
            break

    return "\n".join(reversed(selected))


def _format_buffer_for_decision(buffer: List[Any]) -> str:
    """Render ``_group_history`` entries as ``[sender]: body`` lines.

    Used for the DECISION step (lighter prompt).  Action step uses
    ``render_action_buffer`` from ``listen_prompts`` which adds the
    per-line ``[third-party]`` UNTRUSTED prefix.
    """
    tail = list(buffer)[-_LISTEN_MAX_ENTRIES:]
    lines: List[str] = []
    total = 0
    for entry in tail:
        if not isinstance(entry, dict):
            continue
        sender = str(entry.get("sender", "?"))[:60]
        body = str(entry.get("body", "")).replace("\n", " ")[:400]
        line = f"[{sender}]: {body}"
        if total + len(line) > _LISTEN_MAX_CHARS:
            break
        lines.append(line)
        total += len(line) + 1
    return "\n".join(lines)


def _is_pass_response(text: Optional[str]) -> bool:
    """Return True when the LLM's reply should be treated as 'stay silent'.

    Used by BOTH steps: decision step (non-CHIME = PASS) and action step
    (the LISTEN_INJECTION_GUARD tells the action agent to output the
    literal token PASS for an honourable silent exit).
    """
    if text is None:
        return True
    cleaned = text.strip()
    cleaned = cleaned.strip("`\"' \t").strip()
    if not cleaned:
        return True
    return cleaned.lower() in _LISTEN_PASS_TOKENS


def _is_chime_response(text: Optional[str]) -> bool:
    """Return True when the decision step explicitly said CHIME.

    Strict: only ``CHIME`` (case-insensitive, possibly fenced) counts.
    Anything else (PASS, junk, hallucinated text) is treated as PASS by
    the caller — defensive default keeps the bot quiet when in doubt.
    """
    if text is None:
        return False
    cleaned = text.strip().strip("`\"' \t.").strip()
    return cleaned.lower() == "chime"


def _select_decision_prompt(verbosity: str) -> str:
    """Pick the decision-step prompt based on per-chat verbosity."""
    if verbosity == "aggressive":
        return LISTEN_DECISION_PROMPT_AGGRESSIVE
    return LISTEN_DECISION_PROMPT


def _maybe_dump_listen_prompt(
    config: ListenConfig,
    *args: str,
) -> None:
    """When ``COPAW_LISTEN_DUMP=1``, log prompt + raw reply per step.

    Accepts both v1 and v2 signatures so existing tests still pass:
    - v1: ``(config, prompt_text, raw_response)``
    - v2: ``(config, label, prompt_text, raw_response)``

    Truncated to keep log volume bounded.  Off by default — flip via
    a systemd drop-in when debugging why the LLM keeps PASS-ing.
    """
    if os.environ.get("COPAW_LISTEN_DUMP") != "1":
        return
    if len(args) == 2:
        label = "listen"
        prompt_text, raw_response = args
    elif len(args) == 3:
        label, prompt_text, raw_response = args
    else:
        return
    cap = 3000
    pt = (
        prompt_text
        if len(prompt_text) <= cap
        else prompt_text[:cap] + "...[truncated]"
    )
    rt = (
        raw_response
        if len(raw_response) <= cap
        else raw_response[:cap] + "...[truncated]"
    )
    logger.info(
        "listen_dump[%s] %s:%s verbosity=%s prompt=<<<\n%s\n>>> "
        "raw=<<<\n%s\n>>>",
        label,
        config.channel_name,
        config.chat_id,
        config.verbosity,
        pt,
        rt,
    )


# ---------------------------------------------------------------------------
# Decision step.
# ---------------------------------------------------------------------------


async def _ask_llm_to_chime_in(
    history_text: str,
    config: ListenConfig,
    prior_conversation_text: str = "",
) -> str:
    """Single non-streaming LLM call returning CHIME or PASS (or junk).

    Defensive: callers should treat any non-CHIME output as PASS via
    ``_is_chime_response``.
    """
    agent_id = config.agent_id or ""
    agent_config = load_agent_config(agent_id) if agent_id else None
    language = (
        getattr(agent_config, "language", "en") if agent_config else "en"
    )
    agent_name = (
        getattr(agent_config, "name", "Assistant") or "Assistant"
        if agent_config
        else "Assistant"
    )

    template = _select_decision_prompt(config.verbosity)
    prompt_text = template.format(
        agent_name=agent_name,
        channel_name=config.channel_name,
        language=language,
        prior_conversation=(
            prior_conversation_text or _LISTEN_EMPTY_CONTEXT_MARKER
        ),
        history=history_text,
    )

    model, formatter = create_model_and_formatter(agent_id=agent_id)
    # Bare ReActAgent with empty toolkit and no memory — cheapest way
    # to reuse the main agent's model + formatter without dragging in
    # tools, hooks, or persistence.  One reply call ⇒ one round-trip.
    from agentscope.tool import Toolkit

    sub_agent = ReActAgent(
        name="ListenDecider",
        model=model,
        sys_prompt="You are a quiet observer in a group chat.",
        toolkit=Toolkit(),
        formatter=formatter,
        memory=None,
        max_iters=1,
    )
    response = await sub_agent.reply(
        Msg(name="User", role="user", content=prompt_text),
    )
    raw = "" if response is None else (response.get_text_content() or "")
    _maybe_dump_listen_prompt(config, "decision", prompt_text, raw)
    return raw


async def should_chime_in(
    workspace: "Workspace",
    config: ListenConfig,
) -> bool:
    """Run one decision tick.  Returns True iff the agent should chime.

    Reads:
    - the chat's ``_group_history`` buffer (must be non-empty and have
      grown since ``config.last_seen_ts``).
    - the chat's persisted agent memory (best-effort, for borderline
      cases — provides persona anchoring).

    Side effect: advances ``config.last_seen_ts`` even on PASS, so the
    next tick doesn't re-ask the LLM about the same content.
    """
    channel_manager = getattr(workspace, "channel_manager", None)
    if channel_manager is None:
        return False
    try:
        channel = await channel_manager.get_channel(config.channel_name)
    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.warning(
            "listen: get_channel(%r) failed: %s",
            config.channel_name,
            e,
        )
        return False
    if channel is None:
        logger.debug(
            "listen: channel=%s not registered yet — skipping tick",
            config.channel_name,
        )
        return False

    buffer_map = getattr(channel, "_group_history", None)
    if not isinstance(buffer_map, dict):
        return False
    buffer = buffer_map.get(config.chat_id) or []
    if not buffer:
        logger.info(
            "listen: buffer empty for %s:%s — skipping tick "
            "(no non-mentioned non-slash chatter since last flush)",
            config.channel_name,
            config.chat_id,
        )
        return False

    latest_ts = ""
    for entry in reversed(buffer):
        if isinstance(entry, dict):
            ts = str(entry.get("ts", ""))
            if ts:
                latest_ts = ts
                break
    if latest_ts and latest_ts == config.last_seen_ts:
        logger.debug(
            "listen: buffer for %s:%s unchanged since last tick — skipping",
            config.channel_name,
            config.chat_id,
        )
        return False

    history_text = _format_buffer_for_decision(buffer)
    if not history_text:
        return False

    prior_conversation_text = await _load_session_history(workspace, config)

    try:
        raw = await _ask_llm_to_chime_in(
            history_text,
            config,
            prior_conversation_text=prior_conversation_text,
        )
    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.warning(
            "listen: decision LLM call failed for %s:%s — %s",
            config.channel_name,
            config.chat_id,
            e,
        )
        return False

    # Advance the dedupe cursor whether we chime or not — otherwise the
    # same buffer head would re-trigger the decision on the next tick.
    if latest_ts:
        config.last_seen_ts = latest_ts

    if not _is_chime_response(raw):
        logger.info(
            "listen: decision PASS for %s:%s (raw=%r)",
            config.channel_name,
            config.chat_id,
            raw[:80] if raw else "",
        )
        return False

    logger.info(
        "listen: decision CHIME for %s:%s",
        config.channel_name,
        config.chat_id,
    )
    return True


# ---------------------------------------------------------------------------
# Action step (route D: ephemeral session, full toolkit, append at end).
# ---------------------------------------------------------------------------


def _build_action_agent(
    workspace: "Workspace",
    config: ListenConfig,
    snapshot_memory: Any,
) -> ReActAgent:
    """Build a transient ReActAgent for one chime-in.

    Uses the main agent's toolkit so the action step has the full tool
    stack (web_search, calculators, channel tools, etc.).  Sys prompt
    is the agent's normal sys_prompt + ``LISTEN_INJECTION_GUARD``.

    The ``LISTEN_INJECTION_GUARD`` is the ONLY mitigation against
    prompt injection through untrusted third-party chatter — see
    Codex tension #7 + locked Tension R1 = B (accepted risk).
    """
    agent_id = config.agent_id or ""
    agent_config = load_agent_config(agent_id) if agent_id else None
    agent_name = (
        (getattr(agent_config, "name", None) or "Assistant")
        if agent_config
        else "Assistant"
    )

    main_agent = getattr(workspace, "agent", None)
    base_sys_prompt = (
        getattr(main_agent, "sys_prompt", "") or ""
        if main_agent is not None
        else ""
    )
    if not base_sys_prompt:
        # Fall back to a minimal persona if we can't reach the main
        # agent's sys_prompt — better than an empty system message.
        base_sys_prompt = f"You are {agent_name}, a peer in this chat."

    sys_prompt = base_sys_prompt + LISTEN_INJECTION_GUARD

    # Reuse the main agent's toolkit reference if available — the action
    # agent runs with the full tool stack (CEO + Tension R1 = B: accept
    # risk).  Fall back to a fresh Toolkit if the main agent isn't
    # reachable, which strips tool capability (safer degraded mode).
    if main_agent is not None and getattr(main_agent, "toolkit", None):
        toolkit = main_agent.toolkit
    else:
        from agentscope.tool import Toolkit

        toolkit = Toolkit()

    model, formatter = create_model_and_formatter(agent_id=agent_id)

    return ReActAgent(
        name=agent_name,
        model=model,
        sys_prompt=sys_prompt,
        toolkit=toolkit,
        formatter=formatter,
        memory=snapshot_memory,
        max_iters=_LISTEN_ACTION_MAX_ITERS,
    )


async def _snapshot_session_memory(
    workspace: "Workspace",
    config: ListenConfig,
) -> Any:
    """Build an InMemoryMemory loaded with the chat's persisted state.

    Returns a fresh empty InMemoryMemory on any failure — the action
    agent still runs, just without persona context.
    """
    from agentscope.memory import InMemoryMemory

    memory = InMemoryMemory()
    session_id = config.session_id or ""
    if not session_id:
        return memory
    user_id = config.user_id or session_id

    runner = getattr(workspace, "runner", None)
    session_svc = getattr(runner, "session", None) if runner else None
    if session_svc is None:
        return memory

    try:
        state = await session_svc.get_session_state_dict(
            session_id,
            user_id,
            config.channel_name,
        )
    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.debug(
            "listen: snapshot read failed for %s/%s/%s: %s",
            config.channel_name,
            user_id,
            session_id,
            e,
        )
        return memory

    memory_state = (state or {}).get("agent", {}).get("memory")
    if not memory_state:
        return memory

    try:
        memory.load_state_dict(memory_state, strict=False)
    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.warning(
            "listen: snapshot load_state_dict failed for %s:%s — %s",
            config.channel_name,
            session_id,
            e,
        )
        # Return the empty memory rather than a partially-loaded one —
        # cleaner failure mode.
        from agentscope.memory import InMemoryMemory as _IM

        return _IM()
    return memory


async def _append_chime_to_real_session(
    workspace: "Workspace",
    config: ListenConfig,
    reply_text: str,
) -> bool:
    """Persist the chime-in into the chat's agent memory.

    Re-reads the LATEST session state at append time (not the snapshot
    captured before the action ran), so any user reply that landed
    during the action step is preserved.  We append a single tagged
    assistant ``Msg`` — the synthetic user-turn the action agent saw is
    NEVER persisted (that's the whole point of route D).
    """
    if not reply_text:
        return False
    session_id = config.session_id or ""
    user_id = config.user_id or session_id
    if not session_id:
        return False

    runner = getattr(workspace, "runner", None)
    session_svc = getattr(runner, "session", None) if runner else None
    if session_svc is None:
        return False

    try:
        from agentscope.memory import InMemoryMemory
        from agentscope.message import Msg, TextBlock
    except Exception:  # pylint: disable=broad-exception-caught
        return False

    # Read FRESH state — critical for the append race (Codex tension #1).
    try:
        state = await session_svc.get_session_state_dict(
            session_id,
            user_id,
            config.channel_name,
        )
    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.warning(
            "listen: session read failed before append "
            "(%s/%s/%s): %s",
            config.channel_name,
            user_id,
            session_id,
            e,
        )
        return False

    memory = InMemoryMemory()
    existing = (state or {}).get("agent", {}).get("memory")
    if existing:
        try:
            memory.load_state_dict(existing, strict=False)
        except Exception:  # pylint: disable=broad-exception-caught
            # Corrupted state — start fresh rather than drop the chime.
            memory = InMemoryMemory()

    try:
        listen_msg = Msg(
            name="listen",
            role="assistant",
            content=[
                TextBlock(
                    type="text",
                    text=_LISTEN_REPLY_MEMORY_PREFIX + reply_text,
                ),
            ],
        )
        await memory.add(listen_msg)
    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.warning(
            "listen: building assistant Msg for memory append "
            "failed: %s",
            e,
        )
        return False

    try:
        await session_svc.update_session_state(
            session_id=session_id,
            key="agent.memory",
            value=memory.state_dict(),
            user_id=user_id,
            channel=config.channel_name,
        )
        logger.info(
            "listen: appended chime-in to session memory %s/%s",
            config.channel_name,
            session_id,
        )
        return True
    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.warning(
            "listen: session write failed (%s/%s): %s",
            config.channel_name,
            session_id,
            e,
        )
        return False


async def _dispatch_chime_to_channel(
    workspace: "Workspace",
    config: ListenConfig,
    reply_text: str,
) -> bool:
    """Send ``reply_text`` to the originating chat via the channel.send API.

    Returns True when ``channel.send`` was invoked without raising.
    """
    channel_manager = getattr(workspace, "channel_manager", None)
    if channel_manager is None:
        return False
    try:
        channel = await channel_manager.get_channel(config.channel_name)
    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.warning(
            "listen: get_channel(%r) failed during dispatch: %s",
            config.channel_name,
            e,
        )
        return False
    if channel is None:
        return False

    # to_handle preference mirrors WhatsApp / Signal's send expectations.
    to_handle = (
        config.chat_meta.get("chat_jid")
        or config.chat_meta.get("group_id")
        or config.chat_meta.get("source")
        or config.chat_meta.get("chat_id")
        or config.chat_id
    )
    if not to_handle:
        logger.warning(
            "listen: no to_handle captured for %s:%s — cannot dispatch",
            config.channel_name,
            config.chat_id,
        )
        return False
    try:
        await channel.send(
            to_handle,
            reply_text,
            meta=config.chat_meta or None,
        )
        return True
    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.warning(
            "listen: channel.send to %s:%s failed: %s",
            config.channel_name,
            config.chat_id,
            e,
        )
        return False


async def execute_chime_action(
    workspace: "Workspace",
    config: ListenConfig,
) -> Optional[str]:
    """Run the action step.  Returns the dispatched text or None.

    1. Read group_history buffer fresh.
    2. Snapshot the chat's persisted agent memory.
    3. Build a transient action agent (full toolkit + injection guard).
    4. Run ``agent.reply([UNTRUSTED]-wrapped buffer)``.
    5. If the response is PASS or empty → log + return None (no
       dispatch, no append).
    6. Otherwise dispatch via channel.send and append the assistant
       chime-in to the FRESH session state.

    Bound by ``config.action_timeout_seconds`` — beyond that the room
    has moved on and the chime is stale anyway.
    """
    channel_manager = getattr(workspace, "channel_manager", None)
    if channel_manager is None:
        return None
    try:
        channel = await channel_manager.get_channel(config.channel_name)
    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.warning(
            "listen: get_channel(%r) failed at action time: %s",
            config.channel_name,
            e,
        )
        return None
    if channel is None:
        return None

    buffer_map = getattr(channel, "_group_history", None)
    if not isinstance(buffer_map, dict):
        return None
    buffer = buffer_map.get(config.chat_id) or []
    if not buffer:
        # Buffer was cleared between decision and action.  Skip.
        logger.info(
            "listen: buffer empty at action time for %s:%s — skipping",
            config.channel_name,
            config.chat_id,
        )
        return None

    snapshot_memory = await _snapshot_session_memory(workspace, config)
    action_agent = _build_action_agent(workspace, config, snapshot_memory)

    buffer_msg = Msg(
        name="group_chatter",
        role="user",
        content=render_action_buffer(buffer[-_LISTEN_MAX_ENTRIES:]),
    )

    timeout = max(10, int(config.action_timeout_seconds or 120))
    try:
        response = await asyncio.wait_for(
            action_agent.reply(buffer_msg),
            timeout=timeout,
        )
    except asyncio.TimeoutError:
        logger.warning(
            "listen: action step timed out after %ds for %s:%s",
            timeout,
            config.channel_name,
            config.chat_id,
        )
        return None
    except asyncio.CancelledError:
        raise
    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.warning(
            "listen: action step failed for %s:%s: %s",
            config.channel_name,
            config.chat_id,
            e,
        )
        return None

    raw = "" if response is None else (response.get_text_content() or "")
    _maybe_dump_listen_prompt(
        config,
        "action",
        f"<buffer with {len(buffer[-_LISTEN_MAX_ENTRIES:])} entries>",
        raw,
    )

    if _is_pass_response(raw):
        logger.info(
            "listen: action self-aborted (PASS) for %s:%s",
            config.channel_name,
            config.chat_id,
        )
        return None

    reply = raw.strip().strip("`\"'").strip()
    if not reply:
        return None

    dispatched = await _dispatch_chime_to_channel(workspace, config, reply)
    if not dispatched:
        # Channel rejected the send — skip session append so the main
        # agent's history doesn't claim the bot said something the room
        # never saw.
        return None

    await _append_chime_to_real_session(workspace, config, reply)
    logger.info(
        "listen: chimed in to %s:%s with %d chars",
        config.channel_name,
        config.chat_id,
        len(reply),
    )
    return reply


# ---------------------------------------------------------------------------
# Backwards-compat shims — preserve names the v1 code and existing tests
# import.  These are thin wrappers over the route-D pipeline; new code
# should call ``should_chime_in`` + ``execute_chime_action`` directly.
# ---------------------------------------------------------------------------


async def generate_listen_reply(
    workspace: "Workspace",
    config: ListenConfig,
) -> Optional[str]:
    """V1 compat: decision step only.  Returns ``"CHIME"`` or None.

    The text return value is preserved for older callers that expect a
    string, but its content is no longer the actual chime-in — that's
    generated by ``execute_chime_action`` instead.  Callers that just
    want the boolean should switch to ``should_chime_in``.
    """
    will_chime = await should_chime_in(workspace, config)
    return "CHIME" if will_chime else None


async def deliver_listen_reply(
    workspace: "Workspace",
    config: ListenConfig,
    reply_text: str,  # noqa: ARG001 — preserved for signature stability
) -> bool:
    """V1 compat shim.

    Route D performs dispatch *inside* ``execute_chime_action`` because
    only that step has the real reply text.  Calling this directly
    bypasses the action step, which is almost always wrong; we keep it
    only so any in-flight import doesn't break before callers migrate.
    """
    logger.warning(
        "listen: deliver_listen_reply called directly — "
        "this is route-A-style usage; route D handles dispatch inside "
        "execute_chime_action.  No-op.",
    )
    return False


async def _append_listen_reply_to_session(
    workspace: "Workspace",
    config: ListenConfig,
    reply_text: str,
) -> bool:
    """V1 compat alias for ``_append_chime_to_real_session``."""
    return await _append_chime_to_real_session(workspace, config, reply_text)


# V1 compat aliases — the older test suite imports these names.
# v2 renames are mostly cosmetic; we keep the old symbols so the
# existing tests can run unchanged where the behaviour is preserved.
_format_buffer = _format_buffer_for_decision
_select_prompt_template = _select_decision_prompt
# ``_ask_llm_to_chime_in`` and ``_ask_decision`` refer to the same
# function so older tests that monkeypatch either name still intercept
# the real call site.
_ask_decision = _ask_llm_to_chime_in
