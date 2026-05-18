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
            "listen: failed to materialise InMemoryMemory for " "%s/%s: %s",
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
    prior_conversation_text: str = "",  # noqa: ARG001 — v1 compat
    *,
    workspace: Optional["Workspace"] = None,
) -> str:
    """Single non-streaming LLM call returning CHIME or PASS (or junk).

    When ``workspace`` is provided (the normal in-app path), the
    sub-agent runs against a SNAPSHOT of the chat's persisted memory
    plus the main agent's ``sys_prompt`` and ``name``.  That means the
    decision sees the same persona / past @-mention exchanges the
    action step will see, so its CHIME / PASS call matches how the
    main agent would actually feel about the room.

    When ``workspace`` is None (legacy / unit-test callers that only
    pass history + config), the sub-agent falls back to the v2
    bare-agent + text-rendered prior-conversation path.  Tests that
    monkeypatch this function still work because they replace the
    callable entirely.

    Defensive: callers should treat any non-CHIME output as PASS via
    ``_is_chime_response``.
    """
    agent_id = config.agent_id or ""
    template = _select_decision_prompt(config.verbosity)

    # Branch on whether we have a workspace handle to snapshot.
    if workspace is not None:
        prompt_text, raw = await _ask_with_snapshot(
            workspace,
            config,
            agent_id,
            template,
            history_text,
        )
    else:
        prompt_text, raw = await _ask_with_text_render(
            agent_id,
            config,
            template,
            history_text,
            prior_conversation_text,
        )

    _maybe_dump_listen_prompt(config, "decision", prompt_text, raw)
    return raw


def _disable_thinking_on_model(model: Any) -> None:
    """Best-effort: turn off chain-of-thought / reasoning on a model.

    Decision step outputs a single token (CHIME or PASS); any
    reasoning the model does before that token is pure wasted
    latency.  Different providers expose different knobs:

    - OpenAI reasoning models: ``reasoning_effort`` instance attr.
    - z.ai / GLM via OpenAI-compat: ``extra_body={"thinking": {"type": "disabled"}}``.
    - Qwen / DashScope / Ollama: ``extra_body={"enable_thinking": False}``.

    We drill through the project's wrapper chain
    (``RetryChatModel._inner -> TokenRecordingModelWrapper._model``)
    to reach the real ``OpenAIChatModel`` (or sibling) and mutate its
    ``generate_kwargs`` in place.  The model instance is fresh per
    decision call (``create_model_and_formatter`` returns a new
    object), so this doesn't bleed into other agents.
    """
    # Drill through known wrappers.
    inner = model
    for _ in range(4):  # bounded — guard against weird wrapping cycles
        next_inner = getattr(inner, "_inner", None) or getattr(
            inner,
            "_model",
            None,
        )
        if next_inner is None:
            break
        inner = next_inner

    # Mutate generate_kwargs if the inner model exposes it.
    gk = getattr(inner, "generate_kwargs", None)
    if gk is None:
        return
    extra_body = dict(gk.get("extra_body") or {})
    extra_body.setdefault("thinking", {"type": "disabled"})
    extra_body.setdefault("enable_thinking", False)
    gk["extra_body"] = extra_body
    # Also clear reasoning_effort if the model carries one — for
    # OpenAI reasoning models this is where CoT budget lives.
    if hasattr(inner, "reasoning_effort"):
        try:
            inner.reasoning_effort = None
        except Exception:  # pylint: disable=broad-exception-caught
            pass


async def _ask_with_snapshot(
    workspace: "Workspace",
    config: ListenConfig,
    agent_id: str,
    template: str,
    history_text: str,
) -> tuple[str, str]:
    """Decision step with snapshot memory + main-agent persona.

    Returns ``(prompt_text_for_logging, raw_response)``.
    """
    from agentscope.tool import Toolkit

    snapshot_memory = await _snapshot_session_memory(workspace, config)

    main_agent = getattr(workspace, "agent", None)
    if main_agent is not None:
        agent_name = getattr(main_agent, "name", None) or "Assistant"
        base_sys_prompt = getattr(main_agent, "sys_prompt", "") or ""
    else:
        agent_name = "Assistant"
        base_sys_prompt = ""

    # Decision step gets the main agent's persona-bearing sys_prompt
    # so the LLM knows who it is and how it talks.  No injection guard
    # suffix here — the decision step has no tools, the LISTEN_INJECTION_GUARD
    # only matters for the action step that can call tools.
    decision_sys_prompt = (
        base_sys_prompt
        if base_sys_prompt
        else f"You are {agent_name}, a peer in a group chat."
    )

    # The user-turn prompt no longer carries persona slots; persona is
    # carried by sys_prompt + memory.  Only the {history} slot remains.
    prompt_text = template.format(history=history_text)

    model, formatter = create_model_and_formatter(agent_id=agent_id)
    # Decision step returns a single token; turn off any
    # reasoning/thinking the provider would do otherwise.  Each tick
    # gets a fresh model instance so this mutation is local.
    _disable_thinking_on_model(model)

    sub_agent = ReActAgent(
        name=agent_name,
        model=model,
        sys_prompt=decision_sys_prompt,
        toolkit=Toolkit(),  # still no tools — decision is text-only
        formatter=formatter,
        memory=snapshot_memory,
        max_iters=1,
    )
    response = await sub_agent.reply(
        Msg(name="User", role="user", content=prompt_text),
    )
    raw = "" if response is None else (response.get_text_content() or "")
    return prompt_text, raw


async def _ask_with_text_render(
    agent_id: str,
    config: ListenConfig,
    template: str,
    history_text: str,
    prior_conversation_text: str,
) -> tuple[str, str]:
    """v2 fallback path: bare agent + text-rendered prior conversation.

    Kept for tests that exercise ``_ask_llm_to_chime_in`` directly
    without a workspace handle.  Production callers always go through
    ``_ask_with_snapshot``.
    """
    from agentscope.tool import Toolkit

    agent_config = load_agent_config(agent_id) if agent_id else None
    agent_name = (
        (getattr(agent_config, "name", "Assistant") or "Assistant")
        if agent_config
        else "Assistant"
    )

    # The v2.1 prompt templates dropped the {agent_name} /
    # {prior_conversation} slots; surface them via a synthesised
    # leading line so the legacy text-render path still gets persona
    # signal into the user-turn.
    persona_block = (
        f"(You are {agent_name}.  Previous exchanges in this room "
        f"where you were @-mentioned:\n"
        f"{prior_conversation_text or _LISTEN_EMPTY_CONTEXT_MARKER}\n)\n\n"
    )
    prompt_text = persona_block + template.format(history=history_text)

    model, formatter = create_model_and_formatter(agent_id=agent_id)
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
    return prompt_text, raw


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

    try:
        raw = await _ask_llm_to_chime_in(
            history_text,
            config,
            workspace=workspace,
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


async def _maybe_compact_snapshot(
    workspace: "Workspace",
    config: ListenConfig,
    snapshot_memory: Any,
) -> Any:
    """Compact ``snapshot_memory`` in-place when it exceeds the main
    agent's compaction threshold.

    Reuses ``workspace.agent.context_manager.compact_context`` (the
    same LLM call the main agent runs in its ``pre_reasoning`` hook)
    so listen inherits the agent's compaction prompt + summary
    template.  If the workspace doesn't have a context manager (rare
    — only when the main agent's config disables it), we leave the
    snapshot alone; the action LLM call may fail with
    context-window-exceeded and the action step will log + skip.

    Returns the (possibly compacted) snapshot.  On any failure we
    return the original snapshot so the action can still try its
    luck rather than the whole tick disappearing silently.
    """
    agent_id = config.agent_id or ""
    if not agent_id:
        return snapshot_memory

    try:
        agent_config = load_agent_config(agent_id)
    except Exception:  # pylint: disable=broad-exception-caught
        return snapshot_memory
    if agent_config is None:
        return snapshot_memory

    main_agent = getattr(workspace, "agent", None)
    cm = getattr(main_agent, "context_manager", None) if main_agent else None
    if cm is None:
        return snapshot_memory

    try:
        from ...utils.token_counter import get_token_counter

        token_counter = get_token_counter(agent_config)
    except Exception:  # pylint: disable=broad-exception-caught
        return snapshot_memory

    try:
        messages = await snapshot_memory.get_memory(prepend_summary=False)
    except Exception:  # pylint: disable=broad-exception-caught
        return snapshot_memory
    if not messages:
        return snapshot_memory

    running_config = agent_config.running
    ccc = running_config.light_context_config.context_compact_config

    # Compute the same left-after-sys threshold the main agent uses.
    sys_text = (
        getattr(main_agent, "sys_prompt", "") or ""
        if main_agent is not None
        else ""
    )
    try:
        sys_token_count = (
            await token_counter.count(messages=[], text=sys_text)
            if sys_text
            else 0
        )
    except Exception:  # pylint: disable=broad-exception-caught
        sys_token_count = 0

    context_compact_threshold = int(
        running_config.max_input_length * ccc.compact_threshold_ratio,
    )
    context_compact_reserve = int(
        running_config.max_input_length * ccc.reserve_threshold_ratio,
    )
    left_compact_threshold = context_compact_threshold - sys_token_count
    if left_compact_threshold <= 0:
        return snapshot_memory

    # Split the snapshot via the context manager's own helper so the
    # tail-keep heuristic stays consistent with what the main agent
    # would have done.
    try:
        (
            messages_to_compact,
            messages_to_keep,
            _ctx_total_tokens,
            _ctx_keep_tokens,
        ) = await cm._check_context(  # pylint: disable=protected-access
            messages=messages,
            context_compact_threshold=left_compact_threshold,
            context_compact_reserve=context_compact_reserve,
            as_token_counter=token_counter,
        )
    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.debug(
            "listen: _check_context failed for %s:%s — %s",
            config.channel_name,
            config.chat_id,
            e,
        )
        return snapshot_memory

    if not messages_to_compact:
        return snapshot_memory

    logger.info(
        "listen: compacting snapshot for %s:%s "
        "(compact=%d, keep=%d, threshold=%d)",
        config.channel_name,
        config.chat_id,
        len(messages_to_compact),
        len(messages_to_keep),
        left_compact_threshold,
    )

    try:
        prev_summary = snapshot_memory.get_compressed_summary() or ""
    except Exception:  # pylint: disable=broad-exception-caught
        prev_summary = ""

    try:
        result = await cm.compact_context(
            messages=messages_to_compact,
            previous_summary=prev_summary,
        )
    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.warning(
            "listen: snapshot compaction LLM call failed for %s:%s — %s",
            config.channel_name,
            config.chat_id,
            e,
        )
        return snapshot_memory

    if not result.get("success"):
        logger.info(
            "listen: snapshot compaction skipped (reason=%s) for %s:%s",
            result.get("reason", "unknown"),
            config.channel_name,
            config.chat_id,
        )
        return snapshot_memory

    compact_content = result.get("history_compact") or ""
    if not compact_content:
        return snapshot_memory

    # Apply: drop compacted messages from snapshot.content, set new
    # summary.  We mutate the snapshot in place — it's a transient
    # InMemoryMemory created just for this tick, never persisted.
    keep_ids = {getattr(m, "id", None) for m in messages_to_keep}
    keep_ids.discard(None)
    try:
        snapshot_memory.content = [
            (msg, marks)
            for msg, marks in snapshot_memory.content
            if getattr(msg, "id", None) in keep_ids
        ]
        await snapshot_memory.update_compressed_summary(compact_content)
    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.warning(
            "listen: applying compacted summary to snapshot failed "
            "for %s:%s — %s",
            config.channel_name,
            config.chat_id,
            e,
        )
        return snapshot_memory

    logger.info(
        "listen: snapshot compacted for %s:%s — before=%d after=%d (tokens)",
        config.channel_name,
        config.chat_id,
        result.get("before_tokens", 0),
        result.get("after_tokens", 0),
    )
    return snapshot_memory


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
            "listen: session read failed before append " "(%s/%s/%s): %s",
            config.channel_name,
            user_id,
            session_id,
            e,
        )
        return False

    existing_memory_dict = (state or {}).get("agent", {}).get("memory") or {}
    if not isinstance(existing_memory_dict, dict):
        existing_memory_dict = {}

    memory = InMemoryMemory()
    if existing_memory_dict:
        try:
            memory.load_state_dict(existing_memory_dict, strict=False)
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
            "listen: building assistant Msg for memory append " "failed: %s",
            e,
        )
        return False

    # CRITICAL: preserve qwenpaw-specific agent.memory keys that the
    # base agentscope InMemoryMemory doesn't know about.  Without this,
    # ``_compressed_msg_ids`` (sibling-write tombstones) and
    # ``_compressed_msg_evicted_count`` get wiped on every listen
    # append, defeating the merge-on-save guard that stops a concurrent
    # reply with a pre-compaction baseline from resurrecting compacted
    # messages.  Symptom we hit in prod: 17-msg post-compact state grew
    # back to 223 msgs on the next reply, 5 minutes after compaction.
    #
    # The merge starts from the existing dict and only overlays the
    # keys that the base class actually serialized (content +
    # _compressed_summary).  Unknown keys flow through untouched.
    merged_memory_dict = dict(existing_memory_dict)
    merged_memory_dict.update(memory.state_dict())

    try:
        await session_svc.update_session_state(
            session_id=session_id,
            key="agent.memory",
            value=merged_memory_dict,
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
    # If the snapshot is large enough to threaten the model's context
    # window, compact it before we feed it to the action agent.  The
    # transient agent itself has no LightContextManager hook, so this
    # is the only opportunity to keep its prompt under the limit.
    snapshot_memory = await _maybe_compact_snapshot(
        workspace,
        config,
        snapshot_memory,
    )
    action_agent = _build_action_agent(workspace, config, snapshot_memory)

    buffer_msg = Msg(
        name="group_chatter",
        role="user",
        content=render_action_buffer(buffer[-_LISTEN_MAX_ENTRIES:]),
    )

    # Show a typing indicator in the room while the action agent
    # thinks.  Without this, listen replies look like the bot suddenly
    # blurts out a message — there's no "..." that normal @-mention
    # replies trigger via consume().  Channels that don't support
    # presence return None; stop_typing is a no-op in that case.
    to_handle = (
        config.chat_meta.get("chat_jid")
        or config.chat_meta.get("group_id")
        or config.chat_meta.get("source")
        or config.chat_meta.get("chat_id")
        or config.chat_id
    )
    typing_handle = None
    try:
        typing_handle = await channel.start_typing(
            to_handle,
            meta=config.chat_meta or None,
        )
        if typing_handle is not None:
            logger.info(
                "listen: typing indicator started for %s:%s",
                config.channel_name,
                config.chat_id,
            )
        else:
            logger.debug(
                "listen: channel.start_typing returned None for %s:%s "
                "(channel may not support presence)",
                config.channel_name,
                config.chat_id,
            )
    except Exception as e:  # pylint: disable=broad-exception-caught
        # Indicator is cosmetic — never let a presence failure abort
        # the chime-in.
        logger.warning(
            "listen: start_typing failed for %s:%s — %s",
            config.channel_name,
            config.chat_id,
            e,
        )
        typing_handle = None

    timeout = max(10, int(config.action_timeout_seconds or 120))
    try:
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
    finally:
        # Always stop the indicator — including on PASS / timeout /
        # cancellation paths — so the room doesn't see a stuck "..."
        # after the action quietly aborts.
        if typing_handle is not None:
            try:
                await channel.stop_typing(typing_handle)
                logger.info(
                    "listen: typing indicator stopped for %s:%s",
                    config.channel_name,
                    config.chat_id,
                )
            except Exception as e:  # pylint: disable=broad-exception-caught
                logger.warning(
                    "listen: stop_typing failed for %s:%s — %s",
                    config.channel_name,
                    config.chat_id,
                    e,
                )

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
    # The action agent's memory contains past chime-ins tagged with the
    # ``_LISTEN_REPLY_MEMORY_PREFIX`` so the main agent can tell them
    # apart from real @-mention replies.  The agent sometimes mimics
    # that prefix in its OWN output, which would then leak into the
    # channel.send below as a literal ``"[listen chime-in] ..."``
    # message.  Strip it defensively.
    if reply.startswith(_LISTEN_REPLY_MEMORY_PREFIX):
        reply = reply[len(_LISTEN_REPLY_MEMORY_PREFIX) :].lstrip()
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
