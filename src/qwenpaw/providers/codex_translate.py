# -*- coding: utf-8 -*-
"""Shared OpenAI chat/completions ↔ ChatGPT Codex Responses API
translation helpers.

The ChatGPT backend endpoint that accepts OAuth bearer tokens
(``chatgpt.com/backend-api/codex/responses``) speaks a different
shape from OpenAI's public ``/v1/chat/completions`` — different
field names (``input`` vs ``messages``), different tool-call shapes
(``function_call`` / ``function_call_output`` items vs
``assistant.tool_calls`` / ``role=tool`` messages), different
streaming events (``response.output_text.delta`` vs
``choice.delta.content``).  This module is the single place that
knows how to map between the two, consumed by
:mod:`qwenpaw.providers.codex_oauth_model` — an agentscope
``OpenAIChatModel`` subclass that does the translation in-process,
so CoPaw agents can use ChatGPT OAuth without a separate daemon.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from typing import Any, AsyncIterator

import httpx

logger = logging.getLogger(__name__)

# Models the ChatGPT backend actually serves under Codex OAuth.  Any
# caller-supplied model ID outside this set is silently forced to the
# default (upstream 400s otherwise — ``o3``, ``gpt-5-codex``, etc. are
# rejected server-side for ChatGPT-account OAuth tokens).
# ``DEFAULT_MODEL`` is used only when the caller omits ``model``
# entirely (empty/missing) — a programmer-error fallback.  Picked
# ``gpt-5.2`` because it's the one slug the ChatGPT backend
# currently advertises via ``/codex/models`` with ``visibility=list``
# for consumer accounts; other slugs (gpt-5.4, gpt-5.4-mini) also
# work today but aren't in the published catalogue and may be
# revoked without notice.  No allow-list — we forward whatever the
# caller / UI / discovery says, and the backend's 400 is the source
# of truth for what this account can reach.
DEFAULT_MODEL = "gpt-5.2"


# =========================================================================
# Request translation: chat/completions → Responses API
# =========================================================================


def content_to_plain_text(content: Any) -> str:
    """Flatten chat ``content`` (str OR list of content blocks) into
    plain text.  Image blocks collapse to ``"[image attached]"``
    placeholders; lossy, but the Responses API only accepts strings
    for system / assistant text / tool-result content.  For user
    messages where images must survive, use
    :func:`content_to_responses_items` instead.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if not isinstance(item, dict):
                continue
            t = item.get("type")
            if t in ("text", "input_text", "output_text"):
                parts.append(str(item.get("text", "")))
            elif t in ("image_url", "image", "input_image"):
                parts.append("[image attached]")
        return "".join(parts)
    return ""


def content_to_responses_items(content: Any) -> list[dict]:
    """Translate chat-message ``content`` (for user-role messages)
    into a Responses-API content array.  Supports text and image
    blocks — chat ``image_url`` / Anthropic-style ``image.source``
    both land on Responses' ``input_image``.
    """
    if isinstance(content, str):
        return [{"type": "input_text", "text": content}]
    if not isinstance(content, list):
        return [{"type": "input_text", "text": ""}]

    out: list[dict] = []
    for item in content:
        if not isinstance(item, dict):
            continue
        t = item.get("type")
        if t in ("text", "input_text", "output_text"):
            text = str(item.get("text", ""))
            if text:
                out.append({"type": "input_text", "text": text})
        elif t in ("image_url", "input_image"):
            iu = item.get("image_url")
            if isinstance(iu, dict):
                url = iu.get("url", "")
            else:
                url = str(iu) if iu else ""
            if url:
                out.append({"type": "input_image", "image_url": url})
        elif t == "image":
            source = item.get("source") or {}
            if source.get("type") == "base64":
                mime = source.get("media_type") or "image/png"
                data = source.get("data", "")
                if data:
                    out.append(
                        {
                            "type": "input_image",
                            "image_url": f"data:{mime};base64,{data}",
                        },
                    )
            elif source.get("type") == "url":
                u = source.get("url", "")
                if u:
                    out.append({"type": "input_image", "image_url": u})
    return out or [{"type": "input_text", "text": ""}]


def convert_messages_to_responses_input(
    messages: list[dict],
) -> tuple[str, list[dict]]:
    """Split chat ``messages`` into a single ``instructions`` string
    (concatenated system prompts) and a Responses-API ``input`` list
    that preserves assistant text, tool calls, and tool results in
    their original ordering.
    """
    instructions_parts: list[str] = []
    items: list[dict] = []

    for msg in messages:
        role = msg.get("role")
        content = msg.get("content")
        if role == "system":
            instructions_parts.append(content_to_plain_text(content))
            continue

        if role == "user":
            items.append(
                {
                    "type": "message",
                    "role": "user",
                    "content": content_to_responses_items(content),
                },
            )
            continue

        if role == "assistant":
            text = content_to_plain_text(content)
            if text:
                items.append(
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [
                            {"type": "output_text", "text": text},
                        ],
                    },
                )
            for tc in msg.get("tool_calls") or []:
                fn = tc.get("function") or {}
                items.append(
                    {
                        "type": "function_call",
                        "call_id": tc.get("id")
                        or f"call_{uuid.uuid4().hex[:12]}",
                        "name": fn.get("name", ""),
                        "arguments": fn.get("arguments", "") or "",
                    },
                )
            continue

        if role == "tool":
            items.append(
                {
                    "type": "function_call_output",
                    "call_id": msg.get("tool_call_id", ""),
                    "output": content_to_plain_text(content),
                },
            )
            continue

    instructions = "\n\n".join(p for p in instructions_parts if p.strip())
    return instructions, items


def convert_tools(tools: list[dict] | None) -> list[dict] | None:
    """Chat-completions tools (``{type:"function", function:{name,...}}``)
    → Responses tools (flattened ``{type:"function", name, ...}``).
    """
    if not tools:
        return None
    out: list[dict] = []
    for t in tools:
        if t.get("type") != "function":
            out.append(t)
            continue
        fn = t.get("function") or {}
        out.append(
            {
                "type": "function",
                "name": fn.get("name", ""),
                "description": fn.get("description", ""),
                "parameters": fn.get("parameters")
                or {
                    "type": "object",
                    "properties": {},
                },
                "strict": False,
            },
        )
    return out


def build_responses_body(chat_body: dict) -> dict:
    """Full chat/completions body → Responses API body.

    Always streams upstream; callers that need non-streaming drain
    the SSE stream via :func:`collect_as_chat_completion`.  Drops
    fields the ChatGPT-account backend rejects silently (``max_output_tokens``,
    most sampling knobs) — forwarding them blindly yields 400.
    """
    # Pass whatever the caller asked for straight through to the
    # upstream — the ChatGPT backend is the source of truth for
    # which slugs its account tier can reach.  Silent coercion
    # (``model not in ALLOWED_MODELS → DEFAULT_MODEL``) used to
    # hide mis-configured agents: picking ``gpt-5.5`` in the UI
    # would greenlight the test button while actually running
    # ``gpt-5.4`` under the hood, because this branch rewrote the
    # request behind the caller's back.  Now the backend 400
    # propagates to the UI ("model not supported with a ChatGPT
    # account"), which is the honest answer.  ``ALLOWED_MODELS``
    # stays around as a probe-verified reference set.
    model = chat_body.get("model") or DEFAULT_MODEL
    instructions, input_items = convert_messages_to_responses_input(
        chat_body.get("messages") or [],
    )
    body: dict[str, Any] = {
        "model": model,
        "input": input_items,
        "stream": True,
        "store": False,
        # ChatGPT backend rejects empty instructions with 400.
        "instructions": instructions or "You are a helpful assistant.",
    }

    tools = convert_tools(chat_body.get("tools"))
    if tools:
        body["tools"] = tools

    tool_choice = chat_body.get("tool_choice")
    if tool_choice is not None:
        body["tool_choice"] = tool_choice

    reasoning_effort = chat_body.get("reasoning_effort") or "low"
    body["reasoning"] = {"effort": reasoning_effort}

    # Speed tier — forwarded as ``service_tier`` to the Responses API.
    # The caller passes a friendly value (``fast`` / ``standard`` /
    # ``flex``) and we translate to the wire form the ChatGPT backend
    # actually accepts: Codex CLI maps its ``Fast`` variant to the
    # literal string ``"priority"`` (see codex-rs/core/src/client.rs).
    # On a consumer ChatGPT account (Pro), backend reports
    # ``service_tier: "default"`` on response.completed even when Fast
    # was applied — reporting is misleading but the routing is real
    # (measured ~15-25% throughput gain on gpt-5.4/5.5).
    #
    # ``standard`` resolves to no field at all, matching the Codex CLI
    # default for non-enterprise plans.  Any non-``fast``/``flex``
    # string is dropped with a warning rather than forwarded blindly,
    # since the backend 400s on unknown values (``auto``, ``default``,
    # ``fast``, ``standard`` all rejected on probe).
    speed = chat_body.get("service_tier")
    if speed == "fast":
        body["service_tier"] = "priority"
    elif speed == "flex":
        body["service_tier"] = "flex"
    elif speed and speed not in ("standard",):
        logger.warning(
            "Dropping unknown codex service_tier %r — accepted values "
            "are fast / standard / flex",
            speed,
        )

    return body


# =========================================================================
# Response translation: Responses API SSE → chat/completions chunks
# =========================================================================


class StreamState:
    """Per-request accumulator for the SSE → chat-completions translation.
    Reused across each SSE event callback; owns response_id, created
    timestamp, the in-progress tool_call table, and the final usage /
    finish_reason once ``response.completed`` arrives.
    """

    __slots__ = (
        "model",
        "response_id",
        "created",
        "tool_calls",
        "item_id_to_index",
        "finished",
        "finish_reason",
        "emitted_role",
        "final_usage",
        "active_item_type",
        "active_item_phase",
    )

    def __init__(self, model: str) -> None:
        self.model = model
        self.response_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        self.created = int(time.time())
        # tool_call index → partial {id, name, args}
        self.tool_calls: dict[int, dict[str, Any]] = {}
        # upstream item_id (``fc_xxx``) → tool_call index
        self.item_id_to_index: dict[str, int] = {}
        self.finished = False
        self.finish_reason: str | None = None
        self.emitted_role = False
        self.final_usage: dict[str, Any] | None = None
        # Type of the currently-streaming output item ("message",
        # "reasoning", "function_call", ...).  Tracked so we only
        # forward text deltas that belong to a user-facing message
        # — reasoning summary text emitted by Codex/gpt-5.x for
        # OAuth users gets dropped instead of leaking to the
        # downstream channel.
        self.active_item_type: str | None = None
        # ChatGPT's Responses API tags assistant message items with
        # a ``phase`` field — "commentary" for scratch preamble the
        # model emits before tool calls, "final_answer" for the
        # actual user-facing reply.  Both share the same
        # ``response.output_text.delta`` event stream; only the
        # parent ``output_item.added.item.phase`` distinguishes
        # them.  We capture the phase per item and drop deltas for
        # commentary items so the channel layer never sees Codex
        # scratch text.  Reference: OpenClaw's metadata-based
        # detection (https://github.com/openclaw/openclaw —
        # ``OpenAIResponsesAssistantPhase`` in
        # ``src/agents/openai-ws-connection.ts``).
        self.active_item_phase: str | None = None


def _chat_chunk(
    state: StreamState,
    delta: dict,
    finish_reason: str | None = None,
) -> dict:
    """Build a single ``chat.completion.chunk`` dict with one choice."""
    return {
        "id": state.response_id,
        "object": "chat.completion.chunk",
        "created": state.created,
        "model": state.model,
        "choices": [
            {
                "index": 0,
                "delta": delta,
                "finish_reason": finish_reason,
            },
        ],
    }


async def translate_responses_events_to_chat_chunks(
    upstream: httpx.Response,
    state: StreamState,
) -> AsyncIterator[dict]:
    """Read an upstream Responses-API SSE stream and yield chat/completions
    ``chat.completion.chunk`` dicts in the order a caller would see them
    from ``/v1/chat/completions`` streaming.  The final chunk carries
    ``finish_reason`` and, if available, usage stats.

    Raises :class:`RuntimeError` on ``response.failed`` upstream events.
    """
    if not state.emitted_role:
        state.emitted_role = True
        yield _chat_chunk(state, {"role": "assistant"})

    async for raw_line in upstream.aiter_lines():
        if not raw_line or not raw_line.startswith("data:"):
            continue
        data = raw_line[5:].strip()
        if not data or data == "[DONE]":
            continue
        try:
            ev = json.loads(data)
        except json.JSONDecodeError:
            continue

        ev_type = ev.get("type", "")

        # Text deltas — non-message items (reasoning summary)
        # always drop.  Message items split by ``phase``:
        #   - ``commentary`` (Codex/gpt-5.x scratch preamble like
        #     "Need view_video maybe...") emits as
        #     ``reasoning_content`` — agentscope reads that into
        #     a ``ThinkingBlock`` which the runner adapter turns
        #     into a ``MessageType.REASONING`` event, which the
        #     channel layer already drops from user-facing send
        #     while the Console UI still renders it in the
        #     thinking pane.
        #   - ``final_answer`` (or missing — older models)
        #     emits as ``content`` and reaches the channel as
        #     before.
        # Result: scratch never leaks to chat surfaces but UI
        # operators retain full visibility into the model's
        # internal reasoning.  Reference: OpenClaw's metadata-
        # based phase routing.
        if ev_type == "response.output_text.delta":
            if state.active_item_type not in (None, "message"):
                continue
            delta_text = ev.get("delta", "") or ""
            if not delta_text:
                continue
            if state.active_item_phase == "commentary":
                yield _chat_chunk(
                    state, {"reasoning_content": delta_text},
                )
            else:
                yield _chat_chunk(state, {"content": delta_text})
            continue

        # Item lifecycle
        if ev_type == "response.output_item.added":
            item = ev.get("item") or {}
            item_type = item.get("type", "")
            state.active_item_type = item_type
            # ChatGPT Responses API tags message items with
            # ``phase`` ("commentary" or "final_answer") on the
            # output_item itself.  Capture so the text-delta
            # filter above can route by it.  Default to None
            # for non-message items / models that don't expose
            # the field.
            state.active_item_phase = item.get("phase") if item_type == "message" else None
            if item_type == "function_call":
                idx = len(state.tool_calls)
                item_id = item.get("id", "")
                call_id = (
                    item.get("call_id") or f"call_{uuid.uuid4().hex[:12]}"
                )
                name = item.get("name", "")
                state.tool_calls[idx] = {
                    "id": call_id,
                    "name": name,
                    "args": "",
                }
                if item_id:
                    state.item_id_to_index[item_id] = idx
                yield _chat_chunk(
                    state,
                    {
                        "tool_calls": [
                            {
                                "index": idx,
                                "id": call_id,
                                "type": "function",
                                "function": {"name": name, "arguments": ""},
                            },
                        ],
                    },
                )
            continue

        if ev_type == "response.output_item.done":
            state.active_item_type = None
            state.active_item_phase = None
            continue

        # Tool-call argument deltas
        if ev_type == "response.function_call_arguments.delta":
            item_id = ev.get("item_id", "")
            idx = state.item_id_to_index.get(item_id)
            if idx is None:
                continue
            delta_args = ev.get("delta", "") or ""
            state.tool_calls[idx]["args"] += delta_args
            if delta_args:
                yield _chat_chunk(
                    state,
                    {
                        "tool_calls": [
                            {
                                "index": idx,
                                "function": {"arguments": delta_args},
                            },
                        ],
                    },
                )
            continue

        if ev_type == "response.function_call_arguments.done":
            state.finish_reason = "tool_calls"
            continue

        if ev_type == "response.completed":
            resp = ev.get("response") or {}
            state.final_usage = resp.get("usage")
            if state.finish_reason is None:
                has_tool_call = any(
                    (i.get("type") == "function_call")
                    for i in (resp.get("output") or [])
                )
                state.finish_reason = "tool_calls" if has_tool_call else "stop"
            break

        if ev_type == "response.failed":
            err = (ev.get("response") or {}).get("error") or {}
            raise RuntimeError(
                f"upstream Codex responses failed: {err.get('message') or err}",
            )

    # Final chunk with finish_reason (+ optional usage)
    final = _chat_chunk(state, {}, finish_reason=state.finish_reason or "stop")
    if state.final_usage:
        final["usage"] = {
            "prompt_tokens": state.final_usage.get("input_tokens"),
            "completion_tokens": state.final_usage.get("output_tokens"),
            "total_tokens": state.final_usage.get("total_tokens"),
        }
    yield final


async def collect_as_chat_completion(
    upstream: httpx.Response,
    state: StreamState,
) -> dict:
    """Drain an upstream Responses SSE stream into a single
    non-streaming ``chat.completion`` dict.  Used by callers that
    asked for ``stream=False`` — we always stream upstream (the
    ChatGPT backend requires it) and reassemble here.
    """
    content_parts: list[str] = []
    tool_calls: list[dict] = []

    async for raw_line in upstream.aiter_lines():
        if not raw_line or not raw_line.startswith("data:"):
            continue
        data = raw_line[5:].strip()
        if not data or data == "[DONE]":
            continue
        try:
            ev = json.loads(data)
        except json.JSONDecodeError:
            continue

        t = ev.get("type", "")

        # Same item / phase gates as the streaming path — skip
        # text deltas from non-message items (reasoning summary)
        # AND from message items in ``commentary`` phase (Codex
        # scratch preamble).
        if t == "response.output_text.delta":
            if state.active_item_type not in (None, "message"):
                continue
            if state.active_item_phase == "commentary":
                continue
            content_parts.append(ev.get("delta", "") or "")
            continue

        if t == "response.output_item.added":
            item = ev.get("item") or {}
            item_type = item.get("type", "")
            state.active_item_type = item_type
            state.active_item_phase = (
                item.get("phase") if item_type == "message" else None
            )
            if item_type == "function_call":
                item_id = item.get("id", "")
                idx = len(tool_calls)
                tool_calls.append(
                    {
                        "index": idx,
                        "id": item.get("call_id")
                        or f"call_{uuid.uuid4().hex[:12]}",
                        "type": "function",
                        "function": {
                            "name": item.get("name", ""),
                            "arguments": "",
                        },
                    },
                )
                state.item_id_to_index[item_id] = idx
            continue

        if t == "response.output_item.done":
            state.active_item_type = None
            state.active_item_phase = None
            continue

        if t == "response.function_call_arguments.delta":
            idx = state.item_id_to_index.get(ev.get("item_id", ""))
            if idx is None:
                continue
            tool_calls[idx]["function"]["arguments"] += (
                ev.get(
                    "delta",
                    "",
                )
                or ""
            )
            continue

        if t == "response.completed":
            resp = ev.get("response") or {}
            state.final_usage = resp.get("usage")
            has_tool_call = any(
                (i.get("type") == "function_call")
                for i in (resp.get("output") or [])
            )
            state.finish_reason = "tool_calls" if has_tool_call else "stop"
            break

        if t == "response.failed":
            err = (ev.get("response") or {}).get("error") or {}
            raise RuntimeError(
                f"upstream Codex responses failed: {err.get('message') or err}",
            )

    message: dict[str, Any] = {
        "role": "assistant",
        "content": "".join(content_parts) if content_parts else None,
    }
    if tool_calls:
        for tc in tool_calls:
            tc.pop("index", None)
        message["tool_calls"] = tool_calls
        # OpenAI contract: presence of tool_calls forces the
        # ``finish_reason`` regardless of what we inferred earlier.
        state.finish_reason = "tool_calls"

    body: dict[str, Any] = {
        "id": state.response_id,
        "object": "chat.completion",
        "created": state.created,
        "model": state.model,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": state.finish_reason or "stop",
            },
        ],
    }
    if state.final_usage:
        body["usage"] = {
            "prompt_tokens": state.final_usage.get("input_tokens"),
            "completion_tokens": state.final_usage.get("output_tokens"),
            "total_tokens": state.final_usage.get("total_tokens"),
        }
    return body
