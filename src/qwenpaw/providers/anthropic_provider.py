# -*- coding: utf-8 -*-
"""An Anthropic provider implementation."""

from __future__ import annotations

import contextvars
import json
import logging
import os
import time
from datetime import datetime
from typing import Any, Dict, List

import httpx
from agentscope.model import AnthropicChatModel, ChatModelBase
import anthropic

from qwenpaw.providers.multimodal_prober import (
    ProbeResult,
    _PROBE_IMAGE_B64,
    _IMAGE_PROBE_PROMPT,
    _is_media_keyword_error,
    evaluate_image_probe_answer,
)
from qwenpaw.providers.provider import ModelInfo, Provider
from qwenpaw.exceptions import ModelRefusalException
from qwenpaw.local_models.tag_parser import (
    parse_tool_calls_from_text,
    text_contains_tool_call_tag,
)

logger = logging.getLogger(__name__)

DASHSCOPE_BASE_URLS = (
    "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
    "https://dashscope-us.aliyuncs.com/compatible-mode/v1",
)
CODING_DASHSCOPE_BASE_URL = "https://coding.dashscope.aliyuncs.com/v1"
TOKEN_PLAN_BASE_URL = (
    "https://token-plan.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"
)

# Sentinel value in ``api_key`` that switches the provider into Claude
# Code OAuth mode.  Chosen as a string (not a new pydantic field) so we
# do not have to extend ``ProviderInfo`` and break every serialized
# provider config — users just set the api_key to this literal.
OAUTH_API_KEY_SENTINEL = "oauth"

# Sibling sentinel that switches the provider into Claude Code (acpx)
# mode — same string-based approach so adding a new dispatch path does
# not require a schema migration.  Distinct from
# :data:`OAUTH_API_KEY_SENTINEL` because the two paths share the model
# catalogue but otherwise behave very differently (direct API vs ACP
# subprocess + stateful session registry; see
# :class:`qwenpaw.providers.claude_acpx_model.ClaudeAcpxChatModel`).
ACPX_API_KEY_SENTINEL = "acpx"


# Beta flag that opts a Claude Code OAuth request into "fast mode" —
# the faster-output, 6x-billing variant of Opus 4.6/4.7.  Server only
# honours it when the body also carries top-level ``speed: "fast"`` AND
# the account has Extra usage enabled (otherwise the request 429s with
# ``Extra usage is required for fast mode``).  Confirmed against the
# Claude Code CLI binary (2.1.150) which uses the exact same wire shape.
FAST_MODE_BETA = "fast-mode-2026-02-01"

# Model substrings for which the CLI surfaces fast mode.  Used to gate
# the toggle so flipping ``fast_mode`` on the provider does not poison
# unrelated routes (e.g., a haiku probe inside the same provider).
FAST_MODE_MODELS = ("opus-4-8", "opus-4-7", "opus-4-6")


def _model_supports_fast_mode(model_id: str) -> bool:
    s = (model_id or "").lower()
    return any(tag in s for tag in FAST_MODE_MODELS)


# Tool-name prefix required by Anthropic OAuth billing validation when
# multiple tools are sent.  Claude Code's own tool catalogue uses
# ``mcp_<FirstCharUpper><rest>`` (e.g. ``mcp_Bash``, ``mcp_Read``) and
# Anthropic's validator rejects lowercase-first tool names in this
# mode.  Convention — and opencode-claude-auth's on-the-wire transform
# — is to prefix outbound and strip inbound symmetrically; see
# ``transforms.ts:9-13, 225-269`` in that project.  Note this is NOT
# PascalCase: only character 0 is upper-cased, underscores stay put.
MCP_TOOL_PREFIX = "mcp_"


def _prefix_tool_name(name: str) -> str:
    if not isinstance(name, str) or not name:
        return name
    return f"{MCP_TOOL_PREFIX}{name[:1].upper()}{name[1:]}"


def _unprefix_tool_name_heuristic(name: str) -> str:
    """Heuristic reverse — lowercases the first char after the
    ``mcp_`` prefix.  Correct for lowercase-first originals
    (``"foo"`` → ``"mcp_Foo"`` → ``"foo"``) but LOSSY for
    PascalCase-first originals (``"Edit"`` → ``"mcp_Edit"`` →
    ``"edit"``).  Prefer the per-call forward map when available.
    """
    if not isinstance(name, str) or not name.startswith(MCP_TOOL_PREFIX):
        return name
    tail = name[len(MCP_TOOL_PREFIX) :]
    return f"{tail[:1].lower()}{tail[1:]}"


# Per-call prefixed->original tool-name map.  ClaudeOAuthChatModel
# populates it inside its overridden ``__call__`` so the inbound
# strip pass gets a lossless round-trip even for originally
# PascalCase-first names.  Kept as a ContextVar so concurrent agent
# calls get independent maps.
_TOOL_NAME_REVERSE_MAP: contextvars.ContextVar[
    dict[str, str] | None
] = contextvars.ContextVar("claude_oauth_tool_name_reverse_map", default=None)


def _record_tool_name_mapping(original: str, prefixed: str) -> None:
    reverse = _TOOL_NAME_REVERSE_MAP.get()
    if reverse is not None:
        reverse[prefixed] = original


def _unprefix_tool_name(name: str, reverse: dict[str, str] | None) -> str:
    """Prefer the exact reverse lookup; fall back to the heuristic."""
    if reverse is not None and name in reverse:
        return reverse[name]
    return _unprefix_tool_name_heuristic(name)


def _rewrite_history_tool_names_outbound(
    messages: list[dict],
) -> list[dict]:
    """Walk every assistant-history ``tool_use`` block and rewrite its
    ``name`` with the ``mcp_`` prefix so the whole conversation uses
    the same wire names.  Also records original→prefixed mappings in
    the current-call reverse map so response-side strip is lossless.
    Returns a new list; does not mutate input.
    """
    out: list[dict] = []
    for msg in messages:
        content = msg.get("content")
        if not isinstance(content, list):
            out.append(msg)
            continue
        new_content: list = []
        changed = False
        for block in content:
            if (
                isinstance(block, dict)
                and block.get("type") == "tool_use"
                and isinstance(block.get("name"), str)
                and block["name"]
                and not block["name"].startswith(MCP_TOOL_PREFIX)
            ):
                orig = block["name"]
                prefixed = _prefix_tool_name(orig)
                _record_tool_name_mapping(orig, prefixed)
                new_content.append({**block, "name": prefixed})
                changed = True
            else:
                new_content.append(block)
        out.append({**msg, "content": new_content} if changed else msg)
    return out


def _strip_tool_use_names_inplace(
    resp: Any,
    reverse: dict[str, str] | None = None,
) -> None:
    content = getattr(resp, "content", None)
    if not content:
        return
    for block in content:
        # ChatResponse content blocks behave like dicts (TypedDict).
        if (
            isinstance(block, dict)
            and block.get("type") == "tool_use"
            and isinstance(block.get("name"), str)
            and block["name"].startswith(MCP_TOOL_PREFIX)
        ):
            block["name"] = _unprefix_tool_name(block["name"], reverse)


def _recover_text_tool_calls_inplace(resp: Any) -> None:
    """Recover tool calls that the model emitted as XML inside a text block.

    Some Claude revisions (and Qwen / GLM models distilled on Claude
    transcripts) occasionally emit ``<invoke name="..."><parameter ...>
    </invoke>`` directly inside an assistant ``text`` block instead of as
    a structured ``tool_use`` block.  Without recovery the framework
    parses no tool_use, the call vanishes, and the XML leaks to whatever
    channel rendered the assistant message.

    When this fallback fires we log a WARNING so the leak is visible.
    """
    content = getattr(resp, "content", None)
    if not content:
        return
    if any(
        isinstance(b, dict) and b.get("type") == "tool_use" for b in content
    ):
        # The model already produced structured tool_use blocks — trust
        # those and ignore any stray XML that may also appear in text
        # (avoids double-dispatch).
        return

    new_content: list = []
    recovered_names: list[str] = []
    for block in content:
        if not (isinstance(block, dict) and block.get("type") == "text"):
            new_content.append(block)
            continue
        text = block.get("text") or ""
        if not text_contains_tool_call_tag(text):
            new_content.append(block)
            continue
        parsed = parse_tool_calls_from_text(text)
        if not parsed.tool_calls:
            new_content.append(block)
            continue
        clean_text = parsed.text_before.strip()
        if clean_text:
            new_content.append({**block, "text": clean_text})
        for ptc in parsed.tool_calls:
            new_content.append(
                {
                    "type": "tool_use",
                    "id": ptc.id,
                    "name": ptc.name,
                    "input": ptc.arguments,
                },
            )
            recovered_names.append(ptc.name)
    if recovered_names:
        logger.warning(
            "Recovered %d tool_use block(s) from XML in assistant text "
            "(model emitted prompted-tool-use format instead of native "
            "tool_use): %s",
            len(recovered_names),
            recovered_names,
        )
        resp.content = new_content


def _strip_haiku_incompatible_kwargs(call_kwargs: dict[str, Any]) -> None:
    """Silently remove reasoning-related kwargs that Haiku rejects.

    Anthropic's API rejects with 400 when:
    * ``thinking.type == "adaptive"`` is sent to a Haiku model
      (Haiku only supports manual ``thinking.type="enabled"`` with
      ``budget_tokens``, per the adaptive-thinking docs);
    * ``output_config.effort`` is set on a Haiku model (the effort
      parameter is Opus/Sonnet 4.6+ only).

    Letting those fields reach Haiku would break the UX when a user
    configures reasoning at the *provider* level (via the console
    UI) and then dispatches a call through a Haiku model in that
    same provider.  We strip in place rather than raise so the rest
    of the request (tools, messages, system) still goes through.
    """
    model = call_kwargs.get("model")
    if not isinstance(model, str) or "haiku" not in model.lower():
        return

    thinking = call_kwargs.get("thinking")
    if isinstance(thinking, dict) and thinking.get("type") == "adaptive":
        # Drop the whole thinking block; converting adaptive to a
        # manual ``budget_tokens`` value would require guessing the
        # user's intent, and Haiku-on-by-default is the exception
        # more than the rule.
        call_kwargs.pop("thinking", None)
    elif isinstance(thinking, dict) and "effort" in thinking:
        cleaned = {k: v for k, v in thinking.items() if k != "effort"}
        if cleaned:
            call_kwargs["thinking"] = cleaned
        else:
            call_kwargs.pop("thinking", None)

    oc = call_kwargs.get("output_config")
    if isinstance(oc, dict) and "effort" in oc:
        cleaned_oc = {k: v for k, v in oc.items() if k != "effort"}
        if cleaned_oc:
            call_kwargs["output_config"] = cleaned_oc
        else:
            call_kwargs.pop("output_config", None)


def _inject_identity_system(system: Any, identity: str) -> list[dict]:
    """Prepend the Claude Code identity as its own ``system`` content
    block.  Anthropic validates byte-equality on this first block when
    using OAuth auth — merging it into the caller's system string will
    trigger a 400.
    """
    identity_block = {"type": "text", "text": identity}
    if not system:
        return [identity_block]
    if isinstance(system, str):
        # Idempotent on the string-shape too — callers that send
        # only the identity preamble (or it concatenated to extra
        # rules) should not pay for it twice.
        if system == identity:
            return [identity_block]
        if system.startswith(identity):
            return [
                identity_block,
                {"type": "text", "text": system[len(identity) :].lstrip()},
            ]
        return [identity_block, {"type": "text", "text": system}]
    if isinstance(system, list):
        # Idempotent: don't double-insert if caller already has it.
        if (
            system
            and isinstance(system[0], dict)
            and system[0].get("text") == identity
        ):
            return system
        return [identity_block, *system]
    return [identity_block, {"type": "text", "text": str(system)}]


# ----------------------------------------------------------------------- #
# Prompt caching                                                          #
# ----------------------------------------------------------------------- #
#
# Anthropic allows up to 4 ``cache_control: ephemeral`` breakpoints per
# request.  We spend the budget on:
#
#   1. last tool definition  — caches the entire tools array
#   2. last system block     — caches identity preamble + caller system
#   3. last 2 messages       — rolling window on the conversation tail
#
# A breakpoint marks "cache the prefix up to and including this block".
# On the next turn, the longest matching cached prefix is reused at 10%
# input cost; the trailing delta is processed fresh.  A two-message
# rolling tail anchors a stable cache point even when the most recent
# message changes wildly between turns (e.g., a long tool_result).
#
# Cache writes are billed at 1.25× input on first miss; reads at 0.10×.
# Default TTL is 5 minutes, refreshed on every read.  No beta header
# needed — prompt caching is GA on the OAuth endpoint.

# 1-hour TTL — write cost 2.0× input vs 1.25× for 5-min, but messaging
# channels (Signal / WhatsApp / DingTalk) have idle gaps measured in
# minutes-to-hours where the 5-min cache evaporates and we re-pay the
# full prefix.  Requires ``extended-cache-ttl-2025-04-11`` beta header
# in :data:`CLAUDE_BASE_BETAS`.
_EPHEMERAL_CACHE: dict[str, str] = {"type": "ephemeral", "ttl": "1h"}


def _mark_last_cache(blocks: list[Any]) -> list[Any]:
    """Return a list with ``cache_control: ephemeral`` on the last dict
    element.  No-op when empty / last is not a dict / marker already
    present.  Only the last element is shallow-copied; earlier ones
    pass through by reference.
    """
    if not blocks:
        return blocks
    last = blocks[-1]
    if not isinstance(last, dict):
        return blocks
    if last.get("cache_control") == _EPHEMERAL_CACHE:
        return blocks
    out = list(blocks)
    out[-1] = {**last, "cache_control": _EPHEMERAL_CACHE}
    return out


def _mark_messages_cache(messages: Any) -> Any:
    """Tag ``cache_control: ephemeral`` on the last content block of
    the trailing 1-2 messages.  Returns a new list when changes were
    made; the input list otherwise.  Caller's input is never mutated.
    """
    if not isinstance(messages, list) or not messages:
        return messages
    target_idxs = list(range(max(0, len(messages) - 2), len(messages)))
    out = list(messages)
    changed = False
    for i in target_idxs:
        msg = out[i]
        if not isinstance(msg, dict):
            continue
        content = msg.get("content")
        if isinstance(content, str):
            new_content: list[Any] = [
                {
                    "type": "text",
                    "text": content,
                    "cache_control": _EPHEMERAL_CACHE,
                },
            ]
        elif isinstance(content, list) and content:
            marked = _mark_last_cache(content)
            if marked is content:
                continue
            new_content = marked
        else:
            continue
        out[i] = {**msg, "content": new_content}
        changed = True
    return out if changed else messages


def _add_cache_breakpoints(call_kwargs: dict[str, Any]) -> None:
    """Place the 4-breakpoint cache pattern onto the outbound payload
    in ``call_kwargs`` (last tool, last system block, last 2 messages).
    Mutates ``call_kwargs`` in place.  Idempotent — re-applying does
    nothing because :func:`_mark_last_cache` is a no-op on already-marked
    blocks.

    The system tag is skipped when the list has only one block.  In OAuth
    mode that single block is the Claude Code identity preamble, which
    Anthropic validates byte-for-byte; tagging it risks a 400.  Skipping
    costs ~100 cached tokens in exchange for safety.
    """
    tools = call_kwargs.get("tools")
    if isinstance(tools, list) and tools:
        call_kwargs["tools"] = _mark_last_cache(tools)

    system = call_kwargs.get("system")
    if isinstance(system, list) and len(system) >= 2:
        call_kwargs["system"] = _mark_last_cache(system)

    messages = call_kwargs.get("messages")
    if isinstance(messages, list) and messages:
        call_kwargs["messages"] = _mark_messages_cache(messages)


# Per-call buffer the OAuth wrapper writes cache token counts into so
# the outer ``__call__`` can attach them to ``ChatUsage.metadata`` for
# the recording layer.  ContextVar — concurrent agent calls each see
# their own buffer.
_CURRENT_CACHE_BUF: contextvars.ContextVar[
    dict[str, int] | None
] = contextvars.ContextVar("claude_oauth_cache_buf", default=None)


def _read_cache_into_buf(usage_obj: Any, buf: dict[str, int]) -> None:
    """Copy ``cache_creation_input_tokens`` / ``cache_read_input_tokens``
    off an Anthropic usage object into ``buf``.  Tolerant to missing
    fields and to ``None`` — used on both the non-stream Message.usage
    and the streaming message_start event's ``message.usage``.
    """
    if usage_obj is None:
        return
    for key in ("cache_creation_input_tokens", "cache_read_input_tokens"):
        v = getattr(usage_obj, key, None)
        if v:
            buf[key] = int(v)


def _inject_cache_metadata(resp: Any, buf: dict[str, int]) -> None:
    """Merge ``buf`` into ``resp.usage.metadata`` so
    :class:`TokenRecordingModelWrapper` can record cache token counts
    alongside prompt/completion tokens.  No-op when buf is empty or the
    response has no usage attached (e.g., intermediate stream chunks).
    """
    if not buf:
        return
    usage = getattr(resp, "usage", None)
    if usage is None:
        return
    md = getattr(usage, "metadata", None)
    if not isinstance(md, dict):
        md = {}
    md.update(buf)
    usage.metadata = md


# ------------------------------------------------------------------ #
# Model fallback / safety-reject detection (Fable 5 / Mythos-class)   #
# ------------------------------------------------------------------ #
# Mythos-class models (claude-fable-5) silently fall back to another
# model (e.g. claude-opus-4-8) when a safety classifier blocks the
# request, and may emit unusual ``stop_reason`` values on refusals.
# We log every requested-vs-actual model mismatch and any non-standard
# stop_reason to a JSONL file so pipeline failures can be diagnosed.
_FALLBACK_LOG_PATH = os.path.expanduser(
    "~/.copaw/logs/claude_model_fallback.jsonl",
)
_KNOWN_STOP_REASONS = frozenset(
    {"end_turn", "tool_use", "max_tokens", "stop_sequence", "pause_turn"},
)


def _log_model_anomaly(
    requested_model: str | None,
    actual_model: str | None,
    stop_reason: str | None,
    response_id: str | None,
    kind: str,
) -> None:
    """Append one JSONL record describing a model fallback or an
    unexpected stop_reason.  Best-effort — never raises into the
    request path.
    """
    try:
        entry = {
            "ts": datetime.now().isoformat(timespec="seconds"),
            "kind": kind,  # "model_fallback" | "unusual_stop_reason"
            "requested_model": requested_model,
            "actual_model": actual_model,
            "stop_reason": stop_reason,
            "response_id": response_id,
        }
        logger.warning("Claude model anomaly: %s", entry)
        os.makedirs(os.path.dirname(_FALLBACK_LOG_PATH), exist_ok=True)
        with open(_FALLBACK_LOG_PATH, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception:  # pragma: no cover — logging must never break calls
        logger.debug("fallback-log write failed", exc_info=True)


def _resp_has_visible_content(resp: Any) -> bool:
    """True when *resp* carries a non-empty text or tool_use block."""
    for block in getattr(resp, "content", None) or []:
        btype = getattr(block, "type", None)
        if btype == "tool_use":
            return True
        if btype == "text" and (getattr(block, "text", "") or "").strip():
            return True
    return False


def _check_response_model(
    requested_model: str | None,
    actual_model: str | None,
    stop_reason: str | None,
    response_id: str | None,
) -> None:
    """Compare requested vs actual model + stop_reason and log anomalies."""
    if (
        requested_model
        and actual_model
        and actual_model != requested_model
    ):
        _log_model_anomaly(
            requested_model,
            actual_model,
            stop_reason,
            response_id,
            "model_fallback",
        )
    if stop_reason and stop_reason not in _KNOWN_STOP_REASONS:
        _log_model_anomaly(
            requested_model,
            actual_model,
            stop_reason,
            response_id,
            "unusual_stop_reason",
        )


async def _peek_stream_for_cache(
    sdk_stream: Any,
    cache_buf: dict[str, int] | None,
    requested_model: str | None = None,
) -> Any:
    """Pass-through wrapper around an Anthropic SDK ``AsyncStream`` that
    copies cache token counts out of the ``message_start`` event into
    ``cache_buf``.  Anthropic only reports ``cache_*_input_tokens`` on
    that single event; later ``message_delta`` events carry only output
    token deltas, so a one-shot capture is enough.

    Also watches ``message_start`` for the *actual* responding model and
    ``message_delta`` for the final ``stop_reason`` so Mythos-class
    safety fallbacks (Fable 5 → Opus) get logged.

    Raises :class:`ModelRefusalException` when the stream ends with
    ``stop_reason="refusal"`` without having produced any visible
    content (text or tool_use) — otherwise the agent loop would treat
    the empty response as a normal completion and the channel would go
    silent.
    """
    actual_model: str | None = None
    response_id: str | None = None
    has_visible_content = False
    async for event in sdk_stream:
        etype = getattr(event, "type", None)
        if etype == "message_start":
            msg = getattr(event, "message", None)
            if msg is not None:
                if cache_buf is not None:
                    _read_cache_into_buf(getattr(msg, "usage", None), cache_buf)
                actual_model = getattr(msg, "model", None)
                response_id = getattr(msg, "id", None)
                if (
                    requested_model
                    and actual_model
                    and actual_model != requested_model
                ):
                    _check_response_model(
                        requested_model,
                        actual_model,
                        None,
                        response_id,
                    )
        elif etype == "content_block_start":
            block = getattr(event, "content_block", None)
            if getattr(block, "type", None) == "tool_use":
                has_visible_content = True
        elif etype == "content_block_delta":
            delta = getattr(event, "delta", None)
            if getattr(delta, "text", None) or getattr(
                delta,
                "partial_json",
                None,
            ):
                has_visible_content = True
        elif etype == "message_delta":
            delta = getattr(event, "delta", None)
            stop_reason = getattr(delta, "stop_reason", None)
            if stop_reason and stop_reason not in _KNOWN_STOP_REASONS:
                # Model mismatch (if any) was already logged at
                # message_start — only record the stop_reason here.
                _log_model_anomaly(
                    requested_model,
                    actual_model,
                    stop_reason,
                    response_id,
                    "unusual_stop_reason",
                )
                if stop_reason == "refusal" and not has_visible_content:
                    raise ModelRefusalException(
                        requested_model or actual_model or "unknown",
                        response_id=response_id,
                    )
        yield event


class ClaudeOAuthChatModel(AnthropicChatModel):
    """Claude Code OAuth variant of :class:`AnthropicChatModel`.

    Wraps the underlying anthropic SDK client so that every
    ``messages.create`` call transparently:

    * refreshes the OAuth access_token if it's near expiry and
      updates the live client's ``auth_token`` attribute;
    * prepends ``"You are Claude Code, ..."`` as the first ``system``
      content block — Anthropic rejects OAuth requests that don't
      include it.
    """

    def __init__(
        self,
        *,
        auth: "object",  # ClaudeAuth — avoid hard import cycle at module load
        identity: str,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._auth = auth
        self._identity = identity
        self._install_wrappers()

    def _install_wrappers(self) -> None:
        original_create = self.client.messages.create

        async def _wrapped_create(**call_kwargs: Any) -> Any:
            creds = await self._auth.ensure_fresh()
            # Mutate the live client so in-flight refresh takes effect
            # without rebuilding the HTTP connection pool.
            self.client.auth_token = creds.access_token
            call_kwargs["system"] = _inject_identity_system(
                call_kwargs.get("system"),
                self._identity,
            )
            _strip_haiku_incompatible_kwargs(call_kwargs)
            _add_cache_breakpoints(call_kwargs)

            # Per-call buffer set by the outer ``__call__`` wrapper.
            # Absent when the model is invoked outside our wrapper
            # (e.g., direct ``client.messages.create`` access).
            cache_buf = _CURRENT_CACHE_BUF.get()
            requested_model = call_kwargs.get("model")
            result = await original_create(**call_kwargs)

            if call_kwargs.get("stream"):
                # Always wrap: even without a cache buffer we want
                # fallback/stop_reason anomaly detection (Fable 5).
                return _peek_stream_for_cache(
                    result,
                    cache_buf,
                    requested_model,
                )

            if cache_buf is not None:
                _read_cache_into_buf(
                    getattr(result, "usage", None),
                    cache_buf,
                )
            _check_response_model(
                requested_model,
                getattr(result, "model", None),
                getattr(result, "stop_reason", None),
                getattr(result, "id", None),
            )
            if getattr(
                result,
                "stop_reason",
                None,
            ) == "refusal" and not _resp_has_visible_content(result):
                raise ModelRefusalException(
                    requested_model
                    or getattr(result, "model", None)
                    or "unknown",
                    response_id=getattr(result, "id", None),
                )
            return result

        self.client.messages.create = _wrapped_create  # type: ignore[method-assign]

    # ------------------------------------------------------------- #
    # Tool-name prefix transform (see MCP_TOOL_PREFIX above)         #
    # ------------------------------------------------------------- #

    def _format_tools_json_schemas(
        self,
        schemas: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        formatted = super()._format_tools_json_schemas(schemas)
        for tool in formatted:
            name = tool.get("name")
            if (
                isinstance(name, str)
                and name
                and not name.startswith(MCP_TOOL_PREFIX)
            ):
                prefixed = _prefix_tool_name(name)
                _record_tool_name_mapping(name, prefixed)
                tool["name"] = prefixed
        return formatted

    def _format_tool_choice(self, tool_choice):  # type: ignore[override]
        result = super()._format_tool_choice(tool_choice)
        # Only rewrite the "specific tool" shape — "auto" / "none" /
        # "any" choices carry no tool name.  Also handles the case
        # where ``structured_model`` forces a specific tool name by
        # pydantic class name.
        if (
            isinstance(result, dict)
            and result.get("type") == "tool"
            and isinstance(result.get("name"), str)
            and result["name"]
            and not result["name"].startswith(MCP_TOOL_PREFIX)
        ):
            orig = result["name"]
            prefixed = _prefix_tool_name(orig)
            _record_tool_name_mapping(orig, prefixed)
            result["name"] = prefixed
        return result

    async def __call__(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict] | None = None,
        tool_choice: Any = None,
        structured_model: Any = None,
        **generate_kwargs: Any,
    ) -> Any:
        # Per-call reverse map: ``mcp_PrefixedName`` → original name.
        # Populated as we rewrite history/tools/tool_choice on the way
        # out, consumed on the way back to give a lossless round-trip
        # even for PascalCase-first tool names (which the heuristic
        # strip can't recover unambiguously).
        reverse_map: dict[str, str] = {}
        # Per-call cache token buffer.  The wrapped ``messages.create``
        # writes cache_*_input_tokens here on the way back; we then
        # attach them to the outgoing ChatResponse's usage.metadata so
        # ``TokenRecordingModelWrapper`` can persist them.
        cache_buf: dict[str, int] = {}
        rev_token = _TOOL_NAME_REVERSE_MAP.set(reverse_map)
        cache_token = _CURRENT_CACHE_BUF.set(cache_buf)
        try:
            messages = _rewrite_history_tool_names_outbound(messages)
            result = await super().__call__(
                messages,
                tools=tools,
                tool_choice=tool_choice,
                structured_model=structured_model,
                **generate_kwargs,
            )
        finally:
            _TOOL_NAME_REVERSE_MAP.reset(rev_token)
            _CURRENT_CACHE_BUF.reset(cache_token)

        # Streaming case: close over the captured reverse_map and
        # cache_buf so the generator stays correct even after the
        # ContextVars are reset.
        if self.stream:
            return self._wrap_stream_response(result, reverse_map, cache_buf)
        _recover_text_tool_calls_inplace(result)
        _strip_tool_use_names_inplace(result, reverse_map)
        _inject_cache_metadata(result, cache_buf)
        return result

    @staticmethod
    async def _wrap_stream_response(
        gen: Any,
        reverse_map: dict[str, str],
        cache_buf: dict[str, int],
    ) -> Any:
        async for chunk in gen:
            _recover_text_tool_calls_inplace(chunk)
            _strip_tool_use_names_inplace(chunk, reverse_map)
            _inject_cache_metadata(chunk, cache_buf)
            yield chunk


class _StripApiKeyTransport(httpx.AsyncHTTPTransport):
    """Async transport that removes the x-api-key header from every request.

    Used when auth_mode='auth_token' to avoid sending both x-api-key and
    Authorization headers simultaneously, which some proxies reject.

    The request is reconstructed with ``extensions`` preserved so that
    per-request configuration such as timeouts and SSE hints set by the
    Anthropic SDK are not lost.
    """

    async def handle_async_request(
        self,
        request: httpx.Request,
    ) -> httpx.Response:
        filtered = [
            (k, v)
            for k, v in request.headers.items()
            if k.lower() != "x-api-key"
        ]
        new_request = httpx.Request(
            method=request.method,
            url=request.url,
            headers=filtered,
            content=request.content,
            extensions=request.extensions,
        )
        return await super().handle_async_request(new_request)


class AnthropicProvider(Provider):
    """Provider implementation for Anthropic API."""

    def _get_oauth(self) -> "Any":
        """Build a fresh :class:`ClaudeAuth` instance — cheap (reads
        one small file) and bypasses the need to stash a non-field
        attribute on a pydantic model.  Each call re-reads the
        credentials file, which is what we want when another process
        (the ``claude`` CLI) may have rotated tokens on disk.
        Raises ``FileNotFoundError`` when ``claude login`` has not run.
        """
        from qwenpaw.providers.claude_auth import ClaudeAuth

        return ClaudeAuth()

    @property
    def _is_oauth(self) -> bool:
        return self.api_key == OAUTH_API_KEY_SENTINEL

    @property
    def _is_acpx(self) -> bool:
        """True when the provider is configured to route through the
        ``acpx`` ACP bridge (Claude Code via subprocess) rather than
        Anthropic-direct or Claude OAuth.  Mutually exclusive with
        :attr:`_is_oauth`: the two sentinels are distinct strings.
        """
        return self.api_key == ACPX_API_KEY_SENTINEL

    # Cached AsyncClient for auth_token mode; re-created when auth_mode
    # changes so that the transport is always consistent with the current
    # provider config.
    _strip_http_client: httpx.AsyncClient | None = None

    def _build_default_headers(self) -> Dict[str, str]:
        return dict(self.custom_headers) if self.custom_headers else {}

    def _get_strip_http_client(self) -> httpx.AsyncClient:
        """Return a cached AsyncClient backed by _StripApiKeyTransport."""
        if self._strip_http_client is None:
            self._strip_http_client = httpx.AsyncClient(
                transport=_StripApiKeyTransport(),
            )
        return self._strip_http_client

    def _client(self, timeout: float = 5) -> anthropic.AsyncAnthropic:
        """Build a non-OAuth sync-constructed async client (the common
        path).  OAuth callers must use :meth:`_async_client` instead,
        since the token must be refreshed before the client is used.
        Honours upstream's ``auth_mode='auth_token'`` for bearer-style
        Anthropic-compatible endpoints (e.g. via Cloudflare Workers).
        """
        default_headers = self._build_default_headers()
        if self.auth_mode == "auth_token":
            return anthropic.AsyncAnthropic(
                auth_token=self.api_key,
                base_url=self.base_url,
                default_headers=default_headers,
                http_client=self._get_strip_http_client(),
                timeout=timeout,
            )
        return anthropic.AsyncAnthropic(
            api_key=self.api_key,
            base_url=self.base_url,
            default_headers=default_headers,
            timeout=timeout,
        )

    async def _async_client(
        self,
        timeout: float = 5,
    ) -> anthropic.AsyncAnthropic:
        """Build an async client, refreshing the OAuth token first
        when in OAuth mode."""
        if not self._is_oauth:
            return self._client(timeout=timeout)

        from qwenpaw.providers.claude_auth import ClaudeAuth  # noqa: F401

        auth = self._get_oauth()
        creds = await auth.ensure_fresh()
        return anthropic.AsyncAnthropic(
            api_key=None,
            auth_token=creds.access_token,
            base_url=self.base_url,
            timeout=timeout,
            default_headers=auth.default_headers(),
        )

    @staticmethod
    def _normalize_models_payload(payload: Any) -> List[ModelInfo]:
        if isinstance(payload, dict):
            rows = payload.get("data", [])
        else:
            rows = getattr(payload, "data", payload)

        models: List[ModelInfo] = []
        for row in rows or []:
            model_id = str(
                getattr(row, "id", "") or "",
            ).strip()
            model_name = str(
                getattr(row, "display_name", "") or model_id,
            ).strip()

            if not model_id:
                continue
            models.append(ModelInfo(id=model_id, name=model_name))

        deduped: List[ModelInfo] = []
        seen: set[str] = set()
        for model in models:
            if model.id in seen:
                continue
            seen.add(model.id)
            deduped.append(model)
        return deduped

    async def check_connection(self, timeout: float = 5) -> tuple[bool, str]:
        """Check if Anthropic provider is reachable.

        First tries models.list(); if that endpoint is not supported by the
        proxy (e.g. returns 404/405) falls back to a minimal messages.create
        call so that custom proxies that only expose the messages API still
        pass the connection test.
        """
        client = self._client(timeout=timeout)
        try:
            if self._is_acpx:
                # acpx routes via the local ``npx acpx`` subprocess, not
                # ``base_url``; falling through to ``client.models.list()``
                # would call the OpenAI-shaped Anthropic endpoint with an
                # ``acpx`` sentinel string as the API key and surface
                # "Unknown exception" to the UI.  The dedicated
                # GET /api/providers/claude-acpx/test-connection endpoint
                # owns the real probe (binary present + OAuth creds);
                # here we just signal "supported, defer to that".
                return True, ""
            if self._is_oauth:
                # OAuth: we don't need to round-trip to /v1/models; a
                # successful credential read + (if needed) refresh
                # against auth.claude.ai is proof enough that the
                # subscription is live.  Calling /v1/models with an
                # OAuth token may 401 on accounts whose scope set
                # doesn't include model discovery.
                auth = self._get_oauth()
                await auth.ensure_fresh()
                return True, ""
            client = self._client(timeout=timeout)
            await client.models.list()
            return True, ""
        except FileNotFoundError as e:
            return False, f"Claude Code OAuth not set up: {e}"
        except anthropic.APIStatusError as e:
            # Some proxies don't implement the models endpoint (404/405).
            # Fall back to a lightweight messages probe instead.
            if e.status_code in (404, 405):
                return await self._check_connection_via_messages(client)
            return False, f"Anthropic API error: {e}"
        except anthropic.APIError as e:
            # Network / auth errors from models.list – report directly
            return False, f"Anthropic API error: {e}"
        except Exception:
            return (
                False,
                f"Unknown exception when connecting to `{self.base_url}`",
            )

    async def _check_connection_via_messages(
        self,
        client: anthropic.AsyncAnthropic,
    ) -> tuple[bool, str]:
        """Fallback: check reachability via messages.create."""
        model = self.models[0].id if self.models else "claude-opus-4-5"
        try:
            await client.messages.create(
                model=model,
                max_tokens=1,
                messages=[{"role": "user", "content": "ping"}],
            )
            return True, ""
        except anthropic.APIStatusError as e:
            # 400/404/422: server is reachable and auth is accepted –
            # the model may simply not exist on this proxy, which is fine
            # for a connection check.
            if e.status_code in (400, 404, 422):
                return True, ""
            return False, f"Anthropic API error: {e}"
        except anthropic.APIError as e:
            return False, f"Anthropic API error: {e}"
        except Exception as e:
            return False, f"Unknown exception: {e}"

    async def fetch_models(self, timeout: float = 5) -> List[ModelInfo]:
        """Fetch available models."""
        if self._is_acpx:
            # acpx doesn't expose a discovery endpoint; the configured
            # model list comes from provider_manager defaults.  Same
            # contract as OAuth — empty list means "use configured".
            return []
        if self._is_oauth:
            # Upstream ``GET /v1/models`` with OAuth auth is gated
            # behind ``user:profile`` scope on some subscriptions and
            # returns 403 otherwise.  Returning an empty list signals
            # "discovery unsupported — use the configured model list"
            # which is what openclaw and opencode-claude-auth do.
            return []
        client = self._client(timeout=timeout)
        payload = await client.models.list()
        models = self._normalize_models_payload(payload)
        return models

    async def check_model_connection(
        self,
        model_id: str,
        timeout: float = 5,
    ) -> tuple[bool, str]:
        """Check if a specific model is reachable/usable."""
        target = (model_id or "").strip()
        if not target:
            return False, "Empty model ID"

        if self._is_acpx:
            # The Anthropic SDK ping path doesn't apply — acpx is a
            # subprocess driver, not an HTTP client against base_url.
            # The dedicated GET /api/providers/claude-acpx/test-connection
            # endpoint runs the real binary+OAuth probe.  We trust the
            # configured model list here so the UI doesn't show a red
            # "model unreachable" tile that we can't actually verify.
            return True, ""

        body: dict[str, Any] = {
            "model": target,
            "max_tokens": 1,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "ping",
                        },
                    ],
                },
            ],
            "stream": True,
        }
        if self._is_oauth:
            from qwenpaw.providers.claude_auth import CLAUDE_CODE_IDENTITY

            body["system"] = _inject_identity_system(
                None,
                CLAUDE_CODE_IDENTITY,
            )
        try:
            client = await self._async_client(timeout=timeout)
            resp = await client.messages.create(**body)
            # consume the stream to ensure the model is actually responsive
            async for _ in resp:
                break
            return True, ""
        except FileNotFoundError as e:
            return False, f"Claude Code OAuth not set up: {e}"
        except anthropic.APIError:
            return False, f"Model '{model_id}' is not reachable or usable"
        except Exception:
            return (
                False,
                f"Unknown exception when connecting to model '{model_id}'",
            )

    def get_chat_model_instance(self, model_id: str) -> ChatModelBase:
        from agentscope.model import AnthropicChatModel

        client_kwargs: Dict[str, Any] = {"base_url": self.base_url}

        # Start with any user-defined custom headers
        merged_headers: Dict[str, str] = self._build_default_headers()

        if self.base_url in DASHSCOPE_BASE_URLS:
            merged_headers["x-dashscope-agentapp"] = json.dumps(
                {
                    "agentType": "QwenPaw",
                    "deployType": "UnKnown",
                    "moduleCode": "model",
                    "agentCode": "UnKnown",
                },
                ensure_ascii=False,
            )
        elif self.base_url in (CODING_DASHSCOPE_BASE_URL, TOKEN_PLAN_BASE_URL):
            merged_headers["X-DashScope-Cdpl"] = json.dumps(
                {
                    "agentType": "QwenPaw",
                    "deployType": "UnKnown",
                    "moduleCode": "model",
                    "agentCode": "UnKnown",
                },
                ensure_ascii=False,
            )

        if merged_headers:
            client_kwargs["default_headers"] = merged_headers

        if self.auth_mode == "auth_token":
            client_kwargs["http_client"] = httpx.AsyncClient(
                transport=_StripApiKeyTransport(),
            )
            client_kwargs["auth_token"] = self.api_key
            api_key_arg = None
        else:
            api_key_arg = self.api_key

        effective_generate_kwargs = self.get_effective_generate_kwargs(
            model_id,
        )
        max_tokens = effective_generate_kwargs.pop("max_tokens", 16384)

        if self._is_acpx:
            # Claude Code via the ``acpx`` ACP subprocess bridge.  This
            # branch must come before the OAuth check above only for
            # narrative clarity — the two sentinel strings ("acpx" vs
            # "oauth") are mutually exclusive so order is not load-
            # bearing.  The wrapper class owns its own subprocess /
            # session registry plumbing (see Lane C/D).
            from .claude_acpx_model import ClaudeAcpxChatModel

            return ClaudeAcpxChatModel(
                model_name=model_id,
                stream=True,
                stream_tool_parsing=False,
                client_kwargs=client_kwargs,
                generate_kwargs=self.get_effective_generate_kwargs(model_id),
            )

        if self._is_oauth:
            from qwenpaw.providers.claude_auth import (
                CLAUDE_CODE_IDENTITY,
            )

            auth = self._get_oauth()
            # Seed the SDK client with the currently-cached access
            # token; ClaudeOAuthChatModel's wrapper keeps it fresh on
            # every ``messages.create`` call.
            creds = auth._creds  # type: ignore[attr-defined]
            if creds is None:
                raise RuntimeError(
                    "ClaudeAuth loaded but credentials are empty — "
                    "run `claude login`.",
                )
            client_kwargs["auth_token"] = creds.access_token
            merged_headers = dict(client_kwargs.get("default_headers") or {})
            merged_headers.update(auth.default_headers())
            oauth_generate_kwargs = self.get_effective_generate_kwargs(model_id)
            if self.fast_mode and _model_supports_fast_mode(model_id):
                # Append the fast-mode beta to whatever auth.default_headers
                # already shipped (CLAUDE_BASE_BETAS) so cache-ttl + oauth
                # markers stay intact.
                beta = merged_headers.get("anthropic-beta", "")
                if FAST_MODE_BETA not in beta:
                    merged_headers["anthropic-beta"] = (
                        f"{beta},{FAST_MODE_BETA}" if beta else FAST_MODE_BETA
                    )
                # Top-level body field — anthropic SDK funnels extra_body
                # into the request body root, which is where the CLI puts
                # ``speed`` (see fast-mode-2026-02-01 wire shape).
                extra_body = dict(oauth_generate_kwargs.get("extra_body") or {})
                extra_body["speed"] = "fast"
                oauth_generate_kwargs["extra_body"] = extra_body
            client_kwargs["default_headers"] = merged_headers
            return ClaudeOAuthChatModel(
                auth=auth,
                identity=CLAUDE_CODE_IDENTITY,
                model_name=model_id,
                stream=True,
                api_key=None,
                stream_tool_parsing=False,
                client_kwargs=client_kwargs,
                generate_kwargs=oauth_generate_kwargs,
            )

        return AnthropicChatModel(
            model_name=model_id,
            max_tokens=max_tokens,
            stream=True,
            api_key=api_key_arg,
            stream_tool_parsing=False,
            client_kwargs=client_kwargs,
            generate_kwargs=effective_generate_kwargs,
        )

    async def probe_model_multimodal(
        self,
        model_id: str,
        timeout: float = 60,
        image_only: bool = False,  # pylint: disable=unused-argument
    ) -> ProbeResult:
        """Probe multimodal support using Anthropic messages API format.

        Anthropic does not support video input, so supports_video is
        always False.  Image support is probed by sending a minimal 1x1
        PNG via the Anthropic base64 image source format.
        """
        img_ok, img_msg = await self._probe_image_support(
            model_id,
            timeout,
        )
        return ProbeResult(
            supports_image=img_ok,
            supports_video=False,
            image_message=img_msg,
            video_message="Video not supported by Anthropic",
        )

    async def _probe_image_support(
        self,
        model_id: str,
        timeout: float = 10,
    ) -> tuple[bool, str]:
        """Probe image support via Anthropic messages API.

        Uses a two-stage check (same strategy as OpenAIProvider):
        1. If the API rejects the request (400 / media-keyword error)
           -> not supported.
        2. If accepted, verify the model can *actually perceive* the
           image by asking for the dominant color of a solid-red PNG.
           Some providers silently accept image payloads without
           processing them, so a pure API-error check would produce
           false positives.
        """
        logger.info(
            "Image probe start: model=%s url=%s",
            model_id,
            self.base_url,
        )
        start_time = time.monotonic()
        try:
            client = await self._async_client(timeout=timeout)
            create_kwargs: dict[str, Any] = dict(
                model=model_id,
                max_tokens=200,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": _PROBE_IMAGE_B64,
                                },
                            },
                            {
                                "type": "text",
                                "text": _IMAGE_PROBE_PROMPT,
                            },
                        ],
                    },
                ],
            )
            if self._is_oauth:
                from qwenpaw.providers.claude_auth import CLAUDE_CODE_IDENTITY

                create_kwargs["system"] = _inject_identity_system(
                    None,
                    CLAUDE_CODE_IDENTITY,
                )
            resp = await client.messages.create(**create_kwargs)
            answer = ""
            for block in resp.content:
                if hasattr(block, "text"):
                    answer += block.text
            return evaluate_image_probe_answer(
                answer,
                model_id,
                start_time,
            )
        except anthropic.APIError as e:
            elapsed = time.monotonic() - start_time
            logger.warning(
                "Image probe error: model=%s type=%s msg=%s %.2fs",
                model_id,
                type(e).__name__,
                e,
                elapsed,
            )
            status = getattr(e, "status_code", None)
            if status == 400 or _is_media_keyword_error(e):
                return False, f"Image not supported: {e}"
            return False, f"Probe inconclusive: {e}"
        except Exception as e:
            elapsed = time.monotonic() - start_time
            logger.warning(
                "Image probe error: model=%s type=%s msg=%s %.2fs",
                model_id,
                type(e).__name__,
                e,
                elapsed,
            )
            return False, f"Probe failed: {e}"
