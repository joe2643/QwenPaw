# -*- coding: utf-8 -*-
"""Shared OpenAI chat/completions ↔ ACP (Agent Client Protocol)
translation helpers — acpx-backed Claude Code variant.

Mirrors :mod:`codex_translate` in shape and intent.  Where
``codex_translate`` brokers HTTP↔SSE against ChatGPT's Codex
Responses endpoint, this module brokers stdin/stdout JSON-RPC
against ``acpx claude exec --format json --json-strict``: acpx
spawns the Claude Code ACP adapter as a child, owns the
JSON-RPC channel, and emits one raw ACP JSON-RPC message per
line on stdout (no acpx-specific envelope).

Consumed by a sibling ``ClaudeAcpxChatModel`` (TODO) that
subclasses agentscope's :class:`OpenAIChatModel` so CoPaw
agents can hit Claude Code via ACP without changing call sites.

ACP reference (from upstream ``schema.json``):
  - ``session/prompt`` request — params: ``{sessionId, prompt: ContentBlock[]}``
  - ``session/update`` notification — params: ``{sessionId, update}``
    where ``update.sessionUpdate ∈ { user_message_chunk, agent_message_chunk,
    agent_thought_chunk, tool_call, tool_call_update, plan,
    available_commands_update, current_mode_update,
    config_option_update, session_info_update }``
  - ``session/prompt`` response — result: ``{stopReason}``
    where ``stopReason ∈ { end_turn, max_tokens, max_turn_requests,
    refusal, cancelled }``
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from typing import Any, AsyncIterator

logger = logging.getLogger(__name__)

# Pinned acpx version (plan T4 decision).  acpx is alpha; auto-
# upgrading mid-conversation kills active sessions.  Bump quarterly
# after manual review of the upstream changelog.
#
# Selected 0.6.1 (latest at 2026-04-26) after CLI surface verification:
# ``acpx claude -s <name> --format json --json-strict --ttl <s>`` is
# the per-turn invocation Lane B drives, ``acpx claude sessions close
# <name>`` is the registry tear-down, and ``acpx claude set <key>
# <value>`` is the session/set_config_option call.  Bumping to a newer
# patch release is safe; bumping minor (0.7.x+) requires re-running
# the verification because the CLI surface has churned across minors
# during the alpha.
_PINNED_ACPX_VERSION: str = "0.6.1"

# Stateless one-shot exec — used for the legacy fire-and-forget path
# and for unit tests that don't care about session lifecycle.
DEFAULT_ACPX_CMD: tuple[str, ...] = (
    "npx", f"acpx@{_PINNED_ACPX_VERSION}", "claude", "exec",
    "--format", "json", "--json-strict",
)


def stateful_acpx_cmd(session_name: str) -> tuple[str, ...]:
    """Stateful invocation: ``acpx claude -s <name>`` keeps Claude
    Code's session warm across CoPaw turns.  Used by
    :class:`ClaudeAcpxChatModel` once the session registry has
    decided which session this turn belongs to.
    """
    return (
        "npx", f"acpx@{_PINNED_ACPX_VERSION}", "claude",
        "-s", session_name,
        "--format", "json", "--json-strict",
    )


# =========================================================================
# Request translation: chat/completions → ACP session/prompt content
# =========================================================================


def content_to_acp_blocks(content: Any) -> list[dict]:
    """Translate chat-message ``content`` (str OR list of content
    blocks) into ACP ``ContentBlock[]``.  ACP shapes per spec:
        text          {type:"text", text:str}
        image (b64)   {type:"image", mimeType:str, data:str}
        image (url)   {type:"image", uri:str, mimeType?:str}
        resource_link {type:"resource_link", uri:str, name?:str}

    Lossy on Anthropic-style ``image.source.type == "url"`` — ACP's
    image block prefers base64 + mimeType; URL-only goes through as
    ``resource_link`` since not every Claude Code build resolves
    image URIs server-side.
    """
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    if not isinstance(content, list):
        return [{"type": "text", "text": ""}]

    out: list[dict] = []
    for item in content:
        if not isinstance(item, dict):
            continue
        t = item.get("type")
        if t in ("text", "input_text", "output_text"):
            text = str(item.get("text", ""))
            if text:
                out.append({"type": "text", "text": text})
        elif t in ("image_url", "input_image"):
            iu = item.get("image_url")
            url = iu.get("url", "") if isinstance(iu, dict) else (str(iu) if iu else "")
            if url.startswith("data:"):
                # data:image/png;base64,XXXX → split header from payload
                try:
                    header, b64 = url.split(",", 1)
                    mime = header.split(";")[0].removeprefix("data:") or "image/png"
                    out.append({"type": "image", "mimeType": mime, "data": b64})
                except ValueError:
                    pass
            elif url:
                out.append({"type": "resource_link", "uri": url})
        elif t == "image":
            source = item.get("source") or {}
            if source.get("type") == "base64":
                out.append({
                    "type": "image",
                    "mimeType": source.get("media_type") or "image/png",
                    "data": source.get("data", ""),
                })
            elif source.get("type") == "url" and source.get("url"):
                out.append({"type": "resource_link", "uri": source["url"]})
    return out or [{"type": "text", "text": ""}]


def render_history_for_seed(messages: list[dict]) -> list[dict]:
    """Flatten ``messages`` into a single ACP prompt ContentBlock[]
    suitable for **seeding a fresh acpx Claude Code session** with
    the full conversation transcript.

    Used by :class:`ClaudeAcpxChatModel` on the first turn of a
    conversation (or after drift-driven reseed).  Subsequent turns
    in the same session ship only the trailing tail via
    :func:`extract_tail_from_history`.

    System prompts → leading text block; assistant turns →
    text-block transcript prefixed with ``Assistant:`` / tool
    results inlined; user blocks survive as separate ContentBlocks
    so images stay first-class instead of being collapsed into
    text.

    NOTE: Claude Code's ACP adapter accepts ``systemPrompt`` on
    ``session/new``.  v1 folds system into the seed prompt's
    leading text block; v2 should hoist out (open question 4).
    """
    blocks: list[dict] = []
    transcript_parts: list[str] = []

    def _flush_transcript() -> None:
        if transcript_parts:
            text = "\n\n".join(p for p in transcript_parts if p.strip())
            if text:
                blocks.append({"type": "text", "text": text})
            transcript_parts.clear()

    for msg in messages:
        role = msg.get("role")
        content = msg.get("content")
        if role == "system":
            transcript_parts.append(str(_plain(content)))
            continue
        if role == "assistant":
            text = _plain(content)
            if text:
                transcript_parts.append(f"Assistant: {text}")
            for tc in msg.get("tool_calls") or []:
                fn = tc.get("function") or {}
                transcript_parts.append(
                    f"Assistant tool_call {fn.get('name','')}({fn.get('arguments','')})",
                )
            continue
        if role == "tool":
            transcript_parts.append(f"Tool result: {_plain(content)}")
            continue
        if role == "user":
            # Flush prior transcript so user images/text land in
            # their own blocks rather than getting smushed into the
            # rolling prefix.
            _flush_transcript()
            blocks.extend(content_to_acp_blocks(content))
            continue

    _flush_transcript()
    return blocks or [{"type": "text", "text": ""}]


def extract_tail_from_history(
    messages: list[dict],
    from_idx: int,
) -> list[dict]:
    """Return ACP ContentBlock[] representing only the messages from
    ``from_idx`` onward.  Used by :class:`ClaudeAcpxChatModel` for
    the ``ship_tail`` path: Claude Code's session already has
    everything before ``from_idx``, we just push the new turn(s).

    Mirrors :func:`render_history_for_seed` for messages handled,
    but skips the leading system block (Claude's session keeps
    it from seed) and drops the "Assistant:" transcript prefix on
    pre-existing assistant turns (which by definition aren't in
    the tail — only fresh user / tool messages should be here).
    """
    if from_idx >= len(messages):
        # Caller bug — empty tail means nothing to ship.  Return
        # a single empty text block; spawn layer will replace with
        # ``"(empty prompt)"``.
        return [{"type": "text", "text": ""}]

    blocks: list[dict] = []
    for msg in messages[from_idx:]:
        role = msg.get("role")
        content = msg.get("content")
        if role == "user":
            blocks.extend(content_to_acp_blocks(content))
        elif role == "tool":
            tcid = msg.get("tool_call_id", "")
            text = _plain(content)
            blocks.append({
                "type": "text",
                "text": f"[tool-result tool_call_id={tcid}]\n{text}",
            })
        elif role == "assistant":
            # Hybrid mode (codex C1 resolution): if Claude's session
            # already produced this turn, we shouldn't be re-shipping
            # it.  But on commit_turn the wrapper advances
            # last_shipped_idx past assistant outputs, so this case
            # is a buggy ship_tail that included the assistant reply
            # we just got back.  Defensive: log and include nothing.
            text = _plain(content)
            if text:
                logger.warning(
                    "extract_tail_from_history: unexpected assistant turn "
                    "in tail (idx=%d) — last_shipped_idx not advanced "
                    "after prior turn?",
                    messages.index(msg),
                )
        # role=="system" in tail is also unexpected (system goes in
        # seed); skip silently.
    return blocks or [{"type": "text", "text": ""}]


def _plain(content: Any) -> str:
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


# =========================================================================
# Response translation: ACP session/update → chat/completions chunks
# =========================================================================


class StreamState:
    """Per-request accumulator for ACP → chat-completions translation.
    Mirrors :class:`codex_translate.StreamState` so the downstream
    chunk consumer stays uniform across providers.
    """

    __slots__ = (
        "model",
        "response_id",
        "created",
        "tool_calls",
        "tool_call_id_to_index",
        "finished",
        "finish_reason",
        "emitted_role",
        "session_id",
    )

    def __init__(self, model: str) -> None:
        self.model = model
        self.response_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        self.created = int(time.time())
        # ACP toolCallId (str) → chat-completions tool_call index (int)
        self.tool_calls: dict[int, dict[str, Any]] = {}
        self.tool_call_id_to_index: dict[str, int] = {}
        self.finished = False
        self.finish_reason: str | None = None
        self.emitted_role = False
        self.session_id: str | None = None


def _chat_chunk(state: StreamState, delta: dict, finish_reason: str | None = None) -> dict:
    return {
        "id": state.response_id,
        "object": "chat.completion.chunk",
        "created": state.created,
        "model": state.model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }


# ACP StopReason → OpenAI finish_reason.  ``refusal`` collapses to
# ``stop`` (no OpenAI equivalent; the refusal text is already in
# the agent_message stream).  ``cancelled`` collapses to ``stop``
# too — downstream cancel signaling rides on a separate channel.
_STOP_REASON_MAP = {
    "end_turn": "stop",
    "max_tokens": "length",
    "max_turn_requests": "stop",
    "refusal": "stop",
    "cancelled": "stop",
}


async def translate_acp_updates_to_chat_chunks(
    line_reader: AsyncIterator[str],
    state: StreamState,
) -> AsyncIterator[dict]:
    """Read newline-delimited ACP JSON-RPC messages and yield
    ``chat.completion.chunk`` dicts.  ``line_reader`` is anything
    that yields one JSON line at a time — typically a wrapper
    around ``asyncio.subprocess.PIPE`` reading from
    ``acpx --format json --json-strict`` stdout.

    Filtering policy (matches codex_translate's reasoning gate):
      - ``agent_message_chunk`` → forwarded as ``content`` deltas.
      - ``agent_thought_chunk`` → forwarded as ``reasoning_content``
        deltas (NOT dropped).  Mirrors the codex commit
        ``f70bf8ff fix(codex): route commentary text via
        reasoning_content``: scratch text rides on a separate
        delta field so the Console UI / logs see it, and the
        channel-side suppression (``filter_thinking`` /
        ``MessageType.REASONING`` in ``runner/utils.py`` and
        ``channels/renderer.py``) keeps it off WA / Signal
        user-facing send — same behaviour, two layers.
      - ``tool_call`` → emit a tool_calls delta with name +
        rawInput-as-arguments.
      - ``tool_call_update`` → if status flips to completed/
        failed and we've not yet declared finish_reason, leave it
        alone (final stopReason owns finish_reason).  We do NOT
        forward content/diff blocks back as deltas; the agent
        will narrate results via subsequent agent_message_chunk
        events.
      - ``plan``, ``available_commands_update``,
        ``current_mode_update``, ``config_option_update``,
        ``session_info_update``, ``user_message_chunk`` →
        ignored for chat-completions translation.
    """
    if not state.emitted_role:
        state.emitted_role = True
        yield _chat_chunk(state, {"role": "assistant"})

    async for raw_line in line_reader:
        line = raw_line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            logger.debug("acpx: skipping non-JSON line: %r", line[:200])
            continue

        # Final response: {"jsonrpc":"2.0","id":...,"result":{"stopReason":...}}
        result = msg.get("result")
        if isinstance(result, dict) and "stopReason" in result:
            stop = result.get("stopReason") or "end_turn"
            # ``tool_calls`` finish_reason wins if any tool call
            # was actually emitted this turn — matches OpenAI
            # contract.  Otherwise map per StopReason.
            if state.tool_calls and state.finish_reason is None:
                state.finish_reason = "tool_calls"
            else:
                state.finish_reason = state.finish_reason or _STOP_REASON_MAP.get(stop, "stop")
            break

        # Error response: {"jsonrpc":"2.0","id":...,"error":{...}}
        err = msg.get("error")
        if isinstance(err, dict):
            raise RuntimeError(
                f"acpx claude error: {err.get('message') or err}",
            )

        # Notifications carry the actual session updates.
        method = msg.get("method")
        params = msg.get("params") or {}
        if method != "session/update":
            continue
        if state.session_id is None:
            state.session_id = params.get("sessionId")

        update = params.get("update") or {}
        kind = update.get("sessionUpdate")

        if kind == "agent_message_chunk":
            text = _content_text(update.get("content"))
            if text:
                yield _chat_chunk(state, {"content": text})
            continue

        if kind == "agent_thought_chunk":
            text = _content_text(update.get("content"))
            if text:
                yield _chat_chunk(state, {"reasoning_content": text})
            continue

        if kind == "tool_call":
            tool_call_id = update.get("toolCallId") or f"call_{uuid.uuid4().hex[:12]}"
            name = update.get("title") or ""
            # ACP ``rawInput`` is unstructured JSON.  OpenAI's
            # tool_calls expects ``arguments`` as a JSON string;
            # serialize whatever we got.  ACP also has a ``kind``
            # (file/edit/execute/...) that doesn't map cleanly to
            # function-call semantics — drop it.
            args = update.get("rawInput")
            args_str = json.dumps(args) if args is not None else ""
            idx = len(state.tool_calls)
            state.tool_calls[idx] = {"id": tool_call_id, "name": name, "args": args_str}
            state.tool_call_id_to_index[tool_call_id] = idx
            yield _chat_chunk(state, {
                "tool_calls": [{
                    "index": idx,
                    "id": tool_call_id,
                    "type": "function",
                    "function": {"name": name, "arguments": args_str},
                }],
            })
            continue

        if kind == "tool_call_update":
            # Status-only events; we don't forward result content
            # because Claude narrates results via subsequent
            # agent_message_chunk events.  Bookkeeping only.
            tcid = update.get("toolCallId")
            if tcid and tcid in state.tool_call_id_to_index:
                status = update.get("status")
                if status == "failed":
                    logger.info("acpx: tool_call %s failed", tcid)
            continue

        # Everything else (plan, mode/command/config/session_info
        # updates, user_message_chunk echo) is metadata — ignore.

    # Final chunk with finish_reason.
    yield _chat_chunk(state, {}, finish_reason=state.finish_reason or "stop")


async def collect_as_chat_completion(
    line_reader: AsyncIterator[str],
    state: StreamState,
) -> dict:
    """Drain a JSON-line reader into a single non-streaming
    ``chat.completion`` dict.  Used by callers that asked for
    ``stream=False`` — we always stream the acpx subprocess and
    reassemble here, mirroring :func:`codex_translate.collect_as_chat_completion`.
    """
    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    tool_calls: list[dict] = []
    tool_call_id_to_index: dict[str, int] = {}

    async for raw_line in line_reader:
        line = raw_line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue

        result = msg.get("result")
        if isinstance(result, dict) and "stopReason" in result:
            stop = result.get("stopReason") or "end_turn"
            state.finish_reason = "tool_calls" if tool_calls else _STOP_REASON_MAP.get(stop, "stop")
            break

        err = msg.get("error")
        if isinstance(err, dict):
            raise RuntimeError(f"acpx claude error: {err.get('message') or err}")

        if msg.get("method") != "session/update":
            continue
        update = (msg.get("params") or {}).get("update") or {}
        kind = update.get("sessionUpdate")

        if kind == "agent_message_chunk":
            text = _content_text(update.get("content"))
            if text:
                content_parts.append(text)
        elif kind == "agent_thought_chunk":
            text = _content_text(update.get("content"))
            if text:
                reasoning_parts.append(text)
        elif kind == "tool_call":
            tcid = update.get("toolCallId") or f"call_{uuid.uuid4().hex[:12]}"
            args = update.get("rawInput")
            args_str = json.dumps(args) if args is not None else ""
            tool_call_id_to_index[tcid] = len(tool_calls)
            tool_calls.append({
                "id": tcid,
                "type": "function",
                "function": {"name": update.get("title") or "", "arguments": args_str},
            })
        # everything else: drop.

    message: dict[str, Any] = {
        "role": "assistant",
        "content": "".join(content_parts) if content_parts else None,
    }
    if reasoning_parts:
        message["reasoning_content"] = "".join(reasoning_parts)
    if tool_calls:
        message["tool_calls"] = tool_calls
        state.finish_reason = "tool_calls"

    return {
        "id": state.response_id,
        "object": "chat.completion",
        "created": state.created,
        "model": state.model,
        "choices": [{
            "index": 0,
            "message": message,
            "finish_reason": state.finish_reason or "stop",
        }],
    }


def _content_text(block: Any) -> str:
    """Extract plain text from an ACP ContentBlock.  Non-text
    blocks (image/audio/resource) collapse to a placeholder so
    downstream channels still see something.
    """
    if not isinstance(block, dict):
        return ""
    t = block.get("type")
    if t == "text":
        return str(block.get("text", ""))
    if t in ("image", "audio"):
        return f"[{t} attached]"
    if t in ("resource_link", "resource"):
        return f"[resource {block.get('uri','')}]"
    return ""


# =========================================================================
# Subprocess driver: spawn acpx, expose stdin/stdout JSON-line iter
# =========================================================================


async def spawn_acpx_and_stream(
    prompt_blocks: list[dict],
    *,
    session_name: str | None = None,
    cmd: tuple[str, ...] | None = None,
    cwd: str | None = None,
    env: dict[str, str] | None = None,
) -> tuple[asyncio.subprocess.Process, AsyncIterator[str]]:
    """Spawn an acpx subprocess, feed prompt content via stdin, and
    return ``(process, stdout-line-iterator)``.

    When ``session_name`` is supplied, runs in **stateful mode**:
    ``acpx claude -s <name>``.  Claude Code's session retains
    history across invocations, so the caller is expected to ship
    only the new turn (see :func:`extract_tail_from_history`).
    Used by :class:`ClaudeAcpxChatModel` for the cache-warm path.

    When ``session_name`` is None and ``cmd`` is None, falls back
    to the legacy stateless ``acpx claude exec`` (one-shot, full
    history per call).  Useful for tests and for the rare
    no-session path.

    Explicit ``cmd`` overrides both modes — supply a tuple for
    custom invocations.

    Caller is responsible for awaiting ``process.wait()`` after the
    iterator drains.

    NOTE: acpx's stdin currently expects plain text, NOT ACP
    ContentBlocks — so multimodal prompts (images, resource_link)
    take the lossy path through :func:`_content_text` here.
    Multi-block input is concatenated as a text transcript.
    Open question 3: verify image URI handling against current
    Claude Code adapter; if URIs resolve, switch to ``image`` block.
    """
    if cmd is None:
        cmd = (
            stateful_acpx_cmd(session_name)
            if session_name
            else DEFAULT_ACPX_CMD
        )
    # Collapse blocks → single stdin payload.
    text_payload = "\n\n".join(
        _content_text(b) for b in prompt_blocks if _content_text(b)
    )
    if not text_payload:
        text_payload = "(empty prompt)"

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=cwd,
        env=env,
    )
    assert proc.stdin is not None
    assert proc.stdout is not None
    proc.stdin.write(text_payload.encode("utf-8"))
    await proc.stdin.drain()
    proc.stdin.close()

    async def _lines() -> AsyncIterator[str]:
        assert proc.stdout is not None
        while True:
            raw = await proc.stdout.readline()
            if not raw:
                return
            yield raw.decode("utf-8", errors="replace")

    return proc, _lines()
