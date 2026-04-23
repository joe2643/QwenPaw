# -*- coding: utf-8 -*-
"""
Codex OAuth adapter proxy.

Listens on ``http://localhost:9877/v1/chat/completions`` in the standard
OpenAI chat-completions shape and translates every request to OpenAI's
*Responses* API at ``https://chatgpt.com/backend-api/codex/responses``,
authenticated with the Codex-CLI OAuth token (``~/.codex/auth.json``).

This lets any OpenAI-compatible client (QwenPaw, SkillClaw, custom
scripts) consume a ChatGPT Plus/Pro subscription without knowing about
the Responses-API shape, PKCE, or token refresh.

Why an adapter rather than a direct provider?
    * keeps the token-refresh and base-URL swap in one place
    * QwenPaw's existing ``openai_provider.py`` stays unchanged
    * non-QwenPaw callers can reuse it too

Run::
    python -m qwenpaw.providers.codex_oauth_proxy              # default port 9877
    QWENPAW_CODEX_PROXY_PORT=9878 python -m ...codex_oauth_proxy
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid
from typing import Any, AsyncIterator

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from .codex_auth import CodexAuth

logger = logging.getLogger(__name__)

ALLOWED_MODELS = {"gpt-5.4", "gpt-5.2"}


# =========================================================================
# Request translation: OpenAI chat/completions -> Responses API
# =========================================================================


def _content_to_plain_text(content: Any) -> str:
    """Flatten OpenAI chat content (str OR list of {type,text,...}) to plain
    text.  Non-text items are represented as short placeholders, so this is
    NOT lossless — use :func:`_content_to_responses_items` for multimodal
    input.  Kept for system / assistant / tool-result flattening where
    Codex Responses API only accepts plain strings.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
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


def _content_to_responses_items(content: Any) -> list[dict]:
    """Translate OpenAI chat ``content`` (for user-role messages) into a
    Responses-API content array.  Supports text and image items.

    OpenAI chat/completions shape::

        [{"type":"text", "text":"..."},
         {"type":"image_url", "image_url":{"url":"data:image/png;base64,..." }}]

    Codex Responses shape::

        [{"type":"input_text", "text":"..."},
         {"type":"input_image", "image_url":"data:image/png;base64,..."}]
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
            # OpenAI chat: image_url is either a string OR {"url": "...", "detail": "..."}
            # Codex Responses: input_image expects a plain image_url string.
            iu = item.get("image_url")
            if isinstance(iu, dict):
                url = iu.get("url", "")
            else:
                url = str(iu) if iu else ""
            if url:
                out.append({"type": "input_image", "image_url": url})
        elif t == "image":
            # Generic "image" content with base64 source (Anthropic-style passthrough)
            source = item.get("source") or {}
            if source.get("type") == "base64":
                mime = source.get("media_type") or "image/png"
                data = source.get("data", "")
                if data:
                    out.append({
                        "type": "input_image",
                        "image_url": f"data:{mime};base64,{data}",
                    })
            elif source.get("type") == "url":
                u = source.get("url", "")
                if u:
                    out.append({"type": "input_image", "image_url": u})
    return out or [{"type": "input_text", "text": ""}]


def _convert_messages_to_responses_input(
    messages: list[dict],
) -> tuple[str, list[dict]]:
    """Split out a single ``instructions`` string and a Responses-API
    ``input`` array that preserves assistant text, tool calls, and tool
    results with correct ordering.
    """
    instructions_parts: list[str] = []
    items: list[dict] = []

    for msg in messages:
        role = msg.get("role")
        content = msg.get("content")
        if role == "system":
            instructions_parts.append(_content_to_plain_text(content))
            continue

        if role == "user":
            items.append({
                "type": "message",
                "role": "user",
                "content": _content_to_responses_items(content),
            })
            continue

        if role == "assistant":
            text = _content_to_plain_text(content)
            if text:
                items.append({
                    "type": "message",
                    "role": "assistant",
                    "content": [
                        {"type": "output_text", "text": text},
                    ],
                })
            # Tool calls carried on the same assistant turn become sibling
            # function_call items in Responses input order.
            for tc in msg.get("tool_calls") or []:
                fn = tc.get("function") or {}
                items.append({
                    "type": "function_call",
                    "call_id": tc.get("id") or f"call_{uuid.uuid4().hex[:12]}",
                    "name": fn.get("name", ""),
                    "arguments": fn.get("arguments", "") or "",
                })
            continue

        if role == "tool":
            items.append({
                "type": "function_call_output",
                "call_id": msg.get("tool_call_id", ""),
                "output": _content_to_plain_text(content),
            })
            continue

    instructions = "\n\n".join(p for p in instructions_parts if p.strip())
    return instructions, items


def _convert_tools(tools: list[dict] | None) -> list[dict] | None:
    """OpenAI chat-completions tools (with ``function`` wrapper) ->
    Responses API ``tools`` (flattened)."""
    if not tools:
        return None
    out: list[dict] = []
    for t in tools:
        if t.get("type") != "function":
            out.append(t)
            continue
        fn = t.get("function") or {}
        out.append({
            "type": "function",
            "name": fn.get("name", ""),
            "description": fn.get("description", ""),
            "parameters": fn.get("parameters") or {"type": "object", "properties": {}},
            "strict": False,
        })
    return out


def _build_responses_body(chat_body: dict) -> dict:
    model = chat_body.get("model", "gpt-5.4")
    if model not in ALLOWED_MODELS:
        # Accept any synonyms the user may configure, but force to a
        # ChatGPT-supported model. (gpt-5, o3, gpt-5-codex etc. are
        # rejected server-side.)
        model = "gpt-5.4"
    instructions, input_items = _convert_messages_to_responses_input(
        chat_body.get("messages") or [],
    )
    body: dict[str, Any] = {
        "model": model,
        "input": input_items,
        "stream": True,   # always true upstream — we fake non-stream on the client side
        "store": False,   # required for ChatGPT account
        # ChatGPT backend rejects empty/missing instructions with HTTP 400, so
        # always send a generic system hint when the caller didn't provide one.
        "instructions": instructions or "You are a helpful assistant.",
    }

    tools = _convert_tools(chat_body.get("tools"))
    if tools:
        body["tools"] = tools

    tool_choice = chat_body.get("tool_choice")
    if tool_choice is not None:
        body["tool_choice"] = tool_choice

    reasoning_effort = chat_body.get("reasoning_effort") or "low"
    body["reasoning"] = {"effort": reasoning_effort}

    # NOTE: ChatGPT-backend Codex responses API rejects `max_output_tokens`
    # and silently ignores most sampling knobs (temperature / top_p /
    # frequency_penalty etc.) when called with a ChatGPT-account OAuth
    # token.  We therefore drop every unsupported field rather than
    # forwarding them blindly — upstream returns HTTP 400 otherwise.
    return body


# =========================================================================
# Streaming translation: Responses API SSE -> chat/completions SSE
# =========================================================================


class _StreamState:
    """Carry per-request state while translating upstream events."""

    def __init__(self, model: str) -> None:
        self.model = model
        self.response_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        self.created = int(time.time())
        # tool_call tracking: index -> {id, name, args_accumulated}
        self.tool_calls: dict[int, dict[str, Any]] = {}
        # map upstream item_id (like "fc_xxx") -> index so we can attach deltas
        self.item_id_to_index: dict[str, int] = {}
        self.finished = False
        self.finish_reason: str | None = None
        self.emitted_role = False
        self.final_usage: dict[str, Any] | None = None


def _sse_chunk(state: _StreamState, delta: dict, finish_reason: str | None = None) -> str:
    payload = {
        "id": state.response_id,
        "object": "chat.completion.chunk",
        "created": state.created,
        "model": state.model,
        "choices": [{
            "index": 0,
            "delta": delta,
            "finish_reason": finish_reason,
        }],
    }
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


async def _translate_upstream_sse(
    upstream: httpx.Response,
    state: _StreamState,
) -> AsyncIterator[str]:
    """Read upstream Responses SSE and yield chat-completions SSE chunks."""
    # Emit the initial role delta so clients see an assistant frame up front.
    if not state.emitted_role:
        state.emitted_role = True
        yield _sse_chunk(state, {"role": "assistant"})

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

        # Text deltas
        if ev_type == "response.output_text.delta":
            delta_text = ev.get("delta", "") or ""
            if delta_text:
                yield _sse_chunk(state, {"content": delta_text})
            continue

        # Tool-call item announced
        if ev_type == "response.output_item.added":
            item = ev.get("item") or {}
            if item.get("type") == "function_call":
                idx = len(state.tool_calls)
                item_id = item.get("id", "")
                call_id = item.get("call_id") or f"call_{uuid.uuid4().hex[:12]}"
                name = item.get("name", "")
                state.tool_calls[idx] = {
                    "id": call_id,
                    "name": name,
                    "args": "",
                }
                if item_id:
                    state.item_id_to_index[item_id] = idx
                yield _sse_chunk(state, {
                    "tool_calls": [{
                        "index": idx,
                        "id": call_id,
                        "type": "function",
                        "function": {"name": name, "arguments": ""},
                    }],
                })
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
                yield _sse_chunk(state, {
                    "tool_calls": [{
                        "index": idx,
                        "function": {"arguments": delta_args},
                    }],
                })
            continue

        # Tool-call argument done — treat as finish signal for finish_reason
        if ev_type == "response.function_call_arguments.done":
            state.finish_reason = "tool_calls"
            continue

        # Finalization
        if ev_type == "response.completed":
            resp = ev.get("response") or {}
            state.final_usage = resp.get("usage")
            if state.finish_reason is None:
                # Inspect output items for tool_calls, otherwise "stop"
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

    # Final chunk with finish_reason
    final_chunk = {
        "id": state.response_id,
        "object": "chat.completion.chunk",
        "created": state.created,
        "model": state.model,
        "choices": [{
            "index": 0,
            "delta": {},
            "finish_reason": state.finish_reason or "stop",
        }],
    }
    if state.final_usage:
        final_chunk["usage"] = {
            "prompt_tokens": state.final_usage.get("input_tokens"),
            "completion_tokens": state.final_usage.get("output_tokens"),
            "total_tokens": state.final_usage.get("total_tokens"),
        }
    yield f"data: {json.dumps(final_chunk, ensure_ascii=False)}\n\n"
    yield "data: [DONE]\n\n"


async def _collect_non_streaming(
    upstream: httpx.Response,
    state: _StreamState,
) -> dict:
    """Drain the upstream SSE stream and assemble a non-streaming
    chat/completions response body."""
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

        if t == "response.output_text.delta":
            content_parts.append(ev.get("delta", "") or "")
            continue

        if t == "response.output_item.added":
            item = ev.get("item") or {}
            if item.get("type") == "function_call":
                item_id = item.get("id", "")
                idx = len(tool_calls)
                tool_calls.append({
                    "index": idx,
                    "id": item.get("call_id") or f"call_{uuid.uuid4().hex[:12]}",
                    "type": "function",
                    "function": {"name": item.get("name", ""), "arguments": ""},
                })
                state.item_id_to_index[item_id] = idx
            continue

        if t == "response.function_call_arguments.delta":
            idx = state.item_id_to_index.get(ev.get("item_id", ""))
            if idx is None:
                continue
            tool_calls[idx]["function"]["arguments"] += ev.get("delta", "") or ""
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
        # Presence of tool_calls wins over any finish_reason heuristic —
        # the OpenAI chat/completions contract requires ``tool_calls`` here.
        state.finish_reason = "tool_calls"

    body = {
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
    if state.final_usage:
        body["usage"] = {
            "prompt_tokens": state.final_usage.get("input_tokens"),
            "completion_tokens": state.final_usage.get("output_tokens"),
            "total_tokens": state.final_usage.get("total_tokens"),
        }
    return body


# =========================================================================
# HTTP server
# =========================================================================


def create_app() -> FastAPI:
    app = FastAPI(title="Codex OAuth Adapter", version="0.1.0")

    # Lazy — init on first call so import is cheap + errors surface in logs
    auth_holder: dict[str, CodexAuth] = {}

    def _get_auth() -> CodexAuth:
        if "a" not in auth_holder:
            auth_holder["a"] = CodexAuth()
        return auth_holder["a"]

    @app.get("/healthz")
    async def _healthz():
        try:
            auth = _get_auth()
            creds = await auth.ensure_fresh()
            return {
                "status": "ok",
                "expires_in_s": creds.seconds_until_expiry,
                "account_id": creds.account_id,
            }
        except Exception as e:
            raise HTTPException(status_code=503, detail=str(e)) from e

    @app.get("/v1/models")
    async def _models():
        return {
            "object": "list",
            "data": [
                {"id": m, "object": "model", "owned_by": "openai"}
                for m in sorted(ALLOWED_MODELS)
            ],
        }

    @app.post("/v1/chat/completions")
    async def _chat_completions(request: Request):
        try:
            chat_body = await request.json()
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"invalid JSON body: {e}") from e

        auth = _get_auth()
        headers = await auth.auth_headers()
        upstream_url = f"{auth.base_url}/codex/responses"
        upstream_body = _build_responses_body(chat_body)
        client_wants_stream = bool(chat_body.get("stream", False))

        state = _StreamState(model=upstream_body["model"])

        async def _forward_stream() -> AsyncIterator[str]:
            async with httpx.AsyncClient(timeout=httpx.Timeout(600, connect=30)) as client:
                async with client.stream(
                    "POST", upstream_url, json=upstream_body, headers=headers,
                ) as upstream:
                    if upstream.status_code != 200:
                        err_bytes = await upstream.aread()
                        err_body = err_bytes.decode("utf-8", errors="replace")
                        logger.warning(
                            "[CodexProxy] upstream HTTP %d: %s",
                            upstream.status_code, err_body[:300],
                        )
                        payload = {
                            "error": {
                                "message": f"Codex upstream {upstream.status_code}: {err_body}",
                                "type": "upstream_error",
                                "code": upstream.status_code,
                            },
                        }
                        yield f"data: {json.dumps(payload)}\n\n"
                        yield "data: [DONE]\n\n"
                        return
                    async for chunk in _translate_upstream_sse(upstream, state):
                        yield chunk

        if client_wants_stream:
            return StreamingResponse(_forward_stream(), media_type="text/event-stream")

        # Non-streaming: drain upstream, return a single chat.completion body
        async with httpx.AsyncClient(timeout=httpx.Timeout(600, connect=30)) as client:
            async with client.stream(
                "POST", upstream_url, json=upstream_body, headers=headers,
            ) as upstream:
                if upstream.status_code != 200:
                    err_bytes = await upstream.aread()
                    err_body = err_bytes.decode("utf-8", errors="replace")
                    retry_after = upstream.headers.get("retry-after")
                    resp_headers = {"Retry-After": retry_after} if retry_after else {}
                    return JSONResponse(
                        status_code=upstream.status_code,
                        content={
                            "error": {
                                "message": f"Codex upstream error: {err_body}",
                                "type": "upstream_error",
                                "code": upstream.status_code,
                            },
                        },
                        headers=resp_headers,
                    )
                body = await _collect_non_streaming(upstream, state)
        return JSONResponse(content=body)

    return app


def main() -> None:
    import uvicorn
    port = int(os.environ.get("QWENPAW_CODEX_PROXY_PORT", "9877"))
    host = os.environ.get("QWENPAW_CODEX_PROXY_HOST", "127.0.0.1")
    logging.basicConfig(
        level=os.environ.get("QWENPAW_CODEX_PROXY_LOG", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logger.info("Codex OAuth adapter listening on http://%s:%d", host, port)
    uvicorn.run(create_app(), host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
