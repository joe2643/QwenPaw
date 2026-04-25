# -*- coding: utf-8 -*-
"""OpenAI-compatible HTTP shim backed by Codex OAuth.

Lets external consumers (SkillClaw's PRM scorer, evolve_server's
``EVOLVE_MODEL`` LLM, third-party tools) hit a local
``/v1/chat/completions`` endpoint and have it routed through the
user's ChatGPT subscription via the in-process Codex OAuth bridge —
no separate proxy daemon, no shared API key, no MITM auth handling
copied across components.

Why one centralised shim instead of letting every consumer wire
its own OAuth: the Codex OAuth refresh + ``/codex/responses`` ↔
chat-completions translation already lives in
``codex_translate.py`` and ``codex_oauth_model.py``.  Re-implementing
that for every new component (PRM scorer, summarizer, validator)
duplicates the protocol-version coupling.  This shim keeps that
coupling in one place — anywhere downstream that speaks OpenAI
chat-completions just sets ``base_url`` here.

Endpoint: ``POST /v1/chat/completions``
* Streaming + non-streaming both supported.
* ``model`` field forwarded as-is — caller picks any slug the
  user's ChatGPT account can reach (``gpt-5.4``, ``gpt-5.4-mini``,
  ``gpt-5.3-codex-spark``, etc.).  Wrong slug → upstream 400.
* Auth: opt-in.  When ``QWENPAW_OPENAI_COMPAT_KEY`` env is set,
  caller must send ``Authorization: Bearer <key>``; otherwise
  endpoint is open (localhost-only by default).
"""
from __future__ import annotations

import json
import logging
import os
import time
import uuid
from typing import Any, AsyncIterator, Optional

import httpx
from fastapi import APIRouter, Body, Header, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse

from ...providers.codex_auth import CodexAuth
from ...providers.codex_translate import (
    StreamState,
    build_responses_body,
    collect_as_chat_completion,
    translate_responses_events_to_chat_chunks,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1", tags=["openai-compat"])

# Single shared auth instance — its mtime check picks up
# ``codex login`` rewrites automatically, no restart needed.
_auth: CodexAuth | None = None


def _get_auth() -> CodexAuth:
    """Lazy CodexAuth singleton — first request triggers load."""
    global _auth
    if _auth is None:
        _auth = CodexAuth()
    return _auth


def _check_auth(authorization: Optional[str]) -> None:
    """Optional bearer-token gate.  No env set ⇒ no auth (matches
    the localhost-default deployment).  When set, exact match
    required."""
    expected = os.environ.get("QWENPAW_OPENAI_COMPAT_KEY")
    if not expected:
        return
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(401, detail="missing bearer token")
    if authorization[len("Bearer "):] != expected:
        raise HTTPException(403, detail="invalid bearer token")


@router.post("/chat/completions")
async def chat_completions(
    body: dict = Body(...),
    authorization: Optional[str] = Header(default=None),
) -> Any:
    """OpenAI-compatible chat completions, served via Codex OAuth.

    Returns a streaming SSE response when ``stream: true`` (matches
    the real OpenAI streaming wire format), or a single
    ``ChatCompletion`` JSON object otherwise.
    """
    _check_auth(authorization)

    auth = _get_auth()
    try:
        await auth.ensure_fresh()
    except Exception as e:
        raise HTTPException(
            502,
            detail=f"Codex OAuth refresh failed: {e}",
        ) from e

    headers = await auth.auth_headers()
    upstream_url = f"{auth.base_url}/codex/responses"

    # Build the Responses-API body the same way ``CodexOAuthChatModel``
    # would.  Strip ``stream_options`` — same agentscope-vs-ChatGPT
    # quirk handled in ``codex_oauth_model._wrapped_create``.
    call_kwargs = dict(body)
    call_kwargs.pop("stream_options", None)
    responses_body = build_responses_body(call_kwargs)

    client_wants_stream = bool(call_kwargs.get("stream", False))
    state = StreamState(model=responses_body["model"])

    if client_wants_stream:
        return StreamingResponse(
            _stream_sse(upstream_url, responses_body, headers, state),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            },
        )

    # Non-streaming — drain the upstream SSE into a single
    # ``ChatCompletion`` dict.
    async with httpx.AsyncClient(
        timeout=httpx.Timeout(600, connect=30),
    ) as client:
        async with client.stream(
            "POST", upstream_url, json=responses_body, headers=headers,
        ) as upstream:
            if upstream.status_code != 200:
                err_bytes = await upstream.aread()
                err_body = err_bytes.decode("utf-8", "replace")[:500]
                logger.warning(
                    "[openai-compat] upstream HTTP %d: %s",
                    upstream.status_code,
                    err_body,
                )
                raise HTTPException(
                    upstream.status_code,
                    detail=err_body,
                )
            chat_body = await collect_as_chat_completion(upstream, state)

    return JSONResponse(chat_body)


async def _stream_sse(
    upstream_url: str,
    upstream_body: dict,
    upstream_headers: dict,
    state: StreamState,
) -> AsyncIterator[bytes]:
    """SSE-encode the chat-completions chunk stream.

    Matches the real OpenAI streaming wire format exactly:
    ``data: {chunk_json}\\n\\n`` per chunk, terminated by
    ``data: [DONE]\\n\\n``.
    """
    async with httpx.AsyncClient(
        timeout=httpx.Timeout(600, connect=30),
    ) as client:
        async with client.stream(
            "POST",
            upstream_url,
            json=upstream_body,
            headers=upstream_headers,
        ) as upstream:
            if upstream.status_code != 200:
                err_bytes = await upstream.aread()
                err_body = err_bytes.decode("utf-8", "replace")[:500]
                err = {
                    "error": {
                        "message": err_body,
                        "code": upstream.status_code,
                    },
                }
                yield f"data: {json.dumps(err)}\n\n".encode()
                return

            async for chunk_dict in translate_responses_events_to_chat_chunks(
                upstream, state,
            ):
                yield f"data: {json.dumps(chunk_dict)}\n\n".encode()

    yield b"data: [DONE]\n\n"


@router.get("/models")
async def list_models(
    authorization: Optional[str] = Header(default=None),
) -> dict:
    """OpenAI-compatible model list.  Returns the slugs probed
    against the user's account via the live Codex catalogue, so
    consumers can introspect what's reachable.
    """
    _check_auth(authorization)
    auth = _get_auth()
    try:
        models = await auth.list_models()
    except Exception as e:
        logger.warning("[openai-compat] list_models failed: %s", e)
        models = []
    return {
        "object": "list",
        "data": [
            {
                "id": m.get("slug", ""),
                "object": "model",
                "created": int(time.time()),
                "owned_by": "codex-oauth",
            }
            for m in models
            if m.get("slug")
        ],
    }
