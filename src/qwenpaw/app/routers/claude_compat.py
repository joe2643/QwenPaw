# -*- coding: utf-8 -*-
"""Anthropic-native HTTP shim backed by Claude Code OAuth.

Sibling of :mod:`openai_compat`.  Where that module brokers Codex
OAuth → ``/v1/chat/completions``, this one brokers Claude Code
OAuth → ``/v1/messages`` — the request bytes go to Anthropic
verbatim so every Anthropic-API feature works unchanged,
including ``cache_control: ephemeral`` breakpoints (ACPX path
silently drops these because it has to retranslate into ACP
turn frames; direct API does not).

Why a pure passthrough instead of routing through
``AnthropicProvider``:

- ``AnthropicProvider`` is wired for agent-runtime use (probing,
  ChatModelBase shape, auto cache-tagging tail messages).  Useful
  for the agent, but for an external proxy we want the caller's
  request to land at Anthropic byte-for-byte so SDK upgrades,
  beta headers, and per-block ``cache_control`` work without
  copaw needing to learn the schema.
- ``ClaudeAuth.auth_headers()`` already produces the exact header
  bundle Claude Code CLI sends (``Authorization: Bearer``,
  ``anthropic-version``, ``anthropic-beta``, ``x-app: cli``,
  ``user-agent``) — Cloudflare WAF + Anthropic backend both
  accept it.  Stripping any of those gets you 403s in the
  wild, so reuse the existing helper instead of rebuilding.

Endpoint: ``POST /anthropic/v1/messages``

* Streaming (``stream: true``) returns the upstream SSE byte
  stream untouched — exact wire format match.
* Non-streaming returns the upstream JSON body verbatim.
* Auth: opt-in via ``QWENPAW_ANTHROPIC_COMPAT_KEY`` env, same
  pattern as ``openai_compat._check_auth``.  Unset ⇒ open
  (localhost-only deployment).

Also exposes ``GET /anthropic/v1/models`` returning a hard-coded
list of the slugs the Claude Code subscription accepts — the
upstream API has no ``/v1/models`` endpoint for OAuth tokens,
so we publish what we know works.
"""
from __future__ import annotations

import logging
import os
import time
from typing import AsyncIterator, Optional

import httpx
from fastapi import APIRouter, Body, Header, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse

from ...providers.claude_auth import ClaudeAuth

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/anthropic/v1", tags=["anthropic-compat"])

ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"

# Slugs that the Claude Code subscription bearer can reach.  The
# real Anthropic API has no /v1/models endpoint for OAuth tokens
# (only API-key tokens), so we hand-publish what works.  Update
# when Anthropic ships a new generation.
_KNOWN_MODELS = [
    "claude-opus-4-7",
    "claude-opus-4-6",
    "claude-sonnet-4-6",
    "claude-sonnet-4-5",
    "claude-haiku-4-5",
]

_auth: ClaudeAuth | None = None


def _get_auth() -> ClaudeAuth:
    """Lazy ClaudeAuth singleton — first request triggers
    ``~/.claude/.credentials.json`` load.  Singleton because
    ClaudeAuth holds its own asyncio.Lock for refresh coalescing
    and an mtime-tracked credential cache."""
    global _auth
    if _auth is None:
        _auth = ClaudeAuth()
    return _auth


def _check_auth(authorization: Optional[str]) -> None:
    """Optional bearer-token gate.  No env set ⇒ no auth (matches
    the openai_compat default).  When set, exact match required."""
    expected = os.environ.get("QWENPAW_ANTHROPIC_COMPAT_KEY")
    if not expected:
        return
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(401, detail="missing bearer token")
    if authorization[len("Bearer ") :] != expected:
        raise HTTPException(403, detail="invalid bearer token")


@router.post("/messages")
async def messages(
    body: dict = Body(...),
    authorization: Optional[str] = Header(default=None),
) -> object:
    """Anthropic-compatible messages endpoint via Claude Code OAuth.

    Body is forwarded byte-for-byte (after JSON re-encoding) so
    ``cache_control``, ``thinking``, ``tools``, ``system``, and any
    future Anthropic API field works without code changes here.
    """
    _check_auth(authorization)

    auth = _get_auth()
    try:
        headers = await auth.auth_headers()
    except Exception as e:
        raise HTTPException(
            502,
            detail=f"Claude OAuth refresh failed: {e}",
        ) from e
    # Force identity encoding upstream: httpx negotiates gzip/br
    # by default, but our streaming path forwards raw bytes
    # without re-emitting Content-Encoding, so a compressed body
    # would reach the client as undecodable gibberish.  Cheap to
    # disable — Anthropic responses are small JSON / SSE.
    headers["accept-encoding"] = "identity"

    client_wants_stream = bool(body.get("stream", False))

    if client_wants_stream:
        return StreamingResponse(
            _proxy_sse(body, headers),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            },
        )

    async with httpx.AsyncClient(
        timeout=httpx.Timeout(600, connect=30),
    ) as client:
        upstream = await client.post(
            ANTHROPIC_API_URL,
            json=body,
            headers=headers,
        )
    if upstream.status_code != 200:
        snippet = upstream.text[:500]
        logger.warning(
            "[anthropic-compat] upstream HTTP %d: %s",
            upstream.status_code,
            snippet,
        )
        raise HTTPException(upstream.status_code, detail=snippet)
    return JSONResponse(upstream.json())


async def _proxy_sse(
    upstream_body: dict,
    upstream_headers: dict,
) -> AsyncIterator[bytes]:
    """Forward the upstream Anthropic SSE byte stream untouched.

    Anthropic SSE is event-named (``event: message_start`` etc.) and
    not the same wire format as OpenAI's chat-completions SSE — but
    Anthropic SDK consumers know how to parse it, so we don't
    re-frame.  On upstream error we synthesise a single
    ``event: error`` frame so the client sees structured failure
    instead of a silent stream close.
    """
    async with httpx.AsyncClient(
        timeout=httpx.Timeout(600, connect=30),
    ) as client:
        async with client.stream(
            "POST",
            ANTHROPIC_API_URL,
            json=upstream_body,
            headers=upstream_headers,
        ) as upstream:
            if upstream.status_code != 200:
                err_bytes = await upstream.aread()
                err_body = err_bytes.decode("utf-8", "replace")[:500]
                logger.warning(
                    "[anthropic-compat] stream upstream HTTP %d: %s",
                    upstream.status_code,
                    err_body,
                )
                yield (
                    f"event: error\ndata: "
                    f'{{"type":"error","error":{{'
                    f'"type":"upstream_{upstream.status_code}",'
                    f'"message":{err_body!r}}}}}\n\n'
                ).encode()
                return

            async for chunk in upstream.aiter_raw():
                if chunk:
                    yield chunk


@router.get("/models")
async def list_models(
    authorization: Optional[str] = Header(default=None),
) -> dict:
    """Anthropic-style model list.  Hand-published because the
    OAuth bearer can't introspect ``/v1/models`` upstream."""
    _check_auth(authorization)
    created = int(time.time())
    return {
        "object": "list",
        "data": [
            {
                "id": slug,
                "object": "model",
                "created": created,
                "owned_by": "claude-oauth",
            }
            for slug in _KNOWN_MODELS
        ],
    }
