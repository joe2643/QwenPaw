# -*- coding: utf-8 -*-
"""Codex OAuth adapter proxy.

Thin FastAPI wrapper around :mod:`qwenpaw.providers.codex_translate` —
listens on ``http://localhost:9877/v1/chat/completions`` in OpenAI
chat-completions shape and translates each request to the ChatGPT
backend's Responses API at ``https://chatgpt.com/backend-api/codex/
responses``, authenticated with the Codex-CLI OAuth token
(``~/.codex/auth.json``).

This proxy is useful when:

* You want to point an existing OpenAI-compat client (SkillClaw,
  custom scripts) at a ChatGPT Plus/Pro subscription without
  teaching it about PKCE or the Responses API shape.
* You want a separate process / port for debugging — you can ``curl``
  the local endpoint and see exactly what wire shape comes out.

For in-process use (CoPaw agents talking to ChatGPT OAuth) prefer
:class:`qwenpaw.providers.codex_oauth_model.CodexOAuthChatModel` —
same translation code (``codex_translate`` shared), no extra
process to manage.

Run::

    python -m qwenpaw.providers.codex_oauth_proxy              # default port 9877
    QWENPAW_CODEX_PROXY_PORT=9878 python -m ...codex_oauth_proxy
"""

from __future__ import annotations

import json
import logging
import os
from typing import AsyncIterator

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from .codex_auth import CodexAuth
from .codex_translate import (
    ALLOWED_MODELS,
    StreamState,
    build_responses_body,
    collect_as_chat_completion,
    translate_responses_events_to_chat_chunks,
)

logger = logging.getLogger(__name__)


async def _sse_wrap(
    chunks: AsyncIterator[dict],
) -> AsyncIterator[str]:
    """Wrap chat-completion chunk dicts as SSE ``data:`` lines and
    append the ``[DONE]`` terminator."""
    async for chunk in chunks:
        yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
    yield "data: [DONE]\n\n"


# =========================================================================
# HTTP server
# =========================================================================


def create_app() -> FastAPI:
    app = FastAPI(title="Codex OAuth Adapter", version="0.2.0")

    # Lazy — init on first call so import stays cheap and startup
    # errors surface in logs rather than at import time.
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
            raise HTTPException(
                status_code=400, detail=f"invalid JSON body: {e}",
            ) from e

        auth = _get_auth()
        headers = await auth.auth_headers()
        upstream_url = f"{auth.base_url}/codex/responses"
        upstream_body = build_responses_body(chat_body)
        client_wants_stream = bool(chat_body.get("stream", False))

        state = StreamState(model=upstream_body["model"])

        async def _forward_stream() -> AsyncIterator[str]:
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(600, connect=30),
            ) as client:
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
                                "message": (
                                    f"Codex upstream "
                                    f"{upstream.status_code}: {err_body}"
                                ),
                                "type": "upstream_error",
                                "code": upstream.status_code,
                            },
                        }
                        yield f"data: {json.dumps(payload)}\n\n"
                        yield "data: [DONE]\n\n"
                        return
                    async for sse_line in _sse_wrap(
                        translate_responses_events_to_chat_chunks(
                            upstream, state,
                        ),
                    ):
                        yield sse_line

        if client_wants_stream:
            return StreamingResponse(
                _forward_stream(), media_type="text/event-stream",
            )

        # Non-streaming: drain upstream, return a single chat.completion body.
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(600, connect=30),
        ) as client:
            async with client.stream(
                "POST", upstream_url, json=upstream_body, headers=headers,
            ) as upstream:
                if upstream.status_code != 200:
                    err_bytes = await upstream.aread()
                    err_body = err_bytes.decode("utf-8", errors="replace")
                    retry_after = upstream.headers.get("retry-after")
                    resp_headers = (
                        {"Retry-After": retry_after} if retry_after else {}
                    )
                    return JSONResponse(
                        status_code=upstream.status_code,
                        content={
                            "error": {
                                "message": (
                                    f"Codex upstream error: {err_body}"
                                ),
                                "type": "upstream_error",
                                "code": upstream.status_code,
                            },
                        },
                        headers=resp_headers,
                    )
                body = await collect_as_chat_completion(upstream, state)
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
    logger.info(
        "Codex OAuth adapter listening on http://%s:%d", host, port,
    )
    uvicorn.run(create_app(), host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
