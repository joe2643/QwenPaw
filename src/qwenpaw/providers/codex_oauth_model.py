# -*- coding: utf-8 -*-
"""In-process OpenAI chat-completions ↔ ChatGPT Responses bridge.

Lets CoPaw agents hit a ChatGPT Plus/Pro subscription via Codex OAuth
in-process — no separate daemon needed.  Wraps agentscope's
:class:`OpenAIChatModel` so that every outbound
``client.chat.completions.create`` call is redirected to
``chatgpt.com/backend-api/codex/responses``, translated on both
legs using the shared :mod:`codex_translate` helpers.

Design mirrors :class:`qwenpaw.providers.anthropic_provider.ClaudeOAuthChatModel`:

* Subclass rather than composition — agentscope's response parsing
  logic runs on the same SDK types the real OpenAI client returns,
  so we synthesise those types instead of forking the parser.
* ``_install_wrapper`` mutates ``self.client.chat.completions.create``
  at init time; each call refreshes the OAuth token, builds the
  Responses-API body, and streams the upstream SSE back into SDK
  ``ChatCompletionChunk`` objects (or a single ``ChatCompletion``
  for the non-streaming path).
"""

from __future__ import annotations

import logging
from typing import Any, AsyncIterator

import httpx
from agentscope.model import OpenAIChatModel
from openai.types.chat import ChatCompletion, ChatCompletionChunk

from .codex_translate import (
    StreamState,
    build_responses_body,
    collect_as_chat_completion,
    translate_responses_events_to_chat_chunks,
)

logger = logging.getLogger(__name__)


class CodexOAuthChatModel(OpenAIChatModel):
    """Codex OAuth variant of :class:`OpenAIChatModel`.

    ``auth`` is a :class:`qwenpaw.providers.codex_auth.CodexAuth`
    instance (typed as ``object`` on the signature to avoid a hard
    import cycle at module load).  It must expose the async
    ``ensure_fresh() → creds`` coroutine and ``auth_headers() →
    dict[str, str]`` coroutine plus a synchronous ``base_url``
    attribute, matching the real ``CodexAuth`` interface.
    """

    def __init__(
        self,
        *,
        auth: "object",  # CodexAuth
        **kwargs: Any,
    ) -> None:
        # OpenAI SDK refuses to construct without an ``api_key`` (even
        # though we're about to redirect every request away from its
        # default baseUrl).  Seed with a harmless sentinel — it never
        # reaches the wire since ``_install_wrapper`` replaces
        # ``client.chat.completions.create`` wholesale.
        if kwargs.get("api_key") in (None, ""):
            kwargs["api_key"] = "codex-oauth-unused"
        super().__init__(**kwargs)
        self._auth = auth
        self._install_wrapper()

    def _install_wrapper(self) -> None:
        async def _wrapped_create(**call_kwargs: Any) -> Any:
            # Refresh on the edge: every upstream call gets a fresh
            # token (``ensure_fresh`` no-ops when not near expiry).
            await self._auth.ensure_fresh()  # type: ignore[attr-defined]
            headers = await self._auth.auth_headers()  # type: ignore[attr-defined]
            upstream_url = f"{self._auth.base_url}/codex/responses"  # type: ignore[attr-defined]

            # Strip SDK-specific kwargs the ChatGPT backend rejects.
            # agentscope forwards ``stream_options`` to enable usage
            # telemetry — the Responses API doesn't accept it.
            call_kwargs.pop("stream_options", None)

            responses_body = build_responses_body(call_kwargs)
            client_wants_stream = bool(call_kwargs.get("stream", False))
            state = StreamState(model=responses_body["model"])

            if client_wants_stream:
                return _CodexOAuthAsyncStream(
                    upstream_url=upstream_url,
                    upstream_body=responses_body,
                    headers=headers,
                    state=state,
                )

            # Non-streaming: open the stream, drain it, close it.
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(600, connect=30),
            ) as client:
                async with client.stream(
                    "POST",
                    upstream_url,
                    json=responses_body,
                    headers=headers,
                ) as upstream:
                    _raise_for_upstream_status(upstream)
                    chat_body = await collect_as_chat_completion(
                        upstream, state,
                    )
            return ChatCompletion.model_validate(chat_body)

        self.client.chat.completions.create = _wrapped_create  # type: ignore[method-assign]


def _raise_for_upstream_status(upstream: httpx.Response) -> None:
    """Surface upstream non-200s as an exception the way the OpenAI
    SDK would.  We can't use ``upstream.raise_for_status()`` directly
    because we're inside a streaming context and haven't read the
    body yet — read it first for a useful message.
    """
    if upstream.status_code == 200:
        return
    # aread must happen inside the async caller; callers must hand
    # us an already-opened stream whose first event has arrived.
    raise httpx.HTTPStatusError(
        f"Codex upstream HTTP {upstream.status_code}",
        request=upstream.request,
        response=upstream,
    )


class _CodexOAuthAsyncStream:
    """Async-iterator adapter that looks enough like an OpenAI SDK
    ``AsyncStream[ChatCompletionChunk]`` for agentscope's
    ``_parse_openai_stream_completion_response`` to consume it.

    The real SDK class has extra methods (``__aenter__`` /
    ``__aexit__``, ``close``); agentscope only iterates, so we
    implement the minimum surface and delegate close to the
    underlying httpx client when iteration ends.
    """

    def __init__(
        self,
        upstream_url: str,
        upstream_body: dict,
        headers: dict,
        state: StreamState,
    ) -> None:
        self._upstream_url = upstream_url
        self._upstream_body = upstream_body
        self._headers = headers
        self._state = state
        self._client: httpx.AsyncClient | None = None
        self._stream_ctx: Any = None
        self._upstream: httpx.Response | None = None
        self._iter: AsyncIterator[dict] | None = None

    def __aiter__(self) -> "_CodexOAuthAsyncStream":
        return self

    async def __anext__(self) -> ChatCompletionChunk:
        if self._iter is None:
            # Lazy-open the HTTP stream on first __anext__.
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(600, connect=30),
            )
            self._stream_ctx = self._client.stream(
                "POST",
                self._upstream_url,
                json=self._upstream_body,
                headers=self._headers,
            )
            self._upstream = await self._stream_ctx.__aenter__()
            assert self._upstream is not None
            if self._upstream.status_code != 200:
                # Capture everything we need *before* ``_cleanup``
                # nulls ``self._upstream`` — otherwise the f-string
                # below raises ``AttributeError: 'NoneType'`` and
                # masks the real HTTP status the caller needs.
                status = self._upstream.status_code
                request = self._upstream.request
                response = self._upstream
                err_bytes = await self._upstream.aread()
                err_body = err_bytes.decode("utf-8", errors="replace")
                await self._cleanup()
                raise httpx.HTTPStatusError(
                    f"Codex upstream HTTP {status}: {err_body[:500]}",
                    request=request,
                    response=response,
                )
            self._iter = translate_responses_events_to_chat_chunks(
                self._upstream, self._state,
            )

        assert self._iter is not None
        try:
            chunk_dict = await self._iter.__anext__()
        except StopAsyncIteration:
            await self._cleanup()
            raise

        return ChatCompletionChunk.model_validate(chunk_dict)

    async def _cleanup(self) -> None:
        try:
            if self._stream_ctx is not None:
                await self._stream_ctx.__aexit__(None, None, None)
        finally:
            if self._client is not None:
                await self._client.aclose()
            self._stream_ctx = None
            self._upstream = None
            self._client = None
            self._iter = None

    # Convenience methods some callers expect — agentscope doesn't
    # call these, but defining them avoids AttributeError surprises.
    async def close(self) -> None:
        await self._cleanup()

    async def __aenter__(self) -> "_CodexOAuthAsyncStream":
        return self

    async def __aexit__(self, *_exc: Any) -> None:
        await self._cleanup()
