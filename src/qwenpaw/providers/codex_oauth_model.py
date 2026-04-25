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

import asyncio
import json
import logging
import re
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


# ChatGPT returns:
#   "Error while downloading https://media.example/path. Upstream
#    status code: 403."
# Capture the URL so we can strip it from the request body and retry
# once.  Stops at whitespace OR the literal sentence ". Upstream"
# so trailing punctuation in the message doesn't pollute the URL.
_UNFETCHABLE_URL_RE = re.compile(
    r"Error while downloading (https?://[^\s]+?)\. Upstream",
)

# Cap for the strip-and-retry loop — sized to cover the realistic
# worst case of a long WhatsApp/Signal turn that included several
# images whose signed URLs all aged past expiry, while still
# bounding the total round-trip latency (~3-5s × N).  Beyond this
# something is structurally wrong (model loop bug, all URLs
# revoked, etc.) and we'd rather surface the original error than
# pile up retries.
_MAX_STRIP_RETRIES: int = 5


# Transient upstream-failure retries: ChatGPT's edge sometimes
# returns 502/503/504 for "upstream connect error or
# disconnect/reset before headers" — short-lived backend blips
# rather than anything wrong with our request.  Retry a small
# number of times with exponential backoff before surfacing the
# error to the agent.  Bounded so a sustained outage still
# fails fast instead of holding the user's turn open for minutes.
_TRANSIENT_RETRY_STATUS = {502, 503, 504}
_MAX_TRANSIENT_RETRIES: int = 3
_TRANSIENT_RETRY_BASE_DELAY_S: float = 0.5  # 0.5, 1.0, 2.0 backoff


def _extract_unfetchable_url(error_body: str) -> str | None:
    """Pull the failing image URL out of ChatGPT's 400 body, or
    return ``None`` when the error isn't a download failure (auth,
    rate limit, etc.).
    """
    if not error_body or "Error while downloading" not in error_body:
        return None
    m = _UNFETCHABLE_URL_RE.search(error_body)
    return m.group(1) if m else None


# Detect signed URLs from our media server that are already past
# their ``exp=`` timestamp at request time.  Cheaper than letting
# ChatGPT round-trip and 400 — saves ~3s per stale URL.  Match
# anchors on ``exp=<digits>&sig=<hex>`` because that combination
# is unique to our HMAC signing scheme; we don't want to scrub
# unrelated 3rd-party URLs that happen to carry an ``exp`` query
# param.
_SIGNED_URL_RE = re.compile(
    r"https?://[^\s\"'<>]+?\?[^\s\"'<>]*?exp=(\d+)[^\s\"'<>]*?&sig=[0-9a-f]+",
)


def _prune_expired_signed_urls(
    body: dict,
    now_ts: int | None = None,
) -> int:
    """In-place strip every ``input_image`` whose URL carries an
    expired ``exp=`` query param, replacing it with a text
    placeholder.  Returns the number of items pruned (zero on a
    clean body).  Called once before the first POST so we don't
    pay ChatGPT's ~3 s round-trip for URLs we already know are
    dead.
    """
    import time

    cutoff = now_ts if now_ts is not None else int(time.time())
    pruned = 0
    placeholder = (
        "[image previously sent here is no longer fetchable; "
        "ask the user to re-send if needed]"
    )
    for entry in body.get("input") or []:
        content = entry.get("content")
        if not isinstance(content, list):
            continue
        new_content: list[dict] = []
        for c in content:
            if (
                isinstance(c, dict)
                and c.get("type") == "input_image"
                and isinstance(c.get("image_url"), str)
            ):
                m = _SIGNED_URL_RE.fullmatch(c["image_url"])
                if m and int(m.group(1)) < cutoff:
                    new_content.append(
                        {
                            "type": "input_text",
                            "text": placeholder,
                        },
                    )
                    pruned += 1
                    continue
            new_content.append(c)
        entry["content"] = new_content
    return pruned


def _strip_unfetchable_image_from_body(
    body: dict,
    bad_url: str,
) -> bool:
    """Mutate ``body`` in place to remove every ``input_image`` item
    whose URL matches ``bad_url``.  Replaces each removed image with
    a brief text placeholder so the agent (and the model) still see
    that something used to be there — losing the block silently
    confuses the model when the conversation references "the image
    you just sent".  Returns ``True`` if at least one item was
    stripped, ``False`` otherwise.
    """
    if not bad_url:
        return False
    stripped = False
    placeholder = (
        "[image previously sent here is no longer fetchable; "
        "ask the user to re-send if needed]"
    )
    for entry in body.get("input") or []:
        content = entry.get("content")
        if not isinstance(content, list):
            continue
        new_content: list[dict] = []
        for c in content:
            if (
                isinstance(c, dict)
                and c.get("type") == "input_image"
                and c.get("image_url") == bad_url
            ):
                new_content.append(
                    {"type": "input_text", "text": placeholder},
                )
                stripped = True
            else:
                new_content.append(c)
        entry["content"] = new_content
    return stripped


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
            # Proactive sweep: drop already-expired signed URLs
            # before the first POST so we don't burn a round-trip
            # discovering they're dead.  The reactive strip-and-
            # retry below still covers URLs that pass the cutoff
            # but fail at fetch time (revoked keys, tunnel down,
            # etc.) so this is purely a latency win, not a
            # correctness fix.
            pruned = _prune_expired_signed_urls(responses_body)
            if pruned:
                logger.info(
                    "Codex: pruned %d expired signed URL(s) from "
                    "request before POST",
                    pruned,
                )
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
            # Retry up to ``_MAX_STRIP_RETRIES`` times when ChatGPT
            # bounces because it couldn't download an image URL —
            # one stripped URL per attempt covers turns with
            # several stale signed URLs (multi-image messages,
            # long-history replays).  Strip the offending block,
            # swap in a text placeholder, retry — preserves the
            # rest of the turn.  +1 for the first (un-retried) try.
            attempts_left = _MAX_STRIP_RETRIES + 1
            transient_attempts_left = _MAX_TRANSIENT_RETRIES
            transient_attempt_idx = 0
            while True:
                attempts_left -= 1
                async with httpx.AsyncClient(
                    timeout=httpx.Timeout(600, connect=30),
                ) as client:
                    async with client.stream(
                        "POST",
                        upstream_url,
                        json=responses_body,
                        headers=headers,
                    ) as upstream:
                        try:
                            _raise_for_upstream_status(upstream)
                        except httpx.HTTPStatusError as e:
                            err_body = (await upstream.aread()).decode(
                                "utf-8",
                                errors="replace",
                            )
                            # Transient 5xx — ChatGPT edge had a
                            # short-lived problem reaching its
                            # backend.  Back off and retry without
                            # mutating the request.
                            if (
                                upstream.status_code
                                in _TRANSIENT_RETRY_STATUS
                                and transient_attempts_left > 0
                            ):
                                transient_attempts_left -= 1
                                delay = (
                                    _TRANSIENT_RETRY_BASE_DELAY_S
                                    * (2 ** transient_attempt_idx)
                                )
                                transient_attempt_idx += 1
                                logger.warning(
                                    "Codex upstream HTTP %d (%s) — "
                                    "transient, sleeping %.1fs and retrying "
                                    "(%d left)",
                                    upstream.status_code,
                                    err_body[:120],
                                    delay,
                                    transient_attempts_left,
                                )
                                import asyncio as _asyncio
                                await _asyncio.sleep(delay)
                                state = StreamState(
                                    model=responses_body["model"],
                                )
                                continue
                            bad_url = _extract_unfetchable_url(err_body)
                            if (
                                attempts_left > 0
                                and bad_url
                                and _strip_unfetchable_image_from_body(
                                    responses_body,
                                    bad_url,
                                )
                            ):
                                logger.warning(
                                    "Codex 400 on image URL %s — "
                                    "stripped from request and retrying",
                                    bad_url,
                                )
                                # Reset translator state for the retry so the
                                # second response replaces (not appends to)
                                # the first.
                                state = StreamState(
                                    model=responses_body["model"],
                                )
                                continue
                            raise httpx.HTTPStatusError(
                                f"{e}: {err_body[:500]}",
                                request=e.request,
                                response=e.response,
                            ) from e
                        chat_body = await collect_as_chat_completion(
                            upstream,
                            state,
                        )
                        break
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
        # Allow up to ``_MAX_STRIP_RETRIES`` strip-and-retry attempts
        # on first open when ChatGPT 400s with "Error while
        # downloading <url>".  A turn can carry several stale URLs
        # (multi-image messages, replays of long conversation
        # history), so a single retry isn't enough; the cap guards
        # against retry storms when something else is wrong.
        self._retry_attempts_left: int = _MAX_STRIP_RETRIES
        # Independent budget for transient 5xx retries — kept
        # separate from the strip-and-retry quota because they
        # cover different failure modes (backend blip vs. dead
        # URL) and one shouldn't deplete the other.
        self._transient_attempts_left: int = _MAX_TRANSIENT_RETRIES
        self._transient_attempt_idx: int = 0

    def __aiter__(self) -> "_CodexOAuthAsyncStream":
        return self

    async def _open_stream(self) -> None:
        """Open (or re-open) the upstream HTTP stream into
        ``self._upstream`` / ``self._stream_ctx``.  Pulled out so
        the strip-and-retry path can call it again after mutating
        the body in place.
        """
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

    async def __anext__(self) -> ChatCompletionChunk:
        if self._iter is None:
            # Lazy-open the HTTP stream on first __anext__.
            await self._open_stream()
            assert self._upstream is not None
            while self._upstream.status_code != 200:
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

                # Transient 5xx — ChatGPT edge had a short-lived
                # problem reaching its backend (e.g. "upstream
                # connect error or disconnect/reset before
                # headers").  Back off and re-open the stream
                # without mutating the request.
                if (
                    status in _TRANSIENT_RETRY_STATUS
                    and self._transient_attempts_left > 0
                ):
                    self._transient_attempts_left -= 1
                    delay = (
                        _TRANSIENT_RETRY_BASE_DELAY_S
                        * (2 ** self._transient_attempt_idx)
                    )
                    self._transient_attempt_idx += 1
                    logger.warning(
                        "Codex stream HTTP %d (%s) — transient, "
                        "sleeping %.1fs and retrying (%d left)",
                        status,
                        err_body[:120],
                        delay,
                        self._transient_attempts_left,
                    )
                    await asyncio.sleep(delay)
                    await self._open_stream()
                    assert self._upstream is not None
                    continue

                # Strip-and-retry path: ChatGPT bounced because it
                # couldn't download a URL we sent (most often a
                # signed-URL that expired since the conversation
                # history first got captured).  Replace the broken
                # block with a placeholder so the agent still sees
                # something used to be there, and try once more.
                bad_url = _extract_unfetchable_url(err_body)
                if (
                    self._retry_attempts_left > 0
                    and bad_url
                    and _strip_unfetchable_image_from_body(
                        self._upstream_body,
                        bad_url,
                    )
                ):
                    self._retry_attempts_left -= 1
                    logger.warning(
                        "Codex 400 on image URL %s — stripped from "
                        "request and retrying stream",
                        bad_url,
                    )
                    await self._open_stream()
                    assert self._upstream is not None
                    continue

                raise httpx.HTTPStatusError(
                    f"Codex upstream HTTP {status}: {err_body[:500]}",
                    request=request,
                    response=response,
                )
            self._iter = translate_responses_events_to_chat_chunks(
                self._upstream,
                self._state,
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
