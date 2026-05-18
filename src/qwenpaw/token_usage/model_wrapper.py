# -*- coding: utf-8 -*-
"""Model wrapper that records token usage from LLM responses."""

import logging
import os
from datetime import date, datetime, timezone
from typing import Any, AsyncGenerator, Literal, Type

from agentscope.model import ChatModelBase
from agentscope.model._model_response import ChatResponse
from agentscope.model._model_usage import ChatUsage
from pydantic import BaseModel

from .buffer import _UsageEvent
from .manager import get_token_usage_manager

logger = logging.getLogger(__name__)


def _dump_wire_shape(
    provider_id: str,
    model_name: str,
    messages: list[dict],
    tools: list[dict] | None,
) -> None:
    """Log outgoing wire structure when ``COPAW_WIRE_DUMP`` is set.

    Logs role + block-type-only summary per message, plus tool count.
    Content text is omitted (could leak PII).  Designed to answer the
    question "did video_url survive into the actual API request?" — if
    a wire-dump line for the latest msg shows a ``video_url`` block,
    the formatter pipeline did its job; if not, the block was dropped
    somewhere upstream.
    """
    if not os.environ.get("COPAW_WIRE_DUMP"):
        return
    try:
        for i, m in enumerate(messages or []):
            role = m.get("role")
            content = m.get("content")
            if isinstance(content, list):
                parts = []
                for b in content:
                    if not isinstance(b, dict):
                        parts.append(type(b).__name__)
                        continue
                    btype = b.get("type")
                    if btype == "text":
                        # First 60 chars only — surface text markers like
                        # ``__QWENPAW_VID_*__`` that signal substitute fired
                        # without replace.
                        txt = str(b.get("text") or "")[:60]
                        parts.append(f"text:{txt!r}")
                    elif btype in ("image_url", "video_url"):
                        url = (
                            (b.get(btype) or {}).get("url", "")
                            if isinstance(b.get(btype), dict)
                            else ""
                        )
                        parts.append(f"{btype}:{url[:80]}")
                    else:
                        parts.append(str(btype))
                summary = f"blocks=[{', '.join(parts)}]"
            elif content is None:
                summary = "content=None"
            else:
                summary = (
                    f"content_type={type(content).__name__} "
                    f"len={len(str(content))} preview={str(content)[:60]!r}"
                )
            tool_calls = m.get("tool_calls") or []
            logger.info(
                "wire-dump[%s/%s] msg[%d] role=%s %s tool_calls=%d",
                provider_id,
                model_name,
                i,
                role,
                summary,
                len(tool_calls),
            )
        if tools:
            logger.info(
                "wire-dump[%s/%s] tools=%d",
                provider_id,
                model_name,
                len(tools),
            )
    except Exception as e:  # pragma: no cover - debug-only path
        logger.debug("wire-dump failed: %s", e)

    # COPAW_WIRE_DUMP_FULL=1 also writes the entire messages + tools
    # payload to a timestamped file under /tmp/copaw-wire/ so we can
    # diff it byte-by-byte against a known-good curl request.  Heavy
    # — only enable when actively debugging.
    if os.environ.get("COPAW_WIRE_DUMP_FULL"):
        try:
            import json
            import time
            import uuid

            dump_dir = "/tmp/copaw-wire"
            os.makedirs(dump_dir, exist_ok=True)
            ts = time.strftime("%Y%m%d-%H%M%S")
            slug = uuid.uuid4().hex[:8]
            path = (
                f"{dump_dir}/{ts}-{provider_id.replace('/', '_')}-{slug}.json"
            )
            with open(path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "provider_id": provider_id,
                        "model": model_name,
                        "messages": messages,
                        "tools": tools,
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )
            logger.info(
                "wire-dump-full[%s/%s] → %s",
                provider_id,
                model_name,
                path,
            )
        except Exception as e:  # pragma: no cover
            logger.debug("wire-dump-full failed: %s", e)


def _extract_cache_tokens(metadata: Any) -> tuple[int, int]:
    """Pull (cache_creation, cache_read) token counts off a usage
    ``metadata`` payload that may follow either of two conventions:

    * **Anthropic dict** — written by
      :func:`qwenpaw.providers.anthropic_provider._inject_cache_metadata`
      with explicit ``cache_creation_input_tokens`` /
      ``cache_read_input_tokens`` keys.  Cache writes are billed
      separately and counted client-side.
    * **OpenAI CompletionUsage object** — set by agentscope's
      ``_openai_model`` parser to the raw upstream usage object, which
      exposes ``prompt_tokens_details.cached_tokens``.  OpenAI handles
      caching server-side, so there is no cache-write count to record.

    Returns ``(0, 0)`` for any other shape — silently no-ops on
    providers that have no caching telemetry, rather than crashing the
    recording hot path.
    """
    if metadata is None:
        return 0, 0
    if isinstance(metadata, dict):
        return (
            int(metadata.get("cache_creation_input_tokens", 0) or 0),
            int(metadata.get("cache_read_input_tokens", 0) or 0),
        )
    details = getattr(metadata, "prompt_tokens_details", None)
    if details is not None:
        cached = getattr(details, "cached_tokens", None)
        if cached:
            return 0, int(cached)
    return 0, 0


class TokenRecordingModelWrapper(ChatModelBase):
    """Wraps a ChatModelBase to record token usage on each call."""

    _usage_by_session: dict[str, dict[str, Any]] = {}

    def __init__(self, provider_id: str, model: ChatModelBase) -> None:
        super().__init__(
            model_name=getattr(model, "model_name", "unknown"),
            stream=getattr(model, "stream", True),
        )
        self._model = model
        self._provider_id = provider_id

    def _record_usage(self, usage: ChatUsage | None) -> None:
        """Enqueue a usage event synchronously — never blocks the caller."""
        if usage is None:
            return
        pt = getattr(usage, "input_tokens", 0) or 0
        ct = getattr(usage, "output_tokens", 0) or 0
        # Cache token counts arrive on ``usage.metadata`` in two shapes:
        # the Anthropic dict written by ``ClaudeOAuthChatModel``, or the
        # OpenAI ``CompletionUsage`` object that agentscope's OpenAI
        # parser stashes there for codex-oauth / openai-compat providers.
        cct, crt = _extract_cache_tokens(getattr(usage, "metadata", None))
        if pt <= 0 and ct <= 0 and cct <= 0 and crt <= 0:
            return

        event = _UsageEvent(
            provider_id=self._provider_id,
            model_name=self.model_name,
            prompt_tokens=pt,
            completion_tokens=ct,
            date_str=date.today().isoformat(),
            now_iso=datetime.now(tz=timezone.utc).isoformat(
                timespec="seconds",
            ),
            cache_creation_tokens=cct,
            cache_read_tokens=crt,
        )
        # Fire-and-forget: synchronous put_nowait, ~100 ns, no await needed.
        get_token_usage_manager().enqueue(event)

        usage_data = {
            "provider_id": self._provider_id,
            "model_name": self.model_name,
            "prompt_tokens": pt,
            "completion_tokens": ct,
            "cache_creation_tokens": cct,
            "cache_read_tokens": crt,
            "total_tokens": pt + ct,
        }
        self._store_usage(usage_data)

    @classmethod
    def pop_usage_for_session(cls, session_id: str) -> dict[str, Any] | None:
        return cls._usage_by_session.pop(session_id, None)

    def _store_usage(self, usage: dict[str, Any] | None) -> None:
        from ..app.agent_context import get_current_session_id

        session_id = get_current_session_id()
        if session_id and usage:
            TokenRecordingModelWrapper._usage_by_session[session_id] = usage

    async def __call__(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        tool_choice: Literal["auto", "none", "required"] | str | None = None,
        structured_model: Type[BaseModel] | None = None,
        **kwargs: Any,
    ) -> ChatResponse | AsyncGenerator[ChatResponse, None]:
        # Fix: Omit tool_choice="auto" for vLLM compatibility
        # vLLM without --enable-auto-tool-choice will reject requests when
        # tool_choice="auto" is present, even if tools are provided.
        # By omitting tool_choice when it's "auto", we bypass the check
        # while keeping tools available for correct tool calling behavior.
        if tool_choice == "auto":
            tool_choice = None

        _dump_wire_shape(
            self._provider_id,
            self.model_name,
            messages,
            tools,
        )

        result = await self._model(
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            structured_model=structured_model,
            **kwargs,
        )

        if isinstance(result, AsyncGenerator):
            return self._wrap_stream(result)
        self._record_usage(getattr(result, "usage", None))
        return result

    async def _wrap_stream(
        self,
        stream: AsyncGenerator[ChatResponse, None],
    ) -> AsyncGenerator[ChatResponse, None]:
        last_usage: ChatUsage | None = None
        async for chunk in stream:
            if getattr(chunk, "usage", None) is not None:
                last_usage = chunk.usage
            yield chunk
        self._record_usage(last_usage)
