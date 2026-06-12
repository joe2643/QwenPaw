# -*- coding: utf-8 -*-
"""One-shot fallback-model retry for safety-classifier refusals.

Anthropic's Mythos-class models (claude-fable-*) run a streaming safety
classifier that can end a response with ``stop_reason="refusal"`` and
zero content even on innocuous prompts — the trigger is usually
something persistent in the conversation context, so retrying the same
model is pointless.  The provider wrapper surfaces this as
:class:`~qwenpaw.exceptions.ModelRefusalException` (see
``anthropic_provider``).

:class:`RefusalFallbackChatModel` catches that exception and re-issues
the *exact same call* once on a lazily built sibling model (same
provider, e.g. ``claude-opus-4-8``, which has no Mythos-class streaming
classifier).  If the fallback also refuses — or can't be built — the
original exception propagates so the agent's notice path
(``_build_refusal_reply``) still fires.

When ``notice_text`` is configured, agent-facing calls (those carrying
``tools``) get the notice EMBEDDED into the fallback response as a
leading text block, so the user sees "this reply came from the fallback
model" inside the reply itself.  A separate trailing notice message was
tried first and failed in two ways: channels running the codex-oauth
preamble buffer treat the held reply as preamble and DROP it when the
extra notice message arrives behind it (the notice then *replaces* the
answer), and notices emitted mid-turn get dropped as preamble when a
tool call follows.  Embedding rides inside the same message as the
answer, so neither failure mode exists.  Tool-less internal calls
(title generation, listen decisions, summarize) never get the notice —
their outputs are parsed, not read by users.

Wrapping order (outermost first)::

    RefusalFallbackChatModel
      └─ RetryChatModel              (transient-error retry)
           └─ TokenRecordingModelWrapper
                └─ real ChatModelBase

``ModelRefusalException`` is deliberately *not* retryable in
``RetryChatModel`` (it's a deterministic classifier verdict, not a
transient fault), so it passes straight through to this wrapper.  The
fallback model gets its own TokenRecording/Retry stack via the factory
callable so its usage is attributed to the fallback model's name.
"""

from __future__ import annotations

import logging
from typing import Any, AsyncGenerator, Callable

from agentscope.model import ChatModelBase

from ..exceptions import ModelRefusalException

logger = logging.getLogger(__name__)


class RefusalFallbackChatModel(ChatModelBase):
    """Retry a refused call once on a fallback model.

    Args:
        primary: The fully wrapped primary model (RetryChatModel stack).
        fallback_factory: Zero-arg callable building the fully wrapped
            fallback model.  Called lazily on the first refusal and
            cached — most sessions never pay the construction cost.
        fallback_model_name: Display name of the fallback model, used
            for logging only.
        notice_text: Pre-localized one-liner embedded as the leading
            text block of fallback responses on agent-facing calls
            (see module docstring).  ``None`` disables embedding.
    """

    def __init__(
        self,
        primary: ChatModelBase,
        fallback_factory: Callable[[], ChatModelBase],
        fallback_model_name: str,
        notice_text: str | None = None,
    ) -> None:
        super().__init__(
            model_name=primary.model_name,
            stream=primary.stream,
        )
        # Attribute is named ``_inner`` so wrapper drill-throughs that
        # walk ``_inner``/``_model`` chains (e.g. listen_responder)
        # keep reaching the real model.
        self._inner = primary
        self._fallback_factory = fallback_factory
        self._fallback_model_name = fallback_model_name
        self._notice_text = notice_text
        self._fallback: ChatModelBase | None = None

    @property
    def model_key(self) -> str | None:
        """Capability-cache key — delegate to the primary stack."""
        return getattr(self._inner, "model_key", None)

    @property
    def inner_class(self) -> type:
        """Real model class — delegate for formatter mapping."""
        return getattr(self._inner, "inner_class", self._inner.__class__)

    def _strip_notice_from_history(self, args: tuple, kwargs: dict) -> None:
        """Remove embedded notices from the outbound message payload.

        The notice is for HUMANS: once an embedded notice lands in the
        session history, the next call's context contains assistant
        turns that open with it — and the model starts mimicking it in
        fresh replies (observed live: a reply opening with the notice
        repeated three times).  Stripping it from every outbound text
        block keeps it user-visible in the channel/history while the
        model never sees it.  Mutates the formatted payload in place —
        the formatter builds a fresh payload per call.
        """
        needle = (self._notice_text or "").strip()
        if not needle:
            return
        msgs = kwargs.get("messages")
        if msgs is None and args:
            msgs = args[0]
        if not isinstance(msgs, list):
            return
        for msg in msgs:
            if not isinstance(msg, dict):
                continue
            content = msg.get("content")
            if isinstance(content, str):
                if needle in content:
                    msg["content"] = content.replace(needle, "").strip() or "…"
            elif isinstance(content, list):
                kept = []
                for block in content:
                    if (
                        isinstance(block, dict)
                        and block.get("type") == "text"
                        and needle in (block.get("text") or "")
                    ):
                        text = block["text"].replace(needle, "").strip()
                        if not text:
                            continue  # pure-notice block — drop it
                        block = {**block, "text": text}
                    kept.append(block)
                if kept or content:
                    msg["content"] = kept or [{"type": "text", "text": "…"}]

    def _embed_notice_for(self, args: tuple, kwargs: dict) -> bool:
        """Embed only on agent-facing calls: tools present, free-form
        output.  Structured-output and tool-less internal calls (title
        generation, listen CHIME/PASS decisions) are parsed by code —
        prefixed text would corrupt them.
        """
        if not self._notice_text:
            return False
        if kwargs.get("structured_model") is not None:
            return False
        tools = kwargs.get("tools")
        if tools is None and len(args) >= 2:
            tools = args[1]
        return bool(tools)

    def _prefix_chunk(self, chunk: Any) -> Any:
        """Insert the notice as the leading text block of *chunk*.

        Chunks are cumulative snapshots; inserting a constant block at
        index 0 of every snapshot keeps the sequence consistent for
        downstream delta computation.
        """
        try:
            content = getattr(chunk, "content", None)
            if isinstance(content, list):
                first = content[0] if content else None
                already = (
                    isinstance(first, dict)
                    and first.get("type") == "text"
                    and first.get("text") == self._notice_text
                )
                if not already:
                    chunk.content = [
                        {"type": "text", "text": self._notice_text},
                        *content,
                    ]
        except Exception:  # pragma: no cover — never break the reply path
            logger.debug("refusal-fallback notice embed failed", exc_info=True)
        return chunk

    async def _call_fallback(
        self,
        refusal: ModelRefusalException,
        args: tuple,
        kwargs: dict,
    ) -> Any:
        """Re-issue the call on the fallback model.

        Build failures and fallback API errors re-raise the *original*
        refusal so the agent's notice path stays intact; a refusal from
        the fallback itself propagates as-is (same notice path, fresher
        response_id for the diagnostic log).
        """
        try:
            if self._fallback is None:
                self._fallback = self._fallback_factory()
            fallback = self._fallback
        except Exception:
            logger.warning(
                "Refusal fallback model '%s' could not be built; "
                "surfacing the original refusal",
                self._fallback_model_name,
                exc_info=True,
            )
            raise refusal  # pylint: disable=raise-missing-from
        logger.warning(
            "Model '%s' refused with no content (response_id=%s) — "
            "retrying once on fallback model '%s'",
            self.model_name,
            (getattr(refusal, "details", None) or {}).get("response_id"),
            self._fallback_model_name,
        )
        try:
            return await fallback(*args, **kwargs)
        except ModelRefusalException:
            raise
        except Exception:
            logger.warning(
                "Refusal fallback model '%s' failed; surfacing the "
                "original refusal",
                self._fallback_model_name,
                exc_info=True,
            )
            raise refusal  # pylint: disable=raise-missing-from

    async def __call__(self, *args: Any, **kwargs: Any) -> Any:
        # Strip prior notices from the outbound payload BEFORE the
        # primary call — the fallback call reuses the same (already
        # cleaned) args, so both models get a notice-free context.
        self._strip_notice_from_history(args, kwargs)
        try:
            result = await self._inner(*args, **kwargs)
        except ModelRefusalException as refusal:
            # Non-streaming primary call refused at request time.
            result = await self._call_fallback(refusal, args, kwargs)
            if isinstance(result, AsyncGenerator):
                return self._wrap_fallback_stream(result, args, kwargs)
            if self._embed_notice_for(args, kwargs):
                result = self._prefix_chunk(result)
            return result
        if isinstance(result, AsyncGenerator):
            return self._wrap_stream(result, args, kwargs)
        return result

    async def _wrap_stream(
        self,
        stream: AsyncGenerator[Any, None],
        args: tuple,
        kwargs: dict,
    ) -> AsyncGenerator[Any, None]:
        """Yield primary chunks; on a refusal, switch to the fallback.

        The provider only raises ``ModelRefusalException`` when the
        stream produced *no* visible content, so at the moment of the
        switch the consumer has at most seen empty/thinking chunks —
        each streamed chunk is a cumulative snapshot, so the fallback's
        chunks cleanly replace them.
        """
        try:
            async for chunk in stream:
                yield chunk
            return
        except ModelRefusalException as refusal:
            fb_result = await self._call_fallback(refusal, args, kwargs)
        if isinstance(fb_result, AsyncGenerator):
            async for chunk in self._wrap_fallback_stream(
                fb_result,
                args,
                kwargs,
            ):
                yield chunk
        else:
            if self._embed_notice_for(args, kwargs):
                fb_result = self._prefix_chunk(fb_result)
            yield fb_result

    async def _wrap_fallback_stream(
        self,
        stream: AsyncGenerator[Any, None],
        args: tuple,
        kwargs: dict,
    ) -> AsyncGenerator[Any, None]:
        """Yield fallback chunks, embedding the notice when eligible."""
        embed = self._embed_notice_for(args, kwargs)
        async for chunk in stream:
            yield self._prefix_chunk(chunk) if embed else chunk
