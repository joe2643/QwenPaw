# -*- coding: utf-8 -*-
"""Tests for ``RefusalFallbackChatModel`` and its factory gating.

When a Mythos-class Claude model (claude-fable-*) ends a response with
``stop_reason="refusal"`` and no content, the wrapper re-issues the
same call once on a fallback model (default ``claude-opus-4-8``).  If
the fallback also refuses or can't be built, the original exception
propagates so the agent's refusal notice still fires.

The "completed by fallback" notice is EMBEDDED as a leading text block
of the fallback response on agent-facing calls (those passing
``tools``) — a separate trailing notice message gets dropped or
replaces the real reply under the channels' preamble-buffer logic.
Tool-less internal calls (title generation, listen decisions) and
structured-output calls must never be prefixed: their output is parsed
by code.
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any, AsyncGenerator

import pytest
from agentscope.model import ChatModelBase

import qwenpaw.agents.model_factory as mf
from qwenpaw.exceptions import ModelRefusalException
from qwenpaw.providers.refusal_fallback_model import RefusalFallbackChatModel

NOTICE = "ℹ️ fable refused — completed by opus\n"


def _chunk(*blocks: dict) -> Any:
    return SimpleNamespace(content=list(blocks))


def _text(t: str) -> dict:
    return {"type": "text", "text": t}


class _FakeModel(ChatModelBase):
    """Scriptable fake: yields *chunks* then optionally raises."""

    def __init__(
        self,
        name: str = "claude-fable-5",
        chunks: list[Any] | None = None,
        raise_refusal: bool = False,
        raise_exc: Exception | None = None,
        streaming: bool = True,
    ) -> None:
        super().__init__(model_name=name, stream=streaming)
        self._chunks = chunks or []
        self._raise_refusal = raise_refusal
        self._raise_exc = raise_exc
        self._streaming = streaming
        self.calls = 0

    async def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self.calls += 1
        if not self._streaming:
            if self._raise_refusal:
                raise ModelRefusalException(self.model_name)
            if self._raise_exc:
                raise self._raise_exc
            return self._chunks[-1] if self._chunks else _chunk(_text("ok"))
        return self._gen()

    async def _gen(self) -> AsyncGenerator[Any, None]:
        for c in self._chunks:
            yield c
        if self._raise_refusal:
            raise ModelRefusalException(self.model_name, response_id="msg_r")
        if self._raise_exc:
            raise self._raise_exc


def _wrap(
    primary: _FakeModel,
    fallback: _FakeModel | None,
    factory_exc: Exception | None = None,
    notice: str | None = NOTICE,
) -> tuple[RefusalFallbackChatModel, list[int]]:
    factory_calls: list[int] = []

    def _factory() -> ChatModelBase:
        factory_calls.append(1)
        if factory_exc is not None:
            raise factory_exc
        return fallback

    return (
        RefusalFallbackChatModel(
            primary,
            _factory,
            "claude-opus-4-8",
            notice_text=notice,
        ),
        factory_calls,
    )


async def _drain(result: Any) -> list[Any]:
    if isinstance(result, AsyncGenerator):
        return [c async for c in result]
    return [result]


def _texts(chunk: Any) -> list[str]:
    return [
        b.get("text")
        for b in getattr(chunk, "content", [])
        if isinstance(b, dict) and b.get("type") == "text"
    ]


async def test_clean_stream_passes_through_and_skips_fallback() -> None:
    primary = _FakeModel(chunks=[_chunk(_text("a")), _chunk(_text("ab"))])
    fallback = _FakeModel(name="claude-opus-4-8", chunks=[_chunk(_text("fb"))])
    model, factory_calls = _wrap(primary, fallback)

    out = await _drain(await model("msg", tools=[{"name": "t"}]))

    assert [_texts(c) for c in out] == [["a"], ["ab"]]  # untouched
    assert factory_calls == []  # fallback never built
    assert fallback.calls == 0


async def test_refusal_switches_to_fallback_stream() -> None:
    primary = _FakeModel(raise_refusal=True)
    fallback = _FakeModel(
        name="claude-opus-4-8",
        chunks=[_chunk(_text("fb1")), _chunk(_text("fb1fb2"))],
    )
    model, _ = _wrap(primary, fallback)

    out = await _drain(await model("msg"))

    assert len(out) == 2
    assert primary.calls == 1
    assert fallback.calls == 1


async def test_fallback_refusal_propagates() -> None:
    primary = _FakeModel(raise_refusal=True)
    fallback = _FakeModel(name="claude-opus-4-8", raise_refusal=True)
    model, _ = _wrap(primary, fallback)

    with pytest.raises(ModelRefusalException) as exc_info:
        await _drain(await model("msg"))
    # The fallback's refusal (not the primary's) is what surfaces.
    assert exc_info.value.details["model_name"] == "claude-opus-4-8"


async def test_factory_failure_surfaces_original_refusal() -> None:
    primary = _FakeModel(raise_refusal=True)
    model, _ = _wrap(primary, None, factory_exc=RuntimeError("no provider"))

    with pytest.raises(ModelRefusalException) as exc_info:
        await _drain(await model("msg"))
    assert exc_info.value.details["model_name"] == "claude-fable-5"


async def test_fallback_api_error_surfaces_original_refusal() -> None:
    primary = _FakeModel(raise_refusal=True)
    fallback = _FakeModel(
        name="claude-opus-4-8",
        streaming=False,
        raise_exc=RuntimeError("500"),
    )
    model, _ = _wrap(primary, fallback)

    with pytest.raises(ModelRefusalException) as exc_info:
        await _drain(await model("msg"))
    assert exc_info.value.details["model_name"] == "claude-fable-5"


async def test_fallback_is_built_once_and_cached() -> None:
    primary = _FakeModel(raise_refusal=True)
    fallback = _FakeModel(name="claude-opus-4-8", chunks=[_chunk(_text("fb"))])
    model, factory_calls = _wrap(primary, fallback)

    await _drain(await model("msg"))
    await _drain(await model("msg"))

    assert factory_calls == [1]
    assert fallback.calls == 2


async def test_non_streaming_refusal_uses_fallback() -> None:
    primary = _FakeModel(streaming=False, raise_refusal=True)
    fallback = _FakeModel(
        name="claude-opus-4-8",
        streaming=False,
        chunks=[_chunk(_text("fb-answer"))],
    )
    model, _ = _wrap(primary, fallback)

    out = await _drain(await model("msg"))

    assert _texts(out[0])[-1] == "fb-answer"


def test_model_key_delegates_to_primary() -> None:
    primary = _FakeModel()
    primary.model_key = "claude-oauth:claude-fable-5"  # type: ignore[attr-defined]
    model, _ = _wrap(primary, None)
    assert model.model_key == "claude-oauth:claude-fable-5"


# ---------------------------------------------------------------- #
# Notice embedding                                                  #
# ---------------------------------------------------------------- #


async def test_notice_embedded_on_agent_calls_with_tools() -> None:
    primary = _FakeModel(raise_refusal=True)
    fallback = _FakeModel(
        name="claude-opus-4-8",
        chunks=[_chunk(_text("Hel")), _chunk(_text("Hello"))],
    )
    model, _ = _wrap(primary, fallback)

    out = await _drain(await model("msg", tools=[{"name": "t"}]))

    # Every cumulative chunk leads with the constant notice block.
    assert _texts(out[0]) == [NOTICE, "Hel"]
    assert _texts(out[1]) == [NOTICE, "Hello"]


async def test_notice_not_embedded_without_tools() -> None:
    """Tool-less calls (title gen, listen CHIME/PASS) are parsed by
    code — prefixed text would corrupt them."""
    primary = _FakeModel(raise_refusal=True)
    fallback = _FakeModel(name="claude-opus-4-8", chunks=[_chunk(_text("PASS"))])
    model, _ = _wrap(primary, fallback)

    out = await _drain(await model("msg"))

    assert _texts(out[0]) == ["PASS"]


async def test_notice_not_embedded_for_structured_output() -> None:
    primary = _FakeModel(raise_refusal=True)
    fallback = _FakeModel(name="claude-opus-4-8", chunks=[_chunk(_text("{}"))])
    model, _ = _wrap(primary, fallback)

    out = await _drain(
        await model("msg", tools=[{"name": "t"}], structured_model=object()),
    )

    assert _texts(out[0]) == ["{}"]


async def test_notice_not_embedded_when_disabled() -> None:
    primary = _FakeModel(raise_refusal=True)
    fallback = _FakeModel(name="claude-opus-4-8", chunks=[_chunk(_text("hi"))])
    model, _ = _wrap(primary, fallback, notice=None)

    out = await _drain(await model("msg", tools=[{"name": "t"}]))

    assert _texts(out[0]) == ["hi"]


async def test_notice_embedded_with_tool_use_only_chunks() -> None:
    """A fallback response that's pure tool_use still gets the leading
    text block — the notice shows up when the channel renders it."""
    tool_use = {"type": "tool_use", "id": "tu1", "name": "t", "input": {}}
    primary = _FakeModel(raise_refusal=True)
    fallback = _FakeModel(name="claude-opus-4-8", chunks=[_chunk(tool_use)])
    model, _ = _wrap(primary, fallback)

    out = await _drain(await model("msg", tools=[{"name": "t"}]))

    assert out[0].content[0] == {"type": "text", "text": NOTICE}
    assert out[0].content[1] == tool_use


async def test_notice_embedded_non_streaming() -> None:
    primary = _FakeModel(streaming=False, raise_refusal=True)
    fallback = _FakeModel(
        name="claude-opus-4-8",
        streaming=False,
        chunks=[_chunk(_text("answer"))],
    )
    model, _ = _wrap(primary, fallback)

    out = await _drain(await model("msg", tools=[{"name": "t"}]))

    assert _texts(out[0]) == [NOTICE, "answer"]


async def test_clean_primary_reply_never_gets_notice() -> None:
    primary = _FakeModel(chunks=[_chunk(_text("normal"))])
    model, _ = _wrap(primary, None)

    out = await _drain(await model("msg", tools=[{"name": "t"}]))

    assert _texts(out[0]) == ["normal"]


# ---------------------------------------------------------------- #
# Outbound notice stripping (anti-mimic)                            #
# ---------------------------------------------------------------- #


class _CapturingModel(ChatModelBase):
    """Records the messages payload it was called with."""

    def __init__(self) -> None:
        super().__init__(model_name="claude-fable-5", stream=False)
        self.seen: list[Any] = []

    async def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self.seen.append(args[0] if args else kwargs.get("messages"))
        return _chunk(_text("ok"))


def _notice_block() -> dict:
    return {"type": "text", "text": NOTICE}


async def test_outbound_history_drops_injected_notice_blocks() -> None:
    primary = _CapturingModel()
    model, _ = _wrap(primary, None)
    msgs = [
        {"role": "user", "content": "hi"},
        {
            "role": "assistant",
            "content": [_notice_block(), _text("real answer")],
        },
    ]

    await model(msgs, tools=[{"name": "t"}])

    sent = primary.seen[0]
    assert sent[1]["content"] == [_text("real answer")]


async def test_outbound_history_strips_mimicked_notice_text() -> None:
    """Model-generated copies inside a text block are removed too —
    observed live: a reply opening with the notice repeated 3 times."""
    primary = _CapturingModel()
    model, _ = _wrap(primary, None)
    mimic = f"{NOTICE.strip()} {NOTICE.strip()} actual content"
    msgs = [{"role": "assistant", "content": [_text(mimic)]}]

    await model(msgs, tools=[{"name": "t"}])

    sent = primary.seen[0]
    assert sent[0]["content"][0]["text"] == "actual content"


async def test_outbound_pure_notice_message_gets_placeholder() -> None:
    primary = _CapturingModel()
    model, _ = _wrap(primary, None)
    msgs = [{"role": "assistant", "content": [_notice_block()]}]

    await model(msgs, tools=[{"name": "t"}])

    sent = primary.seen[0]
    # Anthropic rejects empty assistant content — placeholder kept.
    assert sent[0]["content"] == [{"type": "text", "text": "…"}]


async def test_outbound_string_content_stripped() -> None:
    primary = _CapturingModel()
    model, _ = _wrap(primary, None)
    msgs = [{"role": "user", "content": f"context: {NOTICE.strip()} tail"}]

    await model(msgs, tools=[{"name": "t"}])

    assert primary.seen[0][0]["content"] == "context:  tail"


async def test_outbound_untouched_without_notice_config() -> None:
    primary = _CapturingModel()
    model, _ = _wrap(primary, None, notice=None)
    blocks = [_notice_block(), _text("x")]
    msgs = [{"role": "assistant", "content": blocks}]

    await model(msgs, tools=[{"name": "t"}])

    assert primary.seen[0][0]["content"] is blocks  # no-op


# ---------------------------------------------------------------- #
# Factory gating (_wrap_refusal_fallback)                           #
# ---------------------------------------------------------------- #


def test_factory_wraps_fable_models() -> None:
    primary = _FakeModel(name="claude-fable-5")
    wrapped = mf._wrap_refusal_fallback(primary, "claude-oauth", None, None)
    assert isinstance(wrapped, RefusalFallbackChatModel)


def test_factory_builds_localized_notice() -> None:
    primary = _FakeModel(name="claude-fable-5")
    wrapped = mf._wrap_refusal_fallback(
        primary,
        "claude-oauth",
        None,
        None,
        language="zh-CN",
    )
    assert "claude-fable-5" in wrapped._notice_text
    assert "claude-opus-4-8" in wrapped._notice_text
    assert "安全分類器" in wrapped._notice_text


def test_factory_defaults_to_english_notice() -> None:
    primary = _FakeModel(name="claude-fable-5")
    wrapped = mf._wrap_refusal_fallback(primary, "claude-oauth", None, None)
    assert "safety classifier" in wrapped._notice_text


def test_factory_skips_non_fable_models() -> None:
    primary = _FakeModel(name="claude-opus-4-8")
    wrapped = mf._wrap_refusal_fallback(primary, "claude-oauth", None, None)
    assert wrapped is primary


def test_factory_skips_openai_models() -> None:
    primary = _FakeModel(name="gpt-5.2")
    wrapped = mf._wrap_refusal_fallback(primary, "codex-oauth", None, None)
    assert wrapped is primary


def test_factory_disabled_by_empty_fallback_id(monkeypatch) -> None:
    monkeypatch.setattr(mf, "LLM_REFUSAL_FALLBACK_MODEL", "")
    primary = _FakeModel(name="claude-fable-5")
    wrapped = mf._wrap_refusal_fallback(primary, "claude-oauth", None, None)
    assert wrapped is primary


def test_factory_skips_when_fallback_equals_primary(monkeypatch) -> None:
    monkeypatch.setattr(mf, "LLM_REFUSAL_FALLBACK_MODEL", "claude-fable-5")
    primary = _FakeModel(name="claude-fable-5")
    wrapped = mf._wrap_refusal_fallback(primary, "claude-oauth", None, None)
    assert wrapped is primary
