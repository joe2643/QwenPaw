# -*- coding: utf-8 -*-
"""Unit tests for ``view_video``'s fallback-model delegation path.

Covers:
* agent config schema round-trip (``fallback_video_model`` field)
* delegation fires only when primary can't do video AND fallback is set
* fallback model receives a VideoBlock + the caller's prompt
* caller-omitted prompt falls back to the stock describe-everything prompt
* fallback model failure gracefully surfaces the generic placeholder hint
* no delegation when primary already supports video
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, AsyncIterator
from unittest.mock import patch

import pytest

from qwenpaw.agents.tools import view_media as vm
from qwenpaw.agents.tools.view_media import (
    _DEFAULT_VIDEO_FALLBACK_PROMPT,
    view_video,
)
from qwenpaw.config.config import AgentProfileConfig, ModelSlotConfig


# ---------------------------------------------------------------- #
# Schema                                                           #
# ---------------------------------------------------------------- #


class TestAgentProfileConfigFallbackField:
    def test_default_is_none(self):
        p = AgentProfileConfig(id="a", name="A")
        assert p.fallback_video_model is None

    def test_round_trip(self):
        src = {
            "id": "a",
            "name": "A",
            "fallback_video_model": {
                "provider_id": "gemini",
                "model": "gemini-2.5-pro",
            },
        }
        p = AgentProfileConfig.model_validate(src)
        assert isinstance(p.fallback_video_model, ModelSlotConfig)
        assert p.fallback_video_model.provider_id == "gemini"
        assert p.fallback_video_model.model == "gemini-2.5-pro"
        # Serialize back and check idempotency.
        dumped = p.model_dump()
        assert dumped["fallback_video_model"] == src["fallback_video_model"]


# ---------------------------------------------------------------- #
# view_video delegation path                                       #
# ---------------------------------------------------------------- #


class _FakeStreamResponse:
    """Mimics an agentscope streaming chat response — yields one
    chunk whose content has a text block.  Text is cumulative in
    agentscope's streaming contract; we emit the full string in
    a single chunk (equivalent to a complete stream)."""

    def __init__(self, text: str) -> None:
        self._text = text

    def __aiter__(self) -> "_FakeStreamResponse":
        return self

    async def __anext__(self) -> Any:
        if self._text is None:
            raise StopAsyncIteration
        payload = _FakeChunk([{"type": "text", "text": self._text}])
        self._text = None
        return payload


class _FakeChunk:
    def __init__(self, content: list[dict]) -> None:
        self.content = content


class _FakeChatModel:
    """Records the prompt it was called with and returns a canned
    description (or raises when ``fail=True``)."""

    def __init__(self, description: str = "A cat yawns.", fail: bool = False):
        self.description = description
        self.fail = fail
        self.last_messages: list[dict] | None = None

    async def __call__(self, messages: list[dict]) -> Any:
        self.last_messages = messages
        if self.fail:
            raise RuntimeError("simulated fallback API error")
        return _FakeStreamResponse(self.description)


@pytest.fixture
def tmp_video(tmp_path: Path) -> Path:
    """A dummy .mp4 file — ``_validate_media_path`` only checks
    extension + existence + size, so the contents don't matter."""
    p = tmp_path / "sample.mp4"
    p.write_bytes(b"\x00" * 1024)
    return p


@pytest.mark.asyncio
async def test_primary_supports_video_bypasses_fallback(
    tmp_video: Path,
) -> None:
    # When the primary model handles video natively, view_video
    # must return the VideoBlock untouched — no fallback call.
    fake = _FakeChatModel()
    with patch.object(vm, "_check_multimodal_support", return_value=True), \
         patch.object(
             vm, "_resolve_fallback_video_model",
             return_value=(fake, "gemini", "gemini-2.5-pro"),
         ):
        resp = await view_video(str(tmp_video))
    # VideoBlock present; no fallback header text.
    types = [b.get("type") for b in resp.content]
    assert "video" in types
    assert fake.last_messages is None, \
        "fallback model should not have been called"


@pytest.mark.asyncio
async def test_no_fallback_configured_yields_generic_hint(
    tmp_video: Path,
) -> None:
    with patch.object(vm, "_check_multimodal_support", return_value=False), \
         patch.object(vm, "_probe_multimodal_if_needed", return_value=False), \
         patch.object(vm, "_resolve_fallback_video_model", return_value=None):
        resp = await view_video(str(tmp_video))
    # The generic hint contains the telltale "does not appear to support"
    # phrasing from _get_multimodal_fallback_hint.
    texts = [
        b.get("text", "") for b in resp.content if b.get("type") == "text"
    ]
    assert any("multimodal" in t.lower() for t in texts)


@pytest.mark.asyncio
async def test_no_fallback_keeps_videoblock_for_user_display(
    tmp_video: Path,
) -> None:
    # The VideoBlock stays in the ToolResponse so the user / frontend
    # can play the video.  Protection from the 413 ``Request Too
    # Large`` observed on Claude OAuth now lives in the message
    # normalizer (per-media-type strip with path-preserving
    # placeholder) — not in view_video itself.
    with patch.object(vm, "_check_multimodal_support", return_value=False), \
         patch.object(vm, "_probe_multimodal_if_needed", return_value=False), \
         patch.object(vm, "_resolve_fallback_video_model", return_value=None):
        resp = await view_video(str(tmp_video))
    block_types = [b.get("type") for b in resp.content]
    assert "video" in block_types
    texts = [
        b.get("text", "") for b in resp.content if b.get("type") == "text"
    ]
    assert any("multimodal" in t.lower() for t in texts)


@pytest.mark.asyncio
async def test_delegates_with_user_prompt(
    tmp_video: Path,
) -> None:
    fake = _FakeChatModel(description="Detailed description here.")
    user_prompt = "Count how many people appear in this clip."
    with patch.object(vm, "_check_multimodal_support", return_value=False), \
         patch.object(vm, "_probe_multimodal_if_needed", return_value=False), \
         patch.object(
             vm, "_resolve_fallback_video_model",
             return_value=(fake, "gemini", "gemini-2.5-pro"),
         ):
        resp = await view_video(str(tmp_video), prompt=user_prompt)

    # Fallback was called exactly once.
    assert fake.last_messages is not None
    user_msg = fake.last_messages[0]
    assert user_msg["role"] == "user"
    # Content has [VideoBlock, TextBlock(prompt)].
    content = user_msg["content"]
    assert len(content) == 2
    assert content[0]["type"] == "video"
    assert content[1]["type"] == "text"
    assert content[1]["text"] == user_prompt

    # Response contains the fallback's description text.
    texts = [
        b.get("text", "") for b in resp.content if b.get("type") == "text"
    ]
    assert any("Detailed description here." in t for t in texts)
    # And a header line that names the delegate so the primary agent
    # knows where the description came from.
    assert any("gemini-2.5-pro" in t for t in texts)


@pytest.mark.asyncio
async def test_missing_prompt_uses_default(
    tmp_video: Path,
) -> None:
    fake = _FakeChatModel()
    with patch.object(vm, "_check_multimodal_support", return_value=False), \
         patch.object(vm, "_probe_multimodal_if_needed", return_value=False), \
         patch.object(
             vm, "_resolve_fallback_video_model",
             return_value=(fake, "gemini", "gemini-2.5-pro"),
         ):
        await view_video(str(tmp_video))

    assert fake.last_messages is not None
    content = fake.last_messages[0]["content"]
    # The prompt block carries the stock default, verbatim.
    assert content[1]["text"] == _DEFAULT_VIDEO_FALLBACK_PROMPT


@pytest.mark.asyncio
async def test_empty_prompt_falls_back_to_default(
    tmp_video: Path,
) -> None:
    # Whitespace-only prompt → treat as missing; use default.
    fake = _FakeChatModel()
    with patch.object(vm, "_check_multimodal_support", return_value=False), \
         patch.object(vm, "_probe_multimodal_if_needed", return_value=False), \
         patch.object(
             vm, "_resolve_fallback_video_model",
             return_value=(fake, "gemini", "gemini-2.5-pro"),
         ):
        await view_video(str(tmp_video), prompt="   ")
    assert fake.last_messages[0]["content"][1]["text"] == (
        _DEFAULT_VIDEO_FALLBACK_PROMPT
    )


@pytest.mark.asyncio
async def test_fallback_model_failure_falls_back_to_hint(
    tmp_video: Path,
) -> None:
    # Fallback crashes mid-call → user still gets a useful response,
    # not an exception.
    fake = _FakeChatModel(fail=True)
    with patch.object(vm, "_check_multimodal_support", return_value=False), \
         patch.object(vm, "_probe_multimodal_if_needed", return_value=False), \
         patch.object(
             vm, "_resolve_fallback_video_model",
             return_value=(fake, "gemini", "gemini-2.5-pro"),
         ):
        resp = await view_video(str(tmp_video))
    texts = [
        b.get("text", "") for b in resp.content if b.get("type") == "text"
    ]
    assert any("multimodal" in t.lower() for t in texts), (
        "failed fallback should surface the generic hint so the agent "
        "tells the user it can't perceive the video"
    )


@pytest.mark.asyncio
async def test_fallback_empty_response_treated_as_failure(
    tmp_video: Path,
) -> None:
    # A fallback that returns an empty string shouldn't leak an empty
    # description to the caller — treat it as a failure and show the
    # generic hint instead.
    fake = _FakeChatModel(description="   ")
    with patch.object(vm, "_check_multimodal_support", return_value=False), \
         patch.object(vm, "_probe_multimodal_if_needed", return_value=False), \
         patch.object(
             vm, "_resolve_fallback_video_model",
             return_value=(fake, "gemini", "gemini-2.5-pro"),
         ):
        resp = await view_video(str(tmp_video))
    texts = [
        b.get("text", "") for b in resp.content if b.get("type") == "text"
    ]
    assert any("multimodal" in t.lower() for t in texts)


@pytest.mark.asyncio
async def test_url_video_uses_url_source(
    tmp_video: Path,  # unused; fixture kept for symmetry
) -> None:
    fake = _FakeChatModel()
    with patch.object(vm, "_check_multimodal_support", return_value=False), \
         patch.object(vm, "_probe_multimodal_if_needed", return_value=False), \
         patch.object(
             vm, "_resolve_fallback_video_model",
             return_value=(fake, "gemini", "gemini-2.5-pro"),
         ):
        await view_video("https://example.com/clip.mp4")
    # The VideoBlock carried into the fallback request should point
    # at the original URL, not a downloaded path.
    content = fake.last_messages[0]["content"]
    assert content[0]["source"]["url"] == "https://example.com/clip.mp4"


# ---------------------------------------------------------------- #
# Qwen-family shape + media-server signing                         #
# ---------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_qwen_family_gets_signed_url_and_list_shape(
    tmp_video: Path,
) -> None:
    # When the fallback is a Qwen-family provider (Aliyun Bailian /
    # DashScope / Kimi / etc.), view_video must:
    #   1. sign the local path via the media server (public URL)
    #   2. pass the URL inside ``{"type":"video","video":[url]}``
    # Neither the ``source`` shape nor raw local paths reach the
    # upstream API.
    signed = "https://media.example.com/media?path=...&sig=abc"
    fake = _FakeChatModel(description="desc")

    async def _fake_sign(path: str) -> str:
        assert path == str(tmp_video)
        return signed

    with patch.object(vm, "_check_multimodal_support", return_value=False), \
         patch.object(vm, "_probe_multimodal_if_needed", return_value=False), \
         patch.object(
             vm, "_resolve_fallback_video_model",
             return_value=(
                 fake, "bailian-via-skillclaw", "qwen3.6-plus",
             ),
         ), \
         patch("qwenpaw.app.channels.media_utils.resolve_media_url", _fake_sign):
        resp = await view_video(str(tmp_video), prompt="what happens?")

    content = fake.last_messages[0]["content"]
    # Single-video shape — ``type: video_url`` with ``video_url: {url}``.
    # NOT ``type: video`` + list (that's the frame-list mode, which
    # Qwen rejects with "sequence images should be (4, 8000)" when
    # it contains only one URL).
    video_block = content[0]
    assert video_block["type"] == "video_url"
    assert video_block["video_url"] == {"url": signed}
    # No ``source`` leakage / no list wrapping.
    assert "source" not in video_block
    assert "video" not in video_block
    # Prompt block follows.
    assert content[1]["type"] == "text"
    assert content[1]["text"] == "what happens?"
    # Response text was assembled.
    texts = [b.get("text", "") for b in resp.content if b.get("type") == "text"]
    assert any("desc" in t for t in texts)


@pytest.mark.asyncio
async def test_qwen_family_http_url_reaches_upstream_unchanged(
    tmp_video: Path,
) -> None:
    # A video already at an HTTP(S) URL must land in the Qwen
    # request unchanged.  ``resolve_media_url`` short-circuits for
    # URLs internally, but the contract view_video cares about is
    # "the URL I gave you is the URL the upstream sees."
    fake = _FakeChatModel()

    # Emulate ``resolve_media_url``'s real passthrough for URLs.
    async def _passthrough(path: str) -> str:
        return path

    with patch.object(vm, "_check_multimodal_support", return_value=False), \
         patch.object(vm, "_probe_multimodal_if_needed", return_value=False), \
         patch.object(
             vm, "_resolve_fallback_video_model",
             return_value=(fake, "bailian", "qwen3.6-plus"),
         ), \
         patch("qwenpaw.app.channels.media_utils.resolve_media_url", _passthrough):
        await view_video("https://ex.com/clip.mp4", prompt="p")

    video_block = fake.last_messages[0]["content"][0]
    assert video_block["type"] == "video_url"
    assert video_block["video_url"] == {"url": "https://ex.com/clip.mp4"}


@pytest.mark.asyncio
async def test_qwen_family_sign_failure_yields_generic_hint(
    tmp_video: Path,
) -> None:
    # When the media server is unreachable or refuses the path
    # (outside allowed dirs), the Qwen path returns None messages
    # and view_video falls through to the placeholder hint instead
    # of calling the fallback with a broken payload.
    fake = _FakeChatModel()

    async def _fake_sign(path: str) -> str:
        # Mimic real resolve_media_url's failure mode: returns the
        # raw local path verbatim when the media server is
        # unreachable / refuses the file.  The Qwen branch treats
        # that as "can't send to cloud" and bails.
        return path

    with patch.object(vm, "_check_multimodal_support", return_value=False), \
         patch.object(vm, "_probe_multimodal_if_needed", return_value=False), \
         patch.object(
             vm, "_resolve_fallback_video_model",
             return_value=(fake, "aliyun-codingplan", "qwen3.6-plus"),
         ), \
         patch("qwenpaw.app.channels.media_utils.resolve_media_url", _fake_sign):
        resp = await view_video(str(tmp_video))

    # Fallback NOT called (no messages recorded).
    assert fake.last_messages is None
    # Placeholder hint surfaces instead.
    texts = [b.get("text", "") for b in resp.content if b.get("type") == "text"]
    assert any("multimodal" in t.lower() for t in texts)


@pytest.mark.asyncio
async def test_non_qwen_family_uses_original_shape(
    tmp_video: Path,
) -> None:
    # Gemini / unknown providers keep the agentscope VideoBlock as-
    # is; the chat model's own formatter handles translation.
    fake = _FakeChatModel()
    sign_called = {"n": 0}

    async def _fake_sign(path: str):
        sign_called["n"] += 1
        return None

    with patch.object(vm, "_check_multimodal_support", return_value=False), \
         patch.object(vm, "_probe_multimodal_if_needed", return_value=False), \
         patch.object(
             vm, "_resolve_fallback_video_model",
             return_value=(fake, "gemini", "gemini-2.5-pro"),
         ), \
         patch("qwenpaw.app.channels.media_utils.resolve_media_url", _fake_sign):
        await view_video(str(tmp_video))

    # Gemini path doesn't sign.
    assert sign_called["n"] == 0
    video_block = fake.last_messages[0]["content"][0]
    assert video_block["type"] == "video"
    # agentscope-style ``source`` wrapper is preserved.
    assert video_block.get("source", {}).get("url") == str(tmp_video)
