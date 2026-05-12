# -*- coding: utf-8 -*-
"""Unit tests for ``view_image``'s fallback-model delegation path.

Mirrors ``test_view_video_fallback.py`` minus the transcode path
(images don't transcode).  Covers:

* agent config schema round-trip (``fallback_image_model`` field)
* delegation fires only when primary can't do image AND fallback is set
* fallback model receives an ImageBlock + the caller's prompt
* caller-omitted prompt falls back to the stock describe-everything prompt
* fallback failure / empty response gracefully surfaces the generic hint
* Qwen-family path posts the OpenAI-compat ``image_url`` shape via httpx,
  with a signed public URL when given a local file
* non-Qwen path signs the URL and keeps the ImageBlock shape so each
  provider's formatter handles translation
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from qwenpaw.agents.tools import view_media as vm
from qwenpaw.agents.tools.view_media import (
    _DEFAULT_IMAGE_FALLBACK_PROMPT,
    view_image,
)
from qwenpaw.config.config import AgentProfileConfig, ModelSlotConfig


# ---------------------------------------------------------------- #
# Schema                                                           #
# ---------------------------------------------------------------- #


class TestAgentProfileConfigFallbackImageField:
    def test_default_is_none(self):
        p = AgentProfileConfig(id="a", name="A")
        assert p.fallback_image_model is None

    def test_round_trip(self):
        src = {
            "id": "a",
            "name": "A",
            "fallback_image_model": {
                "provider_id": "gemini",
                "model": "gemini-2.5-pro",
            },
        }
        p = AgentProfileConfig.model_validate(src)
        assert isinstance(p.fallback_image_model, ModelSlotConfig)
        assert p.fallback_image_model.provider_id == "gemini"
        assert p.fallback_image_model.model == "gemini-2.5-pro"
        dumped = p.model_dump()
        assert dumped["fallback_image_model"] == src["fallback_image_model"]


# ---------------------------------------------------------------- #
# Shared fakes — agentscope chat model + httpx stubs               #
# ---------------------------------------------------------------- #


class _FakeStreamResponse:
    """Mimics an agentscope streaming chat response — yields one
    chunk whose content has a text block.  Text is cumulative in
    agentscope's streaming contract; emitting the full string in
    a single chunk equals a complete stream."""

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


class _FakeChatClient:
    def __init__(self, base_url: str, api_key: str) -> None:
        self.base_url = base_url
        self.api_key = api_key


class _FakeChatModel:
    def __init__(self, description: str = "A red apple.", fail: bool = False):
        self.description = description
        self.fail = fail
        self.last_messages: list[dict] | None = None
        self.client = _FakeChatClient(
            base_url="http://127.0.0.1:30000/v1",
            api_key="sk-test",
        )

    async def __call__(self, messages: list[dict]) -> Any:
        self.last_messages = messages
        if self.fail:
            raise RuntimeError("simulated fallback API error")
        return _FakeStreamResponse(self.description)


class _FakeHttpxResponse:
    def __init__(self, status_code: int, body: dict) -> None:
        self.status_code = status_code
        self._body = body
        self.text = str(body)

    def json(self) -> dict:
        return self._body


class _FakeHttpxClient:
    def __init__(self, response: _FakeHttpxResponse) -> None:
        self._response = response
        self.last_call: dict | None = None

    async def __aenter__(self) -> "_FakeHttpxClient":
        return self

    async def __aexit__(self, *_exc) -> None:
        return None

    async def post(self, url: str, json: dict, headers: dict):
        self.last_call = {"url": url, "json": json, "headers": headers}
        return self._response


@pytest.fixture
def tmp_image(tmp_path: Path) -> Path:
    p = tmp_path / "sample.png"
    p.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 64)
    return p


# ---------------------------------------------------------------- #
# view_image delegation path                                       #
# ---------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_primary_supports_image_bypasses_fallback(
    tmp_image: Path,
) -> None:
    fake = _FakeChatModel()
    with patch.object(
        vm,
        "_check_multimodal_support",
        return_value=True,
    ), patch.object(
        vm,
        "_resolve_fallback_image_model",
        return_value=(fake, "gemini", "gemini-2.5-pro"),
    ):
        resp = await view_image(
            str(tmp_image),
            prompt="What's the main subject of this image?",
        )
    types = [b.get("type") for b in resp.content]
    assert "image" in types
    assert (
        fake.last_messages is None
    ), "fallback model should not have been called"
    # Caller-supplied prompt must reach the model verbatim so it knows
    # what question to answer — without this the primary-path
    # tool_result is just an image block + "Image loaded:..." label
    # and many vision models reply with a vague acknowledgement
    # instead of analysing (observed in WhatsApp on 2026-05-12).
    texts = [b.get("text", "") for b in resp.content if b.get("type") == "text"]
    assert any(
        "What's the main subject of this image?" in t for t in texts
    ), f"caller prompt missing from response: {texts!r}"


@pytest.mark.asyncio
async def test_primary_supports_image_uses_default_prompt_when_none(
    tmp_image: Path,
) -> None:
    """When caller passes no prompt, primary path still surfaces a
    sensible default instruction so the model has something to do."""
    fake = _FakeChatModel()
    with patch.object(
        vm,
        "_check_multimodal_support",
        return_value=True,
    ), patch.object(
        vm,
        "_resolve_fallback_image_model",
        return_value=(fake, "gemini", "gemini-2.5-pro"),
    ):
        resp = await view_image(str(tmp_image))
    texts = [b.get("text", "") for b in resp.content if b.get("type") == "text"]
    assert any("Describe this image in detail" in t for t in texts), (
        f"default prompt missing from response: {texts!r}"
    )


@pytest.mark.asyncio
async def test_no_fallback_configured_yields_generic_hint(
    tmp_image: Path,
) -> None:
    with patch.object(
        vm,
        "_check_multimodal_support",
        return_value=False,
    ), patch.object(
        vm,
        "_probe_multimodal_if_needed",
        return_value=False,
    ), patch.object(
        vm,
        "_resolve_fallback_image_model",
        return_value=None,
    ):
        resp = await view_image(str(tmp_image))
    texts = [
        b.get("text", "") for b in resp.content if b.get("type") == "text"
    ]
    assert any("multimodal" in t.lower() for t in texts)


@pytest.mark.asyncio
async def test_no_fallback_keeps_imageblock_for_user_display(
    tmp_image: Path,
) -> None:
    with patch.object(
        vm,
        "_check_multimodal_support",
        return_value=False,
    ), patch.object(
        vm,
        "_probe_multimodal_if_needed",
        return_value=False,
    ), patch.object(
        vm,
        "_resolve_fallback_image_model",
        return_value=None,
    ):
        resp = await view_image(str(tmp_image))
    block_types = [b.get("type") for b in resp.content]
    assert "image" in block_types
    texts = [
        b.get("text", "") for b in resp.content if b.get("type") == "text"
    ]
    assert any("multimodal" in t.lower() for t in texts)


@pytest.mark.asyncio
async def test_delegates_with_user_prompt_non_qwen(
    tmp_image: Path,
) -> None:
    fake = _FakeChatModel(description="A bowl of fruit on a wooden table.")
    user_prompt = "What objects are visible in this image?"

    async def _passthrough(path: str) -> str:
        return path

    with patch.object(
        vm,
        "_check_multimodal_support",
        return_value=False,
    ), patch.object(
        vm,
        "_probe_multimodal_if_needed",
        return_value=False,
    ), patch.object(
        vm,
        "_resolve_fallback_image_model",
        return_value=(fake, "gemini", "gemini-2.5-pro"),
    ), patch(
        "qwenpaw.app.channels.media_utils.resolve_media_url",
        _passthrough,
    ):
        resp = await view_image(str(tmp_image), prompt=user_prompt)

    assert fake.last_messages is not None
    user_msg = fake.last_messages[0]
    assert user_msg["role"] == "user"
    content = user_msg["content"]
    assert len(content) == 2
    assert content[0]["type"] == "image"
    assert content[1]["type"] == "text"
    assert content[1]["text"] == user_prompt

    texts = [
        b.get("text", "") for b in resp.content if b.get("type") == "text"
    ]
    assert any("bowl of fruit" in t for t in texts)
    assert any("gemini-2.5-pro" in t for t in texts)


@pytest.mark.asyncio
async def test_missing_prompt_uses_default(
    tmp_image: Path,
) -> None:
    fake = _FakeChatModel()

    async def _passthrough(path: str) -> str:
        return path

    with patch.object(
        vm,
        "_check_multimodal_support",
        return_value=False,
    ), patch.object(
        vm,
        "_probe_multimodal_if_needed",
        return_value=False,
    ), patch.object(
        vm,
        "_resolve_fallback_image_model",
        return_value=(fake, "gemini", "gemini-2.5-pro"),
    ), patch(
        "qwenpaw.app.channels.media_utils.resolve_media_url",
        _passthrough,
    ):
        await view_image(str(tmp_image))

    assert fake.last_messages is not None
    content = fake.last_messages[0]["content"]
    assert content[1]["text"] == _DEFAULT_IMAGE_FALLBACK_PROMPT


@pytest.mark.asyncio
async def test_empty_prompt_falls_back_to_default(
    tmp_image: Path,
) -> None:
    fake = _FakeChatModel()

    async def _passthrough(path: str) -> str:
        return path

    with patch.object(
        vm,
        "_check_multimodal_support",
        return_value=False,
    ), patch.object(
        vm,
        "_probe_multimodal_if_needed",
        return_value=False,
    ), patch.object(
        vm,
        "_resolve_fallback_image_model",
        return_value=(fake, "gemini", "gemini-2.5-pro"),
    ), patch(
        "qwenpaw.app.channels.media_utils.resolve_media_url",
        _passthrough,
    ):
        await view_image(str(tmp_image), prompt="   ")
    assert fake.last_messages[0]["content"][1]["text"] == (
        _DEFAULT_IMAGE_FALLBACK_PROMPT
    )


@pytest.mark.asyncio
async def test_fallback_model_failure_falls_back_to_hint(
    tmp_image: Path,
) -> None:
    fake = _FakeChatModel(fail=True)

    async def _passthrough(path: str) -> str:
        return path

    with patch.object(
        vm,
        "_check_multimodal_support",
        return_value=False,
    ), patch.object(
        vm,
        "_probe_multimodal_if_needed",
        return_value=False,
    ), patch.object(
        vm,
        "_resolve_fallback_image_model",
        return_value=(fake, "gemini", "gemini-2.5-pro"),
    ), patch(
        "qwenpaw.app.channels.media_utils.resolve_media_url",
        _passthrough,
    ):
        resp = await view_image(str(tmp_image))
    texts = [
        b.get("text", "") for b in resp.content if b.get("type") == "text"
    ]
    assert any("multimodal" in t.lower() for t in texts)


@pytest.mark.asyncio
async def test_fallback_empty_response_treated_as_failure(
    tmp_image: Path,
) -> None:
    fake = _FakeChatModel(description="   ")

    async def _passthrough(path: str) -> str:
        return path

    with patch.object(
        vm,
        "_check_multimodal_support",
        return_value=False,
    ), patch.object(
        vm,
        "_probe_multimodal_if_needed",
        return_value=False,
    ), patch.object(
        vm,
        "_resolve_fallback_image_model",
        return_value=(fake, "gemini", "gemini-2.5-pro"),
    ), patch(
        "qwenpaw.app.channels.media_utils.resolve_media_url",
        _passthrough,
    ):
        resp = await view_image(str(tmp_image))
    texts = [
        b.get("text", "") for b in resp.content if b.get("type") == "text"
    ]
    assert any("multimodal" in t.lower() for t in texts)


@pytest.mark.asyncio
async def test_url_image_uses_url_source(
    tmp_image: Path,  # unused; kept for symmetry with the video suite
) -> None:
    fake = _FakeChatModel()

    async def _passthrough(path: str) -> str:
        return path

    with patch.object(
        vm,
        "_check_multimodal_support",
        return_value=False,
    ), patch.object(
        vm,
        "_probe_multimodal_if_needed",
        return_value=False,
    ), patch.object(
        vm,
        "_resolve_fallback_image_model",
        return_value=(fake, "gemini", "gemini-2.5-pro"),
    ), patch(
        "qwenpaw.app.channels.media_utils.resolve_media_url",
        _passthrough,
    ):
        await view_image("https://example.com/cat.png")
    content = fake.last_messages[0]["content"]
    assert content[0]["source"]["url"] == "https://example.com/cat.png"


# ---------------------------------------------------------------- #
# Qwen-family shape + media-server signing                         #
# ---------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_qwen_family_gets_signed_url_and_image_url_shape(
    tmp_image: Path,
) -> None:
    """Qwen-family path posts OpenAI-compat chat/completions directly
    via httpx with ``{"type":"image_url","image_url":{"url":...}}``,
    using a signed public URL from the media server when the source
    is a local file.  Auth header comes from chat_model.client.api_key.
    """
    signed = "https://media.example.com/media?path=...&sig=abc"
    fake = _FakeChatModel(description="desc-unused")
    http = _FakeHttpxClient(
        _FakeHttpxResponse(
            200,
            {"choices": [{"message": {"content": "image description text"}}]},
        ),
    )

    async def _fake_sign(path: str) -> str:
        # Mimic real resolve_media_url: HTTPS / data URLs pass through
        # unchanged so the step-2 normalize + fallback resolve are
        # observably idempotent.
        if path.startswith(("http://", "https://", "data:")):
            return path
        assert path == str(tmp_image)
        return signed

    import httpx as _real_httpx

    with patch.object(
        vm,
        "_check_multimodal_support",
        return_value=False,
    ), patch.object(
        vm,
        "_probe_multimodal_if_needed",
        return_value=False,
    ), patch.object(
        vm,
        "_resolve_fallback_image_model",
        return_value=(
            fake,
            "bailian-via-skillclaw",
            "qwen-vl-max",
        ),
    ), patch(
        "qwenpaw.app.channels.media_utils.resolve_media_url",
        _fake_sign,
    ), patch.object(
        _real_httpx,
        "AsyncClient",
        lambda *a, **kw: http,
    ):
        resp = await view_image(
            str(tmp_image),
            prompt="what's in the picture?",
        )

    assert http.last_call is not None
    body = http.last_call["json"]
    assert body["model"] == "qwen-vl-max"
    content = body["messages"][0]["content"]
    image_block = content[0]
    assert image_block["type"] == "image_url"
    assert image_block["image_url"] == {"url": signed}
    # No legacy shape / no list wrapping.
    assert "source" not in image_block
    assert "image" not in image_block
    assert content[1]["type"] == "text"
    assert content[1]["text"] == "what's in the picture?"
    assert http.last_call["headers"]["Authorization"] == "Bearer sk-test"
    assert http.last_call["url"].endswith("/chat/completions")
    texts = [
        b.get("text", "") for b in resp.content if b.get("type") == "text"
    ]
    assert any("image description text" in t for t in texts)


@pytest.mark.asyncio
async def test_qwen_family_http_url_reaches_upstream_unchanged() -> None:
    fake = _FakeChatModel()
    http = _FakeHttpxClient(
        _FakeHttpxResponse(200, {"choices": [{"message": {"content": "ok"}}]}),
    )

    async def _passthrough(path: str) -> str:
        return path

    import httpx as _real_httpx

    with patch.object(
        vm,
        "_check_multimodal_support",
        return_value=False,
    ), patch.object(
        vm,
        "_probe_multimodal_if_needed",
        return_value=False,
    ), patch.object(
        vm,
        "_resolve_fallback_image_model",
        return_value=(fake, "bailian", "qwen-vl-max"),
    ), patch(
        "qwenpaw.app.channels.media_utils.resolve_media_url",
        _passthrough,
    ), patch.object(
        _real_httpx,
        "AsyncClient",
        lambda *a, **kw: http,
    ):
        await view_image("https://ex.com/cat.png", prompt="p")

    body = http.last_call["json"]
    image_block = body["messages"][0]["content"][0]
    assert image_block["type"] == "image_url"
    assert image_block["image_url"] == {"url": "https://ex.com/cat.png"}


@pytest.mark.asyncio
async def test_qwen_family_sign_failure_yields_generic_hint(
    tmp_image: Path,
) -> None:
    """When ``resolve_media_url`` falls back to returning the raw
    local path (media server unreachable / refused), the Qwen
    branch bails — we don't send a local path to a cloud endpoint."""
    fake = _FakeChatModel()

    async def _fake_sign(path: str) -> str:
        return path  # local path passthrough — Qwen branch must reject

    with patch.object(
        vm,
        "_check_multimodal_support",
        return_value=False,
    ), patch.object(
        vm,
        "_probe_multimodal_if_needed",
        return_value=False,
    ), patch.object(
        vm,
        "_resolve_fallback_image_model",
        return_value=(fake, "aliyun-codingplan", "qwen-vl-max"),
    ), patch(
        "qwenpaw.app.channels.media_utils.resolve_media_url",
        _fake_sign,
    ):
        resp = await view_image(str(tmp_image))

    assert fake.last_messages is None
    texts = [
        b.get("text", "") for b in resp.content if b.get("type") == "text"
    ]
    assert any("multimodal" in t.lower() for t in texts)


@pytest.mark.asyncio
async def test_non_qwen_family_signs_then_passes_through_block(
    tmp_image: Path,
) -> None:
    fake = _FakeChatModel()
    sign_called = {"n": 0}
    signed_url = "https://media.example/local/img.png?sig=ok"

    async def _fake_sign(path: str):
        if path.startswith(("http://", "https://", "data:")):
            return path
        sign_called["n"] += 1
        return signed_url

    with patch.object(
        vm,
        "_check_multimodal_support",
        return_value=False,
    ), patch.object(
        vm,
        "_probe_multimodal_if_needed",
        return_value=False,
    ), patch.object(
        vm,
        "_resolve_fallback_image_model",
        return_value=(fake, "deepseek", "deepseek-vl"),
    ), patch(
        "qwenpaw.app.channels.media_utils.resolve_media_url",
        _fake_sign,
    ):
        await view_image(str(tmp_image))

    assert sign_called["n"] == 1
    image_block = fake.last_messages[0]["content"][0]
    assert image_block["type"] == "image"
    assert image_block.get("source", {}).get("url") == signed_url


@pytest.mark.asyncio
async def test_non_qwen_family_unreachable_signer_preserves_path(
    tmp_image: Path,
) -> None:
    fake = _FakeChatModel()

    async def _passthrough(path: str):
        return path

    with patch.object(
        vm,
        "_check_multimodal_support",
        return_value=False,
    ), patch.object(
        vm,
        "_probe_multimodal_if_needed",
        return_value=False,
    ), patch.object(
        vm,
        "_resolve_fallback_image_model",
        return_value=(fake, "gemini", "gemini-2.5-pro"),
    ), patch(
        "qwenpaw.app.channels.media_utils.resolve_media_url",
        _passthrough,
    ):
        await view_image(str(tmp_image))

    image_block = fake.last_messages[0]["content"][0]
    assert image_block.get("source", {}).get("url") == str(tmp_image)
