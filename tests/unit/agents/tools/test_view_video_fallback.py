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


class _FakeChatClient:
    """OpenAI-SDK-ish ``.client`` stub that the Qwen-family bypass
    reads ``base_url`` / ``api_key`` from.  The bypass never
    actually calls any method on the client; the real HTTP call
    goes through a patched ``httpx.AsyncClient``."""

    def __init__(self, base_url: str, api_key: str) -> None:
        self.base_url = base_url
        self.api_key = api_key


class _FakeChatModel:
    """Records the prompt it was called with and returns a canned
    description (or raises when ``fail=True``)."""

    def __init__(self, description: str = "A cat yawns.", fail: bool = False):
        self.description = description
        self.fail = fail
        self.last_messages: list[dict] | None = None
        # Populate enough fields for both the agentscope code path
        # (non-Qwen providers) and the Qwen-family httpx bypass.
        self.client = _FakeChatClient(
            base_url="http://127.0.0.1:30000/v1",
            api_key="sk-test",
        )

    async def __call__(self, messages: list[dict]) -> Any:
        self.last_messages = messages
        if self.fail:
            raise RuntimeError("simulated fallback API error")
        return _FakeStreamResponse(self.description)


class _FakeHttpxClient:
    """Context-manager stand-in used to intercept the Qwen-family
    httpx POST in tests.  ``response`` is what every ``.post()``
    call returns; ``last_call`` lets tests inspect the request."""

    def __init__(self, response: "_FakeHttpxResponse") -> None:
        self._response = response
        self.last_call: dict | None = None

    async def __aenter__(self) -> "_FakeHttpxClient":
        return self

    async def __aexit__(self, *_exc) -> None:
        return None

    async def post(self, url: str, json: dict, headers: dict):
        self.last_call = {"url": url, "json": json, "headers": headers}
        return self._response


class _FakeHttpxResponse:
    def __init__(self, status_code: int, body: dict) -> None:
        self.status_code = status_code
        self._body = body
        self.text = str(body)

    def json(self) -> dict:
        return self._body


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
    with patch.object(
        vm,
        "_check_multimodal_support",
        return_value=True,
    ), patch.object(
        vm,
        "_resolve_fallback_video_model",
        return_value=(fake, "gemini", "gemini-2.5-pro"),
    ):
        resp = await view_video(str(tmp_video))
    # VideoBlock present; no fallback header text.
    types = [b.get("type") for b in resp.content]
    assert "video" in types
    assert (
        fake.last_messages is None
    ), "fallback model should not have been called"


@pytest.mark.asyncio
async def test_no_fallback_configured_yields_generic_hint(
    tmp_video: Path,
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
        "_resolve_fallback_video_model",
        return_value=None,
    ):
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
        "_resolve_fallback_video_model",
        return_value=None,
    ):
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
        "_resolve_fallback_video_model",
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
        "_resolve_fallback_video_model",
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
        "_resolve_fallback_video_model",
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
        "_resolve_fallback_video_model",
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
        "_resolve_fallback_video_model",
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
        "_resolve_fallback_video_model",
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
async def test_qwen_family_gets_signed_url_and_video_url_shape(
    tmp_video: Path,
) -> None:
    # Qwen-family path posts OpenAI-compat chat/completions directly
    # via httpx (agentscope's formatter would drop ``video_url``
    # blocks as "unsupported block type").  The request must carry:
    #   1. a signed public URL from the media server
    #   2. the single-video shape ``{"type":"video_url","video_url":{"url":...}}``
    #   3. a text prompt block
    # And Authorization header from the chat_model's stored api_key.
    signed = "https://media.example.com/media?path=...&sig=abc"
    fake = _FakeChatModel(description="desc")
    http = _FakeHttpxClient(
        _FakeHttpxResponse(
            200,
            {"choices": [{"message": {"content": "video description text"}}]},
        ),
    )

    async def _fake_sign(path: str) -> str:
        assert path == str(tmp_video)
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
        "_resolve_fallback_video_model",
        return_value=(
            fake,
            "bailian-via-skillclaw",
            "qwen3.6-plus",
        ),
    ), patch(
        "qwenpaw.app.channels.media_utils.resolve_media_url",
        _fake_sign,
    ), patch.object(
        _real_httpx,
        "AsyncClient",
        lambda *a, **kw: http,
    ):
        resp = await view_video(str(tmp_video), prompt="what happens?")

    # HTTPX post was the actual upstream call (bypassing agentscope).
    assert http.last_call is not None, "httpx POST should have been made"
    body = http.last_call["json"]
    assert body["model"] == "qwen3.6-plus"
    content = body["messages"][0]["content"]
    video_block = content[0]
    assert video_block["type"] == "video_url"
    assert video_block["video_url"] == {"url": signed}
    # No ``source`` leakage / no list wrapping / no legacy shape.
    assert "source" not in video_block
    assert "video" not in video_block
    # Prompt block follows.
    assert content[1]["type"] == "text"
    assert content[1]["text"] == "what happens?"
    # Auth header carried through from client.api_key.
    assert http.last_call["headers"]["Authorization"] == "Bearer sk-test"
    # URL composed correctly: base_url + /chat/completions.
    assert http.last_call["url"].endswith("/chat/completions")
    # Response text reached the ToolResponse.
    texts = [
        b.get("text", "") for b in resp.content if b.get("type") == "text"
    ]
    assert any("video description text" in t for t in texts)


@pytest.mark.asyncio
async def test_qwen_family_http_url_reaches_upstream_unchanged(
    tmp_video: Path,
) -> None:
    # A video already at an HTTP(S) URL must land in the Qwen
    # request unchanged.
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
        "_resolve_fallback_video_model",
        return_value=(fake, "bailian", "qwen3.6-plus"),
    ), patch(
        "qwenpaw.app.channels.media_utils.resolve_media_url",
        _passthrough,
    ), patch.object(
        _real_httpx,
        "AsyncClient",
        lambda *a, **kw: http,
    ):
        await view_video("https://ex.com/clip.mp4", prompt="p")

    body = http.last_call["json"]
    video_block = body["messages"][0]["content"][0]
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
        "_resolve_fallback_video_model",
        return_value=(fake, "aliyun-codingplan", "qwen3.6-plus"),
    ), patch(
        "qwenpaw.app.channels.media_utils.resolve_media_url",
        _fake_sign,
    ):
        resp = await view_video(str(tmp_video))

    # Fallback NOT called (no messages recorded).
    assert fake.last_messages is None
    # Placeholder hint surfaces instead.
    texts = [
        b.get("text", "") for b in resp.content if b.get("type") == "text"
    ]
    assert any("multimodal" in t.lower() for t in texts)


@pytest.mark.asyncio
async def test_non_qwen_family_signs_then_passes_through_block(
    tmp_video: Path,
) -> None:
    # Non-Qwen providers (DeepSeek / ZAI / custom OpenAI-compat /
    # Gemini) keep the agentscope VideoBlock shape so each
    # provider's formatter can do its own translation, BUT the
    # source URL is first signed through the media server so
    # downstream endpoints receive a fetchable HTTPS URL instead
    # of a bare local path that the server can't reach.
    fake = _FakeChatModel()
    sign_called = {"n": 0}
    signed_url = "https://media.example/local/abc123?sig=ok"

    async def _fake_sign(path: str):
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
        "_resolve_fallback_video_model",
        return_value=(fake, "deepseek", "deepseek-vl"),
    ), patch(
        "qwenpaw.app.channels.media_utils.resolve_media_url",
        _fake_sign,
    ):
        await view_video(str(tmp_video))

    # Media server is asked to sign exactly once.
    assert sign_called["n"] == 1
    video_block = fake.last_messages[0]["content"][0]
    assert video_block["type"] == "video"
    # Block shape preserved (agentscope ``source`` wrapper);
    # only the URL inside is rewritten to the signed one.
    assert video_block.get("source", {}).get("url") == signed_url


@pytest.mark.asyncio
async def test_non_qwen_family_unreachable_signer_preserves_path(
    tmp_video: Path,
) -> None:
    # When the media server is unreachable ``resolve_media_url``
    # returns the original path — the VideoBlock then passes
    # through unchanged so any formatter that can still handle a
    # local file (e.g. agentscope's Gemini SDK upload) gets one.
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
        "_resolve_fallback_video_model",
        return_value=(fake, "gemini", "gemini-2.5-pro"),
    ), patch(
        "qwenpaw.app.channels.media_utils.resolve_media_url",
        _passthrough,
    ):
        await view_video(str(tmp_video))

    video_block = fake.last_messages[0]["content"][0]
    assert video_block.get("source", {}).get("url") == str(tmp_video)


# ---------------------------------------------------------------- #
# Transcode-on-400 retry path                                      #
# ---------------------------------------------------------------- #


class TestFormatRejectionDetection:
    def test_corrupted_marker_matches(self):
        assert vm._is_format_rejection(
            '{"error":{"message":"Multimodal data is corrupted or '
            'cannot be processed."}}',
        )

    def test_only_mp4_marker_matches(self):
        assert vm._is_format_rejection(
            "invalid video format, only mp4/wmv/mov/avi are supported",
        )

    def test_other_400_does_not_match(self):
        # Auth / rate-limit / shape errors must NOT trigger transcode.
        assert not vm._is_format_rejection("rate limit exceeded")
        assert not vm._is_format_rejection("invalid api key")
        assert not vm._is_format_rejection("")


class _SequencedFakeHttpxClient:
    """Like _FakeHttpxClient but returns a different response on
    each successive POST.  Used to simulate the
    "first call rejects, second call succeeds after transcode" flow.
    """

    def __init__(self, responses: list["_FakeHttpxResponse"]) -> None:
        self._responses = list(responses)
        self.calls: list[dict] = []

    async def __aenter__(self) -> "_SequencedFakeHttpxClient":
        return self

    async def __aexit__(self, *_exc) -> None:
        return None

    async def post(self, url: str, json: dict, headers: dict):
        self.calls.append({"url": url, "json": json, "headers": headers})
        if not self._responses:
            raise RuntimeError("test ran out of canned responses")
        return self._responses.pop(0)


@pytest.mark.asyncio
async def test_qwen_400_corrupted_triggers_transcode_then_retry(
    tmp_video: Path,
) -> None:
    """First mimo POST returns 400 'corrupted' — view_video should
    transcode the local file to H.264-in-MP4, sign the new path,
    and POST again.  Verifies the orchestration: 2 POSTs happen,
    the second one carries a different signed URL, and the final
    text is the one from the second response."""
    fake_chat = _FakeChatModel()

    # Two responses: 400 corrupted, then 200 with text.
    rejection = _FakeHttpxResponse(
        status_code=400,
        body={
            "error": {
                "message": "Multimodal data is corrupted "
                "or cannot be processed.",
            },
        },
    )
    rejection.text = (
        '{"error":{"message":"Multimodal data is corrupted '
        'or cannot be processed."}}'
    )
    success = _FakeHttpxResponse(
        status_code=200,
        body={
            "choices": [
                {
                    "message": {"content": "After transcode: a cat yawns."},
                },
            ],
        },
    )
    fake_http = _SequencedFakeHttpxClient([rejection, success])

    # Sign returns a sentinel URL; the second sign on the
    # transcoded sibling returns a *different* URL so we can verify
    # the retry POSTed against it.
    sign_calls: list[str] = []

    async def _fake_sign(path: str):
        sign_calls.append(path)
        return f"https://media.example/sig?p={Path(path).name}"

    transcode_calls: list[str] = []

    async def _fake_transcode(path: str) -> str | None:
        transcode_calls.append(path)
        # Return a sibling path; the file doesn't have to exist
        # because resolve_media_url is mocked.
        out = str(Path(path).with_name(Path(path).stem + ".h264.mp4"))
        return out

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
        "_resolve_fallback_video_model",
        return_value=(fake_chat, "mimo", "mimo-v2.5"),
    ), patch.object(
        vm,
        "_transcode_to_h264_mp4",
        _fake_transcode,
    ), patch(
        "qwenpaw.app.channels.media_utils.resolve_media_url",
        _fake_sign,
    ), patch(
        "httpx.AsyncClient",
        return_value=fake_http,
    ):
        resp = await view_video(str(tmp_video))

    # Two POSTs: the rejected one and the retry.
    assert (
        len(fake_http.calls) == 2
    ), f"expected 2 mimo POSTs, got {len(fake_http.calls)}"
    # Transcode fired exactly once with the local source.
    assert transcode_calls == [str(tmp_video)]
    # The second POST's video_url points at the *transcoded* file.
    second_url = fake_http.calls[1]["json"]["messages"][0]["content"][0][
        "video_url"
    ]["url"]
    assert "h264.mp4" in second_url
    # Final text comes from the successful retry, not the rejection.
    text = "".join(
        b.get("text", "") for b in resp.content if b.get("type") == "text"
    )
    assert "After transcode" in text


@pytest.mark.asyncio
async def test_describe_video_via_fallback_skips_transcode_for_remote_source():
    """Direct unit test for ``_describe_video_via_fallback``: when
    ``video_block.source.url`` is already a remote URL, the
    transcode branch must be skipped (we have no local file to
    feed to ffmpeg).  The format rejection surfaces as ``None``
    so the caller substitutes the placeholder hint upstream."""
    fake_chat = _FakeChatModel()
    rejection = _FakeHttpxResponse(
        status_code=400,
        body={"error": {"message": "Multimodal data is corrupted"}},
    )
    rejection.text = '{"error":{"message":"Multimodal data is corrupted"}}'
    fake_http = _SequencedFakeHttpxClient([rejection])
    transcode_called = {"n": 0}

    async def _fake_transcode(path: str):
        transcode_called["n"] += 1
        return None

    async def _passthrough(path: str):
        return path  # already a URL → no signing happens

    remote_block = {
        "type": "video",
        "source": {"url": "https://example.com/video.mp4"},
    }

    with patch.object(vm, "_transcode_to_h264_mp4", _fake_transcode), patch(
        "qwenpaw.app.channels.media_utils.resolve_media_url",
        _passthrough,
    ), patch("httpx.AsyncClient", return_value=fake_http):
        result = await vm._describe_video_via_fallback(
            remote_block,
            "describe",
            (fake_chat, "mimo", "mimo-v2.5"),
        )

    assert result is None
    assert len(fake_http.calls) == 1
    assert transcode_called["n"] == 0


@pytest.mark.asyncio
async def test_qwen_other_400_does_not_transcode(
    tmp_video: Path,
) -> None:
    """A non-format 400 (e.g. auth error) must keep the existing
    ``return None`` path — no ffmpeg, no second POST."""
    fake_chat = _FakeChatModel()
    rejection = _FakeHttpxResponse(
        status_code=400,
        body={"error": {"message": "invalid api key"}},
    )
    rejection.text = '{"error":{"message":"invalid api key"}}'
    fake_http = _SequencedFakeHttpxClient([rejection])

    transcode_called = {"n": 0}

    async def _fake_transcode(path: str):
        transcode_called["n"] += 1
        return None

    async def _fake_sign(path: str):
        return "https://media.example/sig"

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
        "_resolve_fallback_video_model",
        return_value=(fake_chat, "mimo", "mimo-v2.5"),
    ), patch.object(
        vm,
        "_transcode_to_h264_mp4",
        _fake_transcode,
    ), patch(
        "qwenpaw.app.channels.media_utils.resolve_media_url",
        _fake_sign,
    ), patch(
        "httpx.AsyncClient",
        return_value=fake_http,
    ):
        await view_video(str(tmp_video))

    assert len(fake_http.calls) == 1
    assert transcode_called["n"] == 0


class TestTranscodeToH264Mp4:
    @pytest.mark.asyncio
    async def test_missing_source_returns_none(self):
        result = await vm._transcode_to_h264_mp4("/nonexistent/foo.mp4")
        assert result is None

    @pytest.mark.asyncio
    async def test_existing_transcode_is_reused(self, tmp_path):
        # Pre-create the .h264.mp4 sibling — should short-circuit
        # without invoking ffmpeg.
        src = tmp_path / "vid.mp4"
        src.write_bytes(b"\x00" * 16)
        out = tmp_path / "vid.h264.mp4"
        out.write_bytes(b"already transcoded")
        # ffmpeg is intentionally NOT patched; if the implementation
        # tries to run it the test would still pass on systems with
        # ffmpeg installed but for the wrong reason.  Instead we
        # patch create_subprocess_exec to fail loudly so we'd notice.
        with patch(
            "asyncio.create_subprocess_exec",
            side_effect=AssertionError("ffmpeg should not run"),
        ):
            result = await vm._transcode_to_h264_mp4(str(src))
        assert result == str(out)
