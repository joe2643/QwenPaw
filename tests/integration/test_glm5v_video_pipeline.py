# -*- coding: utf-8 -*-
"""End-to-end pipeline test for glm-5v-turbo video support.

Two layers:

1. ``test_pipeline_emits_video_url_block``: pure unit, no network.
   Confirms that with ``glm-5v-turbo`` as the active model, CoPaw's
   wrapped OpenAI formatter (``_create_formatter_instance`` →
   ``FileBlockSupportOpenAIChatFormatter``) translates a ``VideoBlock``
   into the OpenAI-compat-with-Qwen-extension wire shape
   ``{"type":"video_url","video_url":{"url":...}}`` that the z.ai
   coding-plan endpoint actually accepts (curl-verified 2026-05-12).

   This is the regression guard for ``glm-5v-turbo``'s hardcoded
   ``supports_video`` flag in ``provider_manager.py`` — before the
   2026-05-12 fix it was ``False``, which made the normalizer
   downgrade every video to a path-preserving text placeholder
   even when probe + curl both said the model handles video fine.

2. ``test_live_video_call``: real network call to z.ai.  Skipped
   unless ``COPAW_VIDEO_E2E=1`` is set.  Decrypts the api key from
   the local CoPaw secret store, signs a test video URL via the
   running media server, and confirms the model returns a non-empty
   description.

Run live test::

    COPAW_VIDEO_E2E=1 COPAW_VIDEO_E2E_FILE=/abs/path/to/clip.mp4 \\
        pytest tests/integration/test_glm5v_video_pipeline.py -k live -s
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

import pytest
from agentscope.message import Msg, TextBlock, VideoBlock
from agentscope.model._openai_model import OpenAIChatModel

from qwenpaw.agents.model_factory import _create_formatter_instance
from qwenpaw.providers.provider_manager import ModelSlotConfig, ProviderManager


_LIVE_ENV = "COPAW_VIDEO_E2E"
_LIVE_FILE_ENV = "COPAW_VIDEO_E2E_FILE"


@pytest.fixture
def glm5v_active():
    """Set glm-5v-turbo as the active model for the duration of one test."""
    mgr = ProviderManager.get_instance()
    saved = mgr.active_model
    mgr.active_model = ModelSlotConfig(
        provider_id="zhipu-intl-codingplan",
        model="glm-5v-turbo",
    )
    try:
        yield
    finally:
        mgr.active_model = saved


def test_pipeline_emits_video_url_block(glm5v_active) -> None:
    """With glm-5v-turbo active, VideoBlock survives normalization and
    the formatter emits the Qwen-style ``video_url`` content block."""
    fmt = _create_formatter_instance(OpenAIChatModel)

    msgs = [
        Msg(
            name="user",
            role="user",
            content=[
                VideoBlock(
                    type="video",
                    source={
                        "type": "url",
                        "url": "https://media.example/clip.mp4",
                    },
                ),
                TextBlock(type="text", text="describe"),
            ],
        ),
    ]
    out = asyncio.run(fmt.format(msgs))

    assert len(out) == 1, f"expected one user message, got {out!r}"
    content = out[0]["content"]
    assert isinstance(content, list)

    video_blocks = [b for b in content if b.get("type") == "video_url"]
    assert len(video_blocks) == 1, (
        f"expected one video_url block, got content={content!r}.  "
        "If this is empty, supports_video for glm-5v-turbo was "
        "downgraded again — check provider_manager.py."
    )
    assert video_blocks[0]["video_url"] == {
        "url": "https://media.example/clip.mp4",
    }

    text_blocks = [b for b in content if b.get("type") == "text"]
    assert any(b.get("text") == "describe" for b in text_blocks)


@pytest.mark.skipif(
    os.environ.get(_LIVE_ENV) != "1",
    reason=f"set {_LIVE_ENV}=1 to run the live z.ai network call",
)
def test_live_video_call(glm5v_active) -> None:
    """Sign a local clip, POST it to z.ai via the same wire shape
    CoPaw's pipeline emits, and assert a non-empty description."""
    import httpx

    from qwenpaw.app.channels.media_utils import resolve_media_url
    from qwenpaw.security.secret_store import decrypt

    src_path = os.environ.get(_LIVE_FILE_ENV) or str(
        next(
            iter(
                sorted(
                    Path("/home/joe/.qwenpaw/media/whatsapp").glob(
                        "wa_vid_*.mp4",
                    ),
                    key=lambda p: p.stat().st_size,
                ),
            ),
        ),
    )
    assert Path(src_path).is_file(), (
        f"video fixture missing: {src_path} — point {_LIVE_FILE_ENV} "
        "at a small .mp4"
    )

    signed_url = asyncio.run(resolve_media_url(src_path))
    assert signed_url.startswith("https://"), (
        f"resolve_media_url did not return a public URL — got {signed_url!r}.  "
        "Make sure CoPaw + Cloudflare tunnel are up."
    )

    provider_json = Path(
        "/home/joe/.qwenpaw.secret/providers/builtin/"
        "zhipu-intl-codingplan.json",
    )
    cfg = json.loads(provider_json.read_text())
    api_key = decrypt(cfg["api_key"])
    base_url = cfg["base_url"].rstrip("/")

    body = {
        "model": "glm-5v-turbo",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "video_url", "video_url": {"url": signed_url}},
                    {
                        "type": "text",
                        "text": "In one sentence: what is in this video?",
                    },
                ],
            },
        ],
        "stream": False,
    }
    resp = httpx.post(
        f"{base_url}/chat/completions",
        json=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        timeout=180.0,
    )
    assert resp.status_code == 200, f"HTTP {resp.status_code}: {resp.text}"
    data = resp.json()
    desc = (
        ((data.get("choices") or [{}])[0].get("message") or {}).get(
            "content",
        )
        or ""
    )
    assert desc.strip(), f"empty description — full response: {data!r}"

    usage = data.get("usage") or {}
    prompt_tokens = usage.get("prompt_tokens", 0)
    # The text prompt alone is ~15 tokens.  z.ai doesn't expose a
    # ``video_tokens`` breakdown the way mimo does, so we infer
    # video consumption from a high prompt_tokens count (each
    # sampled frame burns ~10-50 tokens).  Sub-100 means the model
    # almost certainly ignored the video block.
    assert prompt_tokens > 200, (
        f"prompt_tokens={prompt_tokens} too low — server probably "
        f"ignored the video block.  usage={usage!r}"
    )
    print(f"\n[live z.ai description] {desc[:300]}")
    print(f"[prompt_tokens={prompt_tokens}]")
