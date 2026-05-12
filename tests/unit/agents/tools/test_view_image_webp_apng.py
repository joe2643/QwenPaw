# -*- coding: utf-8 -*-
"""Tests for ``view_image``'s animated WebP → APNG transcode.

Background: z.ai's glm-5v-turbo (and other OpenAI-compat vision
endpoints) reject ``image/webp`` data URLs with animation chunks
(VP8X / ANIM / ANMF) with HTTP 400 ``1210 图片输入格式/解析错误``.
APNG is a backward-compatible PNG extension that preserves every
frame + alpha, and z.ai decodes it end-to-end — curl-verified on
2026-05-12 (PNG first-frame works, GIF rejected, APNG works with
animation pickup, WebP-animated rejected).

These tests cover the local-file branch only.  The HTTP(S)-URL
branch in ``view_image`` leaves the upstream URL alone — we don't
control its Content-Type, so animation handling there is the model
provider's concern.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Iterator

import pytest
from PIL import Image, ImageDraw

from qwenpaw.agents.tools import view_media


def _make_animated_webp(path: Path, n_frames: int = 4, size: int = 32) -> None:
    """Build a tiny multi-frame animated WebP for fixture use.

    Each frame paints a coloured square in a different position so a
    naive single-frame decode is detectable downstream.
    """
    frames = []
    for i in range(n_frames):
        im = Image.new("RGBA", (size, size), (0, 0, 0, 0))
        draw = ImageDraw.Draw(im)
        x0 = (i * (size // n_frames)) % size
        draw.rectangle(
            [x0, 0, x0 + size // n_frames, size],
            fill=(255, 0, 0, 255),
        )
        frames.append(im)
    frames[0].save(
        path,
        format="WEBP",
        save_all=True,
        append_images=frames[1:],
        duration=100,
        loop=0,
    )


def _make_static_webp(path: Path, size: int = 32) -> None:
    """Single-frame WebP — no animation chunks."""
    im = Image.new("RGBA", (size, size), (0, 255, 0, 255))
    im.save(path, format="WEBP")


# ---------------------------------------------------------------------------
# _is_animated_webp header sniff
# ---------------------------------------------------------------------------


def test_is_animated_webp_returns_true_for_animated(tmp_path: Path) -> None:
    src = tmp_path / "anim.webp"
    _make_animated_webp(src)
    assert view_media._is_animated_webp(str(src)) is True


def test_is_animated_webp_returns_false_for_static(tmp_path: Path) -> None:
    src = tmp_path / "static.webp"
    _make_static_webp(src)
    assert view_media._is_animated_webp(str(src)) is False


def test_is_animated_webp_returns_false_for_non_webp(tmp_path: Path) -> None:
    src = tmp_path / "not_webp.png"
    Image.new("RGBA", (8, 8)).save(src)
    assert view_media._is_animated_webp(str(src)) is False


def test_is_animated_webp_returns_false_for_missing(tmp_path: Path) -> None:
    assert view_media._is_animated_webp(str(tmp_path / "nope.webp")) is False


# ---------------------------------------------------------------------------
# _transcode_animated_webp_to_apng
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_transcode_creates_multi_frame_apng(tmp_path: Path) -> None:
    src = tmp_path / "anim.webp"
    _make_animated_webp(src, n_frames=5)

    out = await view_media._transcode_animated_webp_to_apng(str(src))
    assert out is not None
    out_path = Path(out)
    assert out_path.exists()
    assert out_path.suffix == ".apng"
    assert out_path.parent == tmp_path
    # The sibling lives next to the source — no orphan files in tmp.
    assert out_path.stat().st_size > 0

    # Pillow can iterate the frames back out: animation preserved.
    with Image.open(out_path) as im:
        assert getattr(im, "n_frames", 1) == 5


@pytest.mark.asyncio
async def test_transcode_returns_none_for_static_webp(tmp_path: Path) -> None:
    """Static webp has n_frames=1; helper short-circuits with None so
    the caller knows to leave the original alone (no transcode benefit)."""
    src = tmp_path / "static.webp"
    _make_static_webp(src)

    out = await view_media._transcode_animated_webp_to_apng(str(src))
    assert out is None


@pytest.mark.asyncio
async def test_transcode_is_idempotent(tmp_path: Path) -> None:
    """Calling twice should reuse the cached sibling, not re-encode."""
    src = tmp_path / "anim.webp"
    _make_animated_webp(src, n_frames=3)

    out1 = await view_media._transcode_animated_webp_to_apng(str(src))
    assert out1 is not None
    mtime1 = Path(out1).stat().st_mtime_ns

    out2 = await view_media._transcode_animated_webp_to_apng(str(src))
    assert out2 == out1
    mtime2 = Path(out2).stat().st_mtime_ns
    assert mtime2 == mtime1, "transcode should reuse, not re-encode"


@pytest.mark.asyncio
async def test_transcode_returns_none_for_missing_source(tmp_path: Path) -> None:
    out = await view_media._transcode_animated_webp_to_apng(
        str(tmp_path / "ghost.webp"),
    )
    assert out is None


# ---------------------------------------------------------------------------
# view_image swap behavior
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_view_image_swaps_animated_webp_to_apng(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """End-to-end view_image call: animated webp gets swapped to apng
    before the ImageBlock leaves Step 1.  Multimodal capability +
    URL-form normalization are mocked off so we observe the raw
    Step-1 swap, not downstream rewrites.
    """
    src = tmp_path / "sticker.webp"
    _make_animated_webp(src, n_frames=4)

    # Force "primary supports image" so view_image returns the block
    # directly without bouncing through the fallback path.
    monkeypatch.setattr(
        view_media,
        "_check_multimodal_support",
        lambda media_type="image": True,
    )

    # Disable the URL-form normalize so source.url stays a local path
    # we can assert on.  In production it would get signed by the
    # media server, but the swap from .webp to .apng is what matters.
    async def _identity(block):
        return block

    monkeypatch.setattr(view_media, "_to_url_form_block", _identity)

    resp = await view_media.view_image(str(src))

    image_blocks = [
        b for b in resp.content if b.get("type") == "image"
    ]
    assert len(image_blocks) == 1
    url = image_blocks[0]["source"]["url"]
    assert url.endswith(".apng"), (
        f"expected APNG sibling, got {url!r} — webp animation chunks "
        "would be rejected by OpenAI-compat vision endpoints"
    )
    assert Path(url).exists()


@pytest.mark.asyncio
async def test_view_image_leaves_static_webp_unchanged(
    tmp_path: Path,
    monkeypatch,
) -> None:
    src = tmp_path / "static.webp"
    _make_static_webp(src)

    monkeypatch.setattr(
        view_media,
        "_check_multimodal_support",
        lambda media_type="image": True,
    )

    async def _identity(block):
        return block

    monkeypatch.setattr(view_media, "_to_url_form_block", _identity)

    resp = await view_media.view_image(str(src))

    image_blocks = [b for b in resp.content if b.get("type") == "image"]
    assert len(image_blocks) == 1
    url = image_blocks[0]["source"]["url"]
    assert url.endswith(".webp"), (
        f"static webp should not be transcoded — got {url!r}"
    )
