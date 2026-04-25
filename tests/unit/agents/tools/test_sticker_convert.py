# -*- coding: utf-8 -*-
"""Tests for the shared sticker-format conversion core."""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from qwenpaw.agents.tools.sticker_convert import (
    StickerConversionError,
    prepare_sticker_webp,
)


def _png(path: Path, *, size=(1024, 1024), colour=(255, 0, 0, 255)) -> Path:
    Image.new("RGBA", size, colour).save(path)
    return path


def test_prepare_sticker_webp_default_outputs_512_square_png(tmp_path) -> None:
    """Default ``output_format="png"`` produces a 512×512 PNG.
    PNG is what every Signal client renders correctly; user-uploaded
    WebP triggers a voice-message rendering bug on Signal Android
    (see sticker_convert module docstring)."""
    src = _png(tmp_path / "src.png", size=(800, 600))
    out = prepare_sticker_webp(src)
    with Image.open(out) as im:
        assert im.size == (512, 512)
        assert im.format == "PNG"


def test_prepare_sticker_webp_outputs_webp_when_requested(tmp_path) -> None:
    """``output_format="webp"`` keeps WhatsApp's send-as-sticker
    filename convention working for callers targeting that channel."""
    src = _png(tmp_path / "src.png", size=(800, 600))
    out = prepare_sticker_webp(src, output_format="webp")
    with Image.open(out) as im:
        assert im.size == (512, 512)
        assert im.format == "WEBP"


def test_prepare_sticker_webp_default_output_path_is_sticker_png(
    tmp_path,
) -> None:
    """Default suffix is ``.sticker.png`` (Signal-friendly).  Only
    ``output_format="webp"`` produces ``.sticker.webp`` for
    WhatsApp's filename-based send-as-sticker rule."""
    src = _png(tmp_path / "pic.png")
    out = prepare_sticker_webp(src)
    assert out.name == "pic.sticker.png"
    assert out.parent == src.parent
    out_webp = prepare_sticker_webp(src, output_format="webp")
    assert out_webp.name == "pic.sticker.webp"


def test_prepare_sticker_webp_explicit_output_path(tmp_path) -> None:
    src = _png(tmp_path / "src.png")
    dest = tmp_path / "out" / "custom.webp"
    out = prepare_sticker_webp(src, dest)
    assert out == dest
    assert out.is_file()


def test_prepare_sticker_webp_preserves_alpha(tmp_path) -> None:
    """Alpha pixels from the source must survive to the output —
    agents generate transparent stickers and expect them to stay
    transparent after conversion."""
    src = tmp_path / "alpha.png"
    # Top-left pixel transparent, rest opaque red.
    img = Image.new("RGBA", (200, 200), (255, 0, 0, 255))
    img.putpixel((0, 0), (0, 0, 0, 0))
    img.save(src)
    out = prepare_sticker_webp(src)
    with Image.open(out) as im:
        im = im.convert("RGBA")
        # Outside the pasted square (letterbox) must be transparent.
        corner = im.getpixel((0, 0))
        assert corner[3] == 0, f"expected transparent corner, got {corner}"


def test_prepare_sticker_webp_letterboxes_non_square_input(tmp_path) -> None:
    """800×400 input fits at 512×256 with transparent band above
    and below, NOT stretched to 512×512."""
    src = _png(tmp_path / "wide.png", size=(800, 400))
    out = prepare_sticker_webp(src)
    with Image.open(out) as im:
        im = im.convert("RGBA")
        # Middle row of the transparent band should be transparent.
        top_mid = im.getpixel((256, 10))
        bottom_mid = im.getpixel((256, 500))
        assert top_mid[3] == 0, f"top band not transparent: {top_mid}"
        assert bottom_mid[3] == 0, f"bottom band not transparent: {bottom_mid}"


def test_prepare_sticker_webp_rejects_missing_input(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        prepare_sticker_webp(tmp_path / "nope.png")


def test_prepare_sticker_webp_output_fits_under_max(tmp_path) -> None:
    """A realistic input (1024×1024 solid colour) must compress to
    under Signal's 300 KB ceiling."""
    src = _png(tmp_path / "big.png")
    out = prepare_sticker_webp(src)
    assert out.stat().st_size < 300 * 1024


def test_prepare_sticker_webp_accepts_opaque_jpg(tmp_path) -> None:
    """JPG has no alpha channel — the converter must RGBA it so
    the transparent padding still works."""
    src = tmp_path / "opaque.jpg"
    Image.new("RGB", (500, 500), (0, 255, 0)).save(src)
    out = prepare_sticker_webp(src)
    assert out.is_file()
    with Image.open(out) as im:
        assert im.size == (512, 512)


def test_prepare_sticker_webp_raises_for_unreadable(tmp_path) -> None:
    """Not-an-image bytes → Pillow raises, we surface (not silently
    swallow) so callers can decide what to do."""
    src = tmp_path / "junk.png"
    src.write_bytes(b"not an image")
    from PIL import UnidentifiedImageError

    with pytest.raises(UnidentifiedImageError):
        prepare_sticker_webp(src)


def test_prepare_sticker_webp_raises_conversion_error_when_oversize(
    tmp_path,
    monkeypatch,
) -> None:
    """Simulate the pathological case where even quality=35 stays
    above the 300 KB ceiling — must raise
    :class:`StickerConversionError` so the agent knows to simplify
    the input, rather than silently producing a file that will
    fail pack validation later."""
    from qwenpaw.agents.tools import sticker_convert as sc

    src = _png(tmp_path / "src.png")
    # Pin every quality step to "too large".
    monkeypatch.setattr(sc, "_MAX_BYTES", 10)  # make the ceiling unmeetable
    with pytest.raises(StickerConversionError):
        prepare_sticker_webp(src, tmp_path / "out.webp")
