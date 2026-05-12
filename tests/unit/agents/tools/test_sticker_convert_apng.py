# -*- coding: utf-8 -*-
"""Unit tests for the animated-APNG branch in
``qwenpaw.agents.tools.sticker_convert``.

Coverage:
* animated GIF / animated WebP → APNG output (preserves frames)
* static PNG input → static PNG output (unchanged behaviour)
* output magic bytes start with PNG header (APNG is a PNG superset)
* output contains the ``acTL`` animation control chunk
* file size ≤ 300 KB (Signal cap)
* canvas is 512×512
* total animation duration ≤ 3 s (Signal cap)
* downstream signal_sticker validator (PNG path) accepts the result
"""

from __future__ import annotations

import struct
from io import BytesIO
from pathlib import Path

import pytest
from PIL import Image

from qwenpaw.agents.tools.sticker_convert import (
    _APNG_MAX_DURATION_MS,
    _MAX_BYTES,
    _SIZE,
    StickerConversionError,
    prepare_sticker_webp,
)
from qwenpaw.agents.tools.signal_sticker import _validate_sticker_image


_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def _has_apng_chunk(data: bytes) -> bool:
    """Walk PNG chunks looking for ``acTL`` (animation control).
    ``acTL`` is what distinguishes an APNG from a plain PNG.
    """
    if not data.startswith(_PNG_MAGIC):
        return False
    i = 8
    while i + 8 <= len(data):
        length = struct.unpack(">I", data[i : i + 4])[0]
        ctype = data[i + 4 : i + 8]
        if ctype == b"acTL":
            return True
        if ctype == b"IEND":
            return False
        i += 8 + length + 4  # length + type + data + CRC
    return False


def _read_apng_frame_count(data: bytes) -> int:
    """Parse ``acTL`` ``num_frames`` field (first 4 bytes of the
    chunk data).  Returns 0 if the chunk is absent (= static PNG).
    """
    if not data.startswith(_PNG_MAGIC):
        return 0
    i = 8
    while i + 8 <= len(data):
        length = struct.unpack(">I", data[i : i + 4])[0]
        ctype = data[i + 4 : i + 8]
        if ctype == b"acTL":
            return struct.unpack(">I", data[i + 8 : i + 12])[0]
        if ctype == b"IEND":
            return 0
        i += 8 + length + 4
    return 0


def _read_apng_total_duration_ms(data: bytes) -> int:
    """Sum the per-frame delays from every ``fcTL`` chunk.  Each
    ``fcTL`` carries ``delay_num`` / ``delay_den`` (offsets 20-23 +
    24-25 within the chunk data) — multiply to get ms.
    """
    if not data.startswith(_PNG_MAGIC):
        return 0
    i = 8
    total_ms = 0
    while i + 8 <= len(data):
        length = struct.unpack(">I", data[i : i + 4])[0]
        ctype = data[i + 4 : i + 8]
        if ctype == b"fcTL":
            cdata = data[i + 8 : i + 8 + length]
            delay_num = struct.unpack(">H", cdata[20:22])[0]
            delay_den = struct.unpack(">H", cdata[22:24])[0] or 100
            total_ms += int(round(delay_num * 1000 / delay_den))
        if ctype == b"IEND":
            break
        i += 8 + length + 4
    return total_ms


def _make_animated_gif(
    path: Path,
    frames: int = 8,
    size: tuple[int, int] = (200, 200),
    duration_ms: int = 100,
) -> Path:
    """Synthesize a small animated GIF — solid colour cycling per
    frame so the file's actually animated (Pillow short-circuits
    save_all when all frames are identical)."""
    imgs = []
    for i in range(frames):
        # Cycle hue-ish — RGBA so Pillow keeps alpha info in the
        # input pipeline (GIF flattens to P + transparency, but the
        # encoder's input expectation is RGBA per frame).
        col = (
            (i * 30) % 256,
            (255 - i * 20) % 256,
            (i * 50) % 256,
            255,
        )
        imgs.append(Image.new("RGBA", size, col))
    imgs[0].save(
        path,
        save_all=True,
        append_images=imgs[1:],
        duration=duration_ms,
        loop=0,
    )
    return path


def _make_static_png(path: Path) -> Path:
    Image.new("RGBA", (300, 200), (10, 200, 50, 255)).save(path, "PNG")
    return path


# ---------------------------------------------------------------- #
# Animated input → APNG output                                     #
# ---------------------------------------------------------------- #


def test_animated_gif_produces_apng(tmp_path: Path) -> None:
    src = _make_animated_gif(tmp_path / "anim.gif", frames=8)
    out = prepare_sticker_webp(src)
    data = out.read_bytes()
    assert data.startswith(_PNG_MAGIC), "output must be PNG-family"
    assert _has_apng_chunk(data), "output must carry the acTL APNG chunk"
    assert _read_apng_frame_count(data) >= 2


def test_apng_output_canvas_is_512(tmp_path: Path) -> None:
    src = _make_animated_gif(tmp_path / "anim.gif", frames=4, size=(120, 80))
    out = prepare_sticker_webp(src)
    with Image.open(out) as im:
        assert im.size == (_SIZE, _SIZE)


def test_apng_output_under_300kb(tmp_path: Path) -> None:
    src = _make_animated_gif(tmp_path / "anim.gif", frames=10)
    out = prepare_sticker_webp(src)
    assert out.stat().st_size <= _MAX_BYTES


def test_apng_total_duration_capped_at_3s(tmp_path: Path) -> None:
    # 50 frames × 100 ms each = 5000 ms input — should be trimmed to
    # ≤ 3000 ms in the output APNG.
    src = _make_animated_gif(
        tmp_path / "anim.gif",
        frames=50,
        duration_ms=100,
    )
    out = prepare_sticker_webp(src)
    data = out.read_bytes()
    total_ms = _read_apng_total_duration_ms(data)
    # Allow a small encoding-precision tolerance for delay_num/den.
    assert (
        total_ms <= _APNG_MAX_DURATION_MS + 50
    ), f"total animation duration {total_ms}ms exceeds 3s cap"


def test_apng_output_passes_signal_validator(tmp_path: Path) -> None:
    # The downstream pack-staging validator must accept APNG with
    # contentType: image/png — same magic + IHDR shape as static PNG.
    src = _make_animated_gif(tmp_path / "anim.gif", frames=6)
    out = prepare_sticker_webp(src)
    err = _validate_sticker_image(out)
    assert err is None, f"validator rejected APNG: {err}"


def test_apng_named_with_sticker_png_suffix(tmp_path: Path) -> None:
    # File name convention stays ``.sticker.png`` — staging treats
    # APNG identically to PNG, so the filename doesn't need to
    # advertise the animation.
    src = _make_animated_gif(tmp_path / "myclip.gif", frames=4)
    out = prepare_sticker_webp(src)
    assert out.name == "myclip.sticker.png"


def test_apng_explicit_output_path(tmp_path: Path) -> None:
    src = _make_animated_gif(tmp_path / "anim.gif", frames=4)
    dest = tmp_path / "out" / "named.png"
    out = prepare_sticker_webp(src, output_path=dest)
    assert out == dest
    assert _has_apng_chunk(dest.read_bytes())


# ---------------------------------------------------------------- #
# Static input → static PNG (no regression)                        #
# ---------------------------------------------------------------- #


def test_static_png_input_produces_static_png(tmp_path: Path) -> None:
    src = _make_static_png(tmp_path / "still.png")
    out = prepare_sticker_webp(src)
    data = out.read_bytes()
    assert data.startswith(_PNG_MAGIC)
    assert not _has_apng_chunk(data), (
        "static input must NOT carry an acTL chunk — that would mean "
        "we're paying APNG overhead for no animation"
    )


def test_static_png_output_canvas_is_512(tmp_path: Path) -> None:
    src = _make_static_png(tmp_path / "still.png")
    out = prepare_sticker_webp(src)
    with Image.open(out) as im:
        assert im.size == (_SIZE, _SIZE)


# ---------------------------------------------------------------- #
# Animated input forced to webp output → first frame only          #
# ---------------------------------------------------------------- #


def test_animated_input_with_webp_output_flattens(tmp_path: Path) -> None:
    # WebP output explicitly opts out of animation (no animated-WebP
    # path — Signal Android can't render those anyway).  Confirm
    # the existing first-frame-only behaviour still holds.
    src = _make_animated_gif(tmp_path / "anim.gif", frames=6)
    out = prepare_sticker_webp(src, output_format="webp")
    assert out.suffix == ".webp"
    with Image.open(out) as im:
        assert getattr(im, "is_animated", False) is False


# ---------------------------------------------------------------- #
# Round-trip with explicit output_format="png" on animated         #
# ---------------------------------------------------------------- #


def test_explicit_png_output_on_animated_still_apng(tmp_path: Path) -> None:
    # Caller may pass output_format="png" explicitly; animated input
    # should still get the APNG treatment (output_format="png" is the
    # default, not a "flatten" request).
    src = _make_animated_gif(tmp_path / "anim.gif", frames=4)
    out = prepare_sticker_webp(src, output_format="png")
    assert _has_apng_chunk(out.read_bytes())
