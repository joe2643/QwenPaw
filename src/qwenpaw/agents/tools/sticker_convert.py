# -*- coding: utf-8 -*-
"""Image → Signal / WhatsApp sticker-format conversion.

Shared core so both the agent tool wrapper
(``signal_sticker.signal_prepare_sticker_webp``) and the standalone
skill script (``skills/sticker_format-en/scripts/prepare_sticker_webp.py``)
converge on one definition of "valid sticker output".

Output guarantees every caller can rely on:

* 512×512 exactly.
* Recognised sticker magic — ``\\x89PNG…`` for PNG (default) or
  ``RIFF…WEBP`` for WebP (opt-in, WhatsApp-friendly).
* Alpha preserved (transparent pad when the source isn't square).
* File size ≤ ``_MAX_BYTES`` (300 KB — Signal's upload ceiling).

Why PNG is the default:
    Signal's spec accepts both PNG and WebP, but in practice the
    Signal Android client renders user-supplied WebP stickers as
    voice-message blobs when the WebP doesn't come from Signal
    Desktop's own sticker creator (reproducible regardless of
    VP8L vs VP8X encoding).  PNG is byte-for-byte what packs like
    LIHKG Dog use and renders cleanly across every Signal client
    we've tested.  WhatsApp still needs WebP — pass
    ``output_format="webp"`` for that pipeline.

Deliberately stdlib + Pillow only — no CoPaw imports — so the skill
script can execute against any Python environment with Pillow
installed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional, Union

from PIL import Image

_SIZE = 512
_MAX_BYTES = 300 * 1024
_TARGET_BYTES = 100 * 1024
# WebP quality ladder — each step drops ~30% bytes on typical AI-gen
# art.  We stop as soon as we're under ``_TARGET_BYTES`` (nice-to-
# have) but the hard requirement is ``_MAX_BYTES``.  Values chosen
# empirically; anything under 30 looks blocky.
_QUALITY_LADDER = (95, 85, 75, 65, 55, 45, 35)

StickerFormat = Literal["png", "webp"]


class StickerConversionError(RuntimeError):
    """Raised when even the most aggressive encoder step fails to
    bring the file under ``_MAX_BYTES``.  The caller should surface
    this verbatim — the message names the final size so the agent
    can decide whether to reduce input complexity (crop, reduce
    gradients) or skip the sticker entirely.
    """


def prepare_sticker_webp(
    input_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    *,
    output_format: StickerFormat = "png",
) -> Path:
    """Convert an image at ``input_path`` into a sticker-valid file.

    Despite the legacy name, the **default output is PNG**.  PNG
    works on every Signal client we've tested; WebP triggers a
    voice-message rendering bug on Signal Android for any WebP not
    produced by Signal Desktop's own creator.  Pass
    ``output_format="webp"`` to keep the WhatsApp-compatible
    ``.sticker.webp`` convention.

    Args:
        input_path:
            Any Pillow-decodable image: PNG, JPG, WEBP, GIF (first
            frame), BMP, etc.
        output_path:
            Destination file.  When ``None``, the output lands next
            to the source with a ``.sticker.png`` (or
            ``.sticker.webp`` when ``output_format="webp"``) suffix.
            The ``.sticker.webp`` form is what WhatsApp's send-as-
            sticker filename convention recognises.
        output_format:
            ``"png"`` (default) or ``"webp"``.

    Returns:
        Absolute :class:`Path` of the written file.

    Raises:
        :class:`FileNotFoundError`:
            Input file doesn't exist.
        :class:`PIL.UnidentifiedImageError`:
            Input isn't a decodable image.
        :class:`StickerConversionError`:
            Encoded file exceeds 300 KB (relevant for WebP fallback
            cases — PNG with optimize=True almost always fits).
    """
    if output_format not in ("png", "webp"):
        raise ValueError(
            f"output_format must be 'png' or 'webp', got {output_format!r}",
        )

    src_path = Path(input_path).expanduser().resolve()
    if not src_path.is_file():
        raise FileNotFoundError(f"sticker source not found: {src_path}")

    suffix = ".sticker.png" if output_format == "png" else ".sticker.webp"
    if output_path is None:
        out_path = src_path.with_name(f"{src_path.stem}{suffix}")
    else:
        out_path = Path(output_path).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)

    with Image.open(src_path) as raw:
        # GIF / multi-frame: flatten to the first frame.  Animated
        # stickers need a different encoding path (libwebp with
        # frame args) that we haven't scoped yet — degrade rather
        # than fail so the agent gets *something* usable.
        if getattr(raw, "is_animated", False):
            raw.seek(0)
        # Ensure RGBA so the transparent pad step works for
        # opaque JPGs as well as alpha-bearing PNGs.
        img = raw.convert("RGBA")

    # Fit into SIZExSIZE preserving aspect, transparent-pad the rest
    # (Signal recommends the sticker visually fill the canvas but
    # letterboxing is accepted — square crop would distort so
    # avoid it).
    img.thumbnail((_SIZE, _SIZE), Image.LANCZOS)
    canvas = Image.new("RGBA", (_SIZE, _SIZE), (0, 0, 0, 0))
    offset = ((_SIZE - img.width) // 2, (_SIZE - img.height) // 2)
    canvas.paste(img, offset, img)

    if output_format == "png":
        # PNG with optimize=True ~= what LIHKG Dog and other
        # well-known sticker packs ship.  No quality ladder needed
        # — Pillow's PNG encoder + optimize already produces sub-
        # 300 KB output for our 512×512 canvas in practice.
        canvas.save(out_path, "PNG", optimize=True)
        final_size = out_path.stat().st_size
    else:
        final_size = _encode_until_fits_webp(canvas, out_path)

    if final_size > _MAX_BYTES:
        raise StickerConversionError(
            f"Output still {final_size} bytes "
            f"({output_format.upper()}) — exceeds Signal's "
            f"{_MAX_BYTES}-byte limit.  Reduce the input's detail "
            "(flatter colours, fewer gradients) or pre-crop to "
            "simplify.",
        )
    return out_path


def _encode_until_fits_webp(canvas: Image.Image, out_path: Path) -> int:
    """Iteratively drop WebP quality until the file is ≤
    ``_TARGET_BYTES``.  Stops early once the target is hit;
    otherwise keeps going and returns the final size for the
    caller to compare against ``_MAX_BYTES``.
    """
    last_size = -1
    for q in _QUALITY_LADDER:
        canvas.save(out_path, "WEBP", quality=q, method=6)
        last_size = out_path.stat().st_size
        if last_size <= _TARGET_BYTES:
            return last_size
    return last_size
