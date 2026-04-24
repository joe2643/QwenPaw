# -*- coding: utf-8 -*-
"""Image → Signal / WhatsApp sticker-format WebP conversion.

Shared core so both the agent tool wrapper
(``signal_sticker.signal_prepare_sticker_webp``) and the standalone
skill script (``skills/signal_sticker-en/scripts/prepare_sticker_webp.py``)
converge on one definition of "valid sticker output".

Output guarantees every caller can rely on:

* 512×512 exactly.
* ``RIFF…WEBP`` magic.
* Alpha preserved (transparent pad when the source isn't square).
* File size ≤ ``_MAX_BYTES`` (300 KB — Signal's upload ceiling;
  Desktop typically produces ≤100 KB, so we aim for that first and
  degrade quality step-by-step until we fit).

Deliberately stdlib + Pillow only — no CoPaw imports — so the skill
script can execute against any Python environment with Pillow
installed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

from PIL import Image

_SIZE = 512
_MAX_BYTES = 300 * 1024
_TARGET_BYTES = 100 * 1024
# Quality ladder — each step drops ~30% bytes on typical AI-gen art.
# We stop as soon as we're under ``_TARGET_BYTES`` (nice-to-have) but
# the hard requirement is ``_MAX_BYTES``.  Values chosen empirically;
# anything under 30 looks blocky.
_QUALITY_LADDER = (95, 85, 75, 65, 55, 45, 35)


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
) -> Path:
    """Convert an image at ``input_path`` into a sticker-valid WebP.

    Args:
        input_path:
            Any Pillow-decodable image: PNG, JPG, WEBP, GIF (first
            frame), BMP, etc.
        output_path:
            Destination file.  When ``None``, the output lands next
            to the source with a ``.sticker.webp`` suffix (chosen
            to match the WhatsApp channel's sticker-dispatch
            convention — files matching that suffix are routed
            through ``send_sticker`` automatically).

    Returns:
        Absolute :class:`Path` of the written file.

    Raises:
        :class:`FileNotFoundError`:
            Input file doesn't exist.
        :class:`PIL.UnidentifiedImageError`:
            Input isn't a decodable image.
        :class:`StickerConversionError`:
            Even the lowest quality step didn't fit under 300 KB.
    """
    src_path = Path(input_path).expanduser().resolve()
    if not src_path.is_file():
        raise FileNotFoundError(f"sticker source not found: {src_path}")

    if output_path is None:
        # Strip arbitrary existing suffixes and always land on the
        # canonical ``.sticker.webp`` so the file immediately routes
        # through WhatsApp's ``.sticker.webp`` convention and the
        # Signal upload pipeline.  Preserves the original stem.
        out_path = src_path.with_name(f"{src_path.stem}.sticker.webp")
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

    final_size = _encode_until_fits(canvas, out_path)
    if final_size > _MAX_BYTES:
        raise StickerConversionError(
            f"Output still {final_size} bytes after quality=35 — "
            f"exceeds Signal's {_MAX_BYTES}-byte limit.  Reduce the "
            "input's detail (flatter colours, fewer gradients) or "
            "pre-crop to simplify.",
        )
    return out_path


def _encode_until_fits(canvas: Image.Image, out_path: Path) -> int:
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
