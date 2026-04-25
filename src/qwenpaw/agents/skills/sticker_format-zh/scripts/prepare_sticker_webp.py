#!/usr/bin/env python3
"""Convert an image into a Signal/WhatsApp-valid sticker file.

Standalone — depends only on stdlib + Pillow, no CoPaw imports, so
this script can run in any Python environment that has Pillow
installed.  The logic mirrors
``qwenpaw.agents.tools.sticker_convert.prepare_sticker_webp`` and
must stay in lockstep with it.

Default output is **PNG**.  Signal Android renders user-uploaded
WebP stickers as voice-message blobs when the WebP isn't from
Signal Desktop's own creator (independent of VP8L / VP8X
encoding); PNG is the format proven-working third-party packs
ship and renders cleanly across every Signal client.  Pass
``--format webp`` only when targeting WhatsApp (which requires
the ``.sticker.webp`` filename convention).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    from PIL import Image
except ImportError:
    print(
        "Error: Pillow is required. Install with: pip install pillow",
        file=sys.stderr,
    )
    sys.exit(1)

SIZE = 512
MAX_BYTES = 300 * 1024
TARGET_BYTES = 100 * 1024
QUALITY_LADDER = (95, 85, 75, 65, 55, 45, 35)


def _prepare_canvas(src_path: Path) -> Image.Image:
    with Image.open(src_path) as raw:
        if getattr(raw, "is_animated", False):
            raw.seek(0)
        img = raw.convert("RGBA")
    img.thumbnail((SIZE, SIZE), Image.LANCZOS)
    canvas = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    offset = ((SIZE - img.width) // 2, (SIZE - img.height) // 2)
    canvas.paste(img, offset, img)
    return canvas


def _prepare_png(src_path: Path, out_path: Path) -> int:
    canvas = _prepare_canvas(src_path)
    canvas.save(out_path, "PNG", optimize=True)
    return out_path.stat().st_size


def _prepare_webp(src_path: Path, out_path: Path) -> int:
    canvas = _prepare_canvas(src_path)
    last_size = -1
    for q in QUALITY_LADDER:
        canvas.save(out_path, "WEBP", quality=q, method=6)
        last_size = out_path.stat().st_size
        if last_size <= TARGET_BYTES:
            return last_size
    return last_size


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--input", "-i", required=True,
        help="Source image (PNG/JPG/WebP/GIF).",
    )
    ap.add_argument(
        "--output", "-o",
        help=(
            "Destination file. Default: source path with the stem "
            "unchanged and the suffix replaced by '.sticker.png' "
            "(or '.sticker.webp' if --format webp)."
        ),
    )
    ap.add_argument(
        "--format", "-f",
        choices=("png", "webp"),
        default="png",
        help=(
            "Output format. Default 'png' — works on every Signal "
            "client. Use 'webp' for WhatsApp's send-as-sticker "
            "filename convention."
        ),
    )
    args = ap.parse_args()

    src = Path(args.input).expanduser().resolve()
    if not src.is_file():
        print(f"Error: input not found: {src}", file=sys.stderr)
        return 1

    suffix = ".sticker.png" if args.format == "png" else ".sticker.webp"
    if args.output:
        out = Path(args.output).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
    else:
        out = src.with_name(f"{src.stem}{suffix}")

    try:
        if args.format == "png":
            size = _prepare_png(src, out)
        else:
            size = _prepare_webp(src, out)
    except Exception as e:
        print(f"Error: {type(e).__name__}: {e}", file=sys.stderr)
        return 1

    if size > MAX_BYTES:
        print(
            f"Error: output is {size} bytes ({args.format.upper()}); "
            f"exceeds Signal's {MAX_BYTES}-byte limit.  Simplify "
            "the source image (flatten colours, reduce detail) "
            "and retry.",
            file=sys.stderr,
        )
        # Keep the oversize file so the caller can inspect / manually
        # further-optimise it if desired; exit non-zero so the agent
        # knows not to feed it into a sticker pack upload.
        return 1

    print(str(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
