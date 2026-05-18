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

import logging
import os
from pathlib import Path
from typing import Literal, Optional, Union

from PIL import Image

logger = logging.getLogger(__name__)

_SIZE = 512
_MAX_BYTES = 300 * 1024
_TARGET_BYTES = 100 * 1024
# WebP quality ladder — each step drops ~30% bytes on typical AI-gen
# art.  We stop as soon as we're under ``_TARGET_BYTES`` (nice-to-
# have) but the hard requirement is ``_MAX_BYTES``.  Values chosen
# empirically; anything under 30 looks blocky.
_QUALITY_LADDER = (95, 85, 75, 65, 55, 45, 35)

# Signal animated-sticker spec (verified 2026-05 against signalapp
# support docs + laggykiller/sticker-convert's battle-tested preset):
#   * APNG only (NOT animated WebP — Signal Android renders animated
#     WebP the same way it mis-renders static WebP, as a voice-
#     message blob).
#   * ≤300 KB (same cap as static).
#   * ≤3000 ms total duration.
#   * 512×512 px, transparent background.
#   * 1–30 fps.
#   * Palette colour count 32–257 (the encoder's effective tuning knob
#     for size — RGBA is too fat to fit any real animation in 300 KB).
_APNG_MAX_DURATION_MS = 3000
_APNG_MAX_FPS = 30
# Ladder of (mode, colours_or_None, frame_decimation_factor) — try
# each in order, stop at first that fits ``_MAX_BYTES``.
#
# Why we lead with RGBA: per-frame adaptive palette quantization (the
# previous strategy) caused two visible artifacts:
#   * inter-frame palette drift — frame N's palette ≠ frame N+1's
#     palette, so flat colours shift hue between frames ("colour
#     wobble", reported 2026-05-12).
#   * dithering noise — FloydSteinberg adds spatial speckles that
#     turn flat sticker fills into grain.
# RGBA-mode APNG sidesteps both because there's no palette at all.
# It's bigger on disk, but cartoony stickers compress well under
# Pillow's PNG IDAT codec — many fit ≤300 KB without quantization.
#
# When RGBA doesn't fit, we drop to ``P_SHARED``: compute ONE
# palette from all frames concatenated, then map every frame
# against that shared palette.  Inter-frame drift disappears (same
# palette indices mean the same colours) and we can crank the dither
# down because the palette is already adaptive to the actual frame
# content.  Far better than per-frame quantize.
#
# ``decimation`` drops every k-th frame to halve / third / quarter
# the file size while keeping animation length constant (neighbour
# durations get coalesced — see ``_encode_until_fits_apng``).
_APNG_LADDER: tuple[tuple[str, int | None, int], ...] = (
    ("RGBA", None, 1),  # full fidelity, all frames
    ("RGBA", None, 2),  # half frames, still no palette
    ("P_SHARED", 256, 1),  # shared 256-colour palette, all frames
    ("P_SHARED", 128, 1),
    ("P_SHARED", 64, 1),
    ("P_SHARED", 64, 2),
    ("P_SHARED", 32, 2),
    ("P_SHARED", 32, 3),
    ("P_SHARED", 16, 3),
    ("P_SHARED", 16, 4),
)

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

    # Animated source + PNG target → APNG path (preserves animation,
    # which is what Signal Android needs to actually play it back).
    # APNG shares PNG's magic bytes + IHDR layout, so downstream
    # validators (``_detect_sticker_format`` / ``_read_png_dimensions``
    # / ``_validate_sticker_image``) accept it as-is and the staging
    # pipeline writes it with ``contentType: image/png`` — exactly
    # what Signal's CDN + clients expect for animated stickers.
    # WebP output skips animation: even animated WebP triggers the
    # Signal Android voice-message rendering bug, so there's no
    # animated WebP path worth implementing.
    with Image.open(src_path) as raw:
        is_animated = (
            getattr(raw, "is_animated", False)
            and getattr(raw, "n_frames", 1) >= 2
        )
        if is_animated and output_format == "png":
            frames_with_durations = _read_animated_frames(raw)
        else:
            # GIF / multi-frame WebP forced into static output: take
            # frame 0 so the agent still gets something useful.
            if is_animated:
                raw.seek(0)
            frames_with_durations = None
            # Ensure RGBA so the transparent pad step works for
            # opaque JPGs as well as alpha-bearing PNGs.
            img = raw.convert("RGBA")

    if frames_with_durations is not None:
        # Animated APNG path.  Each frame is letterboxed to
        # _SIZE×_SIZE; the ladder encoder handles the 300 KB cap.
        canvases = [
            _letterbox_rgba(f, _SIZE) for f, _ in frames_with_durations
        ]
        durations = [d for _, d in frames_with_durations]
        canvases, durations = _enforce_duration_and_fps(canvases, durations)
        final_size = _encode_until_fits_apng(canvases, durations, out_path)
    else:
        # Static path — fit into SIZExSIZE preserving aspect,
        # transparent-pad the rest (Signal recommends the sticker
        # visually fill the canvas but letterboxing is accepted —
        # square crop would distort so avoid it).
        canvas = _letterbox_rgba(img, _SIZE)
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


def _letterbox_rgba(img: Image.Image, size: int) -> Image.Image:
    """Fit ``img`` into a square ``size``×``size`` RGBA canvas,
    preserving aspect ratio and transparent-padding the rest.
    """
    rgba = img.convert("RGBA")
    rgba.thumbnail((size, size), Image.LANCZOS)
    canvas = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    offset = ((size - rgba.width) // 2, (size - rgba.height) // 2)
    canvas.paste(rgba, offset, rgba)
    return canvas


def _read_animated_frames(
    raw: Image.Image,
) -> list[tuple[Image.Image, int]]:
    """Return ``[(frame_rgba, duration_ms), ...]`` for an open
    animated image.  Durations missing from frame info default to
    100 ms (10 fps) — same fallback Pillow uses internally.

    Caller is responsible for keeping ``raw`` open while iterating;
    we copy each frame to a fresh image so the result outlives the
    file handle.
    """
    n = getattr(raw, "n_frames", 1)
    out: list[tuple[Image.Image, int]] = []
    for i in range(n):
        raw.seek(i)
        # ``info["duration"]`` is per-frame ms.  GIF/animated-WebP
        # both populate it; APNG inputs do via Pillow's ``fcTL``
        # parsing.  Zero means "infinite display" in some formats,
        # which doesn't translate well to APNG — clamp to 1 frame.
        dur = int(raw.info.get("duration") or 100)
        if dur <= 0:
            dur = 100
        out.append((raw.convert("RGBA").copy(), dur))
    return out


def _enforce_duration_and_fps(
    canvases: list[Image.Image],
    durations: list[int],
) -> tuple[list[Image.Image], list[int]]:
    """Apply Signal's animated-sticker time constraints:

    * Total animation length ≤ ``_APNG_MAX_DURATION_MS`` (3 s).
      Trim trailing frames once the running total would exceed.
    * Effective fps ≤ ``_APNG_MAX_FPS`` (30).  If the input plays
      faster on average, decimate uniformly and scale per-frame
      durations so total wall-clock time stays the same.
    """
    if not canvases:
        return canvases, durations

    # Step 1: cap total duration.
    kept_c: list[Image.Image] = []
    kept_d: list[int] = []
    running = 0
    for c, d in zip(canvases, durations):
        if running + d > _APNG_MAX_DURATION_MS and kept_c:
            # Spending more time on the last kept frame is preferable
            # to truncating mid-animation; let the existing frame
            # absorb the remainder.
            kept_d[-1] += _APNG_MAX_DURATION_MS - running
            break
        kept_c.append(c)
        kept_d.append(min(d, _APNG_MAX_DURATION_MS - running))
        running += kept_d[-1]
        if running >= _APNG_MAX_DURATION_MS:
            break
    if not kept_c:
        kept_c = [canvases[0]]
        kept_d = [min(durations[0], _APNG_MAX_DURATION_MS)]

    # Step 2: cap fps.  ``avg_ms_per_frame`` < 1000/_APNG_MAX_FPS
    # means we're playing faster than allowed — drop every k-th
    # frame to bring it in line.
    total = sum(kept_d) or 1
    avg_ms = total / max(1, len(kept_c))
    min_ms_per_frame = 1000.0 / _APNG_MAX_FPS
    if avg_ms < min_ms_per_frame and len(kept_c) > 1:
        k = max(2, int(round(min_ms_per_frame / max(avg_ms, 1))))
        dec_c = kept_c[::k]
        # Coalesce dropped frames' durations into the surviving
        # neighbours so total animation length is preserved.
        dec_d: list[int] = []
        for i in range(0, len(kept_c), k):
            chunk = kept_d[i : i + k]
            dec_d.append(sum(chunk))
        kept_c, kept_d = dec_c, dec_d
    return kept_c, kept_d


def _decimate_frames(
    canvases: list[Image.Image],
    durations: list[int],
    factor: int,
) -> tuple[list[Image.Image], list[int]]:
    """Drop every ``factor``-th frame, coalescing neighbour durations
    so wall-clock animation length is preserved.  ``factor=1`` is a
    no-op.
    """
    if factor == 1:
        return canvases, list(durations)
    out_c = canvases[::factor]
    out_d = []
    for i in range(0, len(canvases), factor):
        out_d.append(sum(durations[i : i + factor]))
    return out_c, out_d


def _have_pngquant() -> bool:
    """Cached: is the system ``pngquant`` binary available?

    ``shutil.which`` is fast (PATH scan) so a per-call lookup is fine;
    we still cache the boolean to keep the encoder hot-path tight when
    running tens of stickers in a batch.
    """
    global _PNGQUANT_AVAILABLE
    if _PNGQUANT_AVAILABLE is None:
        import shutil

        _PNGQUANT_AVAILABLE = shutil.which("pngquant") is not None
    return _PNGQUANT_AVAILABLE


_PNGQUANT_AVAILABLE: bool | None = None


def _shared_palette_quantize_pngquant(
    frames: list[Image.Image],
    colours: int,
) -> list[Image.Image] | None:
    """Quantize via the system ``pngquant`` binary (libimagequant under
    the hood — the same library every serious sticker tool uses).
    Returns the per-frame P-mode list on success, ``None`` on any
    failure so the caller can fall back to Pillow's FASTOCTREE.

    Strategy: same stack-then-crop architecture as the Pillow path —
    we vertically stack RGBA frames into one tall canvas, run pngquant
    on it to get a P-mode result with ONE master palette, then crop
    back into per-frame regions.  pngquant defaults to Floyd-Steinberg
    dither (``--floyd``) which actually *applies* (unlike Pillow's
    FASTOCTREE which silently ignores the dither parameter), so
    gradients stay smooth instead of banding.
    """
    import subprocess
    import tempfile

    if not frames:
        return []
    W, H = frames[0].size
    stacked = Image.new("RGBA", (W, H * len(frames)), (0, 0, 0, 0))
    for i, f in enumerate(frames):
        stacked.paste(f, (0, i * H), f)

    in_fd, in_path_s = tempfile.mkstemp(suffix=".png", prefix="apng-in-")
    out_fd, out_path_s = tempfile.mkstemp(suffix=".png", prefix="apng-out-")
    os.close(in_fd)
    os.close(out_fd)
    in_path = Path(in_path_s)
    out_path = Path(out_path_s)
    try:
        stacked.save(in_path, "PNG")
        # ``--speed=1`` = best quality (slowest); for sticker batches
        # of <20 frames the wall-time cost is negligible vs visual win.
        # ``--strip`` removes ancillary chunks (iCCP, tEXt, etc.) we
        # don't need — saves a few hundred bytes per sticker.
        # We deliberately do NOT pass ``--quality`` so pngquant uses
        # its full quality range; if it can't represent the image at
        # ``colours`` count it'll just produce slightly more error
        # rather than refuse (which is what we want — a banded result
        # is better than no result).
        result = subprocess.run(
            [
                "pngquant",
                "--force",
                "--speed=1",
                "--strip",
                "--output",
                str(out_path),
                str(colours),
                str(in_path),
            ],
            capture_output=True,
            timeout=60,
        )
        if result.returncode != 0:
            logger.warning(
                "pngquant failed (rc=%d): %s",
                result.returncode,
                result.stderr.decode("utf-8", errors="replace")[:300],
            )
            return None
        if not out_path.exists() or out_path.stat().st_size == 0:
            logger.warning("pngquant produced no output for stacked frames")
            return None
        quantized = Image.open(out_path).copy()
        if quantized.mode != "P":
            logger.warning(
                "pngquant output unexpectedly in %s mode (expected P)",
                quantized.mode,
            )
            return None
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as e:
        logger.warning(
            "pngquant invocation error (%s): %s",
            type(e).__name__,
            e,
        )
        return None
    finally:
        try:
            in_path.unlink()
        except OSError:
            pass
        try:
            out_path.unlink()
        except OSError:
            pass

    return [
        quantized.crop((0, i * H, W, (i + 1) * H)).copy()
        for i in range(len(frames))
    ]


def _shared_palette_quantize(
    frames: list[Image.Image],
    colours: int,
) -> list[Image.Image]:
    """Compute ONE adaptive palette from all frames combined, then
    return each frame in P-mode sharing that single palette.

    Prefers the system ``pngquant`` binary (libimagequant — far better
    than Pillow's FASTOCTREE: real Floyd-Steinberg dither + better
    adaptive palette selection) when available, falling back to a
    Pillow-only path so the encoder still works on systems without
    pngquant installed.

    Why shared palette beats per-frame ``quantize(colors=N)``:

    * Per-frame quantize picks a *different* adaptive palette for
      each frame.  Two adjacent frames showing the same flat
      sticker colour can pick subtly different palette entries for
      it, and the result on playback is visible hue wobble.
    * Sharing one palette across all frames means a given source
      RGB value maps to the same palette index in every frame —
      identical colour reads as identical on screen.

    Implementation: vertically stack frames into one tall canvas,
    quantize that whole canvas once to get a P-mode master, then
    *crop* the master back into per-frame regions.  Each crop is a
    view into the same palette — they share the index→colour table
    by construction.
    """
    if _have_pngquant():
        result = _shared_palette_quantize_pngquant(frames, colours)
        if result is not None:
            return result
        # Fall through to Pillow path on pngquant failure.
        logger.info("pngquant fallback to Pillow FASTOCTREE")

    if not frames:
        return []
    W, H = frames[0].size
    stacked = Image.new("RGBA", (W, H * len(frames)), (0, 0, 0, 0))
    for i, f in enumerate(frames):
        stacked.paste(f, (0, i * H), f)
    master = stacked.quantize(
        colors=colours,
        method=Image.Quantize.FASTOCTREE,
        dither=Image.Dither.NONE,
    )
    # Slice the quantized stack back into per-frame regions.  Each
    # crop is P-mode sharing the master's palette table (verified:
    # ``slice.getpalette() == master.getpalette()``).
    return [
        master.crop((0, i * H, W, (i + 1) * H)).copy()
        for i in range(len(frames))
    ]


def _save_apng(
    out_path: Path,
    frames: list[Image.Image],
    durations: list[int],
) -> int:
    """Save ``frames`` as APNG and return the resulting file size.

    APNG-specific writer flags:
    * ``save_all=True, format="PNG"`` — Pillow's APNG codec lives
      on the PNG handler; ``save_all`` is what flips it into APNG.
    * ``loop=0`` — infinite loop, matching Signal Desktop's own
      sticker creator.
    * ``disposal=0`` + ``blend=0`` (OP_NONE + OP_SOURCE) — each
      frame fully overwrites the previous, no compositing.  We
      deliberately avoid ``disposal=2``: Pillow's APNG writer
      pastes a transparent "dispose" buffer between frames and
      that paste fails with ``ValueError: images do not match``
      for P-mode (palette) frames.
    * ``optimize=True`` is intentionally NOT set for P-mode frames
      because it silently lifts them back to RGBA, defeating the
      palette compression.  For RGBA frames we *do* turn it on —
      it shaves real bytes off the IDAT/fdAT streams without
      touching pixel values.
    """
    if not frames:
        raise ValueError("cannot save APNG with no frames")
    head, *tail = frames
    save_kwargs: dict[str, object] = dict(
        format="PNG",
        save_all=True,
        append_images=tail,
        duration=durations,
        loop=0,
        disposal=0,
        blend=0,
    )
    if head.mode == "RGBA":
        save_kwargs["optimize"] = True
    head.save(out_path, **save_kwargs)
    return out_path.stat().st_size


def _encode_until_fits_apng(
    canvases: list[Image.Image],
    durations: list[int],
    out_path: Path,
) -> int:
    """Try each rung of ``_APNG_LADDER`` until the encoded APNG is
    ≤ ``_MAX_BYTES``.  Returns the final size for the caller to
    compare against the hard cap (``StickerConversionError`` is
    raised by the outer function if it still doesn't fit on the
    last rung).

    Rungs are ordered by fidelity, top-down:

    1. RGBA APNG with every frame — most faithful, biggest file.
    2. RGBA with every-other frame — for stickers that are *just*
       over cap at full-RGBA but stay sharp with fewer frames.
    3. Shared-palette quantization, 256 colours down to 16.
       Shared palette eliminates the inter-frame colour drift the
       previous per-frame quantize caused.
    4. Same with progressive frame decimation.

    Anything more aggressive than the last rung is better surfaced
    as a clear failure so the user can simplify the source.
    """
    last_size = -1
    for mode, colours, decimation in _APNG_LADDER:
        dec_canvases, dec_durations = _decimate_frames(
            canvases,
            durations,
            decimation,
        )
        if not dec_canvases:
            continue
        if mode == "RGBA":
            frames = dec_canvases
        else:  # P_SHARED
            assert colours is not None
            frames = _shared_palette_quantize(dec_canvases, colours)
        last_size = _save_apng(out_path, frames, dec_durations)
        if last_size <= _MAX_BYTES:
            return last_size
    return last_size


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
