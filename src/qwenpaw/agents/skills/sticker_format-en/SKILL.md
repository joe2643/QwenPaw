---
name: sticker_format
description: "Use this skill whenever you need to convert any image (PNG, JPG, WEBP, GIF, BMP) into a valid messenger-sticker file. Default output is PNG (Signal-friendly); pass --format webp only when targeting WhatsApp. Triggers: the user asked you to 'make a sticker', 'send this as a sticker', 'turn this image into a sticker'; you just generated an image via codex image gen / dalle / any model and intend to feed it into Signal/WhatsApp sticker pipelines (signal_create_sticker_pack, signal_add_stickers_to_pack, `.sticker.webp` for WhatsApp); an existing sticker-tool call rejected your input with 'not PNG or WebP' / '512x512' / 'too large' errors. Also use when you want to pre-check whether an image would pass sticker validation without uploading. Do NOT use for decorative image processing, thumbnails, or non-sticker resize tasks."
license: Proprietary. See repo LICENSE.
metadata:
  builtin_skill_version: "1.1"
---

# Sticker format conversion

Convert any image into a valid messenger-sticker file:

* **512×512** exactly (transparent-padded when the source isn't square)
* **PNG** (default) or **WebP** (opt-in for WhatsApp); both checked by
  `signal_create_sticker_pack`'s preflight.
* **≤300 KB** (Signal's absolute ceiling; PNG with optimize=True
  almost always fits, WebP uses a quality ladder targeting ≤100 KB
  first and stepping down if needed)
* Alpha preserved when the source has it

## Why PNG by default

Signal's spec accepts both PNG and WebP, but in practice Signal
**Android** renders user-uploaded WebP stickers as **voice-message**
blobs unless the WebP came from Signal Desktop's own creator —
reproducible regardless of VP8L vs VP8X encoding.  PNG is the
format proven-working third-party packs (e.g. LIHKG Dog) use, and
renders cleanly on every Signal client.  WhatsApp still requires
the `.sticker.webp` filename convention to route through its
send-as-sticker path; pass `--format webp` for that case.

## When to use

You almost always need this between image generation and sticker
send, because the downstream sticker tools reject anything that
isn't already in format. Example end-to-end pipeline:

1. Agent generates an image via codex image gen → `/tmp/out.png`
   (typically 1024×1024 PNG)
2. **Run this skill** → `/tmp/out.sticker.png`
3. Feed into `signal_create_sticker_pack` or
   `signal_add_stickers_to_pack` (Signal — accepts PNG or WebP),
   or convert with `--format webp` and pass to `send_file_to_user`
   for WhatsApp (the `.sticker.webp` suffix auto-routes through
   the sticker send path).

## Running the conversion

From this skill directory:

```bash
python scripts/prepare_sticker_webp.py --input /path/to/source.png
# → writes /path/to/source.sticker.png  (default: Signal-friendly PNG)
```

Explicit format / output path:

```bash
# Default PNG, custom output path
python scripts/prepare_sticker_webp.py \
    --input /path/to/source.png \
    --output /path/to/out.sticker.png

# WebP for WhatsApp's send-as-sticker filename convention
python scripts/prepare_sticker_webp.py \
    --input /path/to/source.png \
    --format webp
# → writes /path/to/source.sticker.webp
```

Exit codes:

* `0`: success — wrote the sticker file.
* `1`: file not found / not a decodable image / stays >300 KB —
  stderr carries the reason.

## Input handling

* **PNG/JPG**: direct.
* **Animated GIF / WebP**: first frame only — animated stickers
  need a different libwebp / APNG code path (not yet wired).
* **JPG without alpha**: gets an RGBA conversion so the transparent
  pad works; no visual change.

## Failure modes

* **Oversize PNG**: very rare; the optimized PNG encoder pretty
  much always fits 512×512 under 300 KB.  If it fails, the source
  has so many distinct colours that no encoding will compress it
  enough — simplify (flatten backgrounds, reduce gradients) and
  retry.
* **Oversize WebP**: complex gradients or photographic noise can
  resist compression.  Quality steps: 95 → 85 → 75 → 65 → 55 → 45
  → 35.  If all fail, simplify the source before retrying.
* **Bad input**: "unidentified image" means Pillow couldn't parse
  it — confirm the file is what you think it is.

## Python surface

If your agent context has access to the CoPaw package (any agent
inside a CoPaw runner), you can skip the shell and import:

```python
from qwenpaw.agents.tools.sticker_convert import prepare_sticker_webp
# Default: PNG
prepare_sticker_webp("/tmp/out.png", "/tmp/out.sticker.png")
# Opt-in WebP for WhatsApp
prepare_sticker_webp("/tmp/out.png", "/tmp/out.sticker.webp",
                     output_format="webp")
```

The stdlib-only CLI (`scripts/prepare_sticker_webp.py`) exists for
agents outside the CoPaw venv or running in restricted sandboxes.

## Next step: sending on Signal

This skill stops at producing a valid sticker file.  For the
Signal-side workflow — discovering packs, previewing, sending,
creating your own pack from the files produced here — see the
**`signal_stickers`** skill.  WhatsApp is even simpler: just pass
the `.sticker.webp` to `send_file_to_user`; the filename suffix
auto-routes through the sticker send path.
