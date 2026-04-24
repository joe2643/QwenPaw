---
name: sticker_format
description: "Use this skill whenever you need to convert any image (PNG, JPG, WEBP, GIF, BMP) into a valid messenger-sticker WebP file. Triggers: the user asked you to 'make a sticker', 'send this as a sticker', 'turn this image into a sticker'; you just generated an image via codex image gen / dalle / any model and intend to feed it into Signal/WhatsApp sticker pipelines (signal_create_sticker_pack, signal_add_stickers_to_pack, `.sticker.webp` for WhatsApp); an existing sticker-tool call rejected your input with 'not a WebP' / '512x512' / 'too large' errors. Also use when you want to pre-check whether an image would pass sticker validation without uploading. Do NOT use for decorative image processing, thumbnails, or non-sticker resize tasks."
license: Proprietary. See repo LICENSE.
metadata:
  builtin_skill_version: "1.0"
---

# Sticker format conversion

Convert any image into a valid messenger-sticker WebP:

* **512×512** exactly (transparent-padded when the source isn't square)
* **RIFF/WEBP** magic (what `signal_create_sticker_pack`'s preflight
  checks for)
* **≤300 KB** (Signal's absolute ceiling; we aim ≤100 KB first and
  step quality down only if needed)
* Alpha preserved when the source has it

## When to use

You almost always need this between image generation and sticker
send, because the downstream sticker tools reject anything that
isn't already in format. Example end-to-end pipeline:

1. Agent generates an image via codex image gen → `/tmp/out.png`
   (typically 1024×1024 PNG)
2. **Run this skill** → `/tmp/out.sticker.webp`
3. Feed into `signal_create_sticker_pack` or
   `signal_add_stickers_to_pack` (Signal) or send via
   `send_file_to_user` for WhatsApp (the `.sticker.webp` suffix
   auto-routes through the sticker send path)

## Running the conversion

From this skill directory:

```bash
python scripts/prepare_sticker_webp.py --input /path/to/source.png
# → writes /path/to/source.sticker.webp
```

Explicit output path:

```bash
python scripts/prepare_sticker_webp.py \
    --input /path/to/source.png \
    --output /path/to/out.sticker.webp
```

Exit codes:

* `0`: success — wrote the sticker file.
* `1`: file not found / not a decodable image / stays >300 KB even
  at quality=35 — stderr carries the reason.

## Input handling

* **PNG/JPG**: direct.
* **Animated GIF / WebP**: first frame only — animated stickers
  need a different libwebp code path (not yet wired).
* **JPG without alpha**: gets an RGBA conversion so the transparent
  pad works; no visual change.

## Failure modes

* **Oversize output**: complex gradients or photographic noise can
  resist compression.  Quality steps: 95 → 85 → 75 → 65 → 55 → 45
  → 35.  If all fail, simplify the source (flatten backgrounds,
  reduce detail) before retrying.
* **Bad input**: "unidentified image" means Pillow couldn't parse
  it — confirm the file is what you think it is.

## Python surface

If your agent context has access to the CoPaw package (any agent
inside a CoPaw runner), you can skip the shell and import:

```python
from qwenpaw.agents.tools.sticker_convert import prepare_sticker_webp
prepare_sticker_webp("/tmp/out.png", "/tmp/out.sticker.webp")
```

The stdlib-only CLI (`scripts/prepare_sticker_webp.py`) exists for
agents outside the CoPaw venv or running in restricted sandboxes.
