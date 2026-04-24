# -*- coding: utf-8 -*-
# flake8: noqa: E501
# pylint: disable=line-too-long,too-many-return-statements
"""Signal sticker-pack management tools exposed to the agent.

All five functions auto-lock to the currently-running agent's Signal
channel via :mod:`qwenpaw.app.agent_context` — no cross-channel /
cross-account footguns; agents can't accidentally operate a Signal
account that isn't theirs.  If the agent has no Signal channel
configured (or the subprocess isn't connected), every tool returns
an error ``TextBlock`` rather than raising.

Pack lifecycle in one paragraph:
    ``list`` shows whatever signal-cli has recorded under this
    account (installed + uploaded).  ``preview`` fetches a specific
    sticker's webp bytes so the agent can see what it looks like.
    ``install`` wraps ``addStickerPack`` so the agent can accept a
    pack someone shared.  ``create`` stages local webp files +
    manifest.json into ``{media_dir}/sticker_pack_staging/<uuid>/``
    and calls ``uploadStickerPack``, returning the new ``pack_id`` /
    ``pack_key``.  ``send`` takes a pack reference (``pack_id`` +
    ``sticker_id``) and dispatches it as a true Signal sticker (not
    just an attached webp) to the target handle.
"""

from __future__ import annotations

import json
import logging
import shutil
import struct
import uuid
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from agentscope.message import ImageBlock, TextBlock
from agentscope.tool import ToolResponse

logger = logging.getLogger(__name__)

# Signal's upload service caps each sticker and enforces a square
# layout — we preflight locally so failures come back as a specific
# tool error instead of a cryptic upload 500.  300 KB is the public
# upper bound; Desktop typically targets ≤100 KB but accepts up to
# 300.  512×512 is the documented display size; deviating usually
# produces a pack that renders fine on Desktop but rejects on
# iOS/Android.
_STICKER_MAX_BYTES = 300 * 1024
_STICKER_WIDTH = 512
_STICKER_HEIGHT = 512


def _err(text: str) -> ToolResponse:
    return ToolResponse(content=[TextBlock(type="text", text=text)])


def _ok_text(text: str) -> ToolResponse:
    return ToolResponse(content=[TextBlock(type="text", text=text)])


async def _get_signal_channel() -> Any:
    """Resolve the Signal channel for the currently-running agent.

    Raises ``LookupError`` with a human-facing message when there's
    no active agent context, no Signal channel configured, or the
    signal-cli subprocess isn't connected.  Caller maps that to a
    ``ToolResponse`` error.
    """
    from ...app.agent_context import get_current_agent_id
    from ...app.multi_agent_manager import MultiAgentManager

    agent_id = get_current_agent_id()
    if not agent_id:
        raise LookupError(
            "No active agent context — this tool must run inside a "
            "live agent session (not standalone / not via a worker).",
        )
    workspace = await MultiAgentManager().get_agent(agent_id)
    cm = getattr(workspace, "channel_manager", None)
    if cm is None:
        raise LookupError(
            f"Agent '{agent_id}' has no channel manager — Signal "
            "isn't reachable from this workspace.",
        )
    channel = await cm.get_channel("signal")
    if channel is None:
        raise LookupError(
            f"Agent '{agent_id}' has no Signal channel configured. "
            "Enable channels.signal in agent.json and re-link the "
            "account before calling sticker tools.",
        )
    if not getattr(channel, "enabled", False):
        raise LookupError(
            "Signal channel is present but disabled in config — "
            "toggle it on to use sticker tools.",
        )
    client = getattr(channel, "client", None)
    if client is None or not getattr(client, "connected", False):
        raise LookupError(
            "Signal subprocess isn't connected (signal-cli crashed, "
            "wasn't linked, or is mid-restart).  Check channel "
            "health and retry.",
        )
    return channel


def _parse_signal_art_url(url: str) -> tuple[str, str] | None:
    """Extract ``(pack_id, pack_key)`` from a signal.art share URL.

    Signal uses the URL *fragment* (after ``#``) rather than the
    query string, so :mod:`urlparse` leaves that in ``.fragment``.
    Returns ``None`` when either field is missing — callers should
    treat that as a failed upload.
    """
    try:
        parsed = urlparse(url)
    except Exception:
        return None
    frag = parsed.fragment or ""
    if not frag:
        return None
    qs = parse_qs(frag)
    pid = (qs.get("pack_id") or [""])[0]
    key = (qs.get("pack_key") or [""])[0]
    if not pid or not key:
        return None
    return pid, key


def _read_webp_dimensions(data: bytes) -> tuple[int, int] | None:
    """Parse width/height from a WebP file's leading chunks.

    Supports the three common variants:
        VP8  (lossy, 10-byte frame header, 14-bit dimensions)
        VP8L (lossless, 4-byte signature + packed 14-bit dims)
        VP8X (extended / animated, 24-bit dims)

    Returns ``None`` when the format is unrecognised so the caller
    can choose between blocking or warning.
    """
    if len(data) < 30 or data[0:4] != b"RIFF" or data[8:12] != b"WEBP":
        return None
    chunk = data[12:16]
    if chunk == b"VP8 ":
        # Lossy: 10-byte frame header follows the 4-byte chunk
        # size.  The 3-byte start code sits at offset 23–25 and
        # width/height are the two little-endian uint16s after it,
        # with the top 2 bits used for scaling factors.
        if len(data) < 30:
            return None
        try:
            w = struct.unpack("<H", data[26:28])[0] & 0x3FFF
            h = struct.unpack("<H", data[28:30])[0] & 0x3FFF
            return w, h
        except Exception:
            return None
    if chunk == b"VP8L":
        # Lossless: 1-byte sig (0x2F) then 32 bits packing
        # (width-1, height-1, alpha, version) low-to-high.
        if len(data) < 25 or data[20] != 0x2F:
            return None
        try:
            b1, b2, b3, b4 = data[21], data[22], data[23], data[24]
            w = ((b2 & 0x3F) << 8 | b1) + 1
            h = ((b4 & 0x0F) << 10 | b3 << 2 | (b2 >> 6)) + 1
            return w, h
        except Exception:
            return None
    if chunk == b"VP8X":
        # Extended: dims at offset 24–29 as two 24-bit values.
        if len(data) < 30:
            return None
        try:
            w = int.from_bytes(data[24:27], "little") + 1
            h = int.from_bytes(data[27:30], "little") + 1
            return w, h
        except Exception:
            return None
    return None


def _validate_sticker_webp(path: Path) -> str | None:
    """Preflight a would-be sticker file.

    Returns ``None`` when the file passes; an error string
    otherwise.  Checks applied (strict — better to fail here than
    get an opaque upload 500):

    * File exists and is a regular file.
    * Size ≤ 300 KB.
    * Starts with the ``RIFF....WEBP`` magic signature.
    * Dimensions are exactly 512×512 when we can decode them; when
      we can't (unusual VP8 variant), the check is skipped with a
      warning rather than blocking — Signal's uploader will be the
      final arbiter.
    """
    if not path.is_file():
        return f"sticker file not found: {path}"
    size = path.stat().st_size
    if size == 0:
        return f"sticker file is empty: {path}"
    if size > _STICKER_MAX_BYTES:
        return (
            f"sticker file too large: {size} bytes "
            f"(Signal max {_STICKER_MAX_BYTES} bytes / 300KB): {path}"
        )
    with open(path, "rb") as f:
        head = f.read(128)
    if len(head) < 12 or head[0:4] != b"RIFF" or head[8:12] != b"WEBP":
        return (
            f"sticker file is not a WebP image (expected RIFF/WEBP "
            f"magic): {path}"
        )
    dims = _read_webp_dimensions(head)
    if dims is None:
        logger.warning(
            "signal sticker: could not decode webp dimensions for %s — "
            "letting the upload server be the final arbiter", path,
        )
        return None
    w, h = dims
    if (w, h) != (_STICKER_WIDTH, _STICKER_HEIGHT):
        return (
            f"sticker dimensions must be {_STICKER_WIDTH}x"
            f"{_STICKER_HEIGHT} (got {w}x{h}): {path}"
        )
    return None


# ── Public tool functions ────────────────────────────────────────────

async def signal_prepare_sticker_webp(
    input_path: str,
    output_path: str | None = None,
) -> ToolResponse:
    """Convert any image into a Signal/WhatsApp sticker-format WebP.

    Bridges the gap between image generators (codex image gen /
    dalle / etc.) and the sticker pipeline — every sticker tool
    downstream requires exactly 512×512 WebP ≤300 KB, which image
    generators don't produce by default.  Delegates to the shared
    :func:`~qwenpaw.agents.tools.sticker_convert.prepare_sticker_webp`
    core (same logic as the ``sticker_format`` skill's CLI).

    Args:
        input_path (`str`):
            Source image path. Any Pillow-decodable format works
            (PNG, JPG, WEBP, GIF — first frame only — BMP).
        output_path (`str`, optional):
            Destination file. Defaults to a sibling of the input
            named ``<stem>.sticker.webp`` (which WhatsApp's send
            path recognises as "send as sticker").

    Returns:
        `ToolResponse`:
            TextBlock with the absolute output path on success, or
            a specific error on failure (file missing, unreadable
            image, or sticker still ≥300 KB after the full quality
            ladder).
    """
    from .sticker_convert import (
        StickerConversionError,
        prepare_sticker_webp as _core,
    )

    src = (input_path or "").strip()
    if not src:
        return _err("Error: input_path is required.")
    try:
        out = _core(src, output_path or None)
    except FileNotFoundError as e:
        return _err(f"Error: {e}")
    except StickerConversionError as e:
        return _err(f"Error: {e}")
    except Exception as e:
        return _err(f"Error: failed to convert image: {type(e).__name__}: {e}")
    return _ok_text(str(out))


async def signal_list_sticker_packs() -> ToolResponse:
    """List sticker packs known to this agent's Signal account.

    Shows every pack signal-cli has on file — packs installed from
    share links AND packs this account uploaded itself.  Use this
    to discover ``pack_id`` values to feed into
    ``signal_preview_sticker`` / ``signal_send_sticker``.

    Returns:
        `ToolResponse`:
            TextBlock with a JSON array of
            ``{pack_id, title, author, installed, sticker_count,
            stickers: [{id, emoji}]}``.  Empty array ``[]`` means
            the account has no packs yet.
    """
    try:
        channel = await _get_signal_channel()
    except LookupError as e:
        return _err(f"Error: {e}")

    raw = await channel.client.list_sticker_packs()
    summary = []
    for p in raw:
        if not isinstance(p, dict):
            continue
        stickers = p.get("stickers") or []
        summary.append({
            "pack_id": p.get("packId") or p.get("pack_id") or "",
            "title": p.get("title") or "",
            "author": p.get("author") or "",
            "installed": bool(p.get("installed", False)),
            "sticker_count": len(stickers),
            "stickers": [
                {
                    "id": int(s.get("id", 0)),
                    "emoji": s.get("emoji") or "",
                }
                for s in stickers
                if isinstance(s, dict)
            ],
        })
    return _ok_text(json.dumps(summary, ensure_ascii=False, indent=2))


async def signal_preview_sticker(
    pack_id: str,
    sticker_id: int,
) -> ToolResponse:
    """Fetch one sticker and return its image for the agent to inspect.

    Args:
        pack_id (`str`):
            Hex pack identifier from ``signal_list_sticker_packs``
            or a received sticker's metadata.
        sticker_id (`int`):
            Sticker index within the pack (0-based).

    Returns:
        `ToolResponse`:
            ImageBlock + TextBlock with the sticker's local path.
            The webp is written under the channel's media dir so
            downstream tools (send_file_to_user, view_media, etc.)
            can reference it by path.
    """
    try:
        channel = await _get_signal_channel()
    except LookupError as e:
        return _err(f"Error: {e}")

    try:
        sid = int(sticker_id)
    except (TypeError, ValueError):
        return _err(f"Error: sticker_id must be an integer, got {sticker_id!r}")

    pack_id = (pack_id or "").strip()
    if not pack_id:
        return _err("Error: pack_id is required.")

    path = await channel.client.get_sticker(
        pack_id, sid, channel._media_dir,
    )
    if path is None:
        return _err(
            f"Error: could not fetch sticker {pack_id[:12]}:{sid} "
            "(pack not installed and no key available, or the RPC "
            "failed — try signal_install_sticker_pack first).",
        )
    return ToolResponse(
        content=[
            ImageBlock(
                type="image",
                source={"type": "url", "url": f"file://{path}"},
            ),
            TextBlock(
                type="text",
                text=(
                    f"Sticker {pack_id[:12]}:{sid} at {path}"
                ),
            ),
        ],
    )


async def signal_install_sticker_pack(
    pack_id: str,
    pack_key: str,
) -> ToolResponse:
    """Install a sticker pack someone shared with this account.

    Equivalent to tapping a ``signal.art/addstickers/#pack_id=…&
    pack_key=…`` link on the mobile app.  After install the pack
    shows up in ``signal_list_sticker_packs`` and every sticker in
    it is fetchable via ``signal_preview_sticker`` / sendable via
    ``signal_send_sticker``.

    Args:
        pack_id (`str`):
            Hex pack identifier.
        pack_key (`str`):
            Hex pack key — required (without it the signal.art
            link can't decrypt the pack manifest).

    Returns:
        `ToolResponse`:
            TextBlock confirming install or reporting the failure.
    """
    try:
        channel = await _get_signal_channel()
    except LookupError as e:
        return _err(f"Error: {e}")

    pack_id = (pack_id or "").strip()
    pack_key = (pack_key or "").strip()
    if not pack_id or not pack_key:
        return _err("Error: both pack_id and pack_key are required.")

    ok = await channel.client.add_sticker_pack(pack_id, pack_key)
    if not ok:
        return _err(
            f"Error: addStickerPack failed for pack {pack_id[:12]}. "
            "Common causes: wrong pack_key, pack was deleted, or "
            "signal-cli version mismatch.",
        )
    return _ok_text(
        f"Installed sticker pack {pack_id[:12]}... "
        "(visible in signal_list_sticker_packs now)."
    )


async def signal_create_sticker_pack(
    title: str,
    author: str,
    stickers: list[dict],
) -> ToolResponse:
    """Create and upload a new sticker pack from local webp files.

    Stages the manifest + numbered ``<id>.webp`` copies under
    ``{media_dir}/sticker_pack_staging/<uuid>/`` (kept after upload
    for debugging / re-upload), then calls signal-cli's
    ``uploadStickerPack``.  On success returns the pack's
    ``pack_id`` + ``pack_key`` so the agent can share the install
    URI or use the pack in subsequent sends.

    Args:
        title (`str`):
            Pack title (shown in the sticker picker; 1-32 chars
            recommended).
        author (`str`):
            Pack author / creator name.
        stickers (`list[dict]`):
            Ordered list of ``{"path": str, "emoji": str}`` entries.
            First entry doubles as the pack cover.  Every path must
            pass ``_validate_sticker_webp`` (512×512 webp ≤300KB).
            Max 200 stickers per pack (Signal limit).

    Returns:
        `ToolResponse`:
            TextBlock with ``pack_id``, ``pack_key``, and the
            signal.art install URL on success; specific error on
            validation or upload failure.
    """
    try:
        channel = await _get_signal_channel()
    except LookupError as e:
        return _err(f"Error: {e}")

    title = (title or "").strip()
    author = (author or "").strip()
    if not title or not author:
        return _err(
            "Error: title and author are both required (1-32 chars each).",
        )
    if not isinstance(stickers, list) or not stickers:
        return _err("Error: stickers list must be non-empty.")
    if len(stickers) > 200:
        return _err(
            f"Error: Signal caps packs at 200 stickers (got {len(stickers)}).",
        )

    # Validate every sticker up-front so we never half-build a
    # staging dir.  Collect ALL errors so the agent can fix them
    # in one round rather than whack-a-mole.
    errors: list[str] = []
    resolved: list[tuple[Path, str]] = []
    for idx, item in enumerate(stickers):
        if not isinstance(item, dict):
            errors.append(f"[{idx}] entry must be a dict, got {type(item).__name__}")
            continue
        raw_path = item.get("path") or ""
        emoji = (item.get("emoji") or "").strip()
        if not raw_path:
            errors.append(f"[{idx}] 'path' is required")
            continue
        if not emoji:
            errors.append(f"[{idx}] 'emoji' is required")
            continue
        p = Path(str(raw_path)).expanduser().resolve()
        err = _validate_sticker_webp(p)
        if err:
            errors.append(f"[{idx}] {err}")
            continue
        resolved.append((p, emoji))
    if errors:
        return _err(
            "Error: sticker validation failed:\n  "
            + "\n  ".join(errors),
        )

    # Stage the pack contents.  Keep the dir after upload — signal-cli
    # sometimes wants a retry and the agent may want to inspect the
    # exact bytes that were uploaded.
    staging_root = channel._media_dir / "sticker_pack_staging"
    staging_dir = staging_root / uuid.uuid4().hex
    staging_dir.mkdir(parents=True, exist_ok=True)

    try:
        for i, (src, emoji) in enumerate(resolved):
            shutil.copy2(src, staging_dir / f"{i}.webp")

        manifest = {
            "title": title,
            "author": author,
            "cover": {"id": 0, "emoji": resolved[0][1]},
            "stickers": [
                {"id": i, "emoji": emoji}
                for i, (_, emoji) in enumerate(resolved)
            ],
        }
        manifest_path = staging_dir / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except OSError as e:
        return _err(f"Error: failed to stage sticker pack: {e}")

    url = await channel.client.upload_sticker_pack(str(manifest_path))
    if not url:
        return _err(
            f"Error: uploadStickerPack failed. Staging dir kept at "
            f"{staging_dir} for inspection / retry.",
        )
    parsed = _parse_signal_art_url(url)
    if parsed is None:
        return _err(
            f"Error: upload succeeded but returned URL has no "
            f"pack_id/pack_key fragment: {url}"
        )
    pack_id, pack_key = parsed
    # Structured return so the agent can round-trip any sticker
    # back into ``signal_send_sticker(pack_id, sticker_id, ...)``
    # without guessing the id from upload order.  The JSON is the
    # source of truth; the preamble is human-readable framing so
    # older LLMs that skim text responses still surface something
    # useful.
    payload = {
        "pack_id": pack_id,
        "pack_key": pack_key,
        "install_url": url,
        "title": title,
        "author": author,
        "staged_at": str(staging_dir),
        "stickers": [
            {
                "id": i,
                "emoji": emoji,
                "source_path": str(src),
                "staged_path": str(staging_dir / f"{i}.webp"),
            }
            for i, (src, emoji) in enumerate(resolved)
        ],
    }
    preamble = (
        f"Uploaded sticker pack '{title}' by '{author}' "
        f"({len(resolved)} stickers).\n"
    )
    return _ok_text(
        preamble + json.dumps(payload, ensure_ascii=False, indent=2),
    )


async def signal_add_stickers_to_pack(
    base_pack_id: str,
    new_stickers: list[dict],
) -> ToolResponse:
    """Grow an existing pack by appending new stickers.

    Signal sticker packs are **immutable** — there's no server-side
    "append to existing pack" API.  What this tool does under the
    hood is: (1) download every sticker currently in
    ``base_pack_id`` so we have the bytes locally, (2) stage the
    old stickers + the new ones together with consecutive 0..N
    ids, (3) upload as a fresh pack (new ``pack_id`` /
    ``pack_key``).  The old pack stays on Signal's CDN and keeps
    working for any message already sent; but new sends should
    reference the new pack.

    Args:
        base_pack_id (`str`):
            Hex id of a pack this account already has (installed
            or uploaded).  Use ``signal_list_sticker_packs`` to
            discover.
        new_stickers (`list[dict]`):
            Same shape as ``signal_create_sticker_pack`` expects —
            ``[{"path": "...", "emoji": "🙂"}]``.  Each path must
            pass ``_validate_sticker_webp`` (512×512 webp ≤300KB).

    Returns:
        `ToolResponse`:
            TextBlock with a JSON payload identical in shape to
            ``signal_create_sticker_pack``'s return plus a
            ``previous_pack_id`` field, so the agent can see the
            renumbered sticker ids and know which old pack the
            new one superseded.
    """
    try:
        channel = await _get_signal_channel()
    except LookupError as e:
        return _err(f"Error: {e}")

    base_pack_id = (base_pack_id or "").strip()
    if not base_pack_id:
        return _err("Error: base_pack_id is required.")
    if not isinstance(new_stickers, list) or not new_stickers:
        return _err("Error: new_stickers must be a non-empty list.")

    # Resolve the base pack's metadata so we preserve title/author
    # (otherwise the "grown" pack would look like an unrelated
    # upload in the user's sticker picker).  No match → error.
    raw_packs = await channel.client.list_sticker_packs()
    base_pack: dict | None = None
    for p in raw_packs:
        if not isinstance(p, dict):
            continue
        if (p.get("packId") or p.get("pack_id")) == base_pack_id:
            base_pack = p
            break
    if base_pack is None:
        return _err(
            f"Error: base_pack_id {base_pack_id[:12]}... not found "
            "in this account's sticker packs.  Check "
            "signal_list_sticker_packs output; the pack must be "
            "installed or uploaded by this account before it can "
            "be extended.",
        )

    existing = [
        s for s in (base_pack.get("stickers") or [])
        if isinstance(s, dict)
    ]
    total_count = len(existing) + len(new_stickers)
    if total_count > 200:
        return _err(
            f"Error: combined pack would have {total_count} stickers, "
            "Signal caps at 200.  Create a new pack for the overflow "
            "instead (signal_create_sticker_pack).",
        )

    # Validate new stickers up front so we never partially stage a
    # mix.  Same collector pattern as ``signal_create_sticker_pack``.
    errors: list[str] = []
    resolved_new: list[tuple[Path, str]] = []
    for idx, item in enumerate(new_stickers):
        if not isinstance(item, dict):
            errors.append(
                f"[new {idx}] entry must be a dict, got {type(item).__name__}",
            )
            continue
        raw_path = item.get("path") or ""
        emoji = (item.get("emoji") or "").strip()
        if not raw_path:
            errors.append(f"[new {idx}] 'path' is required")
            continue
        if not emoji:
            errors.append(f"[new {idx}] 'emoji' is required")
            continue
        p = Path(str(raw_path)).expanduser().resolve()
        err = _validate_sticker_webp(p)
        if err:
            errors.append(f"[new {idx}] {err}")
            continue
        resolved_new.append((p, emoji))
    if errors:
        return _err(
            "Error: new sticker validation failed:\n  "
            + "\n  ".join(errors),
        )

    staging_root = channel._media_dir / "sticker_pack_staging"
    staging_dir = staging_root / uuid.uuid4().hex
    staging_dir.mkdir(parents=True, exist_ok=True)

    # Fetch every existing sticker first — the user's account must
    # be able to decrypt the pack (installed OR owned).  If *any*
    # fetch fails we bail out; otherwise we'd quietly lose stickers
    # in the new pack, which is the worst possible UX ("agent
    # ate my crab sticker").
    combined: list[dict] = []
    try:
        for idx, s in enumerate(existing):
            try:
                old_sid = int(s.get("id", idx))
            except (TypeError, ValueError):
                old_sid = idx
            old_emoji = str(s.get("emoji") or "🙂")
            fetched = await channel.client.get_sticker(
                base_pack_id,
                old_sid,
                staging_dir,
            )
            if fetched is None:
                return _err(
                    f"Error: failed to fetch existing sticker "
                    f"{base_pack_id[:12]}:{old_sid} from the base pack. "
                    "Install the pack first (signal_install_sticker_pack "
                    "with pack_key) so signal-cli can decrypt it, then "
                    "retry.",
                )
            # ``get_sticker`` names the file after the *old* sticker
            # id; rename to the new consecutive id for the manifest.
            target = staging_dir / f"{len(combined)}.webp"
            fetched.rename(target)
            combined.append({
                "emoji": old_emoji,
                "staged_path": str(target),
                "source_path": f"pack:{base_pack_id}:{old_sid}",
            })

        for src, emoji in resolved_new:
            target = staging_dir / f"{len(combined)}.webp"
            shutil.copy2(src, target)
            combined.append({
                "emoji": emoji,
                "staged_path": str(target),
                "source_path": str(src),
            })

        manifest = {
            "title": str(base_pack.get("title") or "CoPaw sticker pack"),
            "author": str(base_pack.get("author") or ""),
            "cover": {"id": 0, "emoji": combined[0]["emoji"]},
            "stickers": [
                {"id": i, "emoji": entry["emoji"]}
                for i, entry in enumerate(combined)
            ],
        }
        manifest_path = staging_dir / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except OSError as e:
        return _err(f"Error: failed to stage combined pack: {e}")

    url = await channel.client.upload_sticker_pack(str(manifest_path))
    if not url:
        return _err(
            f"Error: uploadStickerPack failed for combined pack.  "
            f"Staging dir kept at {staging_dir} for inspection.",
        )
    parsed = _parse_signal_art_url(url)
    if parsed is None:
        return _err(
            f"Error: upload succeeded but returned URL has no "
            f"pack_id/pack_key fragment: {url}"
        )
    pack_id, pack_key = parsed
    payload = {
        "pack_id": pack_id,
        "pack_key": pack_key,
        "previous_pack_id": base_pack_id,
        "install_url": url,
        "title": manifest["title"],
        "author": manifest["author"],
        "staged_at": str(staging_dir),
        "stickers": [
            {
                "id": i,
                "emoji": entry["emoji"],
                "source_path": entry["source_path"],
                "staged_path": entry["staged_path"],
            }
            for i, entry in enumerate(combined)
        ],
    }
    preamble = (
        f"Uploaded new pack superseding {base_pack_id[:12]}... with "
        f"{len(combined)} stickers ({len(existing)} existing + "
        f"{len(resolved_new)} new).  Reference the new pack_id for "
        "future sends.\n"
    )
    return _ok_text(
        preamble + json.dumps(payload, ensure_ascii=False, indent=2),
    )


def _resolve_current_signal_target() -> tuple[str, bool] | None:
    """Best-effort: infer ``(to, is_group)`` from the current request.

    Reads :func:`qwenpaw.app.agent_context.get_current_channel_meta`
    — populated by the agent runner from ``request.channel_meta``.
    Returns ``None`` when there's no current request, the current
    request didn't originate from Signal, or the meta dict doesn't
    name a recipient.
    """
    from ...app.agent_context import get_current_channel_meta

    meta = get_current_channel_meta() or {}
    if meta.get("platform") != "signal":
        return None
    group_id = str(meta.get("group_id") or "").strip()
    if group_id:
        return group_id, True
    source = str(meta.get("source") or "").strip()
    if source:
        return source, False
    return None


async def signal_send_sticker(
    pack_id: str,
    sticker_id: int,
    to: str | None = None,
    is_group: bool = False,
) -> ToolResponse:
    """Send a sticker from a pack to a Signal chat.

    The sticker is delivered as a true Signal sticker (with pack
    metadata) — recipients that have the pack installed see the
    pack context; others auto-fetch the sticker image from the
    sticker CDN.

    Args:
        pack_id (`str`):
            Hex pack identifier of a pack this account owns or has
            installed.
        sticker_id (`int`):
            Sticker index within the pack.
        to (`str`, optional):
            Recipient: phone number (``+85251159218``) for DMs, or
            base64 group ID (ending ``==``) for groups.  When
            omitted, auto-resolves to the current Signal request's
            sender (DM source or group id, derived from the agent
            runner's current ``channel_meta``).  Pass explicitly
            when sending to someone other than the current chat.
        is_group (`bool`):
            When True, ``to`` is treated as a group id regardless
            of formatting.  Ignored when ``to`` is auto-resolved
            (we derive ``is_group`` from whether the current
            context has a ``group_id``).

    Returns:
        `ToolResponse`:
            TextBlock confirming the sent timestamp or an error.
    """
    try:
        channel = await _get_signal_channel()
    except LookupError as e:
        return _err(f"Error: {e}")

    pack_id = (pack_id or "").strip()
    if not pack_id:
        return _err("Error: pack_id is required.")
    try:
        sid = int(sticker_id)
    except (TypeError, ValueError):
        return _err(f"Error: sticker_id must be an integer, got {sticker_id!r}")

    target = (to or "").strip()
    if not target:
        resolved = _resolve_current_signal_target()
        if resolved is None:
            return _err(
                "Error: `to` omitted but no current Signal request "
                "context — pass the recipient explicitly "
                "(phone number or group id).",
            )
        target, is_group = resolved

    ts = await channel.client.send_sticker_message(
        target, pack_id, sid, is_group=is_group,
    )
    if ts is None:
        return _err(
            f"Error: sending sticker {pack_id[:12]}:{sid} to {target[:24]} "
            "failed.  Confirm the pack is installed for this account "
            "(signal_list_sticker_packs) and the recipient handle is "
            "reachable.",
        )
    return _ok_text(
        f"Sent sticker {pack_id[:12]}:{sid} to {target[:24]}"
        f"{' (group)' if is_group else ''} at ts={ts}."
    )
