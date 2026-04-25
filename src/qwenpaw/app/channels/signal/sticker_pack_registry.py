# -*- coding: utf-8 -*-
"""Per-account persistent record of Signal sticker packs managed by CoPaw.

signal-cli's local DB holds pack_id / pack_key / encrypted payloads —
everything needed to *re-fetch* a pack from Signal's CDN — but it
doesn't know which packs *this agent* uploaded or whether a newer
pack supersedes an older one (Signal packs are immutable, so
``signal_add_stickers_to_pack`` creates a fresh pack each time;
callers need the lineage to stop referencing stale ids).

This registry fills that gap.  It lives next to the channel's
staging directory (``{media_dir}/sticker_packs.json``), is keyed
by ``pack_id`` for O(1) upsert + lineage bookkeeping, and is
merged into ``signal_list_sticker_packs`` output so the agent
sees the union of "what signal-cli knows" + "what we tagged".

Concurrency: one asyncio.Lock per registry path, held across the
read-modify-write.  Disk writes go through tempfile + os.replace
so partial writes don't corrupt the file even if the process is
killed mid-flight.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

_SCHEMA_VERSION = 1
_FILENAME = "sticker_packs.json"

# Per-path asyncio locks — keyed by the absolute registry path so
# concurrent tool calls on the same channel serialise, but two
# different channels (two signal accounts) don't block each other.
_LOCKS: Dict[str, asyncio.Lock] = {}


def _registry_path(media_dir: Path) -> Path:
    return Path(media_dir) / _FILENAME


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _lock_for(path: Path) -> asyncio.Lock:
    key = str(path.resolve())
    lock = _LOCKS.get(key)
    if lock is None:
        lock = asyncio.Lock()
        _LOCKS[key] = lock
    return lock


def _load_unlocked(path: Path) -> Dict[str, Any]:
    """Read the registry without holding the lock.  Returns an empty
    schema-compliant dict when the file is missing, empty, or
    unparseable (logged so operators can notice corruption)."""
    if not path.is_file():
        return {"schema_version": _SCHEMA_VERSION, "packs": {}}
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as e:
        logger.warning(
            "sticker-pack-registry: read failed for %s: %s",
            path,
            e,
        )
        return {"schema_version": _SCHEMA_VERSION, "packs": {}}
    try:
        data = json.loads(raw) if raw.strip() else {}
    except json.JSONDecodeError as e:
        logger.warning(
            "sticker-pack-registry: %s is malformed (%s) — treating as empty",
            path,
            e,
        )
        return {"schema_version": _SCHEMA_VERSION, "packs": {}}
    if not isinstance(data, dict):
        return {"schema_version": _SCHEMA_VERSION, "packs": {}}
    data.setdefault("schema_version", _SCHEMA_VERSION)
    packs = data.get("packs")
    if not isinstance(packs, dict):
        data["packs"] = {}
    return data


def _write_atomic(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # NamedTemporaryFile + os.replace gives atomic append-or-overwrite
    # semantics on POSIX + Windows; avoids readers observing a half-
    # written JSON if the process dies mid-write.
    fd, tmp = tempfile.mkstemp(
        prefix=".sticker_packs.",
        suffix=".json",
        dir=str(path.parent),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


async def load_registry(media_dir: Path) -> Dict[str, Any]:
    """Return the current registry dict (empty-schema when none exists)."""
    path = _registry_path(Path(media_dir))
    async with _lock_for(path):
        return _load_unlocked(path)


async def upsert_pack(
    media_dir: Path,
    entry: Dict[str, Any],
) -> Dict[str, Any]:
    """Insert or replace a pack record keyed by ``entry['pack_id']``.

    Timestamps (``created_at`` on first insert, ``updated_at`` on
    every write) are stamped automatically — callers pass domain
    fields only.

    Returns the persisted entry (with stamps) so callers can echo it
    back to the agent.
    """
    pack_id = str(entry.get("pack_id") or "").strip()
    if not pack_id:
        raise ValueError("upsert_pack requires entry['pack_id']")

    path = _registry_path(Path(media_dir))
    async with _lock_for(path):
        data = _load_unlocked(path)
        packs = data["packs"]
        now = _now_iso()
        existing = packs.get(pack_id) or {}
        merged = {
            **existing,
            **{k: v for k, v in entry.items() if v is not None},
        }
        merged.setdefault("created_at", existing.get("created_at") or now)
        merged["updated_at"] = now
        merged["pack_id"] = pack_id
        packs[pack_id] = merged
        _write_atomic(path, data)
        return merged


async def mark_superseded(
    media_dir: Path,
    old_pack_id: str,
    new_pack_id: str,
) -> None:
    """Record that ``new_pack_id`` replaces ``old_pack_id``.  No-op if
    the old pack isn't in the registry (e.g. it was installed from
    outside CoPaw)."""
    old_pack_id = (old_pack_id or "").strip()
    new_pack_id = (new_pack_id or "").strip()
    if not old_pack_id or not new_pack_id:
        return

    path = _registry_path(Path(media_dir))
    async with _lock_for(path):
        data = _load_unlocked(path)
        packs = data["packs"]
        entry = packs.get(old_pack_id)
        if entry is None:
            return
        entry["superseded_by"] = new_pack_id
        entry["updated_at"] = _now_iso()
        _write_atomic(path, data)


async def get_pack(
    media_dir: Path,
    pack_id: str,
) -> Optional[Dict[str, Any]]:
    """Return a single pack record, or ``None`` if absent."""
    pack_id = (pack_id or "").strip()
    if not pack_id:
        return None
    path = _registry_path(Path(media_dir))
    async with _lock_for(path):
        data = _load_unlocked(path)
        entry = data["packs"].get(pack_id)
        return dict(entry) if entry else None
