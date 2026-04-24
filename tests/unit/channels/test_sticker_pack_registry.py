# -*- coding: utf-8 -*-
"""Unit tests for the Signal sticker-pack registry."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from qwenpaw.app.channels.signal.sticker_pack_registry import (
    get_pack,
    load_registry,
    mark_superseded,
    upsert_pack,
)


async def test_load_missing_file_returns_empty_schema(tmp_path) -> None:
    data = await load_registry(tmp_path)
    assert data == {"schema_version": 1, "packs": {}}


async def test_upsert_pack_persists_and_stamps_times(tmp_path) -> None:
    entry = await upsert_pack(tmp_path, {
        "pack_id": "PACK1",
        "pack_key": "KEY1",
        "title": "Crabs",
        "source": "uploaded",
    })
    # ``upsert_pack`` returns the persisted entry with time-stamps
    # layered in.
    assert entry["pack_id"] == "PACK1"
    assert "created_at" in entry and "updated_at" in entry
    assert entry["created_at"] == entry["updated_at"]

    data = await load_registry(tmp_path)
    assert set(data["packs"].keys()) == {"PACK1"}
    # File on disk is valid JSON.
    raw = (tmp_path / "sticker_packs.json").read_text()
    assert json.loads(raw)["packs"]["PACK1"]["title"] == "Crabs"


async def test_upsert_second_time_preserves_created_at(tmp_path) -> None:
    first = await upsert_pack(tmp_path, {
        "pack_id": "P", "title": "v1",
    })
    await asyncio.sleep(1.01)  # seconds precision in iso timestamps
    second = await upsert_pack(tmp_path, {
        "pack_id": "P", "title": "v2",
    })
    assert second["created_at"] == first["created_at"]
    assert second["updated_at"] > first["updated_at"]
    assert second["title"] == "v2"


async def test_upsert_without_pack_id_raises(tmp_path) -> None:
    with pytest.raises(ValueError):
        await upsert_pack(tmp_path, {"title": "nope"})


async def test_mark_superseded_updates_old_entry(tmp_path) -> None:
    await upsert_pack(tmp_path, {"pack_id": "OLD", "title": "v1"})
    await mark_superseded(tmp_path, "OLD", "NEW")
    entry = await get_pack(tmp_path, "OLD")
    assert entry is not None
    assert entry["superseded_by"] == "NEW"


async def test_mark_superseded_is_noop_when_old_missing(tmp_path) -> None:
    """Installing outside CoPaw + then extending via
    ``signal_add_stickers_to_pack`` means the base pack was never
    written to the registry.  Marking superseded should silently
    skip (not raise) so the grow path still succeeds."""
    await mark_superseded(tmp_path, "NEVER_SEEN", "NEW")
    data = await load_registry(tmp_path)
    assert data["packs"] == {}


async def test_get_pack_returns_copy_not_reference(tmp_path) -> None:
    await upsert_pack(tmp_path, {"pack_id": "P", "title": "orig"})
    got = await get_pack(tmp_path, "P")
    assert got is not None
    got["title"] = "mutated-in-caller"
    again = await get_pack(tmp_path, "P")
    assert again["title"] == "orig"


async def test_concurrent_upserts_serialise(tmp_path) -> None:
    """Two upserts racing on the same pack_id must both land —
    asyncio.Lock serialises the read-modify-write."""
    async def _task(title):
        await upsert_pack(tmp_path, {"pack_id": "P", "title": title})

    await asyncio.gather(*(_task(f"v{i}") for i in range(20)))
    data = await load_registry(tmp_path)
    assert "P" in data["packs"]
    # Final title is whichever task landed last — we just need the
    # file to be valid JSON with a single entry, no duplicates or
    # corruption.
    assert isinstance(data["packs"]["P"]["title"], str)


async def test_malformed_file_is_treated_as_empty(tmp_path) -> None:
    """A corrupted ``sticker_packs.json`` shouldn't blow up the
    tools — the next upsert silently rebuilds from scratch.
    Better UX than "tools all fail until you rm the file"."""
    (tmp_path / "sticker_packs.json").write_text("{ not json", encoding="utf-8")
    data = await load_registry(tmp_path)
    assert data == {"schema_version": 1, "packs": {}}
    # And a subsequent write recovers cleanly.
    await upsert_pack(tmp_path, {"pack_id": "P"})
    data = await load_registry(tmp_path)
    assert "P" in data["packs"]
