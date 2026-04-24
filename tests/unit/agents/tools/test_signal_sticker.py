# -*- coding: utf-8 -*-
"""Tests for the Signal sticker-pack agent tools."""

from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock

import pytest

from qwenpaw.agents.tools import signal_sticker as st


# ─────────────────────────── fakes ──────────────────────────────────


class _FakeClient:
    """Minimal stand-in for SignalSubprocessClient.

    Every RPC wrapper is an AsyncMock so tests can assert on call
    shape as well as return values.  ``connected`` defaults to
    True so the tool surface's health gate doesn't short-circuit.
    """

    def __init__(self) -> None:
        self.connected = True
        self.list_sticker_packs = AsyncMock(return_value=[])
        self.add_sticker_pack = AsyncMock(return_value=True)
        self.upload_sticker_pack = AsyncMock(return_value=None)
        self.send_sticker_message = AsyncMock(return_value=None)
        self.get_sticker = AsyncMock(return_value=None)


class _FakeChannel:
    def __init__(self, media_dir: Path) -> None:
        self.enabled = True
        self._media_dir = media_dir
        self.client = _FakeClient()


@pytest.fixture
def fake_channel(tmp_path, monkeypatch):
    """Patch ``_get_signal_channel`` to return a controllable fake.

    Returns the channel so tests can configure per-RPC behaviour /
    assert on calls.
    """
    ch = _FakeChannel(media_dir=tmp_path / "media")
    ch._media_dir.mkdir(parents=True, exist_ok=True)

    async def _stub():
        return ch
    monkeypatch.setattr(st, "_get_signal_channel", _stub)
    return ch


# ─────────────────────────── webp helpers ───────────────────────────


def _make_vp8x_webp(w: int, h: int, pad_bytes: int = 0) -> bytes:
    """Build a minimal VP8X-chunk webp with the declared dims.

    Enough bytes for ``_read_webp_dimensions`` to decode dims; not
    enough for a decoder to actually render it — tests just need
    the validator to succeed or fail on size/magic/shape checks.
    """
    # VP8X stores (width-1, height-1) as 24-bit little-endian ints
    # at offsets 24..27 and 27..30 respectively within the file.
    vp8x = (
        b"VP8X"  # chunk fourcc
        + struct.pack("<I", 10)  # chunk size (10 bytes of payload)
        + b"\x00\x00\x00\x00"  # flags + reserved
        + (w - 1).to_bytes(3, "little")
        + (h - 1).to_bytes(3, "little")
    )
    # RIFF header: "RIFF" + uint32 size + "WEBP" + chunks
    body = b"WEBP" + vp8x + (b"\x00" * pad_bytes)
    return b"RIFF" + struct.pack("<I", len(body)) + body


def _write_valid_sticker(path: Path, pad: int = 0) -> Path:
    path.write_bytes(_make_vp8x_webp(512, 512, pad_bytes=pad))
    return path


# ─────────────────────────── url parsing ────────────────────────────


def test_parse_signal_art_url_happy_path() -> None:
    url = "https://signal.art/addstickers/#pack_id=AABB&pack_key=CCDD"
    assert st._parse_signal_art_url(url) == ("AABB", "CCDD")


def test_parse_signal_art_url_missing_fields_returns_none() -> None:
    assert st._parse_signal_art_url(
        "https://signal.art/addstickers/#pack_id=AABB",
    ) is None
    assert st._parse_signal_art_url(
        "https://signal.art/addstickers/",
    ) is None
    assert st._parse_signal_art_url("") is None


# ─────────────────────────── webp validation ────────────────────────


def test_validate_sticker_webp_rejects_missing(tmp_path) -> None:
    err = st._validate_sticker_webp(tmp_path / "nope.webp")
    assert err and "not found" in err


def test_validate_sticker_webp_rejects_empty(tmp_path) -> None:
    p = tmp_path / "empty.webp"
    p.write_bytes(b"")
    err = st._validate_sticker_webp(p)
    assert err and "empty" in err


def test_validate_sticker_webp_rejects_non_webp(tmp_path) -> None:
    p = tmp_path / "fake.webp"
    p.write_bytes(b"\x89PNG\r\n\x1a\n" + b"x" * 40)
    err = st._validate_sticker_webp(p)
    assert err and "not a WebP" in err


def test_validate_sticker_webp_rejects_oversize(tmp_path) -> None:
    p = tmp_path / "big.webp"
    # 512×512 header but padded past 300KB total.
    p.write_bytes(_make_vp8x_webp(512, 512, pad_bytes=400 * 1024))
    err = st._validate_sticker_webp(p)
    assert err and "too large" in err


def test_validate_sticker_webp_rejects_wrong_dimensions(tmp_path) -> None:
    p = tmp_path / "bad.webp"
    p.write_bytes(_make_vp8x_webp(256, 256))
    err = st._validate_sticker_webp(p)
    assert err and "512x512" in err


def test_validate_sticker_webp_accepts_512x512(tmp_path) -> None:
    p = tmp_path / "ok.webp"
    _write_valid_sticker(p)
    assert st._validate_sticker_webp(p) is None


# ─────────────────────────── list ──────────────────────────────────


async def test_list_sticker_packs_empty(fake_channel) -> None:
    fake_channel.client.list_sticker_packs.return_value = []
    res = await st.signal_list_sticker_packs()
    assert res.content[0]["text"].strip() == "[]"


async def test_list_sticker_packs_shapes_output(fake_channel) -> None:
    fake_channel.client.list_sticker_packs.return_value = [
        {
            "packId": "PACKID" * 10,
            "title": "Crabs",
            "author": "Joe",
            "installed": True,
            "stickers": [
                {"id": 0, "emoji": "🦀"},
                {"id": 1, "emoji": "🐚"},
            ],
        },
        # Malformed entry should be skipped silently.
        "not-a-dict",
    ]
    res = await st.signal_list_sticker_packs()
    data = json.loads(res.content[0]["text"])
    assert len(data) == 1
    entry = data[0]
    assert entry["title"] == "Crabs"
    assert entry["sticker_count"] == 2
    assert entry["stickers"] == [
        {"id": 0, "emoji": "🦀"},
        {"id": 1, "emoji": "🐚"},
    ]


# ─────────────────────────── preview ───────────────────────────────


async def test_preview_sticker_returns_image_block_on_success(
    fake_channel,
) -> None:
    sticker_path = fake_channel._media_dir / "signal_sticker_PACK_0.webp"
    sticker_path.write_bytes(_make_vp8x_webp(512, 512))
    fake_channel.client.get_sticker.return_value = sticker_path

    res = await st.signal_preview_sticker("PACKID123", 0)
    kinds = [c.get("type") for c in res.content]
    assert "image" in kinds
    assert "text" in kinds
    # URL points at the actual local file.
    image = next(c for c in res.content if c.get("type") == "image")
    assert str(sticker_path) in image["source"]["url"]


async def test_preview_sticker_rejects_missing_pack_id(fake_channel) -> None:
    res = await st.signal_preview_sticker("   ", 0)
    assert "pack_id is required" in res.content[0]["text"]
    fake_channel.client.get_sticker.assert_not_awaited()


async def test_preview_sticker_rejects_non_int_sticker_id(
    fake_channel,
) -> None:
    res = await st.signal_preview_sticker("PACK", "not-an-int")  # type: ignore[arg-type]
    assert "must be an integer" in res.content[0]["text"]


async def test_preview_sticker_propagates_fetch_failure(fake_channel) -> None:
    fake_channel.client.get_sticker.return_value = None
    res = await st.signal_preview_sticker("PACKID", 3)
    assert "could not fetch sticker" in res.content[0]["text"]


# ─────────────────────────── install ───────────────────────────────


async def test_install_sticker_pack_requires_both_fields(
    fake_channel,
) -> None:
    res = await st.signal_install_sticker_pack("", "KEY")
    assert "both pack_id and pack_key" in res.content[0]["text"]
    fake_channel.client.add_sticker_pack.assert_not_awaited()


async def test_install_sticker_pack_calls_rpc(fake_channel) -> None:
    fake_channel.client.add_sticker_pack.return_value = True
    res = await st.signal_install_sticker_pack("PACKID", "PACKKEY")
    fake_channel.client.add_sticker_pack.assert_awaited_once_with(
        "PACKID", "PACKKEY",
    )
    assert "Installed sticker pack" in res.content[0]["text"]


async def test_install_sticker_pack_surfaces_rpc_failure(
    fake_channel,
) -> None:
    fake_channel.client.add_sticker_pack.return_value = False
    res = await st.signal_install_sticker_pack("PACKID", "PACKKEY")
    assert "addStickerPack failed" in res.content[0]["text"]


# ─────────────────────────── create ────────────────────────────────


async def test_create_sticker_pack_validates_every_sticker_up_front(
    fake_channel, tmp_path,
) -> None:
    good = _write_valid_sticker(tmp_path / "good.webp")
    bad = tmp_path / "bad.webp"
    bad.write_bytes(b"not-webp")

    res = await st.signal_create_sticker_pack(
        "My Pack", "Me",
        stickers=[
            {"path": str(good), "emoji": "🙂"},
            {"path": str(bad), "emoji": "😎"},
            {"path": str(good)},  # missing emoji
        ],
    )
    text = res.content[0]["text"]
    # All three failure modes should appear — no staging should occur.
    assert "[1]" in text and "not a WebP" in text
    assert "[2]" in text and "emoji" in text
    fake_channel.client.upload_sticker_pack.assert_not_awaited()
    assert not (fake_channel._media_dir / "sticker_pack_staging").exists()


async def test_create_sticker_pack_stages_and_uploads(
    fake_channel, tmp_path,
) -> None:
    s0 = _write_valid_sticker(tmp_path / "a.webp")
    s1 = _write_valid_sticker(tmp_path / "b.webp")
    # Upload returns the canonical signal.art URL.
    fake_channel.client.upload_sticker_pack.return_value = (
        "https://signal.art/addstickers/#pack_id=NEWPACK&pack_key=NEWKEY"
    )

    res = await st.signal_create_sticker_pack(
        "Title", "Author",
        stickers=[
            {"path": str(s0), "emoji": "🦀"},
            {"path": str(s1), "emoji": "🐚"},
        ],
    )
    text = res.content[0]["text"]
    # Return shape: human preamble + JSON blob.  The JSON is the
    # agent-consumable surface; the preamble is purely cosmetic.
    payload = json.loads(text[text.index("{"):])
    assert payload["pack_id"] == "NEWPACK"
    assert payload["pack_key"] == "NEWKEY"

    # Staging dir should contain manifest.json + the numbered webp
    # copies, and should survive post-upload for debugging.
    staging_dirs = list(
        (fake_channel._media_dir / "sticker_pack_staging").iterdir(),
    )
    assert len(staging_dirs) == 1
    staged = staging_dirs[0]
    assert (staged / "0.webp").read_bytes() == s0.read_bytes()
    assert (staged / "1.webp").read_bytes() == s1.read_bytes()
    manifest = json.loads((staged / "manifest.json").read_text())
    assert manifest["title"] == "Title"
    # signal-cli's uploadStickerPack requires per-entry ``file`` +
    # ``contentType``.  The ``id`` shape Signal Desktop uses would
    # fail upload with "Must set a 'file' field on each sticker".
    # No explicit cover — signal-cli auto-uses the first sticker.
    assert "cover" not in manifest
    assert manifest["stickers"] == [
        {"file": "0.webp", "contentType": "image/webp", "emoji": "🦀"},
        {"file": "1.webp", "contentType": "image/webp", "emoji": "🐚"},
    ]
    # upload_sticker_pack was called with the manifest path.
    call = fake_channel.client.upload_sticker_pack.await_args
    assert call.args[0] == str(staged / "manifest.json")


async def test_create_sticker_pack_surfaces_upload_failure_with_staging_path(
    fake_channel, tmp_path,
) -> None:
    s0 = _write_valid_sticker(tmp_path / "a.webp")
    fake_channel.client.upload_sticker_pack.return_value = None
    res = await st.signal_create_sticker_pack(
        "T", "A", stickers=[{"path": str(s0), "emoji": "🦀"}],
    )
    text = res.content[0]["text"]
    assert "uploadStickerPack failed" in text
    # Staging dir path should be in the error so the agent can
    # retry or inspect.
    assert "sticker_pack_staging" in text


async def test_create_sticker_pack_rejects_empty_title(fake_channel) -> None:
    res = await st.signal_create_sticker_pack(
        "", "Me", stickers=[{"path": "/x", "emoji": "🙂"}],
    )
    assert "title and author" in res.content[0]["text"]


async def test_create_sticker_pack_rejects_too_many_stickers(
    fake_channel, tmp_path,
) -> None:
    s0 = _write_valid_sticker(tmp_path / "a.webp")
    res = await st.signal_create_sticker_pack(
        "T", "A",
        stickers=[{"path": str(s0), "emoji": "🙂"}] * 201,
    )
    assert "200 stickers" in res.content[0]["text"]


# ─────────────────────────── send ──────────────────────────────────


async def test_send_sticker_dm(fake_channel) -> None:
    fake_channel.client.send_sticker_message.return_value = 1_700_000_000
    res = await st.signal_send_sticker("PACKID", 3, "+85251159218")
    fake_channel.client.send_sticker_message.assert_awaited_once_with(
        "+85251159218", "PACKID", 3, is_group=False,
    )
    assert "Sent sticker" in res.content[0]["text"]
    assert "1700000000" in res.content[0]["text"]


async def test_send_sticker_group(fake_channel) -> None:
    fake_channel.client.send_sticker_message.return_value = 1_700_000_001
    res = await st.signal_send_sticker(
        "PACKID", 0, "GROUPBASE64==", is_group=True,
    )
    fake_channel.client.send_sticker_message.assert_awaited_once_with(
        "GROUPBASE64==", "PACKID", 0, is_group=True,
    )
    assert "(group)" in res.content[0]["text"]


async def test_send_sticker_surfaces_rpc_failure(fake_channel) -> None:
    fake_channel.client.send_sticker_message.return_value = None
    res = await st.signal_send_sticker("PACKID", 0, "+1")
    assert "failed" in res.content[0]["text"]


# ─────────────────────────── channel-missing error path ─────────────


async def test_tools_return_clean_error_when_channel_missing(
    tmp_path, monkeypatch,
) -> None:
    """Every tool should surface a LookupError from
    ``_get_signal_channel`` as a TextBlock error, not an uncaught
    exception — the agent needs a recoverable signal."""
    async def _raise():
        raise LookupError("no signal channel for this agent")
    monkeypatch.setattr(st, "_get_signal_channel", _raise)

    for call in (
        st.signal_list_sticker_packs(),
        st.signal_preview_sticker("PACKID", 0),
        st.signal_install_sticker_pack("PID", "PKEY"),
        st.signal_create_sticker_pack(
            "T", "A", stickers=[{"path": "/x", "emoji": "🙂"}],
        ),
        st.signal_add_stickers_to_pack(
            "BASEID", new_stickers=[{"path": "/x", "emoji": "🙂"}],
        ),
        st.signal_send_sticker("PACKID", 0, "+1"),
    ):
        res = await call
        assert "Error: no signal channel" in res.content[0]["text"]


# ─────────────────────────── prepare_sticker_webp ──────────────────


async def test_prepare_sticker_webp_happy_path(tmp_path) -> None:
    """Any decodable image round-trips through the conversion into
    a 512×512 webp suitable for sticker pack upload."""
    from PIL import Image
    src = tmp_path / "in.png"
    Image.new("RGBA", (800, 600), (255, 0, 0, 255)).save(src)
    res = await st.signal_prepare_sticker_webp(str(src))
    out_path_str = res.content[0]["text"].strip()
    out = Path(out_path_str)
    assert out.is_file()
    # Convert validates => same invariants as pack upload expects.
    assert st._validate_sticker_webp(out) is None


async def test_prepare_sticker_webp_missing_input(tmp_path) -> None:
    res = await st.signal_prepare_sticker_webp(str(tmp_path / "nope.png"))
    assert "Error:" in res.content[0]["text"]
    assert "not found" in res.content[0]["text"]


async def test_prepare_sticker_webp_rejects_empty_input() -> None:
    res = await st.signal_prepare_sticker_webp("")
    assert "input_path is required" in res.content[0]["text"]


# ─────────────────────────── create returns sticker_ids ─────────────


async def test_create_sticker_pack_returns_id_mapping(
    fake_channel, tmp_path,
) -> None:
    """Agents round-trip new packs into ``signal_send_sticker`` using
    the ``stickers[i].id`` field from this return value — without
    it they'd have to guess the id from upload order."""
    s0 = _write_valid_sticker(tmp_path / "a.webp")
    s1 = _write_valid_sticker(tmp_path / "b.webp")
    fake_channel.client.upload_sticker_pack.return_value = (
        "https://signal.art/addstickers/#pack_id=NEW&pack_key=KEY"
    )
    res = await st.signal_create_sticker_pack(
        "Title", "Author",
        stickers=[
            {"path": str(s0), "emoji": "🦀"},
            {"path": str(s1), "emoji": "🐚"},
        ],
    )
    # Payload is a JSON blob after a human-readable preamble line.
    text = res.content[0]["text"]
    blob = text[text.index("{"):]
    payload = json.loads(blob)
    assert payload["pack_id"] == "NEW"
    assert payload["pack_key"] == "KEY"
    assert payload["title"] == "Title"
    assert payload["author"] == "Author"
    assert [s["id"] for s in payload["stickers"]] == [0, 1]
    assert [s["emoji"] for s in payload["stickers"]] == ["🦀", "🐚"]
    # source_path is the caller's original; staged_path is under the
    # channel's staging dir.
    assert payload["stickers"][0]["source_path"] == str(s0)
    assert "sticker_pack_staging" in payload["stickers"][0]["staged_path"]


# ─────────────────────────── send auto-recipient ────────────────────


async def test_send_sticker_auto_resolves_dm_from_context(
    fake_channel, monkeypatch,
) -> None:
    """``to=None`` + DM context → recipient is the DM source,
    is_group=False.  The channel runner populates
    ``get_current_channel_meta`` from ``request.channel_meta``."""
    from qwenpaw.app import agent_context
    monkeypatch.setattr(
        agent_context, "_current_channel_meta",
        agent_context._current_channel_meta,
    )
    agent_context.set_current_channel_meta({
        "platform": "signal",
        "source": "+85298765432",
        "group_id": "",
    })
    try:
        fake_channel.client.send_sticker_message.return_value = 1_700_000_042
        res = await st.signal_send_sticker("PACK", 3)
        fake_channel.client.send_sticker_message.assert_awaited_once_with(
            "+85298765432", "PACK", 3, is_group=False,
        )
        assert "Sent sticker" in res.content[0]["text"]
    finally:
        agent_context.set_current_channel_meta(None)


async def test_send_sticker_auto_resolves_group_from_context(
    fake_channel, monkeypatch,
) -> None:
    """When the current context has a ``group_id``, auto-resolve
    always prefers the group (regardless of whether a ``source``
    also sits in the dict)."""
    from qwenpaw.app import agent_context
    agent_context.set_current_channel_meta({
        "platform": "signal",
        "source": "+85298765432",
        "group_id": "GROUPBASE64==",
    })
    try:
        fake_channel.client.send_sticker_message.return_value = 1
        await st.signal_send_sticker("PACK", 0)
        fake_channel.client.send_sticker_message.assert_awaited_once_with(
            "GROUPBASE64==", "PACK", 0, is_group=True,
        )
    finally:
        agent_context.set_current_channel_meta(None)


async def test_send_sticker_auto_resolve_rejects_when_no_context(
    fake_channel, monkeypatch,
) -> None:
    """No current Signal request + no ``to`` argument → graceful
    error asking for an explicit recipient, not a crash."""
    from qwenpaw.app import agent_context
    agent_context.set_current_channel_meta(None)
    res = await st.signal_send_sticker("PACK", 0)
    assert "no current Signal request context" in res.content[0]["text"]
    fake_channel.client.send_sticker_message.assert_not_awaited()


async def test_send_sticker_auto_resolve_rejects_non_signal_context(
    fake_channel, monkeypatch,
) -> None:
    """Whatsapp/telegram request + no ``to`` → we must not send to
    the *wrong* platform's recipient."""
    from qwenpaw.app import agent_context
    agent_context.set_current_channel_meta({
        "platform": "whatsapp",
        "source": "+1111111",
    })
    try:
        res = await st.signal_send_sticker("PACK", 0)
        assert "no current Signal request context" in res.content[0]["text"]
        fake_channel.client.send_sticker_message.assert_not_awaited()
    finally:
        agent_context.set_current_channel_meta(None)


async def test_send_sticker_explicit_to_overrides_context(
    fake_channel, monkeypatch,
) -> None:
    """Explicit ``to`` always wins over context — agent might want
    to forward a sticker to a different chat."""
    from qwenpaw.app import agent_context
    agent_context.set_current_channel_meta({
        "platform": "signal",
        "source": "+85298765432",
    })
    try:
        fake_channel.client.send_sticker_message.return_value = 1
        await st.signal_send_sticker("PACK", 0, to="+85299999999")
        fake_channel.client.send_sticker_message.assert_awaited_once_with(
            "+85299999999", "PACK", 0, is_group=False,
        )
    finally:
        agent_context.set_current_channel_meta(None)


# ─────────────────────────── add_stickers_to_pack ───────────────────


async def test_add_stickers_to_pack_merges_existing_and_new(
    fake_channel, tmp_path,
) -> None:
    """The new pack should contain ``existing + new`` in that order,
    with ids renumbered 0..N-1; old+new source paths are traceable
    in the return payload so the agent can map tracked state."""
    base_pack_id = "BASE123"
    # Fake pack has 2 existing stickers already.
    fake_channel.client.list_sticker_packs.return_value = [{
        "packId": base_pack_id,
        "title": "Crabs",
        "author": "Joe",
        "stickers": [
            {"id": 0, "emoji": "🦀"},
            {"id": 1, "emoji": "🐚"},
        ],
    }]
    # Simulate successful fetch — signal-cli returns webp bytes.
    async def _fake_get(pack_id, sid, dest_dir, **_kw):
        p = dest_dir / f"old_{sid}.webp"
        _write_valid_sticker(p)
        return p
    fake_channel.client.get_sticker = _fake_get
    fake_channel.client.upload_sticker_pack.return_value = (
        "https://signal.art/addstickers/#pack_id=NEWPACK&pack_key=NEWKEY"
    )

    new_src = _write_valid_sticker(tmp_path / "new_crab.webp")
    res = await st.signal_add_stickers_to_pack(
        base_pack_id,
        new_stickers=[{"path": str(new_src), "emoji": "🦀"}],
    )
    text = res.content[0]["text"]
    payload = json.loads(text[text.index("{"):])
    assert payload["pack_id"] == "NEWPACK"
    assert payload["previous_pack_id"] == base_pack_id
    assert payload["title"] == "Crabs"  # Title preserved from base.
    # 2 existing + 1 new, renumbered 0/1/2.
    assert [s["id"] for s in payload["stickers"]] == [0, 1, 2]
    assert [s["emoji"] for s in payload["stickers"]] == ["🦀", "🐚", "🦀"]
    # Old stickers carry a pack-reference source_path (no original
    # file on disk); new ones carry the agent's input path.
    assert payload["stickers"][0]["source_path"] == f"pack:{base_pack_id}:0"
    assert payload["stickers"][1]["source_path"] == f"pack:{base_pack_id}:1"
    assert payload["stickers"][2]["source_path"] == str(new_src)
    # Staging dir should have the three numbered files + manifest.
    staged = Path(payload["staged_at"])
    assert (staged / "0.webp").is_file()
    assert (staged / "1.webp").is_file()
    assert (staged / "2.webp").is_file()
    assert (staged / "manifest.json").is_file()
    manifest = json.loads((staged / "manifest.json").read_text())
    # Same manifest contract as create: ``file``/``contentType``/
    # ``emoji`` per entry, no explicit cover.
    assert "cover" not in manifest
    assert manifest["stickers"] == [
        {"file": "0.webp", "contentType": "image/webp", "emoji": "🦀"},
        {"file": "1.webp", "contentType": "image/webp", "emoji": "🐚"},
        {"file": "2.webp", "contentType": "image/webp", "emoji": "🦀"},
    ]


async def test_add_stickers_to_pack_rejects_missing_base(
    fake_channel, tmp_path,
) -> None:
    """Agent tried to grow a pack that isn't in listStickerPacks —
    we must not silently create a fresh pack; force them to call
    signal_install_sticker_pack first."""
    fake_channel.client.list_sticker_packs.return_value = []
    s0 = _write_valid_sticker(tmp_path / "a.webp")
    res = await st.signal_add_stickers_to_pack(
        "MISSINGPACK",
        new_stickers=[{"path": str(s0), "emoji": "🦀"}],
    )
    assert "not found" in res.content[0]["text"]
    fake_channel.client.upload_sticker_pack.assert_not_awaited()


async def test_add_stickers_to_pack_rejects_over_cap(
    fake_channel, tmp_path,
) -> None:
    """Signal hard-caps packs at 200 stickers — adding past that
    limit would look like success then reject at upload."""
    fake_channel.client.list_sticker_packs.return_value = [{
        "packId": "P",
        "title": "x",
        "author": "y",
        "stickers": [{"id": i, "emoji": "🙂"} for i in range(199)],
    }]
    s0 = _write_valid_sticker(tmp_path / "a.webp")
    res = await st.signal_add_stickers_to_pack(
        "P",
        new_stickers=[{"path": str(s0), "emoji": "🦀"}] * 5,
    )
    assert "200" in res.content[0]["text"]
    fake_channel.client.upload_sticker_pack.assert_not_awaited()


# ─────────────────────────── pack registry integration ─────────────


async def test_create_pack_writes_registry_entry(
    fake_channel, tmp_path,
) -> None:
    """Every successful upload persists to ``sticker_packs.json`` so
    the agent (and ``signal_list_sticker_packs``) can recover the
    pack_key + source lineage later — signal-cli doesn't track
    who-uploaded-what."""
    s0 = _write_valid_sticker(tmp_path / "a.webp")
    fake_channel.client.upload_sticker_pack.return_value = (
        "https://signal.art/addstickers/#pack_id=REGP&pack_key=REGK"
    )
    await st.signal_create_sticker_pack(
        "T", "A",
        stickers=[{"path": str(s0), "emoji": "🙂"}],
        label="test-pack",
    )
    registry_path = fake_channel._media_dir / "sticker_packs.json"
    assert registry_path.is_file()
    data = json.loads(registry_path.read_text())
    entry = data["packs"]["REGP"]
    assert entry["pack_key"] == "REGK"
    assert entry["source"] == "uploaded"
    assert entry["label"] == "test-pack"
    assert entry["title"] == "T"
    assert entry["stickers"][0]["source_path"] == str(s0)


async def test_install_pack_writes_registry_entry(fake_channel) -> None:
    """Installed packs also land in the registry, tagged
    ``source=installed`` — so the agent can recover a pack_key it
    supplied once even if signal-cli forgets (e.g. relink)."""
    fake_channel.client.add_sticker_pack.return_value = True
    await st.signal_install_sticker_pack(
        "INSP", "INSK", label="friends-stickers",
    )
    registry_path = fake_channel._media_dir / "sticker_packs.json"
    data = json.loads(registry_path.read_text())
    entry = data["packs"]["INSP"]
    assert entry["source"] == "installed"
    assert entry["pack_key"] == "INSK"
    assert entry["label"] == "friends-stickers"
    # Install URL is reconstructed so the agent can re-share it.
    assert "pack_id=INSP" in entry["install_url"]
    assert "pack_key=INSK" in entry["install_url"]


async def test_add_stickers_marks_base_superseded_and_inherits_label(
    fake_channel, tmp_path,
) -> None:
    """Growing a pack: (1) new pack entry persisted, (2) base
    pack's entry annotated with ``superseded_by``, (3) label
    inherits from base when caller didn't pass one."""
    # Pre-seed the base pack in the registry with a label.
    from qwenpaw.app.channels.signal.sticker_pack_registry import upsert_pack
    await upsert_pack(fake_channel._media_dir, {
        "pack_id": "BASE", "pack_key": "BKEY",
        "source": "uploaded", "label": "crabs",
        "title": "Crabs", "author": "me",
    })
    fake_channel.client.list_sticker_packs.return_value = [{
        "packId": "BASE", "title": "Crabs", "author": "me",
        "stickers": [{"id": 0, "emoji": "🦀"}],
    }]

    async def _fake_get(pack_id, sid, dest_dir, **_kw):
        p = dest_dir / f"old_{sid}.webp"
        _write_valid_sticker(p)
        return p
    fake_channel.client.get_sticker = _fake_get
    fake_channel.client.upload_sticker_pack.return_value = (
        "https://signal.art/addstickers/#pack_id=GROWN&pack_key=GK"
    )
    new_src = _write_valid_sticker(tmp_path / "new.webp")
    await st.signal_add_stickers_to_pack(
        "BASE",
        new_stickers=[{"path": str(new_src), "emoji": "🐚"}],
        # label omitted → should inherit "crabs" from base
    )
    data = json.loads(
        (fake_channel._media_dir / "sticker_packs.json").read_text(),
    )
    grown = data["packs"]["GROWN"]
    base = data["packs"]["BASE"]
    assert grown["label"] == "crabs"  # inherited
    assert grown["previous_pack_id"] == "BASE"
    assert base["superseded_by"] == "GROWN"


async def test_add_stickers_explicit_label_overrides_inherit(
    fake_channel, tmp_path,
) -> None:
    from qwenpaw.app.channels.signal.sticker_pack_registry import upsert_pack
    await upsert_pack(fake_channel._media_dir, {
        "pack_id": "BASE", "pack_key": "K",
        "source": "uploaded", "label": "old-name",
    })
    fake_channel.client.list_sticker_packs.return_value = [{
        "packId": "BASE", "title": "x", "author": "y", "stickers": [],
    }]
    fake_channel.client.upload_sticker_pack.return_value = (
        "https://signal.art/addstickers/#pack_id=NEW&pack_key=NK"
    )
    new_src = _write_valid_sticker(tmp_path / "a.webp")
    await st.signal_add_stickers_to_pack(
        "BASE",
        new_stickers=[{"path": str(new_src), "emoji": "🙂"}],
        label="new-name",
    )
    data = json.loads(
        (fake_channel._media_dir / "sticker_packs.json").read_text(),
    )
    assert data["packs"]["NEW"]["label"] == "new-name"


async def test_list_packs_merges_registry_fields(fake_channel) -> None:
    """When signal-cli + the registry both know about a pack, the
    list output merges: signal-cli supplies authoritative
    title/author/emoji; registry supplies label + source_path
    lineage that signal-cli can't know."""
    from qwenpaw.app.channels.signal.sticker_pack_registry import upsert_pack
    await upsert_pack(fake_channel._media_dir, {
        "pack_id": "P1",
        "pack_key": "K1",
        "source": "uploaded",
        "label": "crabs-v1",
        "stickers": [
            {"id": 0, "emoji": "🦀",
             "source_path": "/orig/a.png",
             "staged_path": "/staged/0.webp"},
        ],
    })
    fake_channel.client.list_sticker_packs.return_value = [{
        "packId": "P1",
        "title": "Crabs",
        "author": "Joe",
        "installed": True,
        "stickers": [{"id": 0, "emoji": "🦀"}],
    }]
    res = await st.signal_list_sticker_packs()
    data = json.loads(res.content[0]["text"])
    entry = next(e for e in data if e["pack_id"] == "P1")
    assert entry["title"] == "Crabs"        # from signal-cli
    assert entry["label"] == "crabs-v1"      # from registry
    assert entry["source"] == "uploaded"     # from registry
    assert entry["stickers"][0]["source_path"] == "/orig/a.png"
    assert entry["stickers"][0]["staged_path"] == "/staged/0.webp"


async def test_list_packs_surfaces_registry_only_entries(
    fake_channel,
) -> None:
    """If signal-cli forgets a pack (e.g. re-link cleared its DB)
    but the registry still has it, we surface the registry record
    so the agent can retry install with the preserved pack_key."""
    from qwenpaw.app.channels.signal.sticker_pack_registry import upsert_pack
    await upsert_pack(fake_channel._media_dir, {
        "pack_id": "GHOST",
        "pack_key": "GKEY",
        "source": "uploaded",
        "label": "lost",
        "title": "Lost Pack",
    })
    fake_channel.client.list_sticker_packs.return_value = []
    res = await st.signal_list_sticker_packs()
    data = json.loads(res.content[0]["text"])
    entry = next(e for e in data if e["pack_id"] == "GHOST")
    assert entry["label"] == "lost"
    assert entry["installed"] is False


async def test_list_packs_marks_external_when_signal_cli_only(
    fake_channel,
) -> None:
    """signal-cli knows about a pack (user installed it outside
    CoPaw), registry doesn't → tag ``source=external`` so the
    agent knows it didn't originate from these tools."""
    fake_channel.client.list_sticker_packs.return_value = [{
        "packId": "EXT",
        "title": "Third Party",
        "author": "Random",
        "installed": True,
        "stickers": [],
    }]
    res = await st.signal_list_sticker_packs()
    data = json.loads(res.content[0]["text"])
    entry = next(e for e in data if e["pack_id"] == "EXT")
    assert entry["source"] == "external"
    assert entry["label"] == ""


async def test_add_stickers_to_pack_surfaces_fetch_failure(
    fake_channel, tmp_path,
) -> None:
    """A pack where we can't decrypt one existing sticker must fail
    loud — quietly continuing would "eat" that sticker from the
    new pack."""
    fake_channel.client.list_sticker_packs.return_value = [{
        "packId": "P",
        "title": "x",
        "author": "y",
        "stickers": [{"id": 0, "emoji": "🙂"}],
    }]
    async def _fail(*_a, **_kw):
        return None
    fake_channel.client.get_sticker = _fail
    s0 = _write_valid_sticker(tmp_path / "a.webp")
    res = await st.signal_add_stickers_to_pack(
        "P",
        new_stickers=[{"path": str(s0), "emoji": "🦀"}],
    )
    assert "failed to fetch existing sticker" in res.content[0]["text"]
    fake_channel.client.upload_sticker_pack.assert_not_awaited()
