# -*- coding: utf-8 -*-
# pylint: disable=protected-access
"""Unit tests for WhatsApp channel."""
from __future__ import annotations

import asyncio
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

# ---------------------------------------------------------------------------
# Mock neonize before importing the channel module so the import does not fail
# in environments where neonize is not installed.
# ---------------------------------------------------------------------------

_neonize_mods = [
    "neonize",
    "neonize.aioze",
    "neonize.aioze.client",
    "neonize.events",
    "neonize.utils",
    "neonize.proto",
    "neonize.proto.waE2E",
    "neonize.proto.waE2E.WAWebProtobufsE2E_pb2",
]
for mod in _neonize_mods:
    if mod not in sys.modules:
        sys.modules[mod] = MagicMock()

# Provide lightweight stubs that the channel code actually touches
_utils_mod = sys.modules["neonize.utils"]
_utils_mod.build_jid = lambda user, server: MagicMock(User=user, Server=server)

# Ensure NEONIZE_AVAILABLE is True so WhatsAppChannel can be instantiated
from qwenpaw.app.channels.whatsapp import channel as _wa_mod

_wa_mod.NEONIZE_AVAILABLE = True
_wa_mod.NewAClient = MagicMock

from agentscope_runtime.engine.schemas.agent_schemas import (
    TextContent,
    ImageContent,
    AudioContent,
    VideoContent,
    FileContent,
    ContentType,
)

from qwenpaw.app.channels.whatsapp.channel import (
    WhatsAppChannel,
    _jid_to_str,
    _str_to_jid,
    _is_group_jid,
    _MEDIA_DIR,
    WHATSAPP_MAX_TEXT_LENGTH,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_channel(**overrides: Any) -> WhatsAppChannel:
    """Create a WhatsAppChannel with dummy process handler."""

    async def _noop_process(_request):
        yield  # pragma: no cover

    defaults = {
        "process": _noop_process,
        "enabled": True,
        "auth_dir": tempfile.mkdtemp(),
    }
    defaults.update(overrides)
    ch = WhatsAppChannel(**defaults)
    ch._client = MagicMock()
    ch._connected = True
    return ch


def _make_proto_message(**fields):
    """Build a lightweight mock that behaves like a protobuf WAMessage.

    Usage::

        msg = _make_proto_message(conversation="hello")
        msg = _make_proto_message(
            extendedTextMessage=MagicMock(text="hi", contextInfo=MagicMock(...)),
        )
    """
    msg = MagicMock()
    # HasField returns True only for keys explicitly supplied
    _present = set(fields.keys())
    msg.HasField = lambda name: name in _present
    for k, v in fields.items():
        setattr(msg, k, v)
    # Provide defaults for commonly accessed scalar fields
    if "conversation" not in fields:
        msg.conversation = ""
    return msg


# ===================================================================
# TestExtractMessageContent
# ===================================================================


class TestExtractMessageContent:
    async def test_text_conversation(self):
        ch = _make_channel()
        msg = _make_proto_message(conversation="hello world")
        body, parts = await ch._extract_message_content(
            MagicMock(),
            msg,
            "id1",
        )
        assert body == "hello world"
        assert len(parts) == 1
        assert parts[0].type == ContentType.TEXT
        assert parts[0].text == "hello world"

    async def test_extended_text_message(self):
        ch = _make_channel()
        etm = MagicMock()
        etm.text = "extended hello"
        msg = _make_proto_message(extendedTextMessage=etm)
        msg.conversation = ""
        body, parts = await ch._extract_message_content(
            MagicMock(),
            msg,
            "id2",
        )
        assert body == "extended hello"
        assert any(
            p.text == "extended hello" for p in parts if hasattr(p, "text")
        )

    async def test_image_with_caption(self):
        ch = _make_channel()
        img_msg = MagicMock()
        img_msg.caption = "nice photo"
        client = MagicMock()
        client.download_any = AsyncMock()
        msg = _make_proto_message(imageMessage=img_msg)
        msg.conversation = ""

        body, parts = await ch._extract_message_content(client, msg, "id3")
        # Caption should appear as text
        text_parts = [p for p in parts if hasattr(p, "text")]
        assert any("nice photo" in p.text for p in text_parts)
        # Image content part should be present
        img_parts = [p for p in parts if p.type == ContentType.IMAGE]
        assert len(img_parts) == 1
        client.download_any.assert_called_once()

    async def test_audio_ptt(self):
        ch = _make_channel()
        audio = MagicMock()
        audio.ptt = True
        client = MagicMock()
        client.download_any = AsyncMock()
        msg = _make_proto_message(audioMessage=audio)
        msg.conversation = ""

        body, parts = await ch._extract_message_content(client, msg, "id4")
        audio_parts = [p for p in parts if p.type == ContentType.AUDIO]
        assert len(audio_parts) == 1
        # PTT uses .ogg extension
        assert audio_parts[0].data.endswith(".ogg")

    async def test_location_static(self):
        # Static pin shared via WhatsApp's "Attach > Location" → Send
        # current location.  Without explicit handling these arrive
        # with empty body and get silently dropped before group-history
        # storage; the agent then truthfully replies "I can't see any
        # location" because none ever reached it.
        ch = _make_channel()
        loc = MagicMock()
        loc.degreesLatitude = 35.6586
        loc.degreesLongitude = 139.7454
        loc.name = "Tokyo Tower"
        loc.address = "4-2-8 Shibakoen, Minato City"
        loc.URL = ""
        loc.accuracyInMeters = 0
        loc.speedInMps = 0.0
        loc.isLive = False
        loc.comment = ""
        msg = _make_proto_message(locationMessage=loc)
        msg.conversation = ""

        body, parts = await ch._extract_message_content(
            MagicMock(),
            msg,
            "loc1",
        )
        # Coords + place name must reach body so the per-message
        # log line and the agent both see something non-empty.
        assert "35.658600,139.745400" in body
        assert "Tokyo Tower" in body
        assert "[Location]" in body
        # Same payload as a TextContent block in content_parts so
        # text-only and multimodal models both have it.
        text_parts = [p for p in parts if p.type == ContentType.TEXT]
        assert any("[Location]" in p.text for p in text_parts)
        # Google Maps URL appended for one-click open by humans
        # reading logs and any tool the agent might call.
        assert any("maps.google.com" in p.text for p in text_parts)

    async def test_live_location_standalone_is_suppressed(self):
        # Live location streams every ~5-15s during an active share.
        # Forwarding each one as its own agent turn pollutes history
        # and balloons input token cost.  Standalone updates must be
        # cached only — body+parts empty so the caller's
        # ``if not content_parts: return`` skips dispatch.
        ch = _make_channel()
        loc = MagicMock()
        loc.degreesLatitude = 35.0
        loc.degreesLongitude = 139.0
        loc.accuracyInMeters = 12
        loc.speedInMps = 1.5
        loc.caption = "on the way"
        msg = _make_proto_message(liveLocationMessage=loc)
        msg.conversation = ""

        body, parts = await ch._extract_message_content(
            MagicMock(),
            msg,
            "live1",
            sender_str="alice@s.whatsapp.net",
            chat_str="grp@g.us",
        )
        assert body == ""
        assert parts == []
        # But the fix IS cached for the next real message from alice.
        cached = ch._live_location_cache.get(
            ("grp@g.us", "alice@s.whatsapp.net"),
        )
        assert cached is not None
        cached_text, _ts = cached
        assert "[Live Location]" in cached_text
        assert "on the way" in cached_text

    async def test_live_location_piggybacks_on_next_text(self):
        # The whole point of suppression: when alice TEXTS after
        # streaming her location, the agent sees what she said WITH
        # her position, in one message — not 30 location turns plus
        # one text turn.
        ch = _make_channel()
        sender = "alice@s.whatsapp.net"
        chat = "grp@g.us"
        # First: a live-location update (suppressed, cached)
        loc = MagicMock()
        loc.degreesLatitude = 35.0
        loc.degreesLongitude = 139.0
        loc.accuracyInMeters = 5
        loc.speedInMps = 0.0
        loc.caption = ""
        await ch._extract_message_content(
            MagicMock(),
            _make_proto_message(liveLocationMessage=loc),
            "live1",
            sender_str=sender,
            chat_str=chat,
        )
        # Then: a regular text from the same sender
        text_msg = _make_proto_message(conversation="I'm here, look")
        body, parts = await ch._extract_message_content(
            MagicMock(),
            text_msg,
            "txt1",
            sender_str=sender,
            chat_str=chat,
        )
        assert "I'm here, look" in body
        # Cached fix attached as a separate text block.
        text_parts = [p for p in parts if p.type == ContentType.TEXT]
        assert any("[Live location" in p.text for p in text_parts)
        assert any("[Live Location]" in p.text for p in text_parts)
        assert any("35.000000,139.000000" in p.text for p in text_parts)

    async def test_live_location_no_piggyback_for_other_sender(self):
        # Bob's text must NOT carry alice's cached live location —
        # cache key is (chat, sender), so cross-sender bleed is
        # impossible by construction, but assert it explicitly so a
        # future refactor that flattens to chat-only is caught.
        ch = _make_channel()
        chat = "grp@g.us"
        loc = MagicMock()
        loc.degreesLatitude = 35.0
        loc.degreesLongitude = 139.0
        loc.accuracyInMeters = 0
        loc.speedInMps = 0.0
        loc.caption = ""
        await ch._extract_message_content(
            MagicMock(),
            _make_proto_message(liveLocationMessage=loc),
            "live1",
            sender_str="alice@s.whatsapp.net",
            chat_str=chat,
        )
        body, parts = await ch._extract_message_content(
            MagicMock(),
            _make_proto_message(conversation="hi"),
            "txt1",
            sender_str="bob@s.whatsapp.net",
            chat_str=chat,
        )
        assert body == "hi"
        # No piggyback — bob never shared a location.
        text_parts = [p for p in parts if p.type == ContentType.TEXT]
        assert not any("[Live location" in p.text for p in text_parts)

    async def test_live_location_stale_entry_evicted_on_read(self):
        # 30-min TTL: a fix older than the window must NOT piggyback,
        # AND must be removed from the cache so the dict stays bounded
        # over a long-running session without explicit eviction.
        ch = _make_channel()
        sender = "alice@s.whatsapp.net"
        chat = "grp@g.us"
        # Inject a stale entry directly: 31 minutes old.
        ch._live_location_cache[(chat, sender)] = (
            "[Live Location] 35.0,139.0\nhttps://maps.google.com/?q=35.0,139.0",
            time.monotonic() - 31 * 60,
        )
        body, parts = await ch._extract_message_content(
            MagicMock(),
            _make_proto_message(conversation="hi"),
            "txt1",
            sender_str=sender,
            chat_str=chat,
        )
        assert body == "hi"
        text_parts = [p for p in parts if p.type == ContentType.TEXT]
        assert not any("[Live location" in p.text for p in text_parts)
        # Stale entry cleaned up.
        assert (chat, sender) not in ch._live_location_cache

    async def test_location_uses_proto_url_when_present(self):
        # When WhatsApp's own ``URL`` field is populated (manually-
        # attached link or a third-party share that knows the canonical
        # URL) prefer it over the synthesized maps.google.com fallback
        # — the original may be a maps.app.goo.gl short link the user
        # actually wants to surface.
        ch = _make_channel()
        loc = MagicMock()
        loc.degreesLatitude = 35.0
        loc.degreesLongitude = 139.0
        loc.name = ""
        loc.address = ""
        loc.URL = "https://maps.app.goo.gl/abc123"
        loc.accuracyInMeters = 0
        loc.speedInMps = 0.0
        loc.isLive = False
        loc.comment = ""
        msg = _make_proto_message(locationMessage=loc)
        msg.conversation = ""

        body, _parts = await ch._extract_message_content(
            MagicMock(),
            msg,
            "loc2",
        )
        assert "https://maps.app.goo.gl/abc123" in body
        assert "maps.google.com" not in body

    async def test_audio_non_ptt(self):
        ch = _make_channel()
        audio = MagicMock()
        audio.ptt = False
        client = MagicMock()
        client.download_any = AsyncMock()
        msg = _make_proto_message(audioMessage=audio)
        msg.conversation = ""

        body, parts = await ch._extract_message_content(client, msg, "id5")
        audio_parts = [p for p in parts if p.type == ContentType.AUDIO]
        assert len(audio_parts) == 1
        # Non-PTT uses .m4a extension
        assert audio_parts[0].data.endswith(".m4a")

    async def test_document_path_traversal_sanitized(self):
        ch = _make_channel()
        doc = MagicMock()
        doc.fileName = "../../../etc/passwd"
        client = MagicMock()
        client.download_any = AsyncMock()
        msg = _make_proto_message(documentMessage=doc)
        msg.conversation = ""

        body, parts = await ch._extract_message_content(client, msg, "id6")
        file_parts = [p for p in parts if p.type == ContentType.FILE]
        assert len(file_parts) == 1
        # Path should be sanitized — only the final component "passwd"
        saved_path = Path(file_parts[0].file_url)
        assert saved_path.name == "passwd"
        assert ".." not in str(saved_path)
        # Should be within media dir
        assert str(ch._media_dir) in str(saved_path.parent)


# ===================================================================
# TestExtractQuoteContent
# ===================================================================


class TestExtractQuoteContent:
    async def test_quote_with_text_only(self):
        ch = _make_channel()
        ctx = MagicMock()
        ctx.HasField = lambda name: name == "quotedMessage"
        ctx.participant = "123456@s.whatsapp.net"
        quoted = _make_proto_message(conversation="original text")
        # Ensure text extraction works
        quoted.extendedTextMessage = MagicMock()
        quoted.HasField = (
            lambda name: name == "extendedTextMessage"
            if name == "extendedTextMessage"
            else False
        )
        quoted.extendedTextMessage.text = ""
        # Use conversation path
        quoted.conversation = "original text"
        quoted.HasField = lambda name: False  # No media fields
        ctx.quotedMessage = quoted
        ctx.stanzaId = "stanza1"

        etm = MagicMock()
        etm.text = "reply text"
        etm.contextInfo = ctx
        msg = _make_proto_message(extendedTextMessage=etm)

        parts = await ch._extract_quote_content(MagicMock(), msg)
        assert len(parts) >= 1
        text_parts = [p for p in parts if hasattr(p, "text")]
        assert any("UNTRUSTED reply-to" in p.text for p in text_parts)
        assert any("original text" in p.text for p in text_parts)

    async def test_quote_with_image_download(self):
        ch = _make_channel()
        ctx = MagicMock()
        ctx.HasField = lambda name: name == "quotedMessage"
        ctx.participant = "sender@s.whatsapp.net"
        ctx.stanzaId = "stanza2"

        img = MagicMock()
        img.caption = "photo caption"

        quoted = MagicMock()
        quoted.conversation = ""
        quoted.HasField = lambda name: name == "imageMessage"
        quoted.imageMessage = img
        ctx.quotedMessage = quoted

        etm = MagicMock()
        etm.text = "responding"
        etm.contextInfo = ctx
        msg = _make_proto_message(extendedTextMessage=etm)

        # Simulate download failure (common for quoted messages)
        client = MagicMock()
        client.download_any = AsyncMock(side_effect=Exception("no media key"))

        parts = await ch._extract_quote_content(client, msg)
        assert len(parts) >= 1
        # Should have text description mentioning image
        text_parts = [p for p in parts if hasattr(p, "text")]
        combined = " ".join(p.text for p in text_parts)
        assert "UNTRUSTED reply-to" in combined
        assert "image" in combined.lower()

    async def test_quote_to_static_location(self):
        # Replying to a static pin: WhatsApp embeds the locationMessage
        # proto into the reply's contextInfo, so the quote extractor has
        # raw lat/lng even when the original was an old message.  Quote
        # block must surface coords + place name + a maps URL.
        ch = _make_channel()
        ctx = MagicMock()
        ctx.HasField = lambda name: name == "quotedMessage"
        ctx.participant = "sender@s.whatsapp.net"
        ctx.stanzaId = "stanza_loc"

        loc = MagicMock()
        loc.degreesLatitude = 35.6586
        loc.degreesLongitude = 139.7454
        loc.name = "Tokyo Tower"
        loc.address = "Shibakoen"
        loc.URL = ""
        loc.accuracyInMeters = 0
        loc.speedInMps = 0.0
        loc.isLive = False
        loc.comment = ""

        quoted = MagicMock()
        quoted.conversation = ""
        quoted.HasField = lambda name: name == "locationMessage"
        quoted.locationMessage = loc
        ctx.quotedMessage = quoted

        etm = MagicMock()
        etm.text = "going there"
        etm.contextInfo = ctx
        msg = _make_proto_message(extendedTextMessage=etm)

        parts = await ch._extract_quote_content(MagicMock(), msg)
        text = next(p.text for p in parts if hasattr(p, "text"))
        assert "[Location]" in text
        assert "35.658600,139.745400" in text
        assert "Tokyo Tower" in text
        assert "maps.google.com" in text

    async def test_quote_to_live_location_works_even_when_suppressed(self):
        # The whole point of suppression: standalone liveLocationMessage
        # updates never reach agent history.  But when the user replies
        # to one in WhatsApp's UI later, contextInfo carries a snapshot
        # of the original proto — the quote extractor reads that proto
        # directly, NOT history, so suppression doesn't break this.
        # Reply text must come through alongside the snapshot location.
        ch = _make_channel()
        ctx = MagicMock()
        ctx.HasField = lambda name: name == "quotedMessage"
        ctx.participant = "sender@s.whatsapp.net"
        ctx.stanzaId = "stanza_live"

        loc = MagicMock()
        loc.degreesLatitude = 35.0
        loc.degreesLongitude = 139.0
        loc.accuracyInMeters = 8
        loc.speedInMps = 2.1
        loc.caption = "moving"

        quoted = MagicMock()
        quoted.conversation = ""
        quoted.HasField = lambda name: name == "liveLocationMessage"
        quoted.liveLocationMessage = loc
        ctx.quotedMessage = quoted

        etm = MagicMock()
        etm.text = "you were here"
        etm.contextInfo = ctx
        msg = _make_proto_message(extendedTextMessage=etm)

        parts = await ch._extract_quote_content(MagicMock(), msg)
        text = next(p.text for p in parts if hasattr(p, "text"))
        assert "[Live Location]" in text
        assert "35.000000,139.000000" in text
        assert "±8m" in text
        assert "moving" in text

    async def test_quote_with_image_download_success_emits_path(self):
        # On a successful download the reply-to block MUST surface
        # the file path so the agent can pass it to tools (codex
        # image i2i, describe_image, ocr) — without the path the
        # reference is dead text.
        ch = _make_channel()
        ctx = MagicMock()
        ctx.HasField = lambda name: name == "quotedMessage"
        ctx.participant = "sender@s.whatsapp.net"
        ctx.stanzaId = "stanza_img_ok"

        img = MagicMock()
        img.caption = ""

        quoted = MagicMock()
        quoted.conversation = ""
        quoted.HasField = lambda name: name == "imageMessage"
        quoted.imageMessage = img
        ctx.quotedMessage = quoted

        etm = MagicMock()
        etm.text = "edit this"
        etm.contextInfo = ctx
        msg = _make_proto_message(extendedTextMessage=etm)

        # Write bytes to the target path so the existence check passes.
        async def _fake_download(_proto, *, path):
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_bytes(b"\xff\xd8\xff" + b"x" * 100)

        client = MagicMock()
        client.download_any = AsyncMock(side_effect=_fake_download)

        parts = await ch._extract_quote_content(client, msg)
        # Two parts: text block + ImageContent attached.
        assert any(isinstance(p, ImageContent) for p in parts)
        text = next(p.text for p in parts if hasattr(p, "text"))
        # Path must appear verbatim inside the Media: line.
        assert "Media: image:" in text
        assert "wa_quote_stanza_img_o" in text

    async def test_quote_with_video_download_success_emits_path(self):
        # Generalises the image-path guarantee to the other media
        # types the generalised ``_try_download`` helper now handles.
        ch = _make_channel()
        ctx = MagicMock()
        ctx.HasField = lambda name: name == "quotedMessage"
        ctx.participant = "sender@s.whatsapp.net"
        ctx.stanzaId = "stanza_vid_ok"

        quoted = MagicMock()
        quoted.conversation = ""
        quoted.HasField = lambda name: name == "videoMessage"
        ctx.quotedMessage = quoted

        etm = MagicMock()
        etm.text = "describe this clip"
        etm.contextInfo = ctx
        msg = _make_proto_message(extendedTextMessage=etm)

        async def _fake_download(_proto, *, path):
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_bytes(b"x" * 1024)

        client = MagicMock()
        client.download_any = AsyncMock(side_effect=_fake_download)

        parts = await ch._extract_quote_content(client, msg)
        text = next(p.text for p in parts if hasattr(p, "text"))
        assert "Media: video:" in text
        assert ".mp4" in text

    async def test_quote_with_video_described(self):
        """Quoted video produces 'Media: video' in reply-to block."""
        ch = _make_channel()
        ctx = MagicMock()
        ctx.HasField = lambda name: name == "quotedMessage"
        ctx.participant = "video_sender@s.whatsapp.net"
        ctx.stanzaId = "stanza_video"

        quoted = MagicMock()
        quoted.conversation = ""
        quoted.HasField = lambda name: name == "videoMessage"
        ctx.quotedMessage = quoted

        etm = MagicMock()
        etm.text = "nice clip"
        etm.contextInfo = ctx
        msg = _make_proto_message(extendedTextMessage=etm)

        parts = await ch._extract_quote_content(MagicMock(), msg)
        text_parts = [p for p in parts if hasattr(p, "text")]
        assert len(text_parts) == 1
        assert "UNTRUSTED reply-to" in text_parts[0].text
        assert "Media: video" in text_parts[0].text

    async def test_quote_with_voice_note_described(self):
        """Quoted audio with ptt=True is described as 'voice note'."""
        ch = _make_channel()
        ctx = MagicMock()
        ctx.HasField = lambda name: name == "quotedMessage"
        ctx.participant = "audio_sender@s.whatsapp.net"
        ctx.stanzaId = "stanza_audio"

        audio_msg = MagicMock()
        audio_msg.ptt = True

        quoted = MagicMock()
        quoted.conversation = ""
        quoted.HasField = lambda name: name == "audioMessage"
        quoted.audioMessage = audio_msg
        ctx.quotedMessage = quoted

        etm = MagicMock()
        etm.text = "hear this"
        etm.contextInfo = ctx
        msg = _make_proto_message(extendedTextMessage=etm)

        parts = await ch._extract_quote_content(MagicMock(), msg)
        text = parts[0].text
        assert "Media: voice note" in text

    async def test_quote_with_audio_non_ptt_described(self):
        """Quoted audio with ptt=False is described as 'audio'."""
        ch = _make_channel()
        ctx = MagicMock()
        ctx.HasField = lambda name: name == "quotedMessage"
        ctx.participant = "audio_sender@s.whatsapp.net"
        ctx.stanzaId = "stanza_audio2"

        audio_msg = MagicMock()
        audio_msg.ptt = False

        quoted = MagicMock()
        quoted.conversation = ""
        quoted.HasField = lambda name: name == "audioMessage"
        quoted.audioMessage = audio_msg
        ctx.quotedMessage = quoted

        etm = MagicMock()
        etm.text = ""
        etm.contextInfo = ctx
        msg = _make_proto_message(extendedTextMessage=etm)

        parts = await ch._extract_quote_content(MagicMock(), msg)
        text = parts[0].text
        assert "Media: audio" in text
        assert "voice note" not in text

    async def test_quote_with_document_described(self):
        """Quoted document is described with its filename."""
        ch = _make_channel()
        ctx = MagicMock()
        ctx.HasField = lambda name: name == "quotedMessage"
        ctx.participant = "doc_sender@s.whatsapp.net"
        ctx.stanzaId = "stanza_doc"

        doc_msg = MagicMock()
        doc_msg.fileName = "report.pdf"

        quoted = MagicMock()
        quoted.conversation = ""
        quoted.HasField = lambda name: name == "documentMessage"
        quoted.documentMessage = doc_msg
        ctx.quotedMessage = quoted

        etm = MagicMock()
        etm.text = "read this"
        etm.contextInfo = ctx
        msg = _make_proto_message(extendedTextMessage=etm)

        parts = await ch._extract_quote_content(MagicMock(), msg)
        text = parts[0].text
        assert "file: report.pdf" in text

    async def test_quote_with_sticker_described(self):
        """Quoted sticker is described as 'sticker'."""
        ch = _make_channel()
        ctx = MagicMock()
        ctx.HasField = lambda name: name == "quotedMessage"
        ctx.participant = "s@s.whatsapp.net"
        ctx.stanzaId = "stanza_sticker"

        quoted = MagicMock()
        quoted.conversation = ""
        quoted.HasField = lambda name: name == "stickerMessage"
        ctx.quotedMessage = quoted

        etm = MagicMock()
        etm.text = "lol"
        etm.contextInfo = ctx
        msg = _make_proto_message(extendedTextMessage=etm)

        parts = await ch._extract_quote_content(MagicMock(), msg)
        text = parts[0].text
        assert "Media: sticker" in text

    async def test_no_quoted_message_returns_empty(self):
        ch = _make_channel()
        # Message with no contextInfo
        msg = _make_proto_message(conversation="plain message")
        parts = await ch._extract_quote_content(MagicMock(), msg)
        assert parts == []

    async def test_quote_with_album_describes_counts(self):
        """Replying to a multi-image album: ``albumMessage`` is just
        a count announcement (the actual images arrive as separate
        messages), so the quote block must surface the labelled
        placeholder ``"album with N images + M videos"`` rather
        than returning an empty parts list — without this fix the
        agent loses any signal that the user is referring to a
        multi-media bundle.
        """
        ch = _make_channel()
        ctx = MagicMock()
        ctx.HasField = lambda name: name == "quotedMessage"
        ctx.participant = "alb_sender@s.whatsapp.net"
        ctx.stanzaId = "stanza_album"

        album = MagicMock()
        album.expectedImageCount = 3
        album.expectedVideoCount = 1

        quoted = MagicMock()
        quoted.conversation = ""
        quoted.HasField = lambda name: name == "albumMessage"
        quoted.albumMessage = album
        ctx.quotedMessage = quoted

        etm = MagicMock()
        etm.text = "look at these"
        etm.contextInfo = ctx
        msg = _make_proto_message(extendedTextMessage=etm)

        parts = await ch._extract_quote_content(MagicMock(), msg)
        assert len(parts) >= 1
        text = parts[0].text
        assert "UNTRUSTED reply-to" in text
        assert "Media: album with 3 images + 1 video" in text

    async def test_quote_with_image_only_album(self):
        """Album with images only (no videos) drops the video
        clause from the placeholder."""
        ch = _make_channel()
        ctx = MagicMock()
        ctx.HasField = lambda name: name == "quotedMessage"
        ctx.participant = "alb_sender@s.whatsapp.net"
        ctx.stanzaId = "stanza_album_img"

        album = MagicMock()
        album.expectedImageCount = 4
        album.expectedVideoCount = 0

        quoted = MagicMock()
        quoted.conversation = ""
        quoted.HasField = lambda name: name == "albumMessage"
        quoted.albumMessage = album
        ctx.quotedMessage = quoted

        etm = MagicMock()
        etm.text = ""
        etm.contextInfo = ctx
        msg = _make_proto_message(extendedTextMessage=etm)

        parts = await ch._extract_quote_content(MagicMock(), msg)
        text = parts[0].text
        assert "Media: album with 4 images" in text
        assert "video" not in text

    async def test_inbound_album_message_finds_contextinfo(self):
        """If the user's REPLY message is itself the album header,
        the contextInfo lives on ``albumMessage`` — the field-
        scanning loop in ``_extract_quote_content`` must include
        ``albumMessage`` so the quote is still extracted.
        """
        ch = _make_channel()
        ctx = MagicMock()
        ctx.HasField = lambda name: name == "quotedMessage"
        ctx.participant = "sender@s.whatsapp.net"
        ctx.stanzaId = "stanza_album_inbound"

        # Quoted message is a plain text — we just need to verify
        # the loop reaches contextInfo via the album field.
        quoted = MagicMock()
        quoted.conversation = "earlier text"
        quoted.HasField = lambda name: False
        ctx.quotedMessage = quoted

        album = MagicMock()
        album.contextInfo = ctx
        # Inbound is an album header (no extendedTextMessage).
        msg = _make_proto_message(albumMessage=album)

        parts = await ch._extract_quote_content(MagicMock(), msg)
        assert len(parts) >= 1
        assert "earlier text" in parts[0].text

    async def test_quote_participant_lid_resolution(self):
        ch = _make_channel()
        # Pre-populate LID cache
        ch._lid_cache["123456@lid"] = {"phone": "85251159218", "name": "Alice"}

        ctx = MagicMock()
        ctx.HasField = lambda name: name == "quotedMessage"
        ctx.participant = "123456@lid"
        ctx.stanzaId = "stanza_lid"

        quoted = MagicMock()
        quoted.conversation = "lid message"
        quoted.HasField = lambda name: False
        ctx.quotedMessage = quoted

        etm = MagicMock()
        etm.text = "reply"
        etm.contextInfo = ctx
        msg = _make_proto_message(extendedTextMessage=etm)

        parts = await ch._extract_quote_content(MagicMock(), msg)
        assert len(parts) >= 1
        text_parts = [p for p in parts if hasattr(p, "text")]
        combined = " ".join(p.text for p in text_parts)
        # Should show resolved phone number, not raw LID
        assert "+85251159218" in combined


# ===================================================================
# TestCheckAccess
# ===================================================================


class TestCheckAccess:
    def test_group_policy_open_allows(self):
        ch = _make_channel(group_policy="open")
        assert (
            ch._check_access(
                is_group=True,
                chat_str="groupA@g.us",
                sender_str="user@s.whatsapp.net",
                sender_jid=MagicMock(),
                client=MagicMock(),
                msg=MagicMock(),
                body="hi",
            )
            is True
        )

    def test_group_policy_allowlist_group_in_list(self):
        ch = _make_channel(group_policy="allowlist", groups=["groupA@g.us"])
        assert (
            ch._check_access(
                is_group=True,
                chat_str="groupA@g.us",
                sender_str="user@s.whatsapp.net",
                sender_jid=MagicMock(),
                client=MagicMock(),
                msg=MagicMock(),
                body="hi",
            )
            is True
        )

    def test_group_policy_allowlist_group_not_in_list(self):
        ch = _make_channel(group_policy="allowlist", groups=["groupA@g.us"])
        assert (
            ch._check_access(
                is_group=True,
                chat_str="groupB@g.us",
                sender_str="user@s.whatsapp.net",
                sender_jid=MagicMock(),
                client=MagicMock(),
                msg=MagicMock(),
                body="hi",
            )
            is False
        )

    def test_group_policy_allowlist_empty_groups_blocks_all(self):
        ch = _make_channel(group_policy="allowlist", groups=[])
        assert (
            ch._check_access(
                is_group=True,
                chat_str="anygroup@g.us",
                sender_str="user@s.whatsapp.net",
                sender_jid=MagicMock(),
                client=MagicMock(),
                msg=MagicMock(),
                body="hi",
            )
            is False
        )

    def test_dm_policy_open_allows(self):
        """DM access is not blocked in _check_access (async check in _on_message)."""
        ch = _make_channel(dm_policy="open")
        assert (
            ch._check_access(
                is_group=False,
                chat_str="user@s.whatsapp.net",
                sender_str="user@s.whatsapp.net",
                sender_jid=MagicMock(),
                client=MagicMock(),
                msg=MagicMock(),
                body="hi",
            )
            is True
        )

    def test_group_allow_from_stored(self):
        """group_allow_from is stored on the channel for use in _on_message."""
        ch = _make_channel(group_allow_from=["+85251159218"])
        assert ch._group_allow_from == ["+85251159218"]


# ===================================================================
# TestGroupHistory
# ===================================================================


class TestGroupHistory:
    def test_non_mentioned_message_recorded(self):
        """When bot is NOT mentioned, message should be buffered in history."""
        ch = _make_channel(require_mention=True)
        ch._my_jid = MagicMock(User="botlid")
        ch._bot_lid = "botlid"
        ch._bot_phone = "85200000000"

        chat_str = "group123@g.us"
        history = ch._group_history.setdefault(chat_str, [])
        # Simulate recording (as done in _on_message)
        history.append(
            {"sender": "+85251159218", "body": "hello", "ts": "12345"},
        )
        assert len(ch._group_history[chat_str]) == 1

    def test_history_limit_enforced(self):
        ch = _make_channel()
        ch._group_history_limit = 5
        chat_str = "group123@g.us"
        history = ch._group_history.setdefault(chat_str, [])
        for i in range(10):
            history.append(
                {"sender": f"user{i}", "body": f"msg{i}", "ts": str(i)},
            )
        # Trim like the channel does
        if len(history) > ch._group_history_limit:
            ch._group_history[chat_str] = history[-ch._group_history_limit :]
        assert len(ch._group_history[chat_str]) == 5
        assert ch._group_history[chat_str][0]["body"] == "msg5"

    def test_history_injected_when_mentioned(self):
        """When bot IS mentioned, buffered history should be injected."""
        ch = _make_channel()
        chat_str = "group123@g.us"
        ch._group_history[chat_str] = [
            {"sender": "+852111", "body": "earlier msg 1", "ts": "1"},
            {"sender": "+852222", "body": "earlier msg 2", "ts": "2"},
        ]
        # Simulate the injection logic from _on_message
        history = ch._group_history.get(chat_str, [])
        ctx_lines = []
        for h in history[-10:]:
            ctx_lines.append(f"  {h['sender']}: {h['body']}")
        ctx_text = (
            "--- Recent group messages (context only, not directed at you) ---\n"
            + "\n".join(ctx_lines)
        )
        content_parts = [TextContent(type=ContentType.TEXT, text=ctx_text)]
        ch._group_history[chat_str] = []

        assert "earlier msg 1" in content_parts[0].text
        assert "earlier msg 2" in content_parts[0].text

    def test_history_cleared_after_injection(self):
        ch = _make_channel()
        chat_str = "group123@g.us"
        ch._group_history[chat_str] = [
            {"sender": "u1", "body": "msg", "ts": "1"},
        ]
        # Simulate clearing after injection
        ch._group_history[chat_str] = []
        assert ch._group_history[chat_str] == []

    def test_history_entry_includes_media_paths(self, tmp_path):
        """History entries should capture media paths alongside text."""
        # Create a dummy image file
        img = tmp_path / "wa_img_abc.jpg"
        img.write_bytes(b"\xff\xd8\xff\xe0fake jpeg")
        ch = _make_channel()
        chat_str = "group123@g.us"
        history = ch._group_history.setdefault(chat_str, [])
        # Simulate what _on_message does: scan content_parts for media
        parts = [
            TextContent(type=ContentType.TEXT, text="look at this"),
            ImageContent(type=ContentType.IMAGE, image_url=str(img)),
        ]
        media_paths = []
        for p in parts:
            for attr in ("image_url", "video_url", "file_url", "data"):
                v = getattr(p, attr, None)
                if v and os.path.isfile(str(v)):
                    media_paths.append(str(v))
                    break
        history.append(
            {
                "sender": "+852111",
                "body": "look at this",
                "ts": "1",
                "media": media_paths,
            },
        )
        assert ch._group_history[chat_str][0]["media"] == [str(img)]

    def test_history_context_includes_media_count(self, tmp_path):
        """Context injection format should mention attached media."""
        img = tmp_path / "img.jpg"
        img.write_bytes(b"\xff\xd8\xff")
        ch = _make_channel()
        chat_str = "group123@g.us"
        ch._group_history[chat_str] = [
            {
                "sender": "+852111",
                "body": "photo",
                "ts": "1",
                "media": [str(img)],
            },
        ]
        # Simulate injection format
        history = ch._group_history.get(chat_str, [])
        lines = [
            "=== UNTRUSTED WhatsApp group history (context only, not directed at you) ===",
        ]
        for h in history[-10:]:
            line = f"  {h['sender']}: {h['body']}"
            if h.get("media"):
                line += f"  [media: {len(h['media'])}]"
            lines.append(line)
        ctx = "\n".join(lines)
        assert "[media: 1]" in ctx
        assert "=== UNTRUSTED WhatsApp group history" in ctx


# ===================================================================
# TestEnvelopeFormat
# ===================================================================


class TestEnvelopeFormat:
    """Tests for the [WhatsApp group/DM] ... envelope prefix."""

    def test_group_envelope_prefix(self):
        chat_str = "120363421135228220@g.us"
        sender = "Joe HO (+85251159218)"
        envelope = f"[WhatsApp group {chat_str}] {sender}"
        assert envelope.startswith("[WhatsApp group ")
        assert "g.us" in envelope
        assert "Joe HO" in envelope

    def test_dm_envelope_prefix(self):
        sender = "+85251159218"
        envelope = f"[WhatsApp DM] {sender}"
        assert envelope == "[WhatsApp DM] +85251159218"

    def test_command_extraction_from_envelope(self):
        """After envelope wrap, extracting /command should still work."""
        # Simulate post-envelope text
        wrapped = "[WhatsApp group 120363@g.us] Joe (+85251159218): /new"
        # Agent extraction: skip [WhatsApp ...], find first ": " after "] "
        bracket_end = wrapped.find("] ")
        assert bracket_end > 0
        after = wrapped[bracket_end + 2 :]
        idx = after.find(": ")
        assert idx > 0
        raw = after[idx + 2 :]
        assert raw == "/new"


# ===================================================================
# TestMentionDetection (_is_bot_mentioned)
# ===================================================================


class TestMentionDetection:
    def _setup_channel(self) -> WhatsAppChannel:
        ch = _make_channel()
        ch._my_jid = MagicMock(User="botlid123")
        ch._bot_lid = "botlid123"
        ch._bot_phone = "85200000000"
        return ch

    def test_at_lid_in_body(self):
        ch = self._setup_channel()
        msg = _make_proto_message(conversation="hello @botlid123 test")
        assert ch._is_bot_mentioned(msg, "hello @botlid123 test") is True

    def test_at_phone_in_body(self):
        ch = self._setup_channel()
        msg = _make_proto_message(conversation="hello @85200000000 test")
        assert ch._is_bot_mentioned(msg, "hello @85200000000 test") is True

    def test_at_plus_phone_in_body(self):
        ch = self._setup_channel()
        msg = _make_proto_message(conversation="hello @+85200000000 test")
        assert ch._is_bot_mentioned(msg, "hello @+85200000000 test") is True

    def test_native_mentioned_jid_match(self):
        ch = self._setup_channel()
        ctx = MagicMock()
        jid = MagicMock()
        jid.User = "botlid123"
        ctx.mentionedJID = [jid]
        ctx.HasField = lambda name: False
        ctx.stanzaId = ""
        ctx.participant = ""

        etm = MagicMock()
        etm.text = "hey bot"
        etm.contextInfo = ctx
        msg = _make_proto_message(extendedTextMessage=etm)

        assert ch._is_bot_mentioned(msg, "hey bot") is True

    def test_reply_to_bot_message(self):
        ch = self._setup_channel()
        ctx = MagicMock()
        ctx.mentionedJID = []
        ctx.HasField = lambda name: name == "quotedMessage"
        ctx.stanzaId = "some_stanza"
        ctx.participant = "botlid123@lid"

        etm = MagicMock()
        etm.text = "replying"
        etm.contextInfo = ctx
        msg = _make_proto_message(extendedTextMessage=etm)

        assert ch._is_bot_mentioned(msg, "replying") is True

    def test_no_mention_returns_false(self):
        ch = self._setup_channel()
        msg = _make_proto_message(conversation="hello world")
        assert ch._is_bot_mentioned(msg, "hello world") is False

    def test_reply_to_bot_with_device_suffix(self):
        # Regression guard for the 2026-04-24 bug: WhatsApp JID/LID
        # format is ``<id>:<device>@<server>`` (e.g.
        # ``229661330157571:2@lid`` for the 2nd linked device).  The
        # ConnectedEv handler strips both ``:device`` and ``@server``
        # when stashing ``_bot_lid``, but the old
        # ``_is_bot_mentioned`` quote-participant parser only stripped
        # ``@server`` — leaving ``229661330157571:2`` on one side and
        # ``229661330157571`` on the other, so equality silently
        # failed.  Result: reply-to-bot in groups never triggered a
        # reply.
        ch = self._setup_channel()
        ctx = MagicMock()
        ctx.mentionedJID = []
        ctx.HasField = lambda name: name == "quotedMessage"
        ctx.stanzaId = "some_stanza"
        # Real WhatsApp format — note the ``:2`` device suffix.
        ctx.participant = "botlid123:2@lid"

        etm = MagicMock()
        etm.text = "replying"
        etm.contextInfo = ctx
        msg = _make_proto_message(extendedTextMessage=etm)

        assert ch._is_bot_mentioned(msg, "replying") is True

    def test_mentioned_jid_user_with_device_suffix(self):
        # Same normalization applies to the native ``mentionedJID``
        # path — protobuf gives us ``jid.User`` as ``id:device`` in
        # some WhatsApp deployments.
        ch = self._setup_channel()
        ctx = MagicMock()
        jid = MagicMock()
        jid.User = "botlid123:2"  # device suffix on User
        ctx.mentionedJID = [jid]
        ctx.HasField = lambda name: False
        ctx.stanzaId = ""
        ctx.participant = ""

        etm = MagicMock()
        etm.text = "hey bot"
        etm.contextInfo = ctx
        msg = _make_proto_message(extendedTextMessage=etm)

        assert ch._is_bot_mentioned(msg, "hey bot") is True


# ===================================================================
# TestUpdateConfigDeadClientRestart
# ===================================================================


class TestUpdateConfigDeadClientRestart:
    """Verify ``update_config`` triggers a full restart when the
    neonize client is dead (``_connected=False``).

    Without this, a Console-UI re-pair + config save would preserve
    the zombie client in-place and every send afterwards fails with
    ``websocket not connected`` / ``device JID missing`` — we saw
    this live on 2026-04-24.
    """

    @pytest.mark.asyncio
    async def test_dead_client_forces_full_restart(self):
        ch = _make_channel()
        ch._connected = False  # simulate the zombie state
        result = await ch.update_config(
            {
                "enabled": True,
                "auth_dir": ch._auth_dir,
            },
        )
        # False = caller (service_factories) does clone + replace_channel
        assert result is False

    @pytest.mark.asyncio
    async def test_healthy_client_allows_in_place_patch(self):
        ch = _make_channel()
        ch._connected = True  # client is alive
        ch._client.is_connected = True
        result = await ch.update_config(
            {
                "enabled": True,
                "auth_dir": ch._auth_dir,
                "send_read_receipts": False,  # a soft-patchable field
            },
        )
        assert result is True
        # Soft field took effect without restart.
        assert ch._send_read_receipts is False

    @pytest.mark.asyncio
    async def test_eof_zombie_forces_full_restart(self):
        """The 2026-06-07 zombie: a server-forced logout surfaced only as a
        websocket EOF (no DisconnectedEv), so ``_connected`` stayed stuck at
        True while the underlying socket was dead.  ``update_config`` must
        consult the authoritative ``is_connected`` and force a restart, else
        a Console re-pair + Save keeps the dead client and WhatsApp never
        comes back.
        """
        ch = _make_channel()
        ch._connected = True  # stale — DisconnectedEv never fired
        ch._client.is_connected = False  # whatsmeow says the socket is dead
        result = await ch.update_config(
            {
                "enabled": True,
                "auth_dir": ch._auth_dir,
            },
        )
        assert result is False


# ===================================================================
# TestSend
# ===================================================================


class TestSend:
    async def test_basic_text_send(self):
        ch = _make_channel()
        ch._client.send_message = AsyncMock()
        await ch.send("+85200000000", "hello", {})
        ch._client.send_message.assert_called_once()
        args = ch._client.send_message.call_args[0]
        assert args[1] == "hello"

    async def test_empty_text_noop(self):
        ch = _make_channel()
        ch._client.send_message = AsyncMock()
        await ch.send("+85200000000", "", {})
        ch._client.send_message.assert_not_called()

    async def test_disabled_noop(self):
        ch = _make_channel(enabled=False)
        ch._client = MagicMock()
        ch._client.send_message = AsyncMock()
        await ch.send("+85200000000", "hi")
        ch._client.send_message.assert_not_called()

    async def test_image_path_restricted_to_media_dir(self):
        ch = _make_channel()
        ch._client.send_image = AsyncMock()
        ch._client.send_message = AsyncMock()
        # Image outside media dir should be blocked
        text = "[Image: /etc/passwd] check this"
        await ch.send("+85200000000", text, {})
        ch._client.send_image.assert_not_called()

    async def test_text_chunking(self):
        ch = _make_channel(text_chunk_limit=10)
        ch._client.send_message = AsyncMock()
        text = "AAAAAAAAAA" + "BBBBBBBBBB"  # 20 chars, limit 10
        await ch.send("+85200000000", text, {})
        assert ch._client.send_message.call_count == 2

    async def test_markdown_code_fence_language_removed_for_whatsapp(self):
        ch = _make_channel()
        ch._client.send_message = AsyncMock()
        text = "Here:\n```text\nhello\n```\nDone"

        await ch.send("+85200000000", text, {})

        args = ch._client.send_message.call_args[0]
        assert args[1] == "Here:\n```\nhello\n```\nDone"

    async def test_plain_code_fence_kept_for_whatsapp(self):
        ch = _make_channel()
        ch._client.send_message = AsyncMock()
        text = "```\nhello\n```"

        await ch.send("+85200000000", text, {})

        args = ch._client.send_message.call_args[0]
        assert args[1] == text


class TestParticipantMentionRewrite:
    """Outbound auto-tagging — convert ``+<digits>`` to ``@<digits>`` only
    for digits that match a real group participant.  Pure unit tests against
    the static rewrite helper plus integration tests for the send() path
    that exercise the participant cache via mocked ``get_group_info``.
    """

    def test_rewrite_known_participant(self):
        out = WhatsAppChannel._rewrite_participant_mentions(
            "等 +85260113079 reply 我哋",
            {"85260113079"},
        )
        assert out == "等 @85260113079 reply 我哋"

    def test_rewrite_strips_at_plus_prefix(self):
        out = WhatsAppChannel._rewrite_participant_mentions(
            "@+85260113079 你睇下",
            {"85260113079"},
        )
        assert out == "@85260113079 你睇下"

    def test_unknown_phone_left_alone(self):
        out = WhatsAppChannel._rewrite_participant_mentions(
            "客戶嗰邊嘅電話係 +85211110000",
            {"85260113079"},
        )
        assert out == "客戶嗰邊嘅電話係 +85211110000"

    def test_rewrite_preserves_markdown_around_number(self):
        out = WhatsAppChannel._rewrite_participant_mentions(
            "**+85260113079** 仲未覆",
            {"85260113079"},
        )
        assert out == "**@85260113079** 仲未覆"

    def test_empty_participant_set_no_rewrite(self):
        text = "+85260113079 hi"
        out = WhatsAppChannel._rewrite_participant_mentions(text, set())
        assert out == text

    async def test_send_into_group_tags_participants(self):
        ch = _make_channel()
        ch._client.send_message = AsyncMock()

        # Mock get_group_info to return one phone-server participant.
        participant = MagicMock()
        participant.PhoneNumber = MagicMock(
            User="85260113079",
            Server="s.whatsapp.net",
        )
        participant.JID = MagicMock(User="", Server="")
        participant.LID = MagicMock(User="", Server="")
        info = MagicMock(Participants=[participant])
        ch._client.get_group_info = AsyncMock(return_value=info)

        await ch.send(
            "120363000@g.us",
            "等 +85260113079 reply",
            {"chat_jid": "120363000@g.us"},
        )

        ch._client.send_message.assert_called_once()
        sent_text = ch._client.send_message.call_args[0][1]
        assert sent_text == "等 @85260113079 reply"

    async def test_send_into_group_unknown_phone_passes_through(self):
        ch = _make_channel()
        ch._client.send_message = AsyncMock()
        # Group has a different participant — random number must not get tagged.
        participant = MagicMock()
        participant.PhoneNumber = MagicMock(
            User="85299990000",
            Server="s.whatsapp.net",
        )
        participant.JID = MagicMock(User="", Server="")
        participant.LID = MagicMock(User="", Server="")
        info = MagicMock(Participants=[participant])
        ch._client.get_group_info = AsyncMock(return_value=info)

        await ch.send(
            "120363000@g.us",
            "客戶電話 +85211110000",
            {"chat_jid": "120363000@g.us"},
        )

        sent_text = ch._client.send_message.call_args[0][1]
        assert sent_text == "客戶電話 +85211110000"

    async def test_send_into_dm_skips_participant_lookup(self):
        ch = _make_channel()
        ch._client.send_message = AsyncMock()
        ch._client.get_group_info = AsyncMock()

        await ch.send(
            "+85260113079",
            "hi +85260113079",
            {"chat_jid": "85260113079@s.whatsapp.net"},
        )

        # DM — never call get_group_info, never rewrite.
        ch._client.get_group_info.assert_not_called()
        sent_text = ch._client.send_message.call_args[0][1]
        assert sent_text == "hi +85260113079"

    async def test_get_group_info_failure_falls_back_silent(self):
        """``get_group_info`` failures must not break the send — the message
        goes out without mention tagging rather than crashing the agent
        reply path."""
        ch = _make_channel()
        ch._client.send_message = AsyncMock()
        ch._client.get_group_info = AsyncMock(
            side_effect=RuntimeError("boom"),
        )

        await ch.send(
            "120363000@g.us",
            "等 +85260113079",
            {"chat_jid": "120363000@g.us"},
        )

        ch._client.send_message.assert_called_once()
        sent_text = ch._client.send_message.call_args[0][1]
        # No tagging — but message still went out.
        assert sent_text == "等 +85260113079"

    async def test_participant_cache_reuses_within_ttl(self):
        ch = _make_channel()
        ch._client.send_message = AsyncMock()
        participant = MagicMock()
        participant.PhoneNumber = MagicMock(
            User="85260113079",
            Server="s.whatsapp.net",
        )
        participant.JID = MagicMock(User="", Server="")
        participant.LID = MagicMock(User="", Server="")
        info = MagicMock(Participants=[participant])
        ch._client.get_group_info = AsyncMock(return_value=info)

        # Two consecutive sends → one get_group_info call.
        meta = {"chat_jid": "120363000@g.us"}
        await ch.send("120363000@g.us", "first +85260113079", meta)
        await ch.send("120363000@g.us", "second +85260113079", meta)

        assert ch._client.get_group_info.call_count == 1

    async def test_lid_only_participant_via_lid_cache(self):
        """When a participant is known only by ``@lid`` server, the LID
        cache (populated on inbound) provides the phone digits used for
        matching outbound +<digits> patterns."""
        ch = _make_channel()
        ch._client.send_message = AsyncMock()
        ch._lid_cache["104402249592897@lid"] = {
            "phone": "85260113079",
            "name": "Joe",
        }
        participant = MagicMock()
        # No phone-server entries; only LID server.
        participant.PhoneNumber = MagicMock(User="", Server="")
        participant.JID = MagicMock(User="", Server="")
        participant.LID = MagicMock(
            User="104402249592897",
            Server="lid",
        )
        info = MagicMock(Participants=[participant])
        ch._client.get_group_info = AsyncMock(return_value=info)

        await ch.send(
            "120363000@g.us",
            "等 +85260113079",
            {"chat_jid": "120363000@g.us"},
        )

        sent_text = ch._client.send_message.call_args[0][1]
        assert sent_text == "等 @85260113079"


# ===================================================================
# TestSendMedia — outbound attachments via WhatsAppChannel.send_media
# ===================================================================


class TestSendMedia:
    """Primary outbound media path. Called by base.send_content_parts
    for every non-text block the agent emits (via send_file_to_user
    or directly returning ImageBlock/AudioBlock/VideoBlock/FileBlock)."""

    def _ready_channel(self, tmp_path):
        ch = _make_channel()
        ch._connected = True
        ch._client = MagicMock()
        ch._client.send_image = AsyncMock()
        ch._client.send_video = AsyncMock()
        ch._client.send_audio = AsyncMock()
        ch._client.send_document = AsyncMock()
        f = tmp_path / "a.jpg"
        f.write_bytes(b"\xff\xd8\xff\xe0fake jpeg")
        return ch, f

    async def test_send_image(self, tmp_path):
        ch, f = self._ready_channel(tmp_path)
        part = ImageContent(type=ContentType.IMAGE, image_url=str(f))
        await ch.send_media(
            "12345@s.whatsapp.net",
            part,
            {"chat_jid": "12345@s.whatsapp.net"},
        )
        ch._client.send_image.assert_called_once()
        args = ch._client.send_image.call_args.args
        assert args[1] == str(f)

    async def test_send_video(self, tmp_path):
        ch, _ = self._ready_channel(tmp_path)
        vid = tmp_path / "clip.mp4"
        vid.write_bytes(b"\x00\x00\x00\x20ftypmp42" + b"\x00" * 10)
        part = VideoContent(type=ContentType.VIDEO, video_url=str(vid))
        await ch.send_media(
            "12345@s.whatsapp.net",
            part,
            {"chat_jid": "12345@s.whatsapp.net"},
        )
        ch._client.send_video.assert_called_once()
        assert ch._client.send_document.call_count == 0

    async def test_send_audio(self, tmp_path):
        ch, _ = self._ready_channel(tmp_path)
        aud = tmp_path / "voice.ogg"
        aud.write_bytes(b"OggS" + b"\x00" * 10)
        part = AudioContent(type=ContentType.AUDIO, data=str(aud))
        await ch.send_media(
            "12345@s.whatsapp.net",
            part,
            {"chat_jid": "12345@s.whatsapp.net"},
        )
        ch._client.send_audio.assert_called_once()
        # ptt=True for voice notes
        assert ch._client.send_audio.call_args.kwargs.get("ptt") is True

    async def test_send_file(self, tmp_path):
        ch, _ = self._ready_channel(tmp_path)
        doc = tmp_path / "doc.pdf"
        doc.write_bytes(b"%PDF-1.5")
        part = FileContent(type=ContentType.FILE, file_url=str(doc))
        await ch.send_media(
            "12345@s.whatsapp.net",
            part,
            {"chat_jid": "12345@s.whatsapp.net"},
        )
        ch._client.send_document.assert_called_once()
        assert ch._client.send_image.call_count == 0

    async def test_send_strips_file_scheme(self, tmp_path):
        ch, f = self._ready_channel(tmp_path)
        part = ImageContent(type=ContentType.IMAGE, image_url=f"file://{f}")
        await ch.send_media(
            "12345@s.whatsapp.net",
            part,
            {"chat_jid": "12345@s.whatsapp.net"},
        )
        args = ch._client.send_image.call_args.args
        assert args[1] == str(f)

    async def test_missing_file_noop(self, tmp_path):
        ch, _ = self._ready_channel(tmp_path)
        missing = tmp_path / "gone.jpg"
        part = ImageContent(type=ContentType.IMAGE, image_url=str(missing))
        await ch.send_media(
            "12345@s.whatsapp.net",
            part,
            {"chat_jid": "12345@s.whatsapp.net"},
        )
        ch._client.send_image.assert_not_called()

    async def test_no_path_noop(self, tmp_path):
        ch, _ = self._ready_channel(tmp_path)
        part = ImageContent(type=ContentType.IMAGE, image_url="")
        await ch.send_media(
            "12345@s.whatsapp.net",
            part,
            {"chat_jid": "12345@s.whatsapp.net"},
        )
        ch._client.send_image.assert_not_called()

    async def test_disconnected_noop(self, tmp_path):
        ch, f = self._ready_channel(tmp_path)
        ch._connected = False
        part = ImageContent(type=ContentType.IMAGE, image_url=str(f))
        await ch.send_media(
            "12345@s.whatsapp.net",
            part,
            {"chat_jid": "12345@s.whatsapp.net"},
        )
        ch._client.send_image.assert_not_called()

    async def test_disabled_noop(self, tmp_path):
        ch, f = self._ready_channel(tmp_path)
        ch.enabled = False
        part = ImageContent(type=ContentType.IMAGE, image_url=str(f))
        await ch.send_media(
            "12345@s.whatsapp.net",
            part,
            {"chat_jid": "12345@s.whatsapp.net"},
        )
        ch._client.send_image.assert_not_called()

    async def test_file_fallback_to_file_id(self, tmp_path):
        """FileContent uses file_id when file_url is absent."""
        ch, _ = self._ready_channel(tmp_path)
        doc = tmp_path / "doc.pdf"
        doc.write_bytes(b"%PDF-1.5")
        part = MagicMock()
        part.type = ContentType.FILE
        part.file_url = None
        part.file_id = str(doc)
        await ch.send_media(
            "12345@s.whatsapp.net",
            part,
            {"chat_jid": "12345@s.whatsapp.net"},
        )
        ch._client.send_document.assert_called_once()

    async def test_send_fails_logs_error(self, tmp_path):
        ch, f = self._ready_channel(tmp_path)
        ch._client.send_image = AsyncMock(side_effect=RuntimeError("boom"))
        part = ImageContent(type=ContentType.IMAGE, image_url=str(f))
        # Should not raise — error is caught + logged
        await ch.send_media(
            "12345@s.whatsapp.net",
            part,
            {"chat_jid": "12345@s.whatsapp.net"},
        )


# ===================================================================
# TestChunkText
# ===================================================================


class TestChunkText:
    def test_short_text_not_chunked(self):
        ch = _make_channel()
        assert ch._chunk_text("hello") == ["hello"]

    def test_empty_text(self):
        ch = _make_channel()
        assert ch._chunk_text("") == []

    def test_long_text_chunked(self):
        ch = _make_channel(text_chunk_limit=20)
        text = "A" * 50
        chunks = ch._chunk_text(text)
        assert len(chunks) >= 2
        assert "".join(chunks) == text

    def test_chunking_prefers_newline_break(self):
        ch = _make_channel(text_chunk_limit=30)
        # Chunk limit 30, half = 15. Newline must be at index > 15 to trigger break.
        # "A" * 16 + "\n" puts newline at index 16, which is > 15.
        text = "A" * 16 + "\n" + "B" * 25
        chunks = ch._chunk_text(text)
        assert chunks[0] == "A" * 16


# ===================================================================
# TestTypingLoop
# ===================================================================


class TestTypingLoop:
    async def test_typing_loop_created_and_cancelled(self):
        ch = _make_channel()
        mock_client = MagicMock()
        mock_client._NewAClient__client = MagicMock()
        mock_client._NewAClient__client.SendChatPresence = AsyncMock()
        mock_client.uuid = "test-uuid"

        typing_jid = MagicMock()
        typing_jid.SerializeToString = lambda: b"\x00"

        task = asyncio.create_task(
            ch._typing_loop(mock_client, typing_jid, interval=0.05),
        )
        await asyncio.sleep(0.15)
        assert not task.done()
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass


# ===================================================================
# Utility functions
# ===================================================================


class TestJidUtils:
    def test_jid_to_str(self):
        jid = MagicMock(User="85200000000", Server="s.whatsapp.net")
        assert _jid_to_str(jid) == "85200000000@s.whatsapp.net"

    def test_jid_to_str_no_user(self):
        jid = MagicMock(spec=[])
        result = _jid_to_str(jid)
        assert isinstance(result, str)

    def test_is_group_jid(self):
        jid = MagicMock(Server="g.us")
        assert _is_group_jid(jid) is True

    def test_is_not_group_jid(self):
        jid = MagicMock(Server="s.whatsapp.net")
        assert _is_group_jid(jid) is False


class TestFormatSender:
    def test_with_phone_and_name(self):
        ch = _make_channel()
        ch._lid_cache["123@lid"] = {"phone": "85200000000", "name": "Alice"}
        assert ch._format_sender("123@lid") == "+85200000000 (Alice)"

    def test_with_phone_only(self):
        ch = _make_channel()
        ch._lid_cache["123@lid"] = {"phone": "85200000000", "name": ""}
        assert ch._format_sender("123@lid") == "+85200000000"

    def test_fallback_to_raw(self):
        ch = _make_channel()
        assert ch._format_sender("unknown@lid") == "unknown@lid"


# ===================================================================
# TestStripBotMention
# ===================================================================


class TestStripBotMention:
    """Tests for bot @mention stripping (enables /command detection)."""

    def test_strip_phone_mention_at_start(self):
        ch = _make_channel()
        ch._bot_phone = "817089933036"
        assert ch._strip_bot_mention("@+817089933036 /new") == "/new"

    def test_strip_phone_mention_no_plus(self):
        ch = _make_channel()
        ch._bot_phone = "817089933036"
        assert ch._strip_bot_mention("@817089933036 hello") == "hello"

    def test_strip_lid_mention(self):
        ch = _make_channel()
        ch._bot_lid = "229661330157571"
        assert ch._strip_bot_mention("@229661330157571 /stop") == "/stop"

    def test_strip_both_phone_and_lid(self):
        ch = _make_channel()
        ch._bot_phone = "817089933036"
        ch._bot_lid = "229661330157571"
        assert (
            ch._strip_bot_mention("@+817089933036 @229661330157571 hi") == "hi"
        )

    def test_no_mention_unchanged(self):
        ch = _make_channel()
        ch._bot_phone = "817089933036"
        assert ch._strip_bot_mention("just plain text") == "just plain text"

    def test_no_bot_phone_or_lid_unchanged(self):
        ch = _make_channel()
        ch._bot_phone = ""
        ch._bot_lid = ""
        assert (
            ch._strip_bot_mention("@+817089933036 hi") == "@+817089933036 hi"
        )

    def test_empty_text(self):
        ch = _make_channel()
        ch._bot_phone = "817089933036"
        assert ch._strip_bot_mention("") == ""

    def test_none_text(self):
        ch = _make_channel()
        ch._bot_phone = "817089933036"
        assert ch._strip_bot_mention(None) is None

    def test_different_mention_not_stripped(self):
        ch = _make_channel()
        ch._bot_phone = "817089933036"
        # Mention of a DIFFERENT number should stay
        assert (
            ch._strip_bot_mention("@+85251159218 hello")
            == "@+85251159218 hello"
        )

    def test_mention_in_middle(self):
        ch = _make_channel()
        ch._bot_phone = "817089933036"
        # Mentions are only stripped at the START of the message (so /commands
        # after "@+bot" work). A mid-string mention is left untouched to avoid
        # altering normal conversational text.
        assert (
            ch._strip_bot_mention("hello @+817089933036 world")
            == "hello @+817089933036 world"
        )


# ===================================================================
# TestSlashCommandDetection
# ===================================================================


class TestSlashCommandDetection:
    """Tests for slash command detection after mention strip."""

    def test_slash_command_after_mention(self):
        ch = _make_channel()
        ch._bot_phone = "817089933036"
        body = ch._strip_bot_mention("@+817089933036 /new")
        assert body.lstrip().startswith("/")

    def test_slash_command_no_mention(self):
        ch = _make_channel()
        ch._bot_phone = "817089933036"
        body = ch._strip_bot_mention("/clear")
        assert body.lstrip().startswith("/")

    def test_regular_text_not_command(self):
        ch = _make_channel()
        ch._bot_phone = "817089933036"
        body = ch._strip_bot_mention("@+817089933036 what is this")
        assert not body.lstrip().startswith("/")

    def test_slash_in_middle_not_command(self):
        ch = _make_channel()
        body = "do /not detect this"
        assert not body.lstrip().startswith("/")


# ===================================================================
# TestAckReactions
# ===================================================================


class TestAckReactions:
    """Tests for the thinking/done reaction acknowledgement flow."""

    async def test_send_reaction_calls_build_and_send(self):
        ch = _make_channel()
        client = MagicMock()
        client.build_reaction = AsyncMock(return_value="REACTION_MSG")
        client.send_message = AsyncMock()
        chat_jid = MagicMock()
        sender_jid = MagicMock()
        await ch._send_reaction(client, chat_jid, sender_jid, "MSGID", "🤔")
        client.build_reaction.assert_awaited_once_with(
            chat_jid,
            sender_jid,
            "MSGID",
            "🤔",
        )
        client.send_message.assert_awaited_once_with(chat_jid, "REACTION_MSG")

    async def test_send_reaction_swallows_errors(self):
        ch = _make_channel()
        client = MagicMock()
        client.build_reaction = AsyncMock(side_effect=RuntimeError("boom"))
        # Should not raise
        await ch._send_reaction(
            client,
            MagicMock(),
            MagicMock(),
            "MSGID",
            "🤔",
        )

    async def test_empty_emoji_clears_reaction(self):
        """Passing emoji='' removes any existing reaction — WhatsApp
        convention."""
        ch = _make_channel()
        client = MagicMock()
        client.build_reaction = AsyncMock(return_value="EMPTY")
        client.send_message = AsyncMock()
        await ch._send_reaction(
            client,
            MagicMock(),
            MagicMock(),
            "MSGID",
            "",
        )
        client.build_reaction.assert_called_once()
        assert client.build_reaction.call_args[0][3] == ""

    def test_ack_reactions_configurable(self):
        ch = _make_channel(
            ack_reaction_thinking="⏳",
            ack_reaction_done="✅",
        )
        assert ch._ack_reaction_thinking == "⏳"
        assert ch._ack_reaction_done == "✅"

    def test_ack_reactions_can_be_disabled(self):
        ch = _make_channel(ack_reaction_thinking="", ack_reaction_done="")
        assert ch._ack_reaction_thinking == ""
        assert ch._ack_reaction_done == ""

    def test_error_reaction_configurable(self):
        ch = _make_channel(ack_reaction_error="💥")
        assert ch._ack_reaction_error == "💥"

    def test_error_reaction_default(self):
        ch = _make_channel()
        assert ch._ack_reaction_error == "⚠️"

    def test_error_reaction_can_be_disabled(self):
        ch = _make_channel(ack_reaction_error="")
        assert ch._ack_reaction_error == ""


# ---------------------------------------------------------------------------
# Reply-to trigger message tests
# ---------------------------------------------------------------------------


class TestReplyToTrigger:
    """Tests for the reply-to-trigger-message feature."""

    def test_reply_to_trigger_default_enabled(self):
        ch = _make_channel()
        assert ch._reply_to_trigger is True

    def test_reply_to_trigger_can_disable(self):
        ch = _make_channel(reply_to_trigger=False)
        assert ch._reply_to_trigger is False

    def test_pending_quote_msgs_initialised(self):
        ch = _make_channel()
        assert isinstance(ch._pending_quote_msgs, dict)
        assert len(ch._pending_quote_msgs) == 0

    @pytest.mark.asyncio
    async def test_send_with_quote_calls_build_reply(self):
        """When reply_to_trigger is True and a quote msg exists, first chunk
        should use build_reply_message (awaited) + send_message."""
        ch = _make_channel(reply_to_trigger=True)
        fake_quote = MagicMock()
        ch._pending_quote_msgs["test_chat"] = fake_quote

        # build_reply_message is async
        reply_proto = MagicMock()
        ch._client.build_reply_message = AsyncMock(return_value=reply_proto)
        ch._client.send_message = AsyncMock()

        await ch.send(
            "test_chat",
            "hello world",
            meta={"chat_jid": "test_chat"},
        )

        ch._client.build_reply_message.assert_awaited_once_with(
            message="hello world",
            quoted=fake_quote,
        )
        ch._client.send_message.assert_awaited()
        # Quote msg consumed (popped)
        assert "test_chat" not in ch._pending_quote_msgs

    @pytest.mark.asyncio
    async def test_send_without_quote_sends_normally(self):
        """When no quote msg is pending, send_message is called directly."""
        ch = _make_channel(reply_to_trigger=True)
        ch._client.send_message = AsyncMock()

        await ch.send("test_chat", "hello", meta={})

        ch._client.build_reply_message.assert_not_called()
        ch._client.send_message.assert_awaited()

    @pytest.mark.asyncio
    async def test_send_disabled_skips_quote(self):
        """When reply_to_trigger is False, never quote even if msg pending."""
        ch = _make_channel(reply_to_trigger=False)
        fake_quote = MagicMock()
        ch._pending_quote_msgs["test_chat"] = fake_quote
        ch._client.send_message = AsyncMock()

        await ch.send("test_chat", "hello", meta={"chat_jid": "test_chat"})

        ch._client.build_reply_message.assert_not_called()
        ch._client.send_message.assert_awaited()
        # Quote msg NOT consumed since feature disabled
        assert "test_chat" in ch._pending_quote_msgs

    @pytest.mark.asyncio
    async def test_send_multipart_only_first_chunk_quotes(self):
        """For multi-chunk messages, only the first chunk should quote."""
        ch = _make_channel(reply_to_trigger=True)
        fake_quote = MagicMock()
        long_text = "x" * (WHATSAPP_MAX_TEXT_LENGTH + 100)
        ch._pending_quote_msgs["test_chat"] = fake_quote

        reply_proto = MagicMock()
        ch._client.build_reply_message = AsyncMock(return_value=reply_proto)
        ch._client.send_message = AsyncMock()

        await ch.send("test_chat", long_text, meta={"chat_jid": "test_chat"})

        # build_reply called exactly once (first chunk)
        assert ch._client.build_reply_message.await_count == 1
        # send_message called for all chunks (>= 2)
        assert ch._client.send_message.await_count >= 2

    @pytest.mark.asyncio
    async def test_build_reply_failure_falls_back(self):
        """If build_reply_message raises, should still attempt normal send."""
        ch = _make_channel(reply_to_trigger=True)
        ch._pending_quote_msgs["test_chat"] = MagicMock()
        ch._client.build_reply_message = AsyncMock(
            side_effect=Exception("proto error"),
        )
        ch._client.send_message = AsyncMock()

        # Should not raise — error is caught
        await ch.send("test_chat", "hello", meta={"chat_jid": "test_chat"})


# ---------------------------------------------------------------------------
# Typing loop tests (SendChatPresence panic prevention)
# ---------------------------------------------------------------------------


class TestTypingLoopPresencePanicPrevention:
    """Tests for _typing_loop — especially that cancelled typing does NOT
    send presence type 2 (paused) which causes neonize Go panic."""

    @pytest.mark.asyncio
    async def test_typing_sends_composing_presence(self):
        """While active, typing loop sends presence type 0 (composing)."""
        ch = _make_channel()
        mock_jid = MagicMock()
        mock_jid.SerializeToString = MagicMock(return_value=b"\x00")

        mock_inner = AsyncMock()
        ch._client._NewAClient__client = MagicMock()
        ch._client._NewAClient__client.SendChatPresence = mock_inner
        ch._client.uuid = "test-uuid"

        task = asyncio.create_task(
            ch._typing_loop(ch._client, mock_jid, interval=0.1),
        )
        await asyncio.sleep(0.25)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        # Should have called SendChatPresence with type 0 (composing)
        assert mock_inner.await_count >= 2
        for call in mock_inner.call_args_list:
            args = call[0]
            # 4th arg is presence type: 0=composing, 2=paused
            assert (
                args[3] == 0
            ), f"Expected presence type 0 (composing), got {args[3]}"

    @pytest.mark.asyncio
    async def test_cancelled_typing_does_not_send_paused(self):
        """CRITICAL: Cancelling typing loop must NOT send presence type 2.

        SendChatPresence(type=2) causes neonize Go panic:
        'index out of range [2] with length 2' -> SIGABRT -> process crash.
        """
        ch = _make_channel()
        mock_jid = MagicMock()
        mock_jid.SerializeToString = MagicMock(return_value=b"\x00")

        mock_inner = AsyncMock()
        ch._client._NewAClient__client = MagicMock()
        ch._client._NewAClient__client.SendChatPresence = mock_inner
        ch._client.uuid = "test-uuid"

        task = asyncio.create_task(
            ch._typing_loop(ch._client, mock_jid, interval=0.1),
        )
        await asyncio.sleep(0.15)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        # MUST NOT have any call with presence type 2 (paused)
        for call in mock_inner.call_args_list:
            args = call[0]
            assert args[3] != 2, (
                "FATAL: SendChatPresence called with type=2 (paused). "
                "This causes neonize Go panic -> SIGABRT -> process crash!"
            )

    @pytest.mark.asyncio
    async def test_typing_loop_handles_send_error_gracefully(self):
        """If SendChatPresence raises, loop continues (no crash)."""
        ch = _make_channel()
        mock_jid = MagicMock()
        mock_jid.SerializeToString = MagicMock(return_value=b"\x00")

        call_count = 0

        async def flaky_send(*args):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise Exception("transient error")

        ch._client._NewAClient__client = MagicMock()
        ch._client._NewAClient__client.SendChatPresence = flaky_send
        ch._client.uuid = "test-uuid"

        task = asyncio.create_task(
            ch._typing_loop(ch._client, mock_jid, interval=0.1),
        )
        await asyncio.sleep(0.35)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        # Should have retried after error
        assert (
            call_count >= 2
        ), f"Expected at least 2 calls (1 error + 1 success), got {call_count}"

    @pytest.mark.asyncio
    async def test_typing_loop_cancellation_is_clean(self):
        """Cancelling typing loop should not raise or leave dangling tasks."""
        ch = _make_channel()
        mock_jid = MagicMock()
        mock_jid.SerializeToString = MagicMock(return_value=b"\x00")

        ch._client._NewAClient__client = MagicMock()
        ch._client._NewAClient__client.SendChatPresence = AsyncMock()
        ch._client.uuid = "test-uuid"

        task = asyncio.create_task(
            ch._typing_loop(ch._client, mock_jid, interval=0.1),
        )
        await asyncio.sleep(0.15)
        task.cancel()

        # Should complete without raising
        try:
            await task
        except asyncio.CancelledError:
            pass

        assert task.done()
        assert not task.cancelled()  # CancelledError is caught internally


# ========================================================================
# send_media - sticker filename convention (.sticker.webp)
# ========================================================================


class TestSendMediaStickerConvention:
    """`.sticker.webp` filename convention routes to send_sticker.

    Regular images (including plain .webp) route to send_image.
    Rule is explicit and path-based: no content sniffing, no metadata.
    """

    @pytest.mark.asyncio
    async def test_sticker_webp_suffix_dispatches_send_sticker(self, tmp_path):
        ch = _make_channel()
        ch._client.send_sticker = AsyncMock()
        ch._client.send_image = AsyncMock()

        sticker_path = tmp_path / "crab.sticker.webp"
        sticker_path.write_bytes(b"RIFF\x00\x00\x00\x00WEBP")

        part = ImageContent(
            type=ContentType.IMAGE,
            image_url=str(sticker_path),
        )
        await ch.send_media("1234567890@s.whatsapp.net", part, meta={})

        ch._client.send_sticker.assert_awaited_once()
        assert ch._client.send_sticker.await_args.kwargs.get("passthrough") is True
        ch._client.send_image.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_plain_webp_dispatches_send_image(self, tmp_path):
        """`.webp` WITHOUT `.sticker.` marker => regular image."""
        ch = _make_channel()
        ch._client.send_sticker = AsyncMock()
        ch._client.send_image = AsyncMock()

        img_path = tmp_path / "photo.webp"
        img_path.write_bytes(b"RIFF\x00\x00\x00\x00WEBP")

        part = ImageContent(type=ContentType.IMAGE, image_url=str(img_path))
        await ch.send_media("1234567890@s.whatsapp.net", part, meta={})

        ch._client.send_image.assert_awaited_once()
        ch._client.send_sticker.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_png_dispatches_send_image(self, tmp_path):
        """PNG => send_image (baseline, no regression)."""
        ch = _make_channel()
        ch._client.send_sticker = AsyncMock()
        ch._client.send_image = AsyncMock()

        img_path = tmp_path / "photo.png"
        img_path.write_bytes(b"\x89PNG\r\n\x1a\n")

        part = ImageContent(type=ContentType.IMAGE, image_url=str(img_path))
        await ch.send_media("1234567890@s.whatsapp.net", part, meta={})

        ch._client.send_image.assert_awaited_once()
        ch._client.send_sticker.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_sticker_suffix_case_insensitive(self, tmp_path):
        """`.STICKER.WEBP` (uppercase) still triggers sticker path."""
        ch = _make_channel()
        ch._client.send_sticker = AsyncMock()
        ch._client.send_image = AsyncMock()

        sticker_path = tmp_path / "crab.STICKER.WEBP"
        sticker_path.write_bytes(b"RIFF\x00\x00\x00\x00WEBP")

        part = ImageContent(
            type=ContentType.IMAGE,
            image_url=str(sticker_path),
        )
        await ch.send_media("1234567890@s.whatsapp.net", part, meta={})

        ch._client.send_sticker.assert_awaited_once()
        assert ch._client.send_sticker.await_args.kwargs.get("passthrough") is True
        ch._client.send_image.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_sticker_with_file_scheme_prefix(self, tmp_path):
        """`file://` scheme on sticker path is stripped before suffix check."""
        ch = _make_channel()
        ch._client.send_sticker = AsyncMock()
        ch._client.send_image = AsyncMock()

        sticker_path = tmp_path / "crab.sticker.webp"
        sticker_path.write_bytes(b"RIFF\x00\x00\x00\x00WEBP")

        part = ImageContent(
            type=ContentType.IMAGE,
            image_url="file://" + str(sticker_path),
        )
        await ch.send_media("1234567890@s.whatsapp.net", part, meta={})

        ch._client.send_sticker.assert_awaited_once()
        assert ch._client.send_sticker.await_args.kwargs.get("passthrough") is True
        ch._client.send_image.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_missing_file_skips_both(self, tmp_path):
        """Non-existent file => neither send_sticker nor send_image."""
        ch = _make_channel()
        ch._client.send_sticker = AsyncMock()
        ch._client.send_image = AsyncMock()

        part = ImageContent(
            type=ContentType.IMAGE,
            image_url=str(tmp_path / "does_not_exist.sticker.webp"),
        )
        await ch.send_media("1234567890@s.whatsapp.net", part, meta={})

        ch._client.send_sticker.assert_not_awaited()
        ch._client.send_image.assert_not_awaited()


# ===================================================================
# TestAlbumCollation — multi-image album buffer + flush
# ===================================================================


def _make_album_message(
    expected_images: int,
    expected_videos: int = 0,
    with_quote: bool = False,
):
    """Build a fake WAMessage whose top-level field is ``albumMessage``.

    When ``with_quote=True`` the album header carries a fake
    contextInfo+quotedMessage so the buffer captures a reply block.
    """
    album = MagicMock()
    album.expectedImageCount = expected_images
    album.expectedVideoCount = expected_videos
    if with_quote:
        ctx = MagicMock()
        ctx.HasField = lambda name: name == "quotedMessage"
        ctx.participant = "alice@s.whatsapp.net"
        ctx.stanzaId = "stanza_album_quote"
        quoted = MagicMock()
        quoted.conversation = "earlier reply target"
        quoted.HasField = lambda name: False
        ctx.quotedMessage = quoted
        album.contextInfo = ctx
    return _make_proto_message(albumMessage=album)


def _make_image_child(caption: str = ""):
    """Fake follow-up imageMessage child of an album.  Only the
    ``imageMessage`` field is present so ``_extract_message_content``
    builds a single ImageContent + optional text caption."""
    img = MagicMock()
    img.caption = caption
    img.contextInfo = MagicMock()  # absent quotedMessage
    img.contextInfo.HasField = lambda name: False
    return _make_proto_message(imageMessage=img)


async def _drive_inbound(
    channel,
    msg,
    msg_id="m1",
    sender="alice",
    chat="group@g.us",
    body="",
    parts=None,
    quote_parts=None,
    paths=None,
    is_group=True,
):
    """Invoke the album collation gate the same way ``_on_message``
    does.  Returns the gate's bool decision so tests can assert
    'buffered vs fall-through'.  We bypass the rest of the
    extraction pipeline to keep the surface narrow."""
    return await channel._handle_album_inbound(
        client=MagicMock(),
        message=MagicMock(Info=MagicMock(ID=msg_id)),
        msg=msg,
        msg_id=msg_id,
        sender_jid=sender,
        chat_jid=chat,
        is_group=is_group,
        timestamp=0,
        sender_str=sender,
        chat_str=chat,
        body=body,
        content_parts=parts or [],
        quote_parts=quote_parts or [],
        media_local_paths=paths or [],
    )


@pytest.mark.asyncio
class TestAlbumCollation:
    """Three-image album: header + 3 children should flush exactly
    once with all three images bundled into a single dispatch."""

    async def test_header_alone_buffers_and_returns_true(self):
        ch = _make_channel()
        msg = _make_album_message(expected_images=3)
        result = await _drive_inbound(ch, msg)
        # Header buffered → caller should return early (True).
        assert result is True
        key = ("group@g.us", "alice")
        assert key in ch._album_buffers
        assert ch._album_buffers[key].expected == 3
        assert len(ch._album_buffers[key].gathered_parts) == 0
        # Cancel the timeout to avoid leaks during teardown.
        ch._album_buffers[key].timeout_task.cancel()

    async def test_complete_album_flushes_via_dispatch(self):
        ch = _make_channel()
        # Patch the dispatch helper so we can observe what got
        # called without firing the rest of the pipeline.
        ch._dispatch_inbound_message = AsyncMock()

        # Header (3 images expected)
        await _drive_inbound(ch, _make_album_message(3))
        # 3 children
        for i in range(3):
            child = _make_image_child(caption=("hello" if i == 0 else ""))
            buffered = await _drive_inbound(
                ch,
                child,
                msg_id=f"c{i}",
                parts=[
                    ImageContent(
                        type=ContentType.IMAGE,
                        image_url=f"/tmp/img{i}.jpg",
                    ),
                ],
                paths=[f"/tmp/img{i}.jpg"],
                body=("hello" if i == 0 else ""),
            )
            assert buffered is True

        # Dispatch fired exactly once with merged payload.
        ch._dispatch_inbound_message.assert_awaited_once()
        kwargs = ch._dispatch_inbound_message.await_args.kwargs
        assert len(kwargs["content_parts"]) == 3
        assert all(
            isinstance(p, ImageContent) for p in kwargs["content_parts"]
        )
        assert kwargs["media_local_paths"] == [
            "/tmp/img0.jpg",
            "/tmp/img1.jpg",
            "/tmp/img2.jpg",
        ]
        assert "hello" in kwargs["body"]
        # Buffer cleared after flush.
        assert ("group@g.us", "alice") not in ch._album_buffers

    async def test_album_caption_text_block_survives_collation(self):
        """Regression test for the 2026-04-25 production bug:
        ``_extract_message_content`` appends a TextContent for the
        ``imageMessage.caption``.  Earlier collation only kept
        media parts → the caption block was silently dropped →
        ``_apply_no_text_debounce`` saw a content_parts with no
        text and BUFFERED the request waiting for text that
        never arrived → agent never responded to multi-image
        sends with a caption (the most common case in WhatsApp
        groups).
        """
        ch = _make_channel()
        ch._dispatch_inbound_message = AsyncMock()

        await _drive_inbound(ch, _make_album_message(2))
        # First child arrives with caption + image (the actual
        # shape ``_extract_message_content`` returns).
        await _drive_inbound(
            ch, _make_image_child(caption="@bot here you go"),
            msg_id="c0",
            parts=[
                TextContent(
                    type=ContentType.TEXT, text="@bot here you go",
                ),
                ImageContent(type=ContentType.IMAGE, image_url="/t/i0.jpg"),
            ],
            paths=["/t/i0.jpg"],
            body="@bot here you go",
        )
        # Second child: image only, no caption.
        await _drive_inbound(
            ch, _make_image_child(),
            msg_id="c1",
            parts=[ImageContent(type=ContentType.IMAGE, image_url="/t/i1.jpg")],
            paths=["/t/i1.jpg"],
        )
        ch._dispatch_inbound_message.assert_awaited_once()
        kwargs = ch._dispatch_inbound_message.await_args.kwargs
        # Merged content keeps the caption text block alongside
        # both images — downstream debounce now sees text and
        # forwards immediately instead of stalling.
        types = [type(p).__name__ for p in kwargs["content_parts"]]
        assert types == ["TextContent", "ImageContent", "ImageContent"]
        text_parts = [
            p for p in kwargs["content_parts"]
            if isinstance(p, TextContent)
        ]
        assert text_parts[0].text == "@bot here you go"

    async def test_album_with_reply_preserves_quote_on_flush(self):
        ch = _make_channel()
        ch._dispatch_inbound_message = AsyncMock()

        # Header carries quote_parts (reply context).
        quote = TextContent(
            type=ContentType.TEXT,
            text="=== UNTRUSTED reply-to ===\nFrom: bob",
        )
        await _drive_inbound(
            ch,
            _make_album_message(2),
            quote_parts=[quote],
        )
        # Two image children
        for i in range(2):
            await _drive_inbound(
                ch,
                _make_image_child(),
                msg_id=f"c{i}",
                parts=[
                    ImageContent(
                        type=ContentType.IMAGE,
                        image_url=f"/tmp/img{i}.jpg",
                    ),
                ],
            )

        kwargs = ch._dispatch_inbound_message.await_args.kwargs
        # Quote prepended ahead of all images on the merged turn.
        assert kwargs["content_parts"][0] is quote
        assert all(
            isinstance(p, ImageContent) for p in kwargs["content_parts"][1:]
        )

    async def test_concurrent_albums_from_different_senders_isolated(self):
        ch = _make_channel()
        ch._dispatch_inbound_message = AsyncMock()

        # Two albums interleaved from different senders
        await _drive_inbound(ch, _make_album_message(2), sender="alice")
        await _drive_inbound(ch, _make_album_message(2), sender="bob")
        await _drive_inbound(
            ch,
            _make_image_child(),
            sender="alice",
            msg_id="a1",
            parts=[
                ImageContent(type=ContentType.IMAGE, image_url="/t/a1.jpg"),
            ],
        )
        await _drive_inbound(
            ch,
            _make_image_child(),
            sender="bob",
            msg_id="b1",
            parts=[
                ImageContent(type=ContentType.IMAGE, image_url="/t/b1.jpg"),
            ],
        )
        await _drive_inbound(
            ch,
            _make_image_child(),
            sender="alice",
            msg_id="a2",
            parts=[
                ImageContent(type=ContentType.IMAGE, image_url="/t/a2.jpg"),
            ],
        )
        # Alice's album now complete → first dispatch fires.
        await _drive_inbound(
            ch,
            _make_image_child(),
            sender="bob",
            msg_id="b2",
            parts=[
                ImageContent(type=ContentType.IMAGE, image_url="/t/b2.jpg"),
            ],
        )
        # Bob's album now complete → second dispatch fires.

        assert ch._dispatch_inbound_message.await_count == 2
        # Inspect both calls — each only contains its own sender's images.
        calls = [
            c.kwargs for c in ch._dispatch_inbound_message.await_args_list
        ]
        a_call = next(c for c in calls if c["sender_str"] == "alice")
        b_call = next(c for c in calls if c["sender_str"] == "bob")
        a_urls = [p.image_url for p in a_call["content_parts"]]
        b_urls = [p.image_url for p in b_call["content_parts"]]
        assert a_urls == ["/t/a1.jpg", "/t/a2.jpg"]
        assert b_urls == ["/t/b1.jpg", "/t/b2.jpg"]

    async def test_non_album_message_falls_through(self):
        """Plain text from a sender with no open album buffer must
        return False so normal dispatch proceeds."""
        ch = _make_channel()
        plain = _make_proto_message(conversation="hello")
        result = await _drive_inbound(ch, plain)
        assert result is False
        assert not ch._album_buffers

    async def test_text_during_open_album_falls_through(self):
        """Pure text from a sender mid-album is user-initiated and
        must not be swallowed as a child — text falls through to
        normal dispatch even while the buffer is open."""
        ch = _make_channel()
        await _drive_inbound(ch, _make_album_message(3))  # buffer open
        plain = _make_proto_message(conversation="oh wait nvm")
        result = await _drive_inbound(
            ch,
            plain,
            parts=[TextContent(type=ContentType.TEXT, text="oh wait nvm")],
        )
        assert result is False
        # Buffer untouched (still pending the 3 expected children).
        key = ("group@g.us", "alice")
        assert key in ch._album_buffers
        ch._album_buffers[key].timeout_task.cancel()

    async def test_timeout_flushes_partial_album(self):
        """If only part of the album arrives before the timeout
        fires, flush whatever we've collected — better partial
        than a perpetually-stuck buffer."""
        import asyncio as _asyncio

        ch = _make_channel()
        ch._dispatch_inbound_message = AsyncMock()
        # Compress the timeout so the test runs fast.
        ch._album_timeout_s = 0.1

        await _drive_inbound(ch, _make_album_message(3))
        # Only one child arrives.
        await _drive_inbound(
            ch,
            _make_image_child(),
            msg_id="c0",
            parts=[
                ImageContent(
                    type=ContentType.IMAGE,
                    image_url="/tmp/i0.jpg",
                ),
            ],
        )
        # Wait past the timeout to let the flush task run.
        await _asyncio.sleep(0.3)

        ch._dispatch_inbound_message.assert_awaited_once()
        kwargs = ch._dispatch_inbound_message.await_args.kwargs
        # Partial flush — only the one image we managed to capture.
        assert len(kwargs["content_parts"]) == 1
        assert ("group@g.us", "alice") not in ch._album_buffers

    async def test_new_album_replaces_pending_one(self):
        """A fresh album header from the same sender before the
        previous album finished cancels the old buffer (the only
        sane interpretation: the previous send was lost / aborted)."""
        import asyncio as _asyncio

        ch = _make_channel()
        ch._dispatch_inbound_message = AsyncMock()

        await _drive_inbound(ch, _make_album_message(3))
        first = ch._album_buffers[("group@g.us", "alice")]
        await _drive_inbound(ch, _make_album_message(2))
        second = ch._album_buffers[("group@g.us", "alice")]

        # Yield once so the cancelled task transitions out of the
        # "cancelling" intermediate state into ``cancelled()``.
        await _asyncio.sleep(0)

        assert first is not second
        assert second.expected == 2
        assert first.timeout_task.cancelled() or first.timeout_task.done()
        second.timeout_task.cancel()

    async def test_zero_count_album_falls_through(self):
        """A degenerate album with no expected children isn't
        worth buffering — fall through and let the normal
        ``if not content_parts`` guard drop it."""
        ch = _make_channel()
        msg = _make_album_message(expected_images=0, expected_videos=0)
        result = await _drive_inbound(ch, msg)
        assert result is False
        assert not ch._album_buffers


# ===================================================================
# TestLocalTimestamp + envelope + album reply path lookup
# ===================================================================


class TestFormatLocalTimestamp:
    """The single timestamp formatter all three inbound surfaces
    use (group history, envelope prefix, reply-context)."""

    def test_short_style_includes_zone(self):
        from qwenpaw.app.channels.whatsapp.channel import _format_local_timestamp
        out = _format_local_timestamp(1777106276, style="short")
        # Format: YYYY-MM-DD HH:MM <ZONE>
        assert "-04-" in out
        assert ":" in out
        # Zone abbrev varies by host but should be present.
        assert out.split()[-1].isalpha() or "+" in out

    def test_long_style_uses_chinese_date_format(self):
        from qwenpaw.app.channels.whatsapp.channel import _format_local_timestamp
        out = _format_local_timestamp(1777106276, style="long")
        assert "年" in out and "月" in out and "日" in out
        # H:M:S present
        assert out.count(":") >= 2

    def test_handles_milliseconds(self):
        from qwenpaw.app.channels.whatsapp.channel import _format_local_timestamp
        # WhatsApp sometimes hands us ms-since-epoch
        s_secs = _format_local_timestamp(1777106276, style="short")
        s_ms = _format_local_timestamp(1777106276 * 1000, style="short")
        assert s_secs == s_ms

    def test_invalid_input_returns_empty(self):
        from qwenpaw.app.channels.whatsapp.channel import _format_local_timestamp
        assert _format_local_timestamp(None) == ""
        assert _format_local_timestamp("not a number") == ""


class TestInboundMediaIndex:
    """Cache of inbound message id → local media paths so a later
    reply pointing back at that msg_id (which is what WhatsApp's
    contextInfo.stanzaID gives us) can resolve real files —
    critical for albums where the header proto carries no media
    keys, only an expectedImageCount."""

    def test_record_then_lookup_round_trip(self, tmp_path):
        ch = _make_channel()
        # Use real files so the path-existence filter doesn't drop them.
        f1 = tmp_path / "a.jpg"; f1.write_bytes(b"x")
        f2 = tmp_path / "b.jpg"; f2.write_bytes(b"y")
        ch._record_inbound_media("chat@g.us", "MSG1", [str(f1), str(f2)])
        assert ch._lookup_inbound_media("chat@g.us", "MSG1") == [
            str(f1), str(f2),
        ]

    def test_lookup_drops_paths_that_no_longer_exist(self, tmp_path):
        ch = _make_channel()
        live = tmp_path / "live.jpg"; live.write_bytes(b"x")
        gone = tmp_path / "gone.jpg"  # never created
        ch._record_inbound_media("chat", "M", [str(live), str(gone)])
        assert ch._lookup_inbound_media("chat", "M") == [str(live)]

    def test_fifo_eviction_at_limit(self, tmp_path):
        ch = _make_channel()
        ch._inbound_media_limit = 3
        for i in range(5):
            f = tmp_path / f"{i}.jpg"; f.write_bytes(b"x")
            ch._record_inbound_media("c", f"M{i}", [str(f)])
        # Only the last 3 entries should still be cached.
        assert len(ch._inbound_media) == 3
        assert ("c", "M0") not in ch._inbound_media
        assert ("c", "M1") not in ch._inbound_media
        assert ("c", "M4") in ch._inbound_media

    def test_record_skips_non_existent_paths(self):
        ch = _make_channel()
        ch._record_inbound_media("c", "M", ["/no/such/file.jpg"])
        assert ch._lookup_inbound_media("c", "M") == []


@pytest.mark.asyncio
async def test_quote_with_album_uses_cached_paths(tmp_path):
    """When the user replies to an album whose children we
    previously cached (via ``_record_inbound_media`` from
    ``_flush_album``), the reply context should surface real
    ``image: /path/...`` lines plus inline ImageContent —
    not the opaque ``"album with N images"`` placeholder.
    """
    ch = _make_channel()
    # Pre-populate cache as if the album arrived earlier.
    img1 = tmp_path / "child1.jpg"; img1.write_bytes(b"x")
    img2 = tmp_path / "child2.jpg"; img2.write_bytes(b"y")
    ch._record_inbound_media(
        "grp@g.us", "ALBUM_HDR_ID", [str(img1), str(img2)],
    )

    ctx = MagicMock()
    ctx.HasField = lambda name: name == "quotedMessage"
    ctx.participant = "alice@s.whatsapp.net"
    ctx.stanzaId = "ALBUM_HDR_ID"  # points back at our cached header

    album = MagicMock()
    album.expectedImageCount = 2
    album.expectedVideoCount = 0
    quoted = MagicMock()
    quoted.conversation = ""
    quoted.HasField = lambda name: name == "albumMessage"
    quoted.albumMessage = album
    ctx.quotedMessage = quoted

    etm = MagicMock()
    etm.text = "look at these"
    etm.contextInfo = ctx
    msg = _make_proto_message(extendedTextMessage=etm)

    parts = await ch._extract_quote_content(
        MagicMock(), msg, chat_str="grp@g.us",
    )
    text = parts[0].text
    # Real paths surface in the Media: line, NOT a count placeholder.
    assert str(img1) in text
    assert str(img2) in text
    assert "album with" not in text
    # ImageContent inlined for vision-capable models.
    image_parts = [p for p in parts if isinstance(p, ImageContent)]
    assert len(image_parts) == 2


@pytest.mark.asyncio
async def test_quote_with_album_falls_back_to_count_on_cache_miss():
    """When the cache has nothing for the quoted album header
    (album sent before bot was running, restart cleared the
    index), fall back to the original count-only placeholder
    rather than emitting a half-rendered block."""
    ch = _make_channel()
    # No _record_inbound_media call — cache is empty.

    ctx = MagicMock()
    ctx.HasField = lambda name: name == "quotedMessage"
    ctx.participant = "alice@s.whatsapp.net"
    ctx.stanzaId = "MISSING_ID"

    album = MagicMock()
    album.expectedImageCount = 3
    album.expectedVideoCount = 1
    quoted = MagicMock()
    quoted.conversation = ""
    quoted.HasField = lambda name: name == "albumMessage"
    quoted.albumMessage = album
    ctx.quotedMessage = quoted

    etm = MagicMock()
    etm.text = "x"
    etm.contextInfo = ctx
    msg = _make_proto_message(extendedTextMessage=etm)

    parts = await ch._extract_quote_content(
        MagicMock(), msg, chat_str="grp@g.us",
    )
    text = parts[0].text
    assert "album with 3 images + 1 video" in text


# ===================================================================
# TestSenderFormatting — bot → "(You)" + PushName fallback
# ===================================================================


class TestFormatSender:
    def test_bot_label_renders_as_you(self):
        """Bot's own LID is cached as ``name="bot"`` — _format_sender
        re-labels it ``(You)`` so the agent's prior turns in group
        history read naturally from the model's POV."""
        ch = _make_channel()
        ch._lid_cache["bot@lid"] = {"phone": "817089933036", "name": "bot"}
        assert ch._format_sender("bot@lid") == "+817089933036 (You)"

    def test_human_name_passes_through(self):
        ch = _make_channel()
        ch._lid_cache["x@lid"] = {"phone": "85251159218", "name": "Joe"}
        assert ch._format_sender("x@lid") == "+85251159218 (Joe)"

    def test_phone_only_no_paren(self):
        ch = _make_channel()
        ch._lid_cache["x@lid"] = {"phone": "85251159218", "name": ""}
        assert ch._format_sender("x@lid") == "+85251159218"


@pytest.mark.asyncio
async def test_resolve_lid_falls_back_to_pushname():
    """Saved-contact name (FullName) is empty for group strangers
    — fall through to BusinessName / PushName / FirstName so the
    nickname surfaces alongside the phone instead of leaving
    just the +phone bare."""
    ch = _make_channel()

    # Mock client.contact.get returning an info with only PushName.
    info = MagicMock(spec=["FullName", "BusinessName", "PushName", "FirstName"])
    info.FullName = ""
    info.BusinessName = ""
    info.PushName = "JoeyDDD"
    info.FirstName = ""

    contact = MagicMock()
    contact.get = MagicMock(return_value=info)
    client = MagicMock()
    client.contact = contact
    client.get_pn_from_lid = AsyncMock(return_value=MagicMock(User="85251159218"))

    out = await ch._resolve_lid(client, "abc@lid", MagicMock())
    assert out["phone"] == "85251159218"
    assert out["name"] == "JoeyDDD"


@pytest.mark.asyncio
async def test_resolve_lid_prefers_fullname_when_set():
    """Saved-contact FullName beats PushName in the priority ladder."""
    ch = _make_channel()
    info = MagicMock(spec=["FullName", "BusinessName", "PushName", "FirstName"])
    info.FullName = "Joe Saved"
    info.BusinessName = ""
    info.PushName = "Joey-pushname"
    info.FirstName = "JoeFirst"
    contact = MagicMock()
    contact.get = MagicMock(return_value=info)
    client = MagicMock()
    client.contact = contact
    client.get_pn_from_lid = AsyncMock(return_value=MagicMock(User="852"))

    out = await ch._resolve_lid(client, "x@lid", MagicMock())
    assert out["name"] == "Joe Saved"
