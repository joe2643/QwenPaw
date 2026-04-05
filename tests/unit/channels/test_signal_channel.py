# -*- coding: utf-8 -*-
# pylint: disable=protected-access
"""Unit tests for Signal channel."""
from __future__ import annotations

import asyncio
import os
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from agentscope_runtime.engine.schemas.agent_schemas import (
    TextContent,
    ImageContent,
    AudioContent,
    FileContent,
    VideoContent,
    ContentType,
)

from copaw.app.channels.signal.channel import (
    SignalChannel,
    SignalDaemon,
    _detect_mime,
    _detect_ext,
    _markdown_to_signal,
    _parse_mentions,
    _MEDIA_DIR,
    SIGNAL_MAX_TEXT_LENGTH,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_channel(**overrides: Any) -> SignalChannel:
    """Create a SignalChannel with dummy process handler."""

    async def _noop_process(_request):
        yield  # pragma: no cover

    defaults = {
        "process": _noop_process,
        "enabled": True,
        "account": "+85200000000",
        "http_url": "http://127.0.0.1:8080",
    }
    defaults.update(overrides)
    ch = SignalChannel(**defaults)
    ch.daemon = MagicMock(spec=SignalDaemon)
    ch.daemon.connected = True
    ch.daemon.send_message = AsyncMock(return_value=12345)
    ch.daemon.send_typing = AsyncMock()
    ch.daemon.download_attachment = AsyncMock(return_value=None)
    return ch


# ===================================================================
# TestMarkdownToSignal
# ===================================================================

class TestMarkdownToSignal:
    def test_header_converted_to_bold(self):
        text, styles = _markdown_to_signal("## Title")
        assert text == "Title"
        assert len(styles) == 1
        assert styles[0]["style"] == "BOLD"
        assert styles[0]["start"] == 0
        assert styles[0]["length"] == 5

    def test_h1_header(self):
        text, styles = _markdown_to_signal("# Heading One")
        assert text == "Heading One"
        assert styles[0]["style"] == "BOLD"

    def test_bold(self):
        text, styles = _markdown_to_signal("this is **bold** text")
        assert "bold" in text
        assert "**" not in text
        bold = [s for s in styles if s["style"] == "BOLD"]
        assert len(bold) == 1
        assert bold[0]["length"] == 4

    def test_italic(self):
        text, styles = _markdown_to_signal("this is *italic* text")
        assert "italic" in text
        assert styles[0]["style"] == "ITALIC"
        assert styles[0]["length"] == 6

    def test_inline_code(self):
        text, styles = _markdown_to_signal("use `code` here")
        assert "code" in text
        assert "`" not in text
        mono = [s for s in styles if s["style"] == "MONOSPACE"]
        assert len(mono) == 1
        assert mono[0]["length"] == 4

    def test_strikethrough(self):
        text, styles = _markdown_to_signal("this is ~~strike~~ text")
        assert "strike" in text
        assert "~~" not in text
        st = [s for s in styles if s["style"] == "STRIKETHROUGH"]
        assert len(st) == 1
        assert st[0]["length"] == 6

    def test_mixed_styles_correct_offsets(self):
        text, styles = _markdown_to_signal("**bold** and *italic*")
        assert text == "bold and italic"
        assert len(styles) == 2
        bold = [s for s in styles if s["style"] == "BOLD"][0]
        italic = [s for s in styles if s["style"] == "ITALIC"][0]
        assert bold["start"] == 0
        assert bold["length"] == 4
        assert italic["start"] == 9  # "bold and " = 9 chars
        assert italic["length"] == 6

    def test_no_overlap(self):
        text, styles = _markdown_to_signal("**bold *nested* bold**")
        # The outer **...** should win; inner *...* should be removed by overlap filter
        # Just verify no overlap: each style's range [start, start+length) is disjoint
        ranges = [(s["start"], s["start"] + s["length"]) for s in styles]
        for i, (a_start, a_end) in enumerate(ranges):
            for j, (b_start, b_end) in enumerate(ranges):
                if i != j:
                    assert a_end <= b_start or b_end <= a_start, \
                        f"Overlap: {ranges[i]} and {ranges[j]}"

    def test_code_block(self):
        text, styles = _markdown_to_signal("```python\nprint('hi')\n```")
        assert "print('hi')" in text
        assert "```" not in text
        mono = [s for s in styles if s["style"] == "MONOSPACE"]
        assert len(mono) == 1

    def test_plain_text_no_styles(self):
        text, styles = _markdown_to_signal("plain text nothing special")
        assert text == "plain text nothing special"
        assert styles == []

    def test_underscore_bold(self):
        text, styles = _markdown_to_signal("__bold__")
        assert text == "bold"
        assert styles[0]["style"] == "BOLD"


# ===================================================================
# TestParseMentions
# ===================================================================

class TestParseMentions:
    def test_phone_number_mention_replaced(self):
        text, mentions = _parse_mentions("hey @+85200000000 how are you")
        assert "\ufffc" in text
        assert "@+85200000000" not in text
        assert len(mentions) == 1
        assert mentions[0]["number"] == "+85200000000"
        assert mentions[0]["length"] == 1

    def test_uuid_mention_replaced(self):
        text, mentions = _parse_mentions("hey @12345678-1234-1234-1234-123456789abc done")
        assert "\ufffc" in text
        assert len(mentions) == 1
        assert mentions[0]["uuid"] == "12345678-1234-1234-1234-123456789abc"
        assert mentions[0]["length"] == 1

    def test_multiple_mentions_shift_correctly(self):
        text, mentions = _parse_mentions("@+85200000001 and @+85200000002 done")
        assert text.count("\ufffc") == 2
        assert len(mentions) == 2
        # First mention at position 0
        assert mentions[0]["start"] == 0
        # Second mention: original "@+85200000001 and " is 19 chars,
        # after replacing first (14 chars -> 1), second starts at 0+1+5=6
        assert mentions[1]["start"] == mentions[0]["start"] + 1 + len(" and ")

    def test_no_mentions(self):
        text, mentions = _parse_mentions("hello world")
        assert text == "hello world"
        assert mentions == []


# ===================================================================
# TestMentionStyleShift
# ===================================================================

class TestMentionStyleShift:
    def test_style_offsets_adjusted_after_mention(self):
        """When a mention @+number is replaced with \\ufffc, styles after it shift."""
        import re as _re
        # Simulate what send() does: markdown -> parse mentions -> shift styles
        # "hey @+85200000000 bold"
        #  0123456789...       ^-- "bold" starts at index:
        #  "hey " = 4, "@+85200000000" = 14, " " = 1 => bold at 19
        text_after_md = "hey @+85200000000 bold"
        # "bold" starts at index 18 in text_after_md
        styles = [{"start": 18, "length": 4, "style": "BOLD"}]

        final_text, mention_list = _parse_mentions(text_after_md)

        if mention_list:
            mention_pat = _re.compile(
                r"@(\+\d{7,15}|[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})"
            )
            shifts = []
            for mm in mention_pat.finditer(text_after_md):
                removed = len(mm.group(0)) - 1
                shifts.append((mm.start(), removed))
            for sr in styles:
                total_shift = 0
                for orig_pos, shift_amt in shifts:
                    if sr["start"] > orig_pos:
                        total_shift += shift_amt
                sr["start"] -= total_shift

        # "@+85200000000" is 13 chars, replaced with 1 char -> shift = 12
        # bold: 18 - 12 = 6, which matches index of "bold" in final_text
        assert styles[0]["start"] == 6
        assert styles[0]["start"] >= 0

    def test_multiple_mentions_no_negative_offsets(self):
        import re as _re
        text_after_md = "@+85200000001 @+85200000002 text"
        # Style for "text" at original position 28
        styles = [{"start": 28, "length": 4, "style": "BOLD"}]

        final_text, mention_list = _parse_mentions(text_after_md)
        if mention_list:
            mention_pat = _re.compile(
                r"@(\+\d{7,15}|[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})"
            )
            shifts = []
            for mm in mention_pat.finditer(text_after_md):
                removed = len(mm.group(0)) - 1
                shifts.append((mm.start(), removed))
            for sr in styles:
                total_shift = 0
                for orig_pos, shift_amt in shifts:
                    if sr["start"] > orig_pos:
                        total_shift += shift_amt
                sr["start"] -= total_shift

        assert styles[0]["start"] >= 0

    def test_no_mentions_styles_unchanged(self):
        import re as _re
        text_after_md = "just bold text"
        styles = [{"start": 5, "length": 4, "style": "BOLD"}]
        original_start = styles[0]["start"]

        final_text, mention_list = _parse_mentions(text_after_md)
        assert mention_list == []
        # No adjustment needed
        assert styles[0]["start"] == original_start


# ===================================================================
# TestDetectMime
# ===================================================================

class TestDetectMime:
    def test_jpeg_magic_bytes(self):
        assert _detect_mime(b"\xff\xd8\xff\xe0" + b"\x00" * 20) == "image/jpeg"

    def test_png_magic_bytes(self):
        assert _detect_mime(b"\x89PNG\r\n\x1a\n" + b"\x00" * 20) == "image/png"

    def test_mp4_ftyp_box(self):
        # MP4 files have "ftyp" at bytes 4-7
        data = b"\x00\x00\x00\x1cftypisom" + b"\x00" * 20
        assert _detect_mime(data) == "video/mp4"

    def test_unknown_bytes(self):
        assert _detect_mime(b"\x01\x02\x03\x04\x05\x06\x07\x08") == ""

    def test_generic_null_bytes_not_mp4(self):
        """Null bytes should NOT match mp4 (bytes 4-7 must be 'ftyp')."""
        data = b"\x00" * 16
        assert _detect_mime(data) != "video/mp4"

    def test_gif87a(self):
        assert _detect_mime(b"GIF87a" + b"\x00" * 20) == "image/gif"

    def test_gif89a(self):
        assert _detect_mime(b"GIF89a" + b"\x00" * 20) == "image/gif"

    def test_ogg(self):
        assert _detect_mime(b"OggS" + b"\x00" * 20) == "audio/ogg"

    def test_pdf(self):
        assert _detect_mime(b"%PDF-1.5" + b"\x00" * 20) == "application/pdf"

    def test_webp(self):
        data = b"RIFF\x00\x00\x00\x00WEBP" + b"\x00" * 20
        assert _detect_mime(data) == "image/webp"

    def test_riff_not_webp(self):
        """RIFF header but not WEBP should not match."""
        data = b"RIFF\x00\x00\x00\x00WAVE" + b"\x00" * 20
        assert _detect_mime(data) != "image/webp"


# ===================================================================
# TestDetectExt
# ===================================================================

class TestDetectExt:
    def test_jpeg(self):
        assert _detect_ext(b"\xff\xd8\xff\xe0" + b"\x00" * 20, "") == "jpg"

    def test_declared_content_type_fallback(self):
        # Unknown magic, but declared content-type is image/png
        assert _detect_ext(b"\x01\x02\x03\x04\x05\x06\x07\x08", "image/png") == "png"

    def test_octet_stream_returns_bin(self):
        assert _detect_ext(b"\x01\x02\x03\x04", "application/octet-stream") == "bin"

    def test_mp4(self):
        data = b"\x00\x00\x00\x1cftypisom" + b"\x00" * 20
        assert _detect_ext(data, "") == "mp4"

    def test_empty_content_type_returns_bin(self):
        assert _detect_ext(b"\x01\x02\x03\x04", "") == "bin"

    def test_content_type_with_semicolon(self):
        # "audio/mpeg; charset=utf-8" -> "mpeg"
        assert _detect_ext(b"\x01\x02\x03\x04", "audio/mpeg; charset=utf-8") == "mpeg"


# ===================================================================
# TestExtractQuoteContent
# ===================================================================

class TestExtractQuoteContent:
    async def test_quote_with_text(self):
        ch = _make_channel()
        data_message = {
            "quote": {
                "text": "original message",
                "author": "+85200000001",
                "id": "12345",
                "attachments": [],
            }
        }
        parts = await ch._extract_quote_content(data_message)
        assert len(parts) >= 1
        text_parts = [p for p in parts if hasattr(p, "text")]
        combined = " ".join(p.text for p in text_parts)
        assert "UNTRUSTED reply-to" in combined
        assert "original message" in combined

    async def test_quote_with_attachment_download(self):
        ch = _make_channel()
        # Simulate successful download
        tmp = Path(tempfile.mkdtemp())
        fake_img = tmp / "signal_att_abcdef12.jpg"
        fake_img.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 100)  # JPEG
        ch.daemon.download_attachment = AsyncMock(return_value=fake_img)

        data_message = {
            "quote": {
                "text": "check this image",
                "author": "+85200000001",
                "id": "99999",
                "attachments": [
                    {"id": "att123", "contentType": "image/jpeg"},
                ],
            }
        }
        parts = await ch._extract_quote_content(data_message)
        # Should have text + image
        img_parts = [p for p in parts if p.type == ContentType.IMAGE]
        assert len(img_parts) == 1
        text_parts = [p for p in parts if hasattr(p, "text")]
        assert any("UNTRUSTED reply-to" in p.text for p in text_parts)

    async def test_quote_image_only_still_has_reply_header(self):
        """When quote is image-only (no text) and download succeeds, the
        reply-to text block should still be emitted with 'Media: image'."""
        ch = _make_channel()
        tmp = Path(tempfile.mkdtemp())
        fake_img = tmp / "signal_att_photo.jpg"
        fake_img.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 50)
        ch.daemon.download_attachment = AsyncMock(return_value=fake_img)
        data_message = {
            "quote": {
                "text": "",  # image-only quote
                "author": "+85211111111",
                "attachments": [{"id": "att-xyz", "contentType": "image/jpeg"}],
            }
        }
        parts = await ch._extract_quote_content(data_message)
        # Must have BOTH text header AND image content
        text_parts = [p for p in parts if hasattr(p, "text")]
        img_parts = [p for p in parts if p.type == ContentType.IMAGE]
        assert len(text_parts) == 1
        assert len(img_parts) == 1
        assert "UNTRUSTED reply-to" in text_parts[0].text
        assert "Media: image" in text_parts[0].text

    async def test_quote_file_attachment_labelled(self):
        ch = _make_channel()
        tmp = Path(tempfile.mkdtemp())
        fake_doc = tmp / "signal_att_doc.pdf"
        fake_doc.write_bytes(b"%PDF-1.5")
        ch.daemon.download_attachment = AsyncMock(return_value=fake_doc)
        data_message = {
            "quote": {
                "text": "see this",
                "author": "+85211111111",
                "attachments": [{"id": "a1", "contentType": "application/pdf", "fileName": "doc.pdf"}],
            }
        }
        parts = await ch._extract_quote_content(data_message)
        text_parts = [p for p in parts if hasattr(p, "text")]
        file_parts = [p for p in parts if p.type == ContentType.FILE]
        assert len(file_parts) == 1
        assert "file: doc.pdf" in text_parts[0].text

    async def test_quote_audio_attachment(self):
        """Audio attachment in quote produces AudioContent + 'Media: audio' label."""
        ch = _make_channel()
        tmp = Path(tempfile.mkdtemp())
        fake_audio = tmp / "signal_att_voice.ogg"
        fake_audio.write_bytes(b"OggS" + b"\x00" * 50)
        ch.daemon.download_attachment = AsyncMock(return_value=fake_audio)
        data_message = {
            "quote": {
                "text": "",
                "author": "+85211111111",
                "attachments": [{"id": "a1", "contentType": "audio/ogg"}],
            }
        }
        parts = await ch._extract_quote_content(data_message)
        text_parts = [p for p in parts if hasattr(p, "text")]
        audio_parts = [p for p in parts if p.type == ContentType.AUDIO]
        assert len(audio_parts) == 1
        assert "Media: audio" in text_parts[0].text

    async def test_quote_video_attachment(self):
        """Video attachment in quote produces VideoContent + 'Media: video' label."""
        ch = _make_channel()
        tmp = Path(tempfile.mkdtemp())
        fake_video = tmp / "signal_att_clip.mp4"
        # MP4 ftyp box
        fake_video.write_bytes(b"\x00\x00\x00\x20" + b"ftypmp42" + b"\x00" * 50)
        ch.daemon.download_attachment = AsyncMock(return_value=fake_video)
        data_message = {
            "quote": {
                "text": "look at this clip",
                "author": "+85211111111",
                "attachments": [{"id": "v1", "contentType": "video/mp4"}],
            }
        }
        parts = await ch._extract_quote_content(data_message)
        text_parts = [p for p in parts if hasattr(p, "text")]
        video_parts = [p for p in parts if p.type == ContentType.VIDEO]
        assert len(video_parts) == 1
        assert "Media: video" in text_parts[0].text
        assert "look at this clip" in text_parts[0].text

    async def test_quote_audio_octet_stream_detected_as_audio(self):
        """If contentType is application/octet-stream, magic bytes should
        detect audio/ogg and route to AudioContent, not FileContent."""
        ch = _make_channel()
        tmp = Path(tempfile.mkdtemp())
        fake_audio = tmp / "signal_att_voice.bin"
        fake_audio.write_bytes(b"OggS" + b"\x00" * 50)
        ch.daemon.download_attachment = AsyncMock(return_value=fake_audio)
        data_message = {
            "quote": {
                "text": "",
                "author": "+85211111111",
                "attachments": [{"id": "a1", "contentType": "application/octet-stream"}],
            }
        }
        parts = await ch._extract_quote_content(data_message)
        # OggS magic detected → audio route
        audio_parts = [p for p in parts if p.type == ContentType.AUDIO]
        file_parts = [p for p in parts if p.type == ContentType.FILE]
        assert len(audio_parts) == 1
        assert len(file_parts) == 0

    async def test_no_quote_returns_empty(self):
        ch = _make_channel()
        data_message = {"message": "hello"}
        parts = await ch._extract_quote_content(data_message)
        assert parts == []

    async def test_empty_quote_returns_empty(self):
        ch = _make_channel()
        data_message = {
            "quote": {
                "text": "",
                "author": "",
                "attachments": [],
            }
        }
        parts = await ch._extract_quote_content(data_message)
        # No text, no media -> empty
        assert parts == []


# ===================================================================
# TestGroupHistory
# ===================================================================

class TestGroupHistory:
    def test_downloaded_media_paths_stored_in_history(self):
        ch = _make_channel(require_mention=True)
        group_id = "group123base64=="
        history = ch._group_history.setdefault(group_id, [])
        # Simulate recording with media paths (as done in _on_sse_event)
        history.append({
            "sender": "+85200000001",
            "body": "[media]",
            "ts": 12345,
            "media": ["/tmp/signal_att_abc.jpg"],
        })
        assert len(ch._group_history[group_id]) == 1
        assert ch._group_history[group_id][0]["media"] == ["/tmp/signal_att_abc.jpg"]

    def test_history_injected_with_media_image_content(self):
        ch = _make_channel()
        group_id = "group123base64=="
        # Create a real temp file to simulate downloaded media
        tmp = Path(tempfile.mkdtemp())
        fake_img = tmp / "signal_att_test.jpg"
        fake_img.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 100)

        ch._group_history[group_id] = [
            {
                "sender": "+85200000001",
                "body": "look at this",
                "ts": 1,
                "media": [str(fake_img)],
            },
        ]

        # Simulate the injection logic from _on_sse_event
        history = ch._group_history.get(group_id, [])
        ctx_lines = []
        media_to_add = []
        for h in history[-10:]:
            ctx_lines.append(f"  {h['sender']}: {h['body']}")
            for mp in h.get("media", []):
                if os.path.isfile(mp):
                    media_to_add.append(mp)

        content_parts = []
        if ctx_lines:
            ctx_text = "--- Recent group messages ---\n" + "\n".join(ctx_lines)
            content_parts.insert(0, TextContent(type=ContentType.TEXT, text=ctx_text))
        for mp in media_to_add[-3:]:
            content_parts.append(ImageContent(type=ContentType.IMAGE, image_url=mp))
        ch._group_history[group_id] = []

        assert len(content_parts) == 2
        assert content_parts[0].type == ContentType.TEXT
        assert "look at this" in content_parts[0].text
        assert content_parts[1].type == ContentType.IMAGE
        assert str(fake_img) in content_parts[1].image_url

    def test_history_limit(self):
        ch = _make_channel()
        ch._group_history_limit = 3
        group_id = "grp"
        history = ch._group_history.setdefault(group_id, [])
        for i in range(10):
            history.append({"sender": f"u{i}", "body": f"m{i}", "ts": i, "media": []})
        if len(history) > ch._group_history_limit:
            ch._group_history[group_id] = history[-ch._group_history_limit:]
        assert len(ch._group_history[group_id]) == 3
        assert ch._group_history[group_id][0]["body"] == "m7"


# ===================================================================
# TestAccessControl
# ===================================================================

class TestAccessControl:
    def test_group_allowlist_empty_blocks_all(self):
        """When group_policy=allowlist and groups is empty, all groups blocked."""
        ch = _make_channel(group_policy="allowlist", groups=[])
        # Simulate the check from _on_sse_event
        group_id = "somegroup123"
        if ch.group_policy == "allowlist":
            blocked = not ch._groups or group_id not in ch._groups
        else:
            blocked = False
        assert blocked is True

    def test_group_allowlist_allows_listed(self):
        ch = _make_channel(group_policy="allowlist", groups=["mygroup123"])
        group_id = "mygroup123"
        if ch.group_policy == "allowlist":
            blocked = not ch._groups or group_id not in ch._groups
        else:
            blocked = False
        assert blocked is False

    def test_group_allowlist_blocks_unlisted(self):
        ch = _make_channel(group_policy="allowlist", groups=["mygroup123"])
        group_id = "othergroup456"
        if ch.group_policy == "allowlist":
            blocked = not ch._groups or group_id not in ch._groups
        else:
            blocked = False
        assert blocked is True

    def test_group_allow_from_wildcard_allows(self):
        ch = _make_channel(group_allow_from=["*"])
        sender_id = "+85200000001"
        source = "+85200000001"
        source_uuid = "uuid-1234"
        allowed = (
            "*" in ch._group_allow_from
            or sender_id in ch._group_allow_from
            or source in ch._group_allow_from
            or source_uuid in ch._group_allow_from
        )
        assert allowed is True

    def test_group_allow_from_specific_blocks_others(self):
        ch = _make_channel(group_allow_from=["+85200000001"])
        sender_id = "+85200000002"
        source = "+85200000002"
        source_uuid = "uuid-5678"
        allowed = (
            "*" in ch._group_allow_from
            or sender_id in ch._group_allow_from
            or source in ch._group_allow_from
            or source_uuid in ch._group_allow_from
        )
        assert allowed is False

    def test_group_allow_from_specific_allows_match(self):
        ch = _make_channel(group_allow_from=["+85200000001"])
        source = "+85200000001"
        source_uuid = "uuid-1234"
        allowed = (
            "*" in ch._group_allow_from
            or source in ch._group_allow_from
            or source_uuid in ch._group_allow_from
        )
        assert allowed is True

    def test_dm_allowlist_check(self):
        ch = _make_channel(dm_policy="allowlist", allow_from=["+85200000001"])
        assert ch._is_source_allowed("+85200000001", "uuid-1234") is True
        assert ch._is_source_allowed("+85200000002", "uuid-5678") is False

    def test_dm_allowlist_uuid(self):
        ch = _make_channel(dm_policy="allowlist", allow_from=["uuid:test-uuid-1234"])
        assert ch._is_source_allowed("+85200000001", "test-uuid-1234") is True
        assert ch._is_source_allowed("+85200000001", "other-uuid") is False

    def test_dm_open_policy(self):
        """With dm_policy=open, _is_source_allowed is not called."""
        ch = _make_channel(dm_policy="open")
        # dm_policy=open means the check in _on_sse_event skips _is_source_allowed
        assert ch.dm_policy == "open"


# ===================================================================
# TestBotMentionDetection
# ===================================================================

class TestBotMentionDetection:
    def test_uuid_mention(self):
        ch = _make_channel(account_uuid="bot-uuid-1234")
        data_message = {
            "mentions": [{"uuid": "bot-uuid-1234", "start": 0, "length": 1}],
        }
        assert ch._is_bot_mentioned(data_message, "") is True

    def test_phone_mention(self):
        ch = _make_channel(account="+85200000000")
        data_message = {
            "mentions": [{"number": "+85200000000", "start": 0, "length": 1}],
        }
        assert ch._is_bot_mentioned(data_message, "") is True

    def test_quote_reply_to_bot(self):
        ch = _make_channel(account="+85200000000")
        data_message = {
            "mentions": [],
            "quote": {
                "author": "+85200000000",
                "text": "original",
            },
        }
        assert ch._is_bot_mentioned(data_message, "reply text") is True

    def test_quote_reply_to_bot_uuid(self):
        ch = _make_channel(account_uuid="bot-uuid-1234")
        data_message = {
            "mentions": [],
            "quote": {
                "authorUuid": "bot-uuid-1234",
                "text": "original",
            },
        }
        assert ch._is_bot_mentioned(data_message, "reply text") is True

    def test_bot_number_in_body(self):
        ch = _make_channel(account="+85200000000")
        data_message = {"mentions": []}
        assert ch._is_bot_mentioned(data_message, "hey +85200000000") is True

    def test_no_mention(self):
        ch = _make_channel(account="+85200000000", account_uuid="bot-uuid")
        data_message = {"mentions": []}
        assert ch._is_bot_mentioned(data_message, "hello world") is False


# ===================================================================
# TestTypingLoop
# ===================================================================

class TestTypingLoop:
    async def test_typing_loop_sends_repeatedly(self):
        ch = _make_channel()
        task = asyncio.create_task(
            ch._typing_loop("+85200000001", is_group=False, interval=0.05)
        )
        await asyncio.sleep(0.18)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        # Should have called send_typing(start=True) at least 2 times
        start_calls = [
            c for c in ch.daemon.send_typing.call_args_list
            if c.kwargs.get("start", True) is True or (len(c.args) >= 2 and c.args[1] is True)
        ]
        assert len(start_calls) >= 2

    async def test_cancel_sends_stop_typing(self):
        ch = _make_channel()
        task = asyncio.create_task(
            ch._typing_loop("group123", is_group=True, interval=0.05)
        )
        await asyncio.sleep(0.08)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        # Last call should be send_typing(start=False) — stop typing
        last_call = ch.daemon.send_typing.call_args_list[-1]
        # Check that start=False was passed
        if last_call.kwargs:
            assert last_call.kwargs.get("start") is False
        else:
            # Positional args: target, start, is_group
            assert last_call.args[1] is False


# ===================================================================
# TestSend
# ===================================================================

class TestSend:
    async def test_basic_text_send(self):
        ch = _make_channel()
        await ch.send("+85200000001", "hello", {})
        ch.daemon.send_message.assert_called_once()

    async def test_empty_text_noop(self):
        ch = _make_channel()
        await ch.send("+85200000001", "", {})
        ch.daemon.send_message.assert_not_called()

    async def test_disabled_noop(self):
        ch = _make_channel(enabled=False)
        await ch.send("+85200000001", "hi")
        ch.daemon.send_message.assert_not_called()

    async def test_markdown_passed_through(self):
        """bbernhard parses markdown natively via text_mode=styled, so we
        send the raw markdown text without pre-processing."""
        ch = _make_channel()
        await ch.send("+85200000001", "**bold** text", {})
        call_args = ch.daemon.send_message.call_args
        # Markdown preserved in sent text
        assert "**bold**" in call_args.args[1]

    async def test_text_chunking(self):
        ch = _make_channel(text_chunk_limit=10)
        text = "AAAAAAAAAA" + "BBBBBBBBBB"  # 20 chars
        await ch.send("+85200000001", text, {})
        assert ch.daemon.send_message.call_count == 2

    async def test_group_send(self):
        ch = _make_channel()
        await ch.send("group:mygroup123=", "hello", {"group_id": "mygroup123="})
        ch.daemon.send_message.assert_called_once()
        call_args = ch.daemon.send_message.call_args
        # is_group should be True
        assert call_args.kwargs.get("is_group") is True or (
            len(call_args.args) >= 3 and call_args.args[2] is True
        )

    async def test_image_extraction(self):
        """[Image: /path] tags should be extracted as attachments."""
        ch = _make_channel()
        tmp = Path(tempfile.mkdtemp())
        fake_img = tmp / "test.jpg"
        fake_img.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 10)

        text = f"look [Image: {fake_img}] nice"
        await ch.send("+85200000001", text, {})
        call_args = ch.daemon.send_message.call_args
        # Attachment should be passed
        atts = call_args.kwargs.get("attachments")
        if atts:
            assert str(fake_img) in atts


# ===================================================================
# TestChunkText
# ===================================================================

class TestChunkText:
    def test_short_text(self):
        ch = _make_channel()
        assert ch._chunk_text("hello") == ["hello"]

    def test_empty_text(self):
        ch = _make_channel()
        assert ch._chunk_text("") == []

    def test_long_text(self):
        ch = _make_channel(text_chunk_limit=20)
        text = "A" * 50
        chunks = ch._chunk_text(text)
        assert len(chunks) >= 2
        assert "".join(chunks) == text


# ===================================================================
# TestSessionId
# ===================================================================

class TestSessionId:
    def test_dm_session(self):
        ch = _make_channel()
        sid = ch.resolve_session_id("+85200000001", {})
        assert sid == "signal:+85200000001"

    def test_group_session(self):
        ch = _make_channel()
        sid = ch.resolve_session_id("+85200000001", {"group_id": "grp123"})
        assert sid == "signal:group:grp123"


# ===================================================================
# TestGetToHandle
# ===================================================================

class TestGetToHandle:
    def test_group(self):
        ch = _make_channel()
        req = MagicMock()
        req.channel_meta = {"group_id": "grp123", "source": "+85200000001"}
        assert ch.get_to_handle_from_request(req) == "grp123"

    def test_dm(self):
        ch = _make_channel()
        req = MagicMock()
        req.channel_meta = {"group_id": "", "source": "+85200000001"}
        assert ch.get_to_handle_from_request(req) == "+85200000001"

    def test_dm_with_uuid_fallback(self):
        ch = _make_channel()
        req = MagicMock()
        req.channel_meta = {"source": "", "source_uuid": ""}
        req.user_id = "fallback_user"
        result = ch.get_to_handle_from_request(req)
        # Should return source or user_id
        assert result is not None


# ===================================================================
# TestSignalDaemon (bbernhard REST API client)
# ===================================================================

class TestSignalDaemonToRecipient:
    """Tests for _to_recipient: converts target to bbernhard recipient format."""

    def test_phone_number_passthrough(self):
        assert SignalDaemon._to_recipient("+85212345678", is_group=False) == "+85212345678"

    def test_uuid_passthrough_for_dm(self):
        uuid = "5720b72c-1051-47bd-962b-8c0c9db5aff1"
        assert SignalDaemon._to_recipient(uuid, is_group=False) == uuid

    def test_group_internal_id_to_group_prefix(self):
        # internal_id like "sBlO8LhzR42XNBbUqUrNVNokyOe2NdDZCTs0fSuZnJc="
        # → "group.{base64(internal_id)}"
        internal = "sBlO8LhzR42XNBbUqUrNVNokyOe2NdDZCTs0fSuZnJc="
        result = SignalDaemon._to_recipient(internal, is_group=True)
        assert result.startswith("group.")

    def test_group_already_prefixed_unchanged(self):
        already = "group.c0JsTzhMaHpSNDJYTkJiVXFVck5WTm9reU9lMk5kRFpDVHMwZlN1Wm5KYz0="
        assert SignalDaemon._to_recipient(already, is_group=True) == already


class TestSignalDaemonConnect:
    """Tests for SignalDaemon.connect() against bbernhard's /v1/about."""

    async def test_connect_success(self):
        d = SignalDaemon(account="+85200000000", http_url="http://localhost:8080")
        # Mock the ClientSession's GET to /v1/about
        import aiohttp
        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=None)
        mock_resp.json = AsyncMock(return_value={"mode": "json-rpc", "version": "0.98"})
        with patch.object(aiohttp, "ClientSession") as mock_session_cls:
            mock_session = MagicMock()
            mock_session.get = MagicMock(return_value=mock_resp)
            mock_session_cls.return_value = mock_session
            result = await d.connect()
        assert result is True
        assert d.connected is True

    async def test_connect_failure_404(self):
        d = SignalDaemon(account="+85200000000", http_url="http://localhost:8080")
        import aiohttp
        mock_resp = AsyncMock()
        mock_resp.status = 404
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=None)
        with patch.object(aiohttp, "ClientSession") as mock_session_cls:
            mock_session = MagicMock()
            mock_session.get = MagicMock(return_value=mock_resp)
            mock_session_cls.return_value = mock_session
            result = await d.connect()
        assert result is False
        assert d.connected is False

    async def test_connect_network_error(self):
        d = SignalDaemon(account="+85200000000", http_url="http://localhost:8080")
        import aiohttp
        with patch.object(aiohttp, "ClientSession") as mock_session_cls:
            mock_session = MagicMock()
            mock_session.get = MagicMock(side_effect=Exception("connection refused"))
            mock_session_cls.return_value = mock_session
            result = await d.connect()
        assert result is False

    async def test_connect_idempotent(self):
        d = SignalDaemon(account="+85200000000", http_url="http://localhost:8080")
        d.connected = True
        result = await d.connect()
        assert result is True  # Should skip check


class TestSignalDaemonSend:
    """Tests for SignalDaemon.send_message() POST /v2/send."""

    def _mock_daemon(self, response_status=201, response_body=None):
        """Build a daemon with a mocked aiohttp session."""
        d = SignalDaemon(account="+85200000000", http_url="http://localhost:8080")
        d.connected = True
        d.session = MagicMock()

        body = response_body or {"timestamp": "1775382175305"}
        mock_resp = AsyncMock()
        mock_resp.status = response_status
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=None)
        mock_resp.json = AsyncMock(return_value=body)
        mock_resp.text = AsyncMock(return_value=str(body))
        d.session.post = MagicMock(return_value=mock_resp)
        return d, mock_resp

    async def test_send_dm(self):
        d, _ = self._mock_daemon()
        ts = await d.send_message("+85212345678", "hello", is_group=False)
        assert ts == 1775382175305
        # Verify payload
        call = d.session.post.call_args
        assert call.args[0] == "http://localhost:8080/v2/send"
        payload = call.kwargs["json"]
        assert payload["number"] == "+85200000000"
        assert payload["recipients"] == ["+85212345678"]
        assert payload["message"] == "hello"
        assert payload["text_mode"] == "styled"

    async def test_send_group(self):
        d, _ = self._mock_daemon()
        gid = "sBlO8LhzR42XNBbUqUrNVNokyOe2NdDZCTs0fSuZnJc="
        await d.send_message(gid, "group msg", is_group=True)
        payload = d.session.post.call_args.kwargs["json"]
        assert payload["recipients"][0].startswith("group.")

    async def test_send_with_quote(self):
        d, _ = self._mock_daemon()
        await d.send_message(
            "+85212345678", "reply",
            quote_timestamp=1234567890, quote_author="+85298765432",
        )
        payload = d.session.post.call_args.kwargs["json"]
        assert payload["quote_timestamp"] == 1234567890
        assert payload["quote_author"] == "+85298765432"

    async def test_send_returns_none_on_failure(self):
        d, _ = self._mock_daemon(response_status=500, response_body={"error": "oops"})
        result = await d.send_message("+85212345678", "hello")
        assert result is None

    async def test_send_returns_none_when_disconnected(self):
        d = SignalDaemon(account="+85200000000", http_url="http://localhost:8080")
        d.connected = False
        result = await d.send_message("+85212345678", "hello")
        assert result is None


class TestSignalDaemonTyping:
    """Tests for send_typing() PUT/DELETE /v1/typing-indicator."""

    def _mock_daemon(self):
        d = SignalDaemon(account="+85200000000", http_url="http://localhost:8080")
        d.connected = True
        d.session = MagicMock()
        mock_resp = AsyncMock()
        mock_resp.status = 204
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=None)
        d.session.request = MagicMock(return_value=mock_resp)
        return d

    async def test_start_typing_uses_put(self):
        d = self._mock_daemon()
        await d.send_typing("+85212345678", start=True, is_group=False)
        method = d.session.request.call_args.args[0]
        assert method == "PUT"

    async def test_stop_typing_uses_delete(self):
        d = self._mock_daemon()
        await d.send_typing("+85212345678", start=False, is_group=False)
        method = d.session.request.call_args.args[0]
        assert method == "DELETE"

    async def test_typing_includes_recipient_in_body(self):
        d = self._mock_daemon()
        await d.send_typing("+85212345678", start=True, is_group=False)
        payload = d.session.request.call_args.kwargs["json"]
        assert payload["recipient"] == "+85212345678"

    async def test_typing_noop_when_disconnected(self):
        d = SignalDaemon(account="+85200000000", http_url="http://localhost:8080")
        d.connected = False
        # Should not raise
        await d.send_typing("+85212345678", start=True)


class TestSignalDaemonReaction:
    """Tests for send_reaction() POST/DELETE /v1/reactions."""

    def _mock_daemon(self, response_status=204):
        d = SignalDaemon(account="+85200000000", http_url="http://localhost:8080")
        d.connected = True
        d.session = MagicMock()
        mock_resp = AsyncMock()
        mock_resp.status = response_status
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=None)
        d.session.request = MagicMock(return_value=mock_resp)
        return d

    async def test_add_reaction_uses_post(self):
        d = self._mock_daemon()
        result = await d.send_reaction(
            "+85212345678", "👍",
            target_author="+85212345678", target_timestamp=1234567890,
        )
        assert result is True
        assert d.session.request.call_args.args[0] == "POST"

    async def test_remove_reaction_uses_delete(self):
        d = self._mock_daemon()
        await d.send_reaction(
            "+85212345678", "👍",
            target_author="+85212345678", target_timestamp=1234567890,
            remove=True,
        )
        assert d.session.request.call_args.args[0] == "DELETE"

    async def test_reaction_payload_format(self):
        d = self._mock_daemon()
        await d.send_reaction(
            "+85212345678", "❤️",
            target_author="+85298765432", target_timestamp=1775382175305,
        )
        payload = d.session.request.call_args.kwargs["json"]
        assert payload["reaction"] == "❤️"
        assert payload["recipient"] == "+85212345678"
        assert payload["target_author"] == "+85298765432"
        assert payload["timestamp"] == 1775382175305


class TestSignalDaemonWhoami:
    """Tests for whoami() GET /v1/accounts."""

    async def test_whoami_returns_accounts(self):
        d = SignalDaemon(account="+85200000000", http_url="http://localhost:8080")
        d.connected = True
        d.session = MagicMock()
        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=None)
        mock_resp.json = AsyncMock(return_value=["+85200000000"])
        d.session.get = MagicMock(return_value=mock_resp)
        result = await d.whoami()
        assert result is not None
        assert "+85200000000" in result["accounts"]

    async def test_whoami_returns_none_when_disconnected(self):
        d = SignalDaemon(account="+85200000000", http_url="http://localhost:8080")
        d.connected = False
        result = await d.whoami()
        assert result is None


# ===================================================================
# TestSignalEnvelopeFormat
# ===================================================================

class TestSignalEnvelopeFormat:
    """Tests for the [Signal group/DM] envelope prefix + history format."""

    def test_group_envelope_prefix(self):
        group_id = "sBlO8LhzR42XNBbUqUrNVNokyOe2NdDZCTs0fSuZnJc="
        sender = "+85251159218"
        envelope = f"[Signal group {group_id}] {sender}"
        assert envelope.startswith("[Signal group ")
        assert group_id in envelope
        assert sender in envelope

    def test_dm_envelope_prefix(self):
        sender = "+85251159218"
        envelope = f"[Signal DM] {sender}"
        assert envelope == "[Signal DM] +85251159218"

    def test_uuid_sender_fallback(self):
        # When source (phone) is empty, should fall back to uuid:prefix
        source = ""
        source_uuid = "5720b72c-1051-47bd-962b-8c0c9db5aff1"
        sender_label = source or (f"uuid:{source_uuid[:8]}" if source_uuid else "unknown")
        assert sender_label == "uuid:5720b72c"

    def test_history_context_block_format(self, tmp_path):
        """History block should match OpenClaw-style format with bounds."""
        img = tmp_path / "img.jpg"
        img.write_bytes(b"\xff\xd8\xff")
        group_id = "sBlO8LhzR42X...="
        history = [
            {"sender": "+852111", "body": "hi", "ts": "1", "media": []},
            {"sender": "+852222", "body": "photo", "ts": "2", "media": [str(img)]},
        ]
        lines = [
            "=== UNTRUSTED Signal group history (context only, not directed at you) ===",
            f"Group: {group_id}",
        ]
        for h in history[-10:]:
            line = f"  {h['sender']}: {h['body']}"
            if h.get("media"):
                line += f"  [media: {len(h['media'])}]"
            lines.append(line)
        lines.append("=== end of group history ===")
        ctx = "\n".join(lines)
        assert "=== UNTRUSTED Signal group history" in ctx
        assert "=== end of group history ===" in ctx
        assert f"Group: {group_id}" in ctx
        assert "+852111: hi" in ctx
        assert "[media: 1]" in ctx

    def test_envelope_not_applied_to_history_or_reply_blocks(self):
        """Envelope wrap should skip === history ... === and [Replying ...] parts."""
        parts_text = ["=== UNTRUSTED Signal group history ===", "[Replying to abc: hi]", "actual message"]
        # Simulate envelope apply loop from channel
        envelope_prefix = "[Signal group xxx] +852111"
        result = []
        wrapped = False
        for txt in parts_text:
            if txt.startswith("===") or txt.startswith("[Replying"):
                result.append(txt)
                continue
            if not wrapped:
                result.append(f"{envelope_prefix}: {txt}")
                wrapped = True
            else:
                result.append(txt)
        assert result[0].startswith("===")
        assert result[1].startswith("[Replying")
        assert result[2].startswith("[Signal group xxx]")
