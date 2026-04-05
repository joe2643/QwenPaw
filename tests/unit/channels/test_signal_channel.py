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
        assert "Replying to" in combined
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
        assert any("Replying to" in p.text for p in text_parts)

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

    async def test_markdown_converted(self):
        ch = _make_channel()
        await ch.send("+85200000001", "**bold** text", {})
        call_args = ch.daemon.send_message.call_args
        # Text should have markdown stripped
        sent_text = call_args.args[1] if len(call_args.args) > 1 else call_args.kwargs.get("text", "")
        # The message arg is positional arg index 1
        assert "**" not in str(call_args)

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
