# -*- coding: utf-8 -*-
"""Tests for media block compaction in MemoryCompactionHook."""

import os

import pytest
from agentscope.message import Msg, TextBlock, ImageBlock, VideoBlock

from qwenpaw.agents.hooks.memory_compaction import MemoryCompactionHook


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_text_msg(role, text):
    return Msg(name=role, role=role, content=[TextBlock(type="text", text=text)])


def _make_video_msg(role, path):
    return Msg(
        name=role, role="assistant",
        content=[
            TextBlock(type="text", text=f"Video loaded: {os.path.basename(path)}"),
            VideoBlock(type="video", source={"type": "url", "url": path}),
        ],
    )


def _make_image_msg(role, path):
    return Msg(
        name=role, role="assistant",
        content=[
            TextBlock(type="text", text=f"Image loaded: {os.path.basename(path)}"),
            ImageBlock(type="image", source={"type": "url", "url": path}),
        ],
    )


def _make_mixed_msg(role, text, video_path):
    return Msg(
        name=role, role="assistant",
        content=[
            TextBlock(type="text", text=text),
            VideoBlock(type="video", source={"type": "url", "url": video_path}),
            TextBlock(type="text", text="some follow-up text"),
        ],
    )


def _has_media_block(msg):
    if not isinstance(msg.content, list):
        return False
    return any(
        isinstance(b, dict) and b.get("type") in ("video", "image")
        for b in msg.content
    )


def _count_media_blocks(messages):
    return sum(1 for m in messages if _has_media_block(m))


def _get_placeholder_texts(msg):
    if not isinstance(msg.content, list):
        return []
    return [
        b.get("text", "")
        for b in msg.content
        if isinstance(b, dict) and b.get("type") == "text"
    ]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestCompactMediaBlocks:

    def test_no_messages(self):
        assert MemoryCompactionHook._compact_media_blocks([], recent_n=2) == 0

    def test_no_media_blocks(self):
        msgs = [_make_text_msg("user", "hello"), _make_text_msg("assistant", "hi")]
        assert MemoryCompactionHook._compact_media_blocks(msgs, recent_n=1) == 0

    def test_video_compacted_in_old_messages(self):
        msgs = [
            _make_text_msg("user", "show me the video"),
            _make_video_msg("assistant", "/tmp/big_video.mp4"),
            _make_text_msg("user", "thanks"),
            _make_text_msg("assistant", "no problem"),
        ]
        result = MemoryCompactionHook._compact_media_blocks(msgs, recent_n=2)
        assert result == 1
        assert not _has_media_block(msgs[1])
        texts = _get_placeholder_texts(msgs[1])
        assert any("big_video.mp4" in t for t in texts)
        assert any("removed from context" in t for t in texts)

    def test_image_compacted_in_old_messages(self):
        msgs = [
            _make_image_msg("assistant", "/tmp/screenshot.png"),
            _make_text_msg("user", "nice"),
            _make_text_msg("assistant", "thanks"),
        ]
        result = MemoryCompactionHook._compact_media_blocks(msgs, recent_n=2)
        assert result == 1
        assert not _has_media_block(msgs[0])
        texts = _get_placeholder_texts(msgs[0])
        assert any("screenshot.png" in t for t in texts)

    def test_placeholder_is_prepended_not_inline(self):
        # After compaction the <system-note> placeholder must sit at
        # the *top* of the message content list, before any
        # surviving user text.  Inline placement (where the image
        # was) is subtly worse for the model: the agent reads text
        # first, then the bracketed placeholder mid-stream, and has
        # to back-patch its interpretation.  Prepending frames the
        # whole turn with "these media were viewed and removed"
        # before any other content.
        msgs = [
            _make_text_msg("user", "filler-top"),
            # Synthesise a message whose content has text -> image ->
            # text, to prove the placeholder rises to the top even
            # when the image was originally in the middle.
            _make_text_msg("user", "here is the pic"),
        ]
        msgs[1].content = [
            {"type": "text", "text": "before"},
            {
                "type": "image",
                "source": {"type": "url", "url": "/tmp/middle.png"},
            },
            {"type": "text", "text": "after"},
        ]
        msgs.append(_make_text_msg("assistant", "ack"))

        MemoryCompactionHook._compact_media_blocks(msgs, recent_n=1)
        content = msgs[1].content

        # First block must be the system-note placeholder.
        assert content[0]["type"] == "text"
        assert "<system-note>" in content[0]["text"]
        assert "middle.png" in content[0]["text"]
        # The surviving user text follows, in original order.
        assert content[1]["text"] == "before"
        assert content[2]["text"] == "after"
        # And no media block survives.
        assert all(b.get("type") != "image" for b in content)

    def test_placeholder_is_wrapped_in_system_note_tag(self):
        # The placeholder has to be easily distinguishable from user
        # text — otherwise the model quotes "[Image was viewed: ...
        # — removed from context]" back at the user as if it were a
        # real explanation.  Tag convention matches the other
        # out-of-band injections (<system-hint>, <system-info>).
        msgs = [
            _make_image_msg("user", "/tmp/receipt.png"),
            _make_text_msg("assistant", "got it"),
            _make_text_msg("user", "thanks"),
        ]
        MemoryCompactionHook._compact_media_blocks(msgs, recent_n=1)
        texts = _get_placeholder_texts(msgs[0])
        joined = "\n".join(texts)
        assert "<system-note>" in joined
        assert "</system-note>" in joined
        # The human-readable marker must still be inside the tag so
        # the model's summarisation heuristics still notice it.
        assert "receipt.png" in joined

    def test_recent_media_preserved(self):
        msgs = [
            _make_text_msg("user", "old"),
            _make_video_msg("assistant", "/tmp/old.mp4"),
            _make_text_msg("user", "new"),
            _make_video_msg("assistant", "/tmp/new.mp4"),
        ]
        result = MemoryCompactionHook._compact_media_blocks(msgs, recent_n=2)
        assert result == 1
        assert not _has_media_block(msgs[1])
        assert _has_media_block(msgs[3])

    def test_all_recent_nothing_compacted(self):
        msgs = [_make_video_msg("a", "/tmp/a.mp4"), _make_video_msg("a", "/tmp/b.mp4")]
        assert MemoryCompactionHook._compact_media_blocks(msgs, recent_n=5) == 0
        assert _count_media_blocks(msgs) == 2

    def test_mixed_content_only_media_removed(self):
        msgs = [
            _make_mixed_msg("assistant", "here is the analysis", "/tmp/vid.mp4"),
            _make_text_msg("user", "recent"),
        ]
        result = MemoryCompactionHook._compact_media_blocks(msgs, recent_n=1)
        assert result == 1
        assert not _has_media_block(msgs[0])
        texts = _get_placeholder_texts(msgs[0])
        assert any("here is the analysis" in t for t in texts)
        assert any("some follow-up text" in t for t in texts)
        assert any("vid.mp4" in t for t in texts)

    def test_multiple_media_blocks_compacted(self):
        msgs = [
            _make_video_msg("a", "/tmp/v1.mp4"),
            _make_image_msg("a", "/tmp/img1.png"),
            _make_video_msg("a", "/tmp/v2.mp4"),
            _make_text_msg("user", "recent"),
        ]
        result = MemoryCompactionHook._compact_media_blocks(msgs, recent_n=1)
        assert result == 3
        assert _count_media_blocks(msgs) == 0

    def test_placeholder_contains_basename(self):
        long_path = "/home/joe/.qwenpaw/workspaces/default/media/downloads/very_long_video_name.mp4"
        msgs = [_make_video_msg("a", long_path), _make_text_msg("user", "recent")]
        MemoryCompactionHook._compact_media_blocks(msgs, recent_n=1)
        texts = _get_placeholder_texts(msgs[0])
        assert any("very_long_video_name.mp4" in t for t in texts)
        # Full path should NOT be in placeholder (privacy fix)
        assert not any(long_path in t for t in texts)

    def test_dict_style_video_block(self):
        msg = Msg(
            name="assistant", role="assistant",
            content=[
                {"type": "text", "text": "here"},
                {"type": "video", "source": {"type": "url", "url": "/tmp/dict_vid.mp4"}},
            ],
        )
        msgs = [msg, _make_text_msg("user", "recent")]
        assert MemoryCompactionHook._compact_media_blocks(msgs, recent_n=1) == 1

    def test_recent_n_zero_compacts_everything(self):
        msgs = [_make_video_msg("a", "/tmp/a.mp4"), _make_video_msg("a", "/tmp/b.mp4")]
        assert MemoryCompactionHook._compact_media_blocks(msgs, recent_n=0) == 2
        assert _count_media_blocks(msgs) == 0

    def test_idempotent(self):
        msgs = [_make_video_msg("a", "/tmp/v.mp4"), _make_text_msg("user", "recent")]
        first = MemoryCompactionHook._compact_media_blocks(msgs, recent_n=1)
        assert first == 1
        second = MemoryCompactionHook._compact_media_blocks(msgs, recent_n=1)
        assert second == 0
