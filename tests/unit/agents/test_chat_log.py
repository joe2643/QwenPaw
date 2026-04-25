# -*- coding: utf-8 -*-
"""Unit tests for ``qwenpaw.agents.chat_log``.

Covers:
* append/read round-trip
* HINT-mark filtering
* malformed-line tolerance (partial write from a SIGKILL'd tail)
* watermark filtering on the reconcile path (mtime-based)
* msg.id dedup against current memory
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

from agentscope.message import Msg

from qwenpaw.agents.chat_log import (
    append_to_log,
    chat_log_path,
    collect_unpersisted,
    read_log,
)


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    return tmp_path


def _msg(text: str, role: str = "user", msg_id: str | None = None) -> Msg:
    """Build a Msg matching agentscope's ctor signature.  ``id`` defaults
    to a uuid agentscope assigns; we override only when a test cares."""
    m = Msg(name=role, content=text, role=role)
    if msg_id is not None:
        m.id = msg_id
    return m


def _text(msg_or_dict) -> str:
    """Pull the text content out of a Msg or its serialised dict.

    agentscope keeps ``content`` as a plain string when constructed
    with a string arg, and as a list[ContentBlock] when constructed
    with structured blocks — both shapes need to round-trip through
    the log identically.  Tests use this helper so they don't have
    to branch on which shape happens to land.
    """
    content = (
        msg_or_dict["msg"]["content"]
        if isinstance(msg_or_dict, dict) and "msg" in msg_or_dict
        else getattr(msg_or_dict, "content", msg_or_dict)
    )
    if isinstance(content, str):
        return content
    if isinstance(content, list) and content:
        first = content[0]
        if isinstance(first, dict):
            return first.get("text", "")
        return getattr(first, "text", "")
    return ""


class TestAppendRead:
    def test_round_trip_single_message(self, workspace: Path):
        append_to_log(workspace, "chat-A", _msg("hi"))
        entries = read_log(workspace, "chat-A")
        assert len(entries) == 1
        assert _text(entries[0]) == "hi"
        assert entries[0]["msg"]["role"] == "user"
        assert entries[0]["marks"] == []

    def test_round_trip_list(self, workspace: Path):
        append_to_log(
            workspace,
            "chat-B",
            [_msg("first"), _msg("second", role="assistant")],
        )
        entries = read_log(workspace, "chat-B")
        assert len(entries) == 2
        assert entries[0]["msg"]["role"] == "user"
        assert entries[1]["msg"]["role"] == "assistant"

    def test_appends_preserve_order(self, workspace: Path):
        for i in range(5):
            append_to_log(workspace, "chat-C", _msg(f"msg-{i}"))
        entries = read_log(workspace, "chat-C")
        assert [_text(e) for e in entries] == [
            f"msg-{i}" for i in range(5)
        ]

    def test_marks_persisted_as_list(self, workspace: Path):
        append_to_log(workspace, "chat-D", _msg("hint"), marks="hint")
        append_to_log(workspace, "chat-E", _msg("none"), marks=None)
        append_to_log(workspace, "chat-F", _msg("multi"), marks=["a", "b"])

        assert read_log(workspace, "chat-D")[0]["marks"] == ["hint"]
        assert read_log(workspace, "chat-E")[0]["marks"] == []
        assert read_log(workspace, "chat-F")[0]["marks"] == ["a", "b"]

    def test_none_input_is_noop(self, workspace: Path):
        append_to_log(workspace, "chat-G", None)
        assert not chat_log_path(workspace, "chat-G").exists()

    def test_empty_list_is_noop(self, workspace: Path):
        append_to_log(workspace, "chat-H", [])
        assert not chat_log_path(workspace, "chat-H").exists()

    def test_skips_none_in_list(self, workspace: Path):
        append_to_log(workspace, "chat-I", [_msg("kept"), None])
        entries = read_log(workspace, "chat-I")
        assert len(entries) == 1
        assert _text(entries[0]) == "kept"


class TestReadResilience:
    def test_skips_malformed_lines(self, workspace: Path):
        """A SIGKILL during write can leave a half-line at EOF.  The
        reader must skip it without dropping valid earlier lines."""
        append_to_log(workspace, "chat-X", _msg("good"))
        path = chat_log_path(workspace, "chat-X")
        # Append a deliberately broken trailing line.
        with path.open("a", encoding="utf-8") as f:
            f.write('{"ts": "2026-01-01", "msg": {bro\n')

        entries = read_log(workspace, "chat-X")
        assert len(entries) == 1
        assert _text(entries[0]) == "good"

    def test_returns_empty_for_missing_file(self, workspace: Path):
        assert read_log(workspace, "never-written") == []


class TestCollectUnpersisted:
    def test_no_log_returns_empty(self, workspace: Path):
        out = collect_unpersisted(
            workspace, "chat-none", session_json_path=None,
            memory_msg_ids=set(),
        )
        assert out == []

    def test_no_session_json_returns_all_non_hint(self, workspace: Path):
        """No session.json yet (first turn ever) ⇒ everything in log is
        unpersisted by definition.  Watermark just doesn't apply."""
        append_to_log(workspace, "c", _msg("u1"))
        append_to_log(workspace, "c", _msg("a1", role="assistant"))
        append_to_log(workspace, "c", _msg("hint"), marks="hint")

        out = collect_unpersisted(
            workspace, "c", session_json_path=None,
            memory_msg_ids=set(),
        )
        roles = [m.role for m in out]
        # HINT skipped; user + assistant kept.
        assert "user" in roles
        assert "assistant" in roles
        assert len(out) == 2

    def test_watermark_skips_old_entries(
        self, workspace: Path, tmp_path: Path,
    ):
        """Entries with ts <= session.json mtime are persisted."""
        # First batch of writes.
        append_to_log(workspace, "c", _msg("old-1"))
        append_to_log(workspace, "c", _msg("old-2"))

        # Simulate a successful save_session_state landing at "now".
        sess_path = tmp_path / "sess.json"
        sess_path.write_text("{}")
        # Wait long enough that subsequent log entries get a strictly
        # later ISO timestamp.  ISO precision is microseconds and
        # ``time.sleep(0.01)`` is enough on every supported platform.
        time.sleep(0.05)

        # New writes that happen after the save (i.e. SIGKILL casualties
        # if the next save never lands).
        append_to_log(workspace, "c", _msg("new-1"))
        append_to_log(workspace, "c", _msg("new-2", role="assistant"))

        out = collect_unpersisted(
            workspace, "c", sess_path, memory_msg_ids=set(),
        )
        texts = [_text(m) for m in out]
        assert texts == ["new-1", "new-2"]

    def test_dedup_by_msg_id(self, workspace: Path):
        """msg.ids already in memory aren't re-injected even if their
        log ts is past the watermark."""
        m1 = _msg("dup", msg_id="abc-123")
        m2 = _msg("fresh", msg_id="def-456")
        append_to_log(workspace, "c", m1)
        append_to_log(workspace, "c", m2)

        out = collect_unpersisted(
            workspace, "c",
            session_json_path=None,
            memory_msg_ids={"abc-123"},
        )
        ids = [m.id for m in out]
        assert ids == ["def-456"]

    def test_hint_marks_excluded(self, workspace: Path):
        append_to_log(workspace, "c", _msg("real"))
        append_to_log(workspace, "c", _msg("nudge"), marks="hint")
        append_to_log(workspace, "c", _msg("nudge2"), marks=["HINT"])

        out = collect_unpersisted(
            workspace, "c",
            session_json_path=None,
            memory_msg_ids=set(),
        )
        texts = [_text(m) for m in out]
        assert texts == ["real"]


class TestPathLayout:
    def test_path_under_chats_subdir(self, workspace: Path):
        p = chat_log_path(workspace, "abc-123")
        assert p.parent == workspace / "chats"
        assert p.name == "abc-123.jsonl"

    def test_creates_parent_on_append(self, workspace: Path):
        # Parent dir doesn't exist yet — append must create it.
        assert not (workspace / "chats").exists()
        append_to_log(workspace, "fresh", _msg("hi"))
        assert (workspace / "chats").is_dir()
        assert (workspace / "chats" / "fresh.jsonl").is_file()
