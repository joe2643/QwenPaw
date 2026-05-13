# -*- coding: utf-8 -*-
"""Regression test for the chat-log reconciler watermark path.

Bug:
    ``_reconcile_chat_log_into_memory`` used to compute the session.json
    path as ``sessions/{fname}`` regardless of channel.  For
    channel-namespaced WhatsApp/Signal sessions, the saver writes to
    ``sessions/{channel}/{fname}`` while leaving the legacy flat copy
    stale.  Reading mtime from the stale flat file made the watermark
    hours-to-days old, so every chat_log entry written since was
    re-injected as "unpersisted" — symptomatic as "context recovers
    full session after /compact".

This test pins the path-selection logic so the bug can't sneak back.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from qwenpaw.agents.react_agent import QwenPawAgent


def _touch(path: Path, mtime: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{}")
    os.utime(path, (mtime, mtime))


def test_channel_namespaced_path_preferred_when_present(tmp_path: Path):
    """When both flat and channel-namespaced copies exist, pick the
    channel one — that's where the saver writes."""
    session_id = "whatsapp:group:1@g.us"
    user_id = "group:1@g.us"
    channel = "whatsapp"

    # Build both files; flat is older (yesterday), namespaced is fresh.
    yesterday = time.time() - 86400
    now = time.time()
    from qwenpaw.app.runner.session import sanitize_filename

    safe_sid = sanitize_filename(session_id)
    safe_uid = sanitize_filename(user_id)
    fname = f"{safe_uid}_{safe_sid}.json"

    flat = tmp_path / "sessions" / fname
    namespaced = tmp_path / "sessions" / sanitize_filename(channel) / fname
    _touch(flat, yesterday)
    _touch(namespaced, now)

    resolved = QwenPawAgent._resolve_watermark_path(
        tmp_path,
        session_id,
        user_id,
        channel,
    )
    assert resolved == namespaced
    # And the mtime read from it is the fresh one.
    assert abs(os.path.getmtime(resolved) - now) < 1


def test_falls_back_to_flat_path_before_migration(tmp_path: Path):
    """First turn after enabling per-channel layout: only the legacy
    flat file exists.  Use it so the watermark isn't empty."""
    session_id = "whatsapp:+85251159218"
    user_id = "+85251159218"
    channel = "whatsapp"

    from qwenpaw.app.runner.session import sanitize_filename

    safe_sid = sanitize_filename(session_id)
    safe_uid = sanitize_filename(user_id)
    fname = f"{safe_uid}_{safe_sid}.json"

    flat = tmp_path / "sessions" / fname
    _touch(flat, time.time() - 3600)

    resolved = QwenPawAgent._resolve_watermark_path(
        tmp_path,
        session_id,
        user_id,
        channel,
    )
    assert resolved == flat


def test_no_channel_uses_flat_layout(tmp_path: Path):
    """Console / agent-chat sessions have no channel — flat is canonical."""
    session_id = "console-test"
    user_id = "default"

    from qwenpaw.app.runner.session import sanitize_filename

    safe_sid = sanitize_filename(session_id)
    safe_uid = sanitize_filename(user_id)
    fname = f"{safe_uid}_{safe_sid}.json"

    expected = tmp_path / "sessions" / fname
    resolved = QwenPawAgent._resolve_watermark_path(
        tmp_path,
        session_id,
        user_id,
        channel="",
    )
    assert resolved == expected


def test_returns_namespaced_path_when_nothing_exists(tmp_path: Path):
    """When neither file exists yet, prefer the channel-namespaced
    path — caller (collect_unpersisted) treats a missing file as
    'no watermark' and proceeds correctly."""
    resolved = QwenPawAgent._resolve_watermark_path(
        tmp_path,
        "whatsapp:group:1@g.us",
        "group:1@g.us",
        "whatsapp",
    )
    assert "whatsapp" in resolved.parts
    assert not resolved.exists()
