# -*- coding: utf-8 -*-
"""Tests for the per-chat busy gate (``is_chat_busy``).

Listen v2 uses this instead of the workspace-global ``is_agent_busy``
so a user @-mention in one chat doesn't pause listen ticks in another.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from qwenpaw.agents.memory.proactive.proactive_utils import is_chat_busy


class _FakeTaskTracker:
    def __init__(self, active_keys: list[str]):
        self._active = active_keys

    async def list_active_tasks(self) -> list[str]:
        return list(self._active)


async def test_is_chat_busy_returns_false_when_workspace_has_no_tracker():
    ws = SimpleNamespace(task_tracker=None)
    assert await is_chat_busy(ws, "12345@g.us") is False


async def test_is_chat_busy_returns_false_when_no_active_tasks():
    ws = SimpleNamespace(task_tracker=_FakeTaskTracker([]))
    assert await is_chat_busy(ws, "12345@g.us") is False


async def test_is_chat_busy_returns_true_on_chat_id_match():
    ws = SimpleNamespace(
        task_tracker=_FakeTaskTracker(["whatsapp:12345@g.us:reply-42"]),
    )
    assert await is_chat_busy(ws, "12345@g.us") is True


async def test_is_chat_busy_returns_true_on_session_id_match():
    """Run-keys vary by call site; the helper accepts session_id as a
    fallback needle so any consistent identifier in the run_key trips
    the gate."""
    ws = SimpleNamespace(
        task_tracker=_FakeTaskTracker(["streaming:whatsapp:group:abc"]),
    )
    busy = await is_chat_busy(
        ws,
        chat_id="some-chat-id",
        session_id="whatsapp:group:abc",
    )
    assert busy is True


async def test_is_chat_busy_ignores_unrelated_active_tasks():
    """A user @-mention in CHAT A must not register as busy for CHAT B."""
    ws = SimpleNamespace(
        task_tracker=_FakeTaskTracker(
            ["whatsapp:chat-A:reply-1", "cron:job-42"],
        ),
    )
    busy = await is_chat_busy(ws, "chat-B")
    assert busy is False


async def test_is_chat_busy_swallows_tracker_errors():
    """A transient task_tracker hiccup must NOT stall listen forever —
    the helper returns False on any exception so the tick proceeds."""

    class _BoomTracker:
        async def list_active_tasks(self):
            raise RuntimeError("tracker is sad")

    ws = SimpleNamespace(task_tracker=_BoomTracker())
    assert await is_chat_busy(ws, "12345@g.us") is False


async def test_is_chat_busy_returns_false_when_all_needles_empty():
    ws = SimpleNamespace(
        task_tracker=_FakeTaskTracker(["anything"]),
    )
    assert await is_chat_busy(ws, "", session_id="", user_id="") is False


pytestmark = pytest.mark.asyncio
