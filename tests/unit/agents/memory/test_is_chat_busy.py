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


# ---------------------------------------------------------------------------
# ChatSpec.id resolution path (regression for the "listen fires during
# compaction" symptom — run_key is a UUID that doesn't contain any of
# the channel-side identifiers, so substring match alone always missed).
# ---------------------------------------------------------------------------


class _FakeChat:
    def __init__(self, chat_id, session_id, user_id, channel):
        self.id = chat_id
        self.session_id = session_id
        self.user_id = user_id
        self.channel = channel


class _FakeChatManager:
    def __init__(self, chats):
        self._chats = chats

    async def list_chats(self):
        return list(self._chats)


async def test_resolves_chatspec_id_by_session_id():
    """Real bug we hit: a normal @-mention reply was running for this
    chat, but ``is_chat_busy`` returned False because the run_key in
    task_tracker was the ChatSpec UUID, not the channel session URI.
    Listen kept firing during the agent's compaction LLM call."""

    chat_uuid = "abc12345-1234-1234-1234-1234567890ab"
    ws = SimpleNamespace(
        task_tracker=_FakeTaskTracker([chat_uuid]),
        chat_manager=_FakeChatManager(
            [
                _FakeChat(
                    chat_id=chat_uuid,
                    session_id="whatsapp:group:120363@g.us",
                    user_id="group:120363@g.us",
                    channel="whatsapp",
                ),
            ],
        ),
    )

    busy = await is_chat_busy(
        ws,
        chat_id="120363@g.us",
        session_id="whatsapp:group:120363@g.us",
        user_id="group:120363@g.us",
        channel="whatsapp",
    )
    assert busy is True


async def test_chatspec_match_blocks_when_parallel_child_run_key():
    """Parallel runs use ``{parent}::run:{hex}`` as the concrete key.
    The match must catch those too — otherwise listen would fire
    against a chat that's mid-reply in a parallel child."""

    parent_uuid = "abc12345-1234-1234-1234-1234567890ab"
    child_key = f"{parent_uuid}::run:deadbeefcafe"
    ws = SimpleNamespace(
        task_tracker=_FakeTaskTracker([child_key]),
        chat_manager=_FakeChatManager(
            [
                _FakeChat(
                    chat_id=parent_uuid,
                    session_id="whatsapp:group:1@g.us",
                    user_id="group:1@g.us",
                    channel="whatsapp",
                ),
            ],
        ),
    )

    busy = await is_chat_busy(
        ws,
        chat_id="1@g.us",
        session_id="whatsapp:group:1@g.us",
        channel="whatsapp",
    )
    assert busy is True


async def test_chatspec_match_rejects_wrong_channel():
    """A chat with same session_id but DIFFERENT channel must NOT
    block — channels are isolated."""

    chat_uuid = "abc12345-1234-1234-1234-1234567890ab"
    ws = SimpleNamespace(
        task_tracker=_FakeTaskTracker([chat_uuid]),
        chat_manager=_FakeChatManager(
            [
                _FakeChat(
                    chat_id=chat_uuid,
                    session_id="whatsapp:group:1@g.us",
                    user_id="group:1@g.us",
                    channel="signal",  # different channel
                ),
            ],
        ),
    )

    busy = await is_chat_busy(
        ws,
        chat_id="1@g.us",
        session_id="whatsapp:group:1@g.us",
        channel="whatsapp",
    )
    assert busy is False


async def test_falls_back_to_substring_match_when_no_chat_manager():
    """Without a chat_manager (unit tests, partial workspace), the
    substring path is the only signal we have.  Keep it working."""

    ws = SimpleNamespace(
        task_tracker=_FakeTaskTracker(["streaming:whatsapp:group:abc"]),
        chat_manager=None,
    )

    busy = await is_chat_busy(
        ws,
        chat_id="any",
        session_id="whatsapp:group:abc",
    )
    assert busy is True


pytestmark = pytest.mark.asyncio
