# -*- coding: utf-8 -*-
"""Type definitions for the listen-mode feature."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Literal, Optional

# ``normal``  — neutral threshold: chime in when there's a clear hook,
#               stay quiet otherwise.  Default.
# ``aggressive`` — lean toward speaking on every tick unless safety
#                  rails kick in.  Used by ``/listen aggressive ...``
#                  for chats where the user wants the bot to feel
#                  present.
Verbosity = Literal["normal", "aggressive"]


@dataclass
class ListenConfig:
    """Per-chat listen configuration.

    Keyed in ``listen_configs`` by ``f"{channel_name}:{chat_id}"``.

    Fields snapshot the originating request so the trigger loop can
    deliver back to the same chat without needing fresh request context.
    """

    enabled: bool = True
    interval_minutes: int = 5
    channel_name: str = ""
    chat_id: str = ""
    chat_meta: Dict[str, Any] = field(default_factory=dict)
    agent_id: str = ""
    verbosity: Verbosity = "normal"
    # Captured at /listen enable time so the trigger loop can read the
    # chat's persisted agent memory via
    # ``workspace.runner.session.get_session_state_dict(...)``.  Without
    # these, the LLM only sees the in-memory non-mention buffer and has
    # no context about prior bot conversations in the room.
    session_id: str = ""
    user_id: str = ""
    last_fire: Optional[datetime] = None
    # Highest message timestamp we have already inspected.  Used so the
    # next tick can skip when the buffer hasn't grown since last fire
    # (no new chatter ⇒ nothing new to chime in on).
    last_seen_ts: str = ""
    # When the last action step actually dispatched a chime-in (not just
    # PASSed).  Drives the ``min_chime_gap_seconds`` throttle so bursts
    # of replies don't pile up if the room is fast-moving.
    last_chime_ts: Optional[datetime] = None
    # Minimum seconds between actual chime-ins.  Decision step still
    # fires on the normal cadence, but the throttle short-circuits before
    # the action step even when the LLM said CHIME.
    min_chime_gap_seconds: int = 300
    # Hard cap on the action-step react-agent reply.  120s comfortably
    # covers 1-2 tool calls; beyond that the room has moved on and the
    # chime is stale anyway.
    action_timeout_seconds: int = 120
