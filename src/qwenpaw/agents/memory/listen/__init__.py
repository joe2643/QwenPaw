# -*- coding: utf-8 -*-
"""Listen-mode memory submodule for CoPaw agents.

Listen mode is a per-chat timer that, every N minutes, hands the
channel's in-memory ``_group_history`` buffer (the rolling tail of
non-@mentioned chatter — see ``BaseChannel._group_history``) to a small
LLM call that decides whether to chime in.  If the LLM returns a reply,
it is delivered back to the same chat via ``channel.send``.

Listen is independent of :mod:`proactive`:

* ``/proactive`` is workspace-wide and timer-driven on idle.  It nudges
  the user with task follow-ups when the agent has been quiet for N
  minutes.
* ``/listen`` is per-chat and timer-driven on a fixed cadence.  It reads
  the buffered chatter and lets the agent join the conversation
  naturally without being @-mentioned.

Both can be enabled at the same time and they share no state.
"""
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from .listen_types import ListenConfig
    from .listen_trigger import (
        disable_listen_for_chat,
        enable_listen_for_chat,
        listen_configs,
        listen_tasks,
        listen_trigger_loop,
    )

__all__ = [
    "ListenConfig",
    "enable_listen_for_chat",
    "disable_listen_for_chat",
    "listen_trigger_loop",
    "listen_configs",
    "listen_tasks",
]


def __getattr__(name: str):
    # Lazy re-export: avoid pulling in model_factory / agentscope at
    # import time of ``qwenpaw.agents.memory.__init__``.
    if name == "ListenConfig":
        from .listen_types import ListenConfig as _LC

        return _LC
    if name in (
        "enable_listen_for_chat",
        "disable_listen_for_chat",
        "listen_trigger_loop",
        "listen_configs",
        "listen_tasks",
    ):
        from . import listen_trigger as _t

        return getattr(_t, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
