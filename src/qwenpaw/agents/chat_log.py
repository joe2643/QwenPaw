# -*- coding: utf-8 -*-
"""Append-only JSONL chat log per chat_id, with reply-time reconciliation
back into agent memory.

**Why this exists.**  Console UI's chat history endpoint at
``runner/api.py:get_chat`` reads from the agent's working ``memory``
(via the latest persisted ``session.json``) — not from an
append-only log.  That has two failure modes:

1. **Memory compaction drops original turns.**  Once
   ``MemPalacePreCompactHook`` (or agentscope's ``_summarizing``)
   summarises old turns, the original messages are gone from
   ``agent.memory.content``.  ``session.json`` only holds the
   compacted state, so the UI no longer sees the originals.

2. **Mid-query SIGKILL skips the persistence finally block.**  The
   runner saves ``session.json`` only inside its query-handler
   ``finally``.  If the process is hard-killed during reasoning
   (or even gracefully terminated past the 30s drain window), the
   in-flight turn's user message + any partial reasoning chunks
   never reach disk — both the UI and the agent itself lose them.

**Design.**

* Wrap ``agent.memory.add`` so every non-transient call also appends
  a JSON line to ``<workspace>/chats/<chat_id>.jsonl``.  Wrapping at
  the memory layer (not via reply-level hooks) means we capture
  user input, every per-iteration assistant reasoning Msg, every
  tool result, and Phase A's failure tombstones — anything that
  flows through ``memory.add`` lands in the log.

* HINT-marked messages are filtered out — agentscope marks them
  with ``_MemoryMark.HINT`` precisely so it can ``delete_by_mark``
  them after one use; persisting them would just bloat the log
  with transient nudges the agent never re-reads.

* On the next ``reply()``, before the agent processes new user
  input, scan the log for entries whose timestamp is greater than
  ``session.json``'s mtime — those are the ones added between the
  last successful save and now (i.e. SIGKILL casualties).  Inject
  them back via the **unwrapped** ``memory.add`` so we don't
  re-append.  Compaction is NOT re-injected because compaction
  runs at end of reply, before save_session_state — its summary
  ends up in ``session.json``, with the original messages' log
  entries' ts < session.json mtime, so they're skipped.

* UI chat history endpoint reads from the log when it exists,
  falling back to the legacy memory-based view for chats that
  predate this module.  (See ``runner/api.py:get_chat`` for the
  reader-side wiring.)

**What this does NOT solve.**

* ``session.json`` itself is still the agent's bootstrap.  A
  partial assistant Msg with a dangling ``tool_use`` block whose
  matching ``tool_result`` was never written can confuse the LLM
  on the next prompt.  A follow-up ``X2`` could detect dangling
  tool_use and synthesise an interrupt-style ``tool_result`` —
  out of scope for this module.
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from agentscope.message import Msg

logger = logging.getLogger(__name__)

# Suffix used by every chat-log file.  Lives next to ``sessions/`` under
# the workspace dir; one file per ``chat_id``.
LOG_SUBDIR = "chats"
LOG_SUFFIX = ".jsonl"


def chat_log_path(workspace_dir: str | Path, chat_id: str) -> Path:
    """Resolve the JSONL path for *chat_id* under *workspace_dir*."""
    return Path(workspace_dir) / LOG_SUBDIR / f"{chat_id}{LOG_SUFFIX}"


def _serialise_msg(msg: Msg) -> dict[str, Any]:
    """Serialise a Msg to a dict the log can store as one JSON line.

    Prefers :meth:`Msg.to_dict` when available (agentscope ≥ 0.x);
    falls back to a manual minimal projection for stand-in objects in
    tests.
    """
    if hasattr(msg, "to_dict") and callable(msg.to_dict):
        return msg.to_dict()
    return {
        "id": getattr(msg, "id", None),
        "name": getattr(msg, "name", None),
        "role": getattr(msg, "role", None),
        "content": getattr(msg, "content", None),
    }


def _deserialise_msg(data: dict[str, Any]) -> Msg:
    """Reconstruct a Msg from a serialised dict.

    Uses :meth:`Msg.from_dict` when present (the symmetric counterpart
    to ``to_dict``); otherwise builds a minimal Msg from the fields we
    know agentscope's ``__init__`` accepts.
    """
    if hasattr(Msg, "from_dict") and callable(getattr(Msg, "from_dict")):
        try:
            return Msg.from_dict(data)
        except Exception:  # pylint: disable=broad-exception-caught
            pass
    return Msg(
        name=data.get("name") or data.get("role") or "system",
        content=data.get("content") or [],
        role=data.get("role") or "system",
    )


def append_to_log(
    workspace_dir: str | Path,
    chat_id: str,
    memories: Msg | list[Msg] | None,
    marks: Optional[Any] = None,
) -> None:
    """Append one or more Msg objects to ``<workspace>/chats/<chat_id>.jsonl``.

    Synchronous on purpose — JSON serialisation + a single ``write``
    is cheap, and putting it on a thread pool would re-introduce
    ordering bugs (two concurrent ``memory.add`` calls could land out
    of order in the log).  The file open mode is ``"a"`` so OS-level
    write atomicity holds for lines under PIPE_BUF (4KB on Linux);
    larger Msgs are still safe because we hold the file handle for
    the whole batch.

    HINT-marked messages are filtered out by the caller, not here —
    keeping that policy at the call site means a future caller that
    does want HINTs persisted (e.g. for debugging) doesn't have to
    fight a hard-coded filter.

    Best-effort: any exception is logged and swallowed.  The chat
    log is a recovery aid; failing to write must NEVER block the
    actual ``memory.add`` that already succeeded.
    """
    if memories is None:
        return
    msgs = memories if isinstance(memories, list) else [memories]
    msgs = [m for m in msgs if m is not None]
    if not msgs:
        return

    try:
        path = chat_log_path(workspace_dir, chat_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).isoformat()
        # Normalise marks to a list[str] for the on-disk shape.
        mark_list: list[str]
        if marks is None:
            mark_list = []
        elif isinstance(marks, (list, tuple, set)):
            mark_list = [str(m) for m in marks]
        else:
            mark_list = [str(marks)]

        with path.open("a", encoding="utf-8") as f:
            for m in msgs:
                line = json.dumps(
                    {
                        "ts": ts,
                        "marks": mark_list,
                        "msg": _serialise_msg(m),
                    },
                    ensure_ascii=False,
                )
                f.write(line + "\n")
    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.warning(
            "chat_log: append failed for chat=%s: %s "
            "(memory.add already succeeded; UI may be missing this turn)",
            chat_id,
            e,
        )


def read_log(
    workspace_dir: str | Path,
    chat_id: str,
) -> list[dict[str, Any]]:
    """Read all entries for *chat_id*, oldest first.

    Returns a list of envelope dicts (each ``{"ts": ..., "marks":
    [...], "msg": {...}}``).  Skips malformed lines silently — a
    partial line from a half-written tail (process killed mid-write)
    must not blow up the whole reader.
    """
    path = chat_log_path(workspace_dir, chat_id)
    if not path.exists():
        return []
    out: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except OSError as e:
        logger.warning("chat_log: read failed for chat=%s: %s", chat_id, e)
    return out


def collect_unpersisted(
    workspace_dir: str | Path,
    chat_id: str,
    session_json_path: Optional[str | Path],
    memory_msg_ids: set[str],
) -> list[Msg]:
    """Build the list of Msg objects in the log that haven't yet been
    persisted into ``session.json``.

    Watermark policy: the boundary between "persisted" and "lost in
    flight" is ``session.json``'s mtime.  Anything in the log whose
    ``ts`` is on-or-before that mtime was already part of a successful
    ``save_session_state`` call (or pre-dates it).  Anything strictly
    after that mtime was added by ``memory.add`` between the last
    save and the next process death.

    A second filter — ``msg.id not in memory_msg_ids`` — guards the
    case where ``session.json`` was just loaded back (so its msg ids
    *are* the current memory's ids) yet has the same mtime as a
    preceding log entry due to clock granularity.  Belt-and-braces.

    HINT-marked entries are skipped — they're transient nudges the
    parent ``ReActAgent`` would have ``delete_by_mark``-ed already.
    """
    entries = read_log(workspace_dir, chat_id)
    if not entries:
        return []

    watermark_iso: Optional[str] = None
    if session_json_path is not None:
        try:
            mtime = os.path.getmtime(str(session_json_path))
            watermark_iso = datetime.fromtimestamp(
                mtime, tz=timezone.utc,
            ).isoformat()
        except OSError:
            watermark_iso = None

    unpersisted: list[Msg] = []
    for entry in entries:
        # Filter: HINT marks
        marks = entry.get("marks") or []
        if "hint" in [str(m).lower() for m in marks]:
            continue

        # Filter: watermark
        ts = entry.get("ts") or ""
        if watermark_iso is not None and ts and ts <= watermark_iso:
            continue

        msg_data = entry.get("msg") or {}
        msg_id = msg_data.get("id")

        # Filter: already in memory
        if msg_id and msg_id in memory_msg_ids:
            continue

        try:
            unpersisted.append(_deserialise_msg(msg_data))
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.debug(
                "chat_log: skipping malformed entry for chat=%s: %s",
                chat_id, e,
            )
            continue

    return unpersisted
