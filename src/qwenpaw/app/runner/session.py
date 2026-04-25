# -*- coding: utf-8 -*-
"""Safe JSON session with filename sanitization for cross-platform
compatibility.

Windows filenames cannot contain: \\ / : * ? " < > |
This module wraps agentscope's SessionBase so that session_id and user_id
are sanitized before being used as filenames.
"""
import os
import re
import json
import logging

from typing import Union, Sequence

import aiofiles
from agentscope.session import SessionBase
from agentscope_runtime.engine.schemas.exception import ConfigurationException
from ...exceptions import AgentStateError

logger = logging.getLogger(__name__)


def _safe_json_loads(content: str, filepath: str = "") -> dict:
    """Parse JSON with corruption recovery.

    Attempts standard ``json.loads`` first.  If that fails due to
    trailing garbage (a common symptom of concurrent-write race
    conditions), falls back to ``raw_decode`` to extract the first
    valid JSON object.  If the file is completely unparseable, returns
    an empty dict and logs a warning so callers never crash.

    Args:
        content: Raw file content.
        filepath: Used only for log messages.

    Returns:
        Parsed dict, or ``{}`` when the content is beyond recovery.
    """
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    # Try to extract the first valid JSON object.
    try:
        result, _ = json.JSONDecoder().raw_decode(content)
        logger.warning(
            "Session file %s had corrupted JSON. "
            "Recovered first valid object via raw_decode.",
            filepath,
        )
        return result
    except json.JSONDecodeError:
        logger.warning(
            "Session file %s is completely corrupted and could not "
            "be recovered. Returning empty dict.",
            filepath,
        )
        return {}


# Characters forbidden in Windows filenames
_UNSAFE_FILENAME_RE = re.compile(r'[\\/:*?"<>|]')


def sanitize_filename(name: str) -> str:
    """Replace characters that are illegal in Windows filenames with ``--``.

    >>> sanitize_filename('discord:dm:12345')
    'discord--dm--12345'
    >>> sanitize_filename('normal-name')
    'normal-name'
    """
    return _UNSAFE_FILENAME_RE.sub("--", name)


class SafeJSONSession(SessionBase):
    """SessionBase subclass with filename sanitization and async file I/O.

    Overrides all file-reading/writing methods to use :mod:`aiofiles` so
    that disk I/O does not block the event loop.
    """

    def __init__(
        self,
        save_dir: str = "./",
    ) -> None:
        """Initialize the JSON session class.

        Args:
            save_dir (`str`, defaults to `"./"):
                The directory to save the session state.
        """
        self.save_dir = save_dir

    def _get_save_path(self, session_id: str, user_id: str) -> str:
        """Return a filesystem-safe save path.

        Overrides the parent implementation to ensure the generated
        filename is valid on Windows, macOS and Linux.
        """
        os.makedirs(self.save_dir, exist_ok=True)
        safe_sid = sanitize_filename(session_id)
        safe_uid = sanitize_filename(user_id) if user_id else ""
        if safe_uid:
            file_path = f"{safe_uid}_{safe_sid}.json"
        else:
            file_path = f"{safe_sid}.json"
        return os.path.join(self.save_dir, file_path)

    async def save_session_state(
        self,
        session_id: str,
        user_id: str = "",
        **state_modules_mapping,
    ) -> None:
        """Save state modules to a JSON file using atomic write +
        rotating ``.prev`` backup.

        **Why the ``.prev`` rotation exists.**  Between 10:00-13:00
        on 2026-04-24 we observed multiple ``Session file ... does
        not exist`` log entries on the DM session file between a
        known-good ``Saved session state ... successfully`` and the
        next inbound message's load.  No code path in CoPaw or the
        systemd unit was found that deletes session files, but the
        file was demonstrably gone on disk.  Rather than keep
        searching for the deleter, this routine now (a) writes via
        ``<path>.tmp`` + ``os.replace`` so a crash mid-write never
        leaves an empty file, and (b) keeps the prior full version
        at ``<path>.prev`` so the matching ``load_session_state``
        can fall through to it when the primary vanishes — context
        survives even when we don't yet know what's erasing it.
        """
        state_dicts = {
            name: state_module.state_dict()
            for name, state_module in state_modules_mapping.items()
        }
        session_save_path = self._get_save_path(session_id, user_id=user_id)
        prev_path = session_save_path + ".prev"
        tmp_path = session_save_path + ".tmp"

        # **Write-first, rotate-after ordering is load-bearing.**  The
        # original ordering did ``os.replace(primary → .prev)`` THEN
        # wrote the new file — a SIGKILL between those two steps (we
        # saw one on 2026-04-24 when ``systemctl stop`` escalated
        # after ``shutdown_all_runners`` took too long) left the
        # primary gone, the ``.prev`` intact but behind, and a huge
        # chunk of recent history (compaction-resistant but not
        # flushed to ``.prev`` yet) wiped.  The safe sequence:
        #
        #   1. Write new state to ``<path>.tmp``  (primary untouched)
        #   2. ``shutil.copy2(primary → .prev)``  (backup without
        #      removing primary, so primary is NEVER unreferenced)
        #   3. ``os.replace(.tmp → primary)``     (atomic swap;
        #      primary becomes new state, no window where it's gone)
        #
        # Any crash during step 1 leaves primary intact; crash during
        # step 2 leaves primary intact and ``.prev`` may be stale (no
        # loss); crash during step 3 leaves either the old primary
        # (if rename hasn't happened) or the new primary (if it
        # has) — never a nothing-at-all window.

        # Step 1: write new state to tmp
        with open(
            tmp_path,
            "w",
            encoding="utf-8",
        ) as f:
            f.write(json.dumps(state_dicts, ensure_ascii=False))

        # Step 2: non-destructive copy of current primary to .prev
        #         (only if primary already exists — first save skips)
        if os.path.exists(session_save_path):
            try:
                import shutil

                shutil.copy2(session_save_path, prev_path)
            except Exception as e:
                logger.warning(
                    "save_session_state: failed to copy %s → .prev "
                    "(backup will be stale on next load): %s",
                    session_save_path,
                    e,
                )

        # Step 3: atomic swap tmp → primary
        os.replace(tmp_path, session_save_path)

        logger.info(
            "Saved session state to %s successfully.",
            session_save_path,
        )

    def _recover_primary_from_prev_if_missing(
        self,
        session_save_path: str,
    ) -> None:
        """Restore the primary session file from its ``.prev`` sibling
        when the primary has disappeared.

        **Why this matters.**  Three code paths read the primary
        session file: :meth:`load_session_state` (agent boot),
        :meth:`update_session_state` (key-scoped mutation), and
        :meth:`get_session_state_dict` (Console UI history).  If only
        ``load_session_state`` falls back to ``.prev``, the other two
        paths observe a missing primary and silently start from an
        empty dict.  When ``update_session_state`` then writes back,
        it overwrites the (still fine) ``.prev`` companion on its
        *next* rotation with an empty history — data loss.  This
        helper gives all three paths a common recovery step.
        """
        prev_path = session_save_path + ".prev"
        if os.path.exists(session_save_path):
            return
        if not os.path.exists(prev_path):
            return
        try:
            import shutil

            shutil.copy2(prev_path, session_save_path)
            logger.warning(
                "Session file %s was missing; recovered from %s "
                "— some tail turns since the last rotation may "
                "be lost.",
                session_save_path,
                prev_path,
            )
        except Exception as e:
            logger.error(
                "failed to recover %s from %s: %s",
                session_save_path,
                prev_path,
                e,
            )

    async def load_session_state(
        self,
        session_id: str,
        user_id: str = "",
        allow_not_exist: bool = True,
        **state_modules_mapping,
    ) -> None:
        """Load state modules from a JSON file using async I/O.

        Falls through to the ``.prev`` sibling written by
        :meth:`save_session_state` when the primary is missing — the
        symptom we're covering: a live session file that had been
        saved successfully moments earlier is sometimes gone at load
        time (root cause still under investigation, see the save
        routine's docstring).  A stale ``.prev`` is much better than
        an empty agent memory on restart.
        """
        session_save_path = self._get_save_path(session_id, user_id=user_id)

        # Primary missing but backup present → recover.
        self._recover_primary_from_prev_if_missing(session_save_path)

        if os.path.exists(session_save_path):
            async with aiofiles.open(
                session_save_path,
                "r",
                encoding="utf-8",
                errors="surrogatepass",
            ) as f:
                content = await f.read()
                states = _safe_json_loads(content, session_save_path)

            for name, state_module in state_modules_mapping.items():
                if name in states:
                    state_module.load_state_dict(states[name])
            logger.info(
                "Load session state from %s successfully.",
                session_save_path,
            )

        elif allow_not_exist:
            logger.info(
                "Session file %s does not exist. Skip loading session state.",
                session_save_path,
            )

        else:
            raise AgentStateError(
                session_id=session_id,
                message=(
                    f"Failed to load session state for file "
                    f"{session_save_path} because it does not exist"
                ),
            )

    async def update_session_state(
        self,
        session_id: str,
        key: Union[str, Sequence[str]],
        value,
        user_id: str = "",
        create_if_not_exist: bool = True,
    ) -> None:
        session_save_path = self._get_save_path(session_id, user_id=user_id)
        prev_path = session_save_path + ".prev"
        tmp_path = session_save_path + ".tmp"

        # If primary was wiped but ``.prev`` survived, resurrect before
        # we read — otherwise we would start from ``{}`` and the
        # write-back below would destroy the surviving history.  This
        # is the bug that wiped the WhatsApp DM on ``/new`` commands
        # when the primary was missing: reader saw empty, ``/new``
        # then stored ``memory.content=[]`` over the primary, and a
        # later ``save_session_state`` copied that empty primary to
        # ``.prev`` too — both files now empty.
        self._recover_primary_from_prev_if_missing(session_save_path)

        if os.path.exists(session_save_path):
            async with aiofiles.open(
                session_save_path,
                "r",
                encoding="utf-8",
                errors="surrogatepass",
            ) as f:
                content = await f.read()
                states = _safe_json_loads(content, session_save_path)

        else:
            if not create_if_not_exist:
                raise AgentStateError(
                    session_id=session_id,
                    message=f"Session file {session_save_path} does not exist",
                )
            states = {}

        path = key.split(".") if isinstance(key, str) else list(key)
        if not path:
            raise ConfigurationException(
                message="key path is empty",
            )

        cur = states
        for k in path[:-1]:
            if k not in cur or not isinstance(cur[k], dict):
                cur[k] = {}
            cur = cur[k]

        cur[path[-1]] = value

        # Same write ordering as save_session_state: tmp → copy2 →
        # atomic replace.  Direct overwrite of the primary leaves a
        # window where the primary is mid-write and reads (including
        # concurrent Console UI history fetches) can see a truncated
        # file or zero bytes on crash.
        with open(tmp_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(states, ensure_ascii=False))

        if os.path.exists(session_save_path):
            try:
                import shutil

                shutil.copy2(session_save_path, prev_path)
            except Exception as e:
                logger.warning(
                    "update_session_state: failed to copy %s → .prev: %s",
                    session_save_path,
                    e,
                )

        os.replace(tmp_path, session_save_path)

        logger.info(
            "Updated session state key '%s' in %s successfully.",
            key,
            session_save_path,
        )

    async def get_session_state_dict(
        self,
        session_id: str,
        user_id: str = "",
        allow_not_exist: bool = True,
    ) -> dict:
        """Return the session state dict from the JSON file.

        Args:
            session_id (`str`):
                The session id.
            user_id (`str`, default to `""`):
                The user ID for the storage.
            allow_not_exist (`bool`, defaults to `True`):
                Whether to allow the session to not exist. If `False`, raises
                an error if the session does not exist.

        Returns:
            `dict`:
                The session state dict loaded from the JSON file. Returns an
                empty dict if the file does not exist and
                `allow_not_exist=True`.
        """
        session_save_path = self._get_save_path(session_id, user_id=user_id)

        # Same ``.prev`` recovery as load_session_state.  Console UI
        # hits this path for /api/chats/{id} — without the fallback,
        # the UI paints a blank chat whenever the primary transiently
        # vanishes, which is what the user sees as "session drop".
        self._recover_primary_from_prev_if_missing(session_save_path)

        if os.path.exists(session_save_path):
            async with aiofiles.open(
                session_save_path,
                "r",
                encoding="utf-8",
                errors="surrogatepass",
            ) as file:
                content = await file.read()
                states = _safe_json_loads(content, session_save_path)

            logger.info(
                "Get session state dict from %s successfully.",
                session_save_path,
            )
            return states

        if allow_not_exist:
            logger.info(
                "Session file %s does not exist. Return empty state dict.",
                session_save_path,
            )
            return {}

        raise AgentStateError(
            session_id=session_id,
            message=(
                f"Failed to get session state for file {session_save_path} "
                f"because it does not exist"
            ),
        )
