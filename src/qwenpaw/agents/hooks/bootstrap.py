# -*- coding: utf-8 -*-
"""Bootstrap hook for first-time user interaction guidance.

This hook checks for BOOTSTRAP.md on the first user interaction and
prepends guidance to help set up the agent's identity and preferences.
"""
import logging
from pathlib import Path
from typing import Any

from ..prompt import build_bootstrap_guidance
from ..utils import (
    is_first_user_interaction,
    prepend_to_message_content,
)

logger = logging.getLogger(__name__)


class _MemPalaceDisabled(Exception):
    """Sentinel raised when MemPalace is disabled in agent config."""


class BootstrapHook:
    """Hook for bootstrap guidance on first user interaction.

    This hook looks for a BOOTSTRAP.md file in the working directory
    and if found, prepends guidance to the first user message to help
    establish the agent's identity and user preferences.
    """

    def __init__(
        self,
        working_dir: Path,
        language: str = "zh",
    ):
        """Initialize bootstrap hook.

        Args:
            working_dir: Working directory containing BOOTSTRAP.md
            language: Language code for bootstrap guidance (en/zh)
        """
        self.working_dir = working_dir
        self.language = language

    async def __call__(
        self,
        agent,
        kwargs: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Check and load BOOTSTRAP.md on first user interaction.

        Args:
            agent: The agent instance
            kwargs: Input arguments to the _reasoning method

        Returns:
            None (hook doesn't modify kwargs)
        """
        try:
            bootstrap_path = self.working_dir / "BOOTSTRAP.md"
            bootstrap_completed_flag = (
                self.working_dir / ".bootstrap_completed"
            )

            # MemPalace wake-up context — inject L0+L1 into SYSTEM PROMPT
            # System prompt survives memory compaction; user messages do not.
            # We monkey-patch _build_sys_prompt so future rebuild_sys_prompt
            # calls also preserve the wake-up. We also detect compaction events
            # via memory.get_compressed_summary() markers and refresh wake-up
            # so the prompt reflects the latest palace state after compaction.
            # Gated on agent_config.mempalace.enabled (same flag as diary hooks
            # in react_agent.py) — when disabled, skip the wake-up entirely
            # (WAL recovery + BOOTSTRAP.md below still run).
            try:
                _mp_cfg = getattr(
                    getattr(agent, "_agent_config", None),
                    "mempalace",
                    None,
                )
                _mp_enabled = bool(
                    _mp_cfg is not None
                    and getattr(_mp_cfg, "enabled", False),
                )
                if not _mp_enabled:
                    raise _MemPalaceDisabled
                import subprocess
                import sys

                def _fetch_wakeup_block():
                    try:
                        result = subprocess.run(
                            [sys.executable, "-m", "mempalace", "wake-up"],
                            capture_output=True,
                            text=True,
                            timeout=15,
                        )
                        if result.returncode != 0 or not result.stdout.strip():
                            logger.warning(
                                "MemPalace wake-up subprocess: rc=%s stdout=%dB stderr=%s",
                                result.returncode,
                                len(result.stdout or ""),
                                (result.stderr or "")[:500],
                            )
                            return None
                        lines = result.stdout.strip().split("\n")
                        wakeup = "\n".join(
                            line
                            for line in lines
                            if not line.startswith("Wake-up text")
                            and not line.startswith("===")
                        ).strip()
                        if not wakeup or len(wakeup) <= 50:
                            logger.warning(
                                "MemPalace wake-up too short: len=%d preview=%r",
                                len(wakeup),
                                wakeup[:200],
                            )
                            return None
                        return "\n\n## MemPalace Wake-up Context\n" + wakeup
                    except Exception as _e:
                        logger.warning(
                            "MemPalace wake-up fetch failed: %s",
                            _e,
                        )
                        return None

                def _log_wakeup(reason: str, length: int):
                    try:
                        from datetime import datetime

                        log_path = Path.home() / ".mempalace" / "hook.log"
                        with open(log_path, "a") as f:
                            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            f.write(
                                f"{ts} | INFO | Bootstrap: wake-up {reason} ({length} chars)\n",  # noqa: E501
                            )
                    except Exception:
                        pass

                # Detect compaction via summary marker
                current_summary = ""
                try:
                    if hasattr(agent.memory, "get_compressed_summary"):
                        current_summary = (
                            agent.memory.get_compressed_summary() or ""
                        )
                except Exception:
                    pass
                summary_marker = (
                    hash(current_summary) if current_summary else 0
                )

                first_time = not getattr(
                    agent,
                    "_mempalace_wakeup_patched",
                    False,
                )
                last_marker = getattr(agent, "_mempalace_summary_marker", None)
                summary_changed = (
                    not first_time
                    and last_marker is not None
                    and summary_marker != last_marker
                )

                if first_time or summary_changed:
                    wakeup_block = _fetch_wakeup_block()
                    if wakeup_block:
                        # Build a closure that holds the LATEST wakeup block.
                        # We mutate a list cell so future refreshes can update it  # noqa: E501
                        # without re-monkey-patching.
                        if first_time:
                            wakeup_cell = [wakeup_block]
                            agent._mempalace_wakeup_cell = wakeup_cell
                            original_build = agent._build_sys_prompt

                            def _patched_build_sys_prompt(
                                _orig=original_build,
                                _cell=wakeup_cell,
                            ):
                                base = _orig()
                                # Strip any previous wake-up block before appending fresh one  # noqa: E501
                                marker = "\n\n## MemPalace Wake-up Context\n"
                                if marker in base:
                                    base = base.split(marker, 1)[0]
                                return base + _cell[0]

                            agent._build_sys_prompt = _patched_build_sys_prompt
                            agent._mempalace_wakeup_patched = True
                        else:
                            # Already patched — just refresh the cell
                            agent._mempalace_wakeup_cell[0] = wakeup_block

                        agent.rebuild_sys_prompt()
                        agent._mempalace_summary_marker = summary_marker

                        reason = (
                            "injected into sys_prompt"
                            if first_time
                            else "refreshed after compaction"
                        )
                        logger.info(
                            "MemPalace wake-up %s (%d chars)",
                            reason,
                            len(wakeup_block),
                        )
                        _log_wakeup(reason, len(wakeup_block))
                    elif first_time:
                        logger.warning(
                            "MemPalace wake-up: fetch returned nothing",
                        )
                else:
                    # Steady state — keep tracking the marker
                    if first_time is False and last_marker is None:
                        agent._mempalace_summary_marker = summary_marker
            except _MemPalaceDisabled:
                pass
            except Exception as e:
                logger.warning("MemPalace wake-up skipped: %s", e)

            # Check WAL for crash recovery (runs every session, not
            # just first).  Critically, scope by the agent's current
            # request session_id so a crashed tool_start from another
            # channel/chat can't bleed into this session's recovery
            # prompt — see the cross-channel bug mode called out in
            # ``tool_wal.py``'s module docstring.
            try:
                from .tool_wal import SessionWAL

                _ctx = getattr(agent, "_request_context", None) or {}
                _sid = _ctx.get("session_id") or None
                crash_report = SessionWAL.get_crash_report(
                    self.working_dir,
                    session_id=_sid,
                )
                if crash_report:
                    logger.warning(f"WAL crash detected: {crash_report[:200]}")
                    messages = await agent.memory.get_memory()
                    for msg in messages:
                        if getattr(msg, "role", None) == "user":
                            prepend_to_message_content(msg, crash_report)
                            break
            except Exception as e:
                logger.debug(f"WAL crash check skipped: {e}")

            # Check if bootstrap has already been triggered before
            if bootstrap_completed_flag.exists():
                return None

            if not bootstrap_path.exists():
                return None

            messages = await agent.memory.get_memory()
            if not is_first_user_interaction(messages):
                return None

            bootstrap_guidance = build_bootstrap_guidance(
                self.language,
            )

            logger.debug(
                "Found BOOTSTRAP.md [%s], prepending guidance",
                self.language,
            )

            system_prompt_count = sum(
                1 for msg in messages if msg.role == "system"
            )
            for msg in messages[system_prompt_count:]:
                if msg.role == "user":
                    prepend_to_message_content(msg, bootstrap_guidance)
                    break

            logger.debug("Bootstrap guidance prepended to first user message")

            # Create completion flag to prevent repeated triggering
            bootstrap_completed_flag.touch()
            logger.debug("Created bootstrap completion flag")

        except Exception as e:
            logger.error(
                "Failed to process bootstrap: %s",
                e,
                exc_info=True,
            )

        return None
