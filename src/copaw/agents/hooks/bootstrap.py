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

            # MemPalace wake-up context — inject L0+L1 on first user message
            # We prepend to the first user message instead of setting sys_prompt
            # because sys_prompt is a read-only @property in agentscope ReActAgent.
            try:
                import subprocess
                result = subprocess.run(
                    ["python3", "-m", "mempalace", "wake-up"],
                    capture_output=True, text=True, timeout=15,
                )
                if result.returncode == 0 and result.stdout.strip():
                    lines = result.stdout.strip().split("\n")
                    wakeup = "\n".join(
                        l for l in lines
                        if not l.startswith("Wake-up text") and not l.startswith("===")
                    ).strip()
                    if wakeup and len(wakeup) > 50:
                        # Inject into the first user message in memory (like WAL crash report)
                        messages = await agent.memory.get_memory()
                        for msg in messages:
                            if getattr(msg, "role", None) == "user":
                                from ..utils.message_processing import prepend_to_message_content
                                wakeup_block = "## MemPalace Wake-up Context\n" + wakeup
                                prepend_to_message_content(msg, wakeup_block)
                                logger.info("MemPalace wake-up injected (%d chars)", len(wakeup))
                                # Log to mempalace hook.log too for visibility
                                try:
                                    from datetime import datetime
                                    log_path = Path.home() / ".mempalace" / "hook.log"
                                    with open(log_path, "a") as f:
                                        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                        f.write(f"{ts} | INFO | Bootstrap: wake-up injected ({len(wakeup)} chars)\n")
                                except Exception:
                                    pass
                                break
                        else:
                            logger.debug("MemPalace wake-up: no user message yet to prepend to")
                else:
                    logger.warning("MemPalace wake-up returned non-zero: %s", result.stderr[:200])
            except subprocess.TimeoutExpired:
                logger.warning("MemPalace wake-up timed out (>15s)")
            except Exception as e:
                logger.warning("MemPalace wake-up skipped: %s", e)

            # Check WAL for crash recovery (runs every session, not just first)
            try:
                from .tool_wal import SessionWAL
                crash_report = SessionWAL.get_crash_report(self.working_dir)
                if crash_report:
                    logger.warning(f"WAL crash detected: {crash_report[:200]}")
                    messages = await agent.memory.get_memory()
                    for msg in messages:
                        if getattr(msg, 'role', None) == 'user':
                            from ..prompt import prepend_to_message_content
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
