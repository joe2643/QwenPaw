# -*- coding: utf-8 -*-
"""Utility functions for app routers."""

import asyncio
import logging
import traceback
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fastapi import Request
    from .multi_agent_manager import MultiAgentManager

logger = logging.getLogger(__name__)


def schedule_agent_reload(request: "Request", agent_id: str) -> None:
    """Schedule an agent reload in background (non-blocking).

    This is a common pattern used across multiple endpoints to reload
    agent configuration after making changes. The reload happens
    asynchronously without blocking the API response.

    IMPORTANT: This function extracts manager and agent_id from the
    request context before creating the background task, to avoid
    accessing request/workspace objects after their lifecycle ends.

    Args:
        request: FastAPI request object (must have multi_agent_manager)
        agent_id: Agent ID to reload

    Example:
        >>> from qwenpaw.app.utils import schedule_agent_reload
        >>> save_agent_config(workspace.agent_id, agent_config)
        >>> schedule_agent_reload(request, workspace.agent_id)
    """
    # Caller diagnostic: reload storms are hard to trace because the trigger
    # runs in the background — log the caller frame at schedule time so the
    # source is visible even if the actual reload logs later.
    if logger.isEnabledFor(logging.INFO):
        caller_frame = traceback.extract_stack()[-2]
        endpoint = ""
        try:
            url = getattr(request, "url", None)
            method = getattr(request, "method", None)
            if url is not None and method is not None:
                endpoint = f"{method} {url.path}"
        except Exception:  # pragma: no cover - defensive
            pass
        logger.info(
            "schedule_agent_reload: agent=%s endpoint=%r caller=%s:%d in %s",
            agent_id,
            endpoint,
            Path(caller_frame.filename).name,
            caller_frame.lineno,
            caller_frame.name,
        )

    # Extract manager before creating background task (defensive)
    manager: "MultiAgentManager" = getattr(
        request.app.state,
        "multi_agent_manager",
        None,
    )

    if manager is None:
        logger.warning(
            f"Cannot schedule agent reload for '{agent_id}': "
            "MultiAgentManager not initialized in app state",
        )
        return

    async def reload_in_background():
        try:
            await manager.reload_agent(agent_id)
        except Exception as e:
            logger.warning(
                f"Background reload failed for agent '{agent_id}': {e}",
                exc_info=True,
            )

    asyncio.create_task(reload_in_background())
