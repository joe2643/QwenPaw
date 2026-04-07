# -*- coding: utf-8 -*-
"""Service factory functions for workspace components.

Factory functions are used by Workspace._register_services() to create
and initialize service components. Extracted from local functions to
improve testability and code organization.
"""

from typing import TYPE_CHECKING
import logging

if TYPE_CHECKING:
    from .workspace import Workspace

logger = logging.getLogger(__name__)


async def create_mcp_service(ws: "Workspace", mcp):
    """Initialize MCP manager and attach to runner.

    Args:
        ws: Workspace instance
        mcp: MCPClientManager instance
    """
    # pylint: disable=protected-access
    if ws._config.mcp:
        try:
            await mcp.init_from_config(ws._config.mcp)
            logger.debug(f"MCP initialized for agent: {ws.agent_id}")
        except Exception as e:
            logger.warning(f"Failed to init MCP: {e}")
    ws._service_manager.services["runner"].set_mcp_manager(mcp)
    # pylint: enable=protected-access


async def create_chat_service(ws: "Workspace", service):
    """Create and attach chat manager, or reuse existing one.

    Args:
        ws: Workspace instance
        service: Existing ChatManager if reused, None if creating new
    """
    # pylint: disable=protected-access
    from ..runner.manager import ChatManager
    from ..runner.repo.json_repo import JsonChatRepository

    if service is not None:
        # Reused ChatManager - just wire to new runner
        cm = service
        logger.info(f"Reusing ChatManager for {ws.agent_id}")
    else:
        # Create new ChatManager
        chats_path = str(ws.workspace_dir / "chats.json")
        chat_repo = JsonChatRepository(chats_path)
        cm = ChatManager(repo=chat_repo)
        ws._service_manager.services["chat_manager"] = cm
        logger.info(f"ChatManager created: {chats_path}")

    # Always wire to new runner
    ws._service_manager.services["runner"].set_chat_manager(cm)
    # pylint: enable=protected-access


async def create_channel_service(ws: "Workspace", _):
    """Create channel manager if configured.

    Args:
        ws: Workspace instance
        _: Unused service parameter

    Returns:
        ChannelManager instance or None if not configured
    """
    # pylint: disable=protected-access
    if not ws._config.channels:
        return None

    from ...config import Config, update_last_dispatch
    from ..channels.manager import ChannelManager
    from ..channels.utils import make_process_from_runner

    temp_config = Config(channels=ws._config.channels)
    runner = ws._service_manager.services["runner"]

    def on_last_dispatch(channel, user_id, session_id):
        update_last_dispatch(
            channel=channel,
            user_id=user_id,
            session_id=session_id,
            agent_id=ws.agent_id,
        )

    cm = ChannelManager.from_config(
        process=make_process_from_runner(runner),
        config=temp_config,
        on_last_dispatch=on_last_dispatch,
        workspace_dir=ws.workspace_dir,
    )
    ws._service_manager.services["channel_manager"] = cm

    # Inject workspace into ChannelManager and all channels
    cm.set_workspace(ws)

    # Inject workspace into runner for control command handlers
    runner.set_workspace(ws)

    return cm
    # pylint: enable=protected-access


async def create_agent_config_watcher(ws: "Workspace", _):
    """Create agent config watcher if channel/cron exists.

    Args:
        ws: Workspace instance
        _: Unused service parameter

    Returns:
        AgentConfigWatcher instance or None if not needed
    """
    # pylint: disable=protected-access
    channel_mgr = ws._service_manager.services.get("channel_manager")
    cron_mgr = ws._service_manager.services.get("cron_manager")

    if not (channel_mgr or cron_mgr):
        return None

    from ..agent_config_watcher import AgentConfigWatcher

    watcher = AgentConfigWatcher(
        agent_id=ws.agent_id,
        workspace_dir=ws.workspace_dir,
        channel_manager=channel_mgr,
        cron_manager=cron_mgr,
    )
    ws._service_manager.services["agent_config_watcher"] = watcher
    return watcher
    # pylint: enable=protected-access


async def create_mcp_config_watcher(ws: "Workspace", _):
    """Create MCP config watcher if MCP manager exists.

    Args:
        ws: Workspace instance
        _: Unused service parameter

    Returns:
        MCPConfigWatcher instance or None if not needed
    """
    # pylint: disable=protected-access
    mcp_mgr = ws._service_manager.services.get("mcp_manager")
    if not mcp_mgr:
        return None

    from ..mcp.watcher import MCPConfigWatcher
    from ...config.config import load_agent_config

    def mcp_config_loader():
        agent_config = load_agent_config(ws.agent_id)
        return agent_config.mcp

    watcher = MCPConfigWatcher(
        mcp_manager=mcp_mgr,
        config_loader=mcp_config_loader,
        config_path=ws.workspace_dir / "agent.json",
    )
    ws._service_manager.services["mcp_config_watcher"] = watcher
    return watcher
    # pylint: enable=protected-access

async def reload_channel_service(ws, cm) -> None:
    """Update reused channel_manager to point to the new runner.

    When channel_manager is reused during hot-reload, the channels
    still reference the old runner (now stopped). This swaps the
    process callback on all channels to the new runner.
    """
    from ..channels.utils import make_process_from_runner
    import logging
    _logger = logging.getLogger(__name__)

    runner = ws._service_manager.services.get("runner")
    if not runner:
        _logger.warning("channel_manager reload: no runner found, skipping")
        return

    # Verify runner is healthy before swapping
    health = getattr(runner, "_health", None)
    _logger.info(
        "channel_manager reload: runner id=%s health=%s",
        id(runner), health,
    )
    if not health:
        _logger.error("channel_manager reload: new runner is NOT healthy, skipping swap")
        return

    new_process = make_process_from_runner(runner)
    for ch in cm.channels:
        old_id = id(getattr(ch, "_process", None))
        ch._process = new_process
        _logger.debug("channel_manager reload: %s _process %s -> %s", ch.channel, old_id, id(new_process))
    cm.set_workspace(ws)
    _logger.info(
        "channel_manager reload: updated %d channels to new runner (id=%s)",
        len(cm.channels), id(runner),
    )

async def create_media_server(ws, _):
    """Create embedded media server if enabled in config."""
    config = ws._config
    running = getattr(config, "running", None) or getattr(config, "agents", None)
    if running is None:
        return None
    ms_cfg = getattr(running, "media_server", None)
    if ms_cfg is None or not getattr(ms_cfg, "enabled", False):
        return None

    from urllib.parse import urlparse
    from ..media_server import MediaServer

    # Parse host/port from server_url config instead of hardcoding
    server_url = getattr(ms_cfg, "server_url", "") or ""
    parsed = urlparse(server_url)
    port = parsed.port or 8089
    host = parsed.hostname or "127.0.0.1"

    return MediaServer.get_or_create(
        agent_id=ws.agent_id,
        host=host,
        port=port,
        secret=ms_cfg.media_secret,
        allowed_dirs=list(ms_cfg.allowed_dirs),
        max_size_mb=ms_cfg.max_size_mb,
        tunnel_domain=ms_cfg.tunnel_domain,
    )
