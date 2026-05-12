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


async def create_channel_service(ws: "Workspace", existing_cm):
    """Create channel manager if configured, or reuse existing one.

    Args:
        ws: Workspace instance
        existing_cm: Existing ChannelManager if reused, None if creating new

    Returns:
        ChannelManager instance or None if not configured
    """
    # pylint: disable=protected-access
    if not ws._config.channels:
        return None

    from ...config import Config, update_last_dispatch
    from ..channels.manager import ChannelManager
    from ..channels.utils import make_process_from_runner

    runner = ws._service_manager.services["runner"]

    if existing_cm is not None:
        # Reused from previous workspace — channels will be updated by
        # reload_channel_service() after start_all() completes.
        cm = existing_cm
        ws._service_manager.services["channel_manager"] = cm
    else:
        # Fresh start — create new ChannelManager
        temp_config = Config(channels=ws._config.channels)

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

    # Always inject workspace into ChannelManager, all channels, and runner
    cm.set_workspace(ws)
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


async def reload_channel_service(
    ws,
    cm,
) -> None:
    # pylint: disable=protected-access,redefined-outer-name,reimported
    """Update reused channel_manager to point to the new runner AND
    propagate the new agent config down to each reused channel.

    When channel_manager is reused during hot-reload, the channels still
    reference the old runner (now stopped) AND hold the config snapshot
    they were constructed with. This function:

    1. Swaps the process callback on all channels to the new runner.
    2. Propagates the new per-channel config via ``update_config`` (in-place)
       falling back to ``clone`` + ``replace_channel`` (full restart) for
       fields that cannot be patched without re-init. Without step 2, saving
       a channel config in the Console (which routes through
       ``multi_agent_manager.reload_agent`` → here) would silently drop the
       change — the file on disk gets the new values but the running channel
       keeps its old ones (observed 2026-04-17 on Signal: clearing
       ``allow_from`` didn't take effect until next process restart).
    3. Removes channels that have been disabled (``enabled=false``) since
       the manager was first built, and adds channels that have become
       enabled. Without these two steps, hot-reload could only mutate the
       set of channels that already existed at process start — adding a
       new agent or enabling a previously-off channel required a full
       process restart (observed 2026-05-07: a hot-added ``trader``
       agent's telegram bot never started polling because its
       ``ChannelManager`` was inherited from another agent and had no
       telegram entry).
    """
    from ..channels.manager import ChannelManager
    from ..channels.registry import get_channel_registry
    from ..channels.utils import make_process_from_runner
    from ...config import get_available_channels, update_last_dispatch

    _logger = logger  # reuse module-level logger

    runner = ws._service_manager.services.get("runner")
    if not runner:
        _logger.warning("channel_manager reload: no runner found, skipping")
        return

    new_process = make_process_from_runner(runner)
    new_channels_config = getattr(ws._config, "channels", None)

    def _get_ch_cfg(name: str):
        if new_channels_config is None:
            return None
        cfg = getattr(new_channels_config, name, None)
        if cfg is None:
            extra = (
                getattr(new_channels_config, "__pydantic_extra__", None) or {}
            )
            cfg = extra.get(name)
        return cfg

    def _is_enabled(cfg) -> bool:
        if cfg is None:
            return False
        if isinstance(cfg, dict):
            return bool(cfg.get("enabled", False))
        return bool(getattr(cfg, "enabled", False))

    # 1+2+3a. Walk existing channels: swap process; remove if disabled;
    # otherwise propagate the new config (in-place or via clone+replace).
    # Snapshot list — `replace_channel`/`remove_channel` mutate
    # `cm.channels` mid-iteration.
    snapshot = list(cm.channels)
    for ch in snapshot:
        old_id = id(getattr(ch, "_process", None))
        ch._process = new_process
        _logger.debug(
            "channel_manager reload: %s _process %s -> %s",
            ch.channel,
            old_id,
            id(new_process),
        )
        new_ch_cfg = _get_ch_cfg(ch.channel)
        if not _is_enabled(new_ch_cfg):
            try:
                removed = await cm.remove_channel(ch.channel)
            except Exception:
                _logger.exception(
                    "channel_manager reload: failed to remove channel %s",
                    ch.channel,
                )
                continue
            if removed:
                _logger.info(
                    "channel_manager reload: %s removed (no longer enabled)",
                    ch.channel,
                )
            continue
        try:
            applied_in_place = await ch.update_config(new_ch_cfg)
        except Exception:
            _logger.exception(
                "channel_manager reload: update_config raised for %s",
                ch.channel,
            )
            continue
        if applied_in_place:
            _logger.info(
                "channel_manager reload: %s config updated in-place",
                ch.channel,
            )
            continue
        # update_config returned False — requires clone + replace
        try:
            new_ch = ch.clone(new_ch_cfg)
            await cm.replace_channel(new_ch)
            _logger.info(
                "channel_manager reload: %s replaced (full restart) to "
                "pick up config change",
                ch.channel,
            )
        except Exception:
            _logger.exception(
                "channel_manager reload: failed to replace channel %s; "
                "running channel may have stale config",
                ch.channel,
            )

    # 3b. Build any newly-enabled channels not previously in cm.channels
    # (e.g. a hot-added agent enabling its first telegram bot).
    current_names = {ch.channel for ch in cm.channels}
    available = get_available_channels()
    show_tool_details = getattr(ws._config, "show_tool_details", True)

    def _on_last_dispatch(channel, user_id, session_id):
        update_last_dispatch(
            channel=channel,
            user_id=user_id,
            session_id=session_id,
            agent_id=ws.agent_id,
        )

    for key, ch_cls in get_channel_registry().items():
        if key in current_names or key not in available:
            continue
        ch_cfg = _get_ch_cfg(key)
        if not _is_enabled(ch_cfg):
            continue
        new_ch = ChannelManager.build_one_channel(
            key=key,
            ch_cls=ch_cls,
            ch_cfg=ch_cfg,
            process=new_process,
            on_last_dispatch=_on_last_dispatch,
            show_tool_details=show_tool_details,
            workspace_dir=ws.workspace_dir,
        )
        if new_ch is None:
            continue
        try:
            # replace_channel's "no existing" branch appends + starts the
            # new channel, so this also handles the cold-add case.
            await cm.replace_channel(new_ch)
            _logger.info(
                "channel_manager reload: %s built and started (hot-add)",
                key,
            )
        except Exception:
            _logger.exception(
                "channel_manager reload: failed to start newly enabled "
                "channel %s",
                key,
            )

    # set_workspace also injects `_workspace` into channels appended above
    # via replace_channel, so it must run after the hot-add loop.
    cm.set_workspace(ws)
    _logger.info(
        "channel_manager reload: now managing %d channels (id=%s)",
        len(cm.channels),
        id(runner),
    )
