# -*- coding: utf-8 -*-

import logging
from datetime import datetime, timezone
from pathlib import Path as _P
from typing import Any, List, Optional

from fastapi import APIRouter, Body, Depends, HTTPException, Path, Request
from pydantic import BaseModel, Field

from ..utils import schedule_agent_reload
from ...config import (
    load_config,
    save_config,
    ChannelConfig,
    ChannelConfigUnion,
    get_available_channels,
    ToolGuardConfig,
    ToolGuardRuleConfig,
)
from ..channels.registry import BUILTIN_CHANNEL_KEYS
from ...config.timezone import normalize_tz
from ...config.config import (
    MediaServerConfig,
    AgentsLLMRoutingConfig,
    ConsoleConfig,
    DingTalkConfig,
    DiscordConfig,
    FeishuConfig,
    HeartbeatConfig,
    IMessageChannelConfig,
    MatrixConfig,
    MattermostConfig,
    MQTTConfig,
    QQConfig,
    SignalConfig,
    SIPChannelConfig,
    SkillScannerConfig,
    SkillScannerWhitelistEntry,
    TelegramConfig,
    VoiceChannelConfig,
    WecomConfig,
    WhatsAppConfig,
    WeixinConfig,
    XiaoYiConfig,
)
from ...agents.acp.core import ACPConfig, ACPAgentConfig

logger = logging.getLogger(__name__)

from .schemas_config import (
    ChannelHealthResponse,
    ChannelRestartResponse,
    HeartbeatBody,
)
from ..channels.qrcode_auth_handler import (
    QRCODE_AUTH_HANDLERS,
    generate_qrcode_image,
)

router = APIRouter(prefix="/config", tags=["config"])


_CHANNEL_CONFIG_CLASS_MAP = {
    "telegram": TelegramConfig,
    "dingtalk": DingTalkConfig,
    "discord": DiscordConfig,
    "feishu": FeishuConfig,
    "qq": QQConfig,
    "imessage": IMessageChannelConfig,
    "console": ConsoleConfig,
    "voice": VoiceChannelConfig,
    "sip": SIPChannelConfig,
    "mattermost": MattermostConfig,
    "mqtt": MQTTConfig,
    "matrix": MatrixConfig,
    "wecom": WecomConfig,
    "whatsapp": WhatsAppConfig,
    "weixin": WeixinConfig,
    "xiaoyi": XiaoYiConfig,
    "signal": SignalConfig,
}
_ALLOWED_ACP_TOOL_PARSE_MODES = {
    "call_title",
    "update_detail",
    "call_detail",
}


@router.get(
    "/channels",
    summary="List all channels",
    description="Retrieve configuration for all available channels",
)
async def list_channels(request: Request) -> dict:
    """List all channel configs (filtered by available channels)."""
    from ..agent_context import get_agent_for_request

    agent = await get_agent_for_request(request)
    agent_config = agent.config
    available = get_available_channels()

    # Get channel configs from agent's config (with fallback to empty)
    channels_config = agent_config.channels
    if channels_config is None:
        # No channels config yet, use empty defaults
        all_configs = {}
    else:
        all_configs = channels_config.model_dump()
        extra = getattr(channels_config, "__pydantic_extra__", None) or {}
        all_configs.update(extra)

    # Return all available channels (use default config if not saved)
    result = {}
    for key in available:
        if key in all_configs:
            channel_data = (
                dict(all_configs[key])
                if isinstance(all_configs[key], dict)
                else all_configs[key]
            )
        else:
            # Channel registered but no config saved yet, use empty default
            channel_data = {"enabled": False, "bot_prefix": ""}
        if isinstance(channel_data, dict):
            channel_data["isBuiltin"] = key in BUILTIN_CHANNEL_KEYS
        result[key] = channel_data

    return result


@router.get(
    "/channels/types",
    summary="List channel types",
    description="Return all available channel type identifiers",
)
async def list_channel_types() -> List[str]:
    """Return available channel type identifiers (env-filtered)."""
    return list(get_available_channels())


@router.put(
    "/channels",
    response_model=ChannelConfig,
    summary="Update all channels",
    description="Update configuration for all channels at once",
)
async def put_channels(
    request: Request,
    channels_config: ChannelConfig = Body(
        ...,
        description="Complete channel configuration",
    ),
) -> ChannelConfig:
    """Update all channel configs."""
    from ..agent_context import get_agent_for_request
    from ...config.config import save_agent_config

    agent = await get_agent_for_request(request)
    agent.config.channels = channels_config
    save_agent_config(agent.agent_id, agent.config)

    # Hot reload config (async, non-blocking)
    schedule_agent_reload(request, agent.agent_id)

    return channels_config


# ── Channel health check & restart ─────────────────────────────────────────


async def _resolve_channel_manager(
    request: Request,
    channel_name: str = Path(
        ...,
        description="Name of the channel",
        min_length=1,
    ),
):
    """Shared dependency: validate channel name and return channel_manager."""
    from ..agent_context import get_agent_for_request

    available = get_available_channels()
    if channel_name not in available:
        raise HTTPException(
            status_code=404,
            detail=f"Channel '{channel_name}' not available",
        )

    agent = await get_agent_for_request(request)
    channel_manager = agent.channel_manager
    if channel_manager is None:
        raise HTTPException(
            status_code=503,
            detail="Channel manager not initialized",
        )
    return channel_manager


@router.get(
    "/channels/{channel_name}/health",
    response_model=ChannelHealthResponse,
    summary="Health check for a channel",
    description="Return the runtime health status of a specific channel",
)
async def get_channel_health(
    channel_name: str = Path(
        ...,
        description="Name of the channel to check",
        min_length=1,
    ),
    channel_manager=Depends(_resolve_channel_manager),
) -> ChannelHealthResponse:
    """Return health status for a specific channel."""
    try:
        return await channel_manager.get_channel_health(
            channel_name,
        )
    except KeyError as exc:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Channel '{channel_name}' is not running."
                " It may be disabled or not configured."
            ),
        ) from exc


@router.post(
    "/channels/{channel_name}/restart",
    response_model=ChannelRestartResponse,
    summary="Restart a channel",
    description=(
        "Stop and re-start a specific channel" " without restarting the agent"
    ),
)
async def restart_channel(
    channel_name: str = Path(
        ...,
        description="Name of the channel to restart",
        min_length=1,
    ),
    channel_manager=Depends(_resolve_channel_manager),
) -> ChannelRestartResponse:
    """Restart a specific channel."""
    try:
        return await channel_manager.restart_channel(
            channel_name,
        )
    except KeyError as exc:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Channel '{channel_name}' is not running."
                " It may be disabled or not configured."
            ),
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=(f"Failed to restart channel" f" '{channel_name}': {exc}"),
        ) from exc


# ── Unified QR code endpoints for all channels ─────────────────────────────


@router.get(
    "/channels/{channel}/qrcode",
    summary="Get channel authorization QR code",
    description=(
        "Fetch a QR code image (base64 PNG) for the given channel. "
        "Supported channels: " + ", ".join(QRCODE_AUTH_HANDLERS.keys())
    ),
)
async def get_channel_qrcode(request: Request, channel: str) -> dict:
    """Return {qrcode_img, poll_token} for the requested channel."""
    handler = QRCODE_AUTH_HANDLERS.get(channel)
    if handler is None:
        raise HTTPException(
            status_code=404,
            detail=f"QR code not supported for channel: {channel}",
        )

    result = await handler.fetch_qrcode(request)
    qrcode_img = generate_qrcode_image(result.scan_url)
    return {"qrcode_img": qrcode_img, "poll_token": result.poll_token}


@router.get(
    "/channels/{channel}/qrcode/status",
    summary="Poll channel QR code authorization status",
)
async def get_channel_qrcode_status(
    request: Request,
    channel: str,
    token: str,
) -> dict:
    """Return {status, credentials} for the requested channel."""
    handler = QRCODE_AUTH_HANDLERS.get(channel)
    if handler is None:
        raise HTTPException(
            status_code=404,
            detail=f"QR code not supported for channel: {channel}",
        )

    result = await handler.poll_status(token, request)
    return {"status": result.status, "credentials": result.credentials}


@router.get(
    "/channels/{channel_name}",
    response_model=ChannelConfigUnion,
    summary="Get channel config",
    description="Retrieve configuration for a specific channel by name",
)
async def get_channel(
    request: Request,
    channel_name: str = Path(
        ...,
        description="Name of the channel to retrieve",
        min_length=1,
    ),
) -> ChannelConfigUnion:
    """Get a specific channel config by name."""
    from ..agent_context import get_agent_for_request

    available = get_available_channels()
    if channel_name not in available:
        raise HTTPException(
            status_code=404,
            detail=f"Channel '{channel_name}' not found",
        )

    agent = await get_agent_for_request(request)
    channels = agent.config.channels
    if channels is None:
        raise HTTPException(
            status_code=404,
            detail=f"Channel '{channel_name}' not configured",
        )

    single_channel_config = getattr(channels, channel_name, None)
    if single_channel_config is None:
        extra = getattr(channels, "__pydantic_extra__", None) or {}
        single_channel_config = extra.get(channel_name)
    if single_channel_config is None:
        raise HTTPException(
            status_code=404,
            detail=f"Channel '{channel_name}' not found",
        )
    return single_channel_config


@router.put(
    "/channels/{channel_name}",
    response_model=ChannelConfigUnion,
    summary="Update channel config",
    description="Update configuration for a specific channel by name",
)
async def put_channel(
    request: Request,
    channel_name: str = Path(
        ...,
        description="Name of the channel to update",
        min_length=1,
    ),
    single_channel_config: dict = Body(
        ...,
        description="Updated channel configuration",
    ),
) -> ChannelConfigUnion:
    """Update a specific channel config by name."""
    from ..agent_context import get_agent_for_request
    from ...config.config import save_agent_config

    available = get_available_channels()
    if channel_name not in available:
        raise HTTPException(
            status_code=404,
            detail=f"Channel '{channel_name}' not found",
        )

    agent = await get_agent_for_request(request)

    # Initialize channels if not exists
    if agent.config.channels is None:
        agent.config.channels = ChannelConfig()

    config_class = _CHANNEL_CONFIG_CLASS_MAP.get(channel_name)
    if config_class is not None:
        channel_config = config_class(**single_channel_config)
    else:
        # For custom channels, just use the dict
        channel_config = single_channel_config

    # Set channel config in agent's config
    setattr(agent.config.channels, channel_name, channel_config)
    save_agent_config(agent.agent_id, agent.config)

    # WhatsApp: if the user just re-linked via the Console QR / pair flow,
    # a temporary pairing client (see /channels/whatsapp/qrcode and
    # /channels/whatsapp/pair) is still connected to the freshly-paired
    # device.  WhatsApp permits only one socket per device and neonize holds
    # an exclusive SQLite lock on neonize.db, so the channel restart the
    # reload below triggers would collide with it (one connection gets
    # kicked with a stream-end / inbound messages silently dropped).  Tear
    # the pairing client down first so the restarted channel owns the single
    # connection cleanly and Save actually brings WhatsApp back. No-op when
    # no pairing is in progress.
    if channel_name == "whatsapp":
        await _teardown_whatsapp_pair_client(agent.agent_id)

    # Hot reload config (async, non-blocking)
    schedule_agent_reload(request, agent.agent_id)

    return channel_config


@router.get(
    "/acp",
    response_model=ACPConfig,
    summary="Get ACP config",
    description="Retrieve ACP configuration for current agent",
)
async def get_acp_config(request: Request) -> ACPConfig:
    """Return ACP config for the current agent."""
    from ..agent_context import get_agent_for_request

    agent = await get_agent_for_request(request)
    return agent.config.acp or ACPConfig()


@router.put(
    "/acp",
    response_model=ACPConfig,
    summary="Update ACP config",
    description="Update ACP configuration for current agent",
)
async def put_acp_config(
    request: Request,
    acp_config: ACPConfig = Body(
        ...,
        description="Complete ACP configuration",
    ),
) -> ACPConfig:
    """Update ACP config for the current agent."""
    from ..agent_context import get_agent_for_request
    from ...config.config import save_agent_config

    agent = await get_agent_for_request(request)
    agent.config.acp = acp_config
    save_agent_config(agent.agent_id, agent.config)
    schedule_agent_reload(request, agent.agent_id)
    return agent.config.acp


@router.get(
    "/acp/{agent_name}",
    response_model=ACPAgentConfig,
    summary="Get ACP agent config",
    description="Retrieve ACP configuration for a specific ACP agent",
)
async def get_acp_agent_config(
    request: Request,
    agent_name: str = Path(
        ...,
        description="Name of the ACP agent to retrieve",
        min_length=1,
    ),
) -> ACPAgentConfig:
    """Return config for one ACP agent."""
    from ..agent_context import get_agent_for_request

    agent = await get_agent_for_request(request)
    acp_config = agent.config.acp or ACPConfig()
    acp_agent = acp_config.agents.get(agent_name)
    if acp_agent is None:
        raise HTTPException(
            status_code=404,
            detail=f"ACP agent '{agent_name}' not found",
        )
    return acp_agent


@router.put(
    "/acp/{agent_name}",
    response_model=ACPAgentConfig,
    summary="Update ACP agent config",
    description="Update ACP configuration for a specific ACP agent",
)
async def put_acp_agent_config(
    request: Request,
    agent_name: str = Path(
        ...,
        description="Name of the ACP agent to update",
        min_length=1,
    ),
    acp_agent_config: ACPAgentConfig = Body(
        ...,
        description="Updated ACP agent configuration",
    ),
) -> ACPAgentConfig:
    """Update config for one ACP agent."""
    from ..agent_context import get_agent_for_request
    from ...config.config import save_agent_config

    if acp_agent_config.tool_parse_mode not in _ALLOWED_ACP_TOOL_PARSE_MODES:
        raise HTTPException(
            status_code=400,
            detail=(
                "Invalid tool_parse_mode. Allowed values: "
                + ", ".join(sorted(_ALLOWED_ACP_TOOL_PARSE_MODES))
            ),
        )

    agent = await get_agent_for_request(request)
    if agent.config.acp is None:
        agent.config.acp = ACPConfig()

    agent_name = agent_name.strip()
    if not agent_name:
        raise HTTPException(
            status_code=400,
            detail="ACP agent name cannot be empty",
        )

    agent.config.acp.agents[agent_name] = acp_agent_config
    save_agent_config(agent.agent_id, agent.config)
    schedule_agent_reload(request, agent.agent_id)
    return agent.config.acp.agents[agent_name]


@router.get(
    "/heartbeat",
    summary="Get heartbeat config",
    description="Return current heartbeat config (interval, target, etc.)",
)
async def get_heartbeat(request: Request) -> Any:
    """Return effective heartbeat config (from file or default)."""
    from ..agent_context import get_agent_for_request
    from ...config.config import HeartbeatConfig as HeartbeatConfigModel

    agent = await get_agent_for_request(request)
    hb = agent.config.heartbeat
    if hb is None:
        # Use default if not configured
        hb = HeartbeatConfigModel()
    return hb.model_dump(mode="json", by_alias=True)


@router.put(
    "/heartbeat",
    summary="Update heartbeat config",
    description="Update heartbeat and hot-reload the scheduler",
)
async def put_heartbeat(
    request: Request,
    body: HeartbeatBody = Body(..., description="Heartbeat configuration"),
) -> Any:
    """Update heartbeat config and reschedule the heartbeat job."""
    from ..agent_context import get_agent_for_request
    from ...config.config import save_agent_config

    agent = await get_agent_for_request(request)
    hb = HeartbeatConfig(
        enabled=body.enabled,
        every=body.every,
        target=body.target,
        active_hours=body.active_hours,
    )
    agent.config.heartbeat = hb
    save_agent_config(agent.agent_id, agent.config)

    # Reschedule heartbeat (async, non-blocking)
    import asyncio

    async def reschedule_in_background():
        try:
            if agent.cron_manager is not None:
                await agent.cron_manager.reschedule_heartbeat()
        except Exception as e:
            import logging

            logging.getLogger(__name__).warning(
                f"Background reschedule failed: {e}",
            )

    asyncio.create_task(reschedule_in_background())

    return hb.model_dump(mode="json", by_alias=True)


@router.post(
    "/heartbeat/run",
    summary="Run heartbeat now",
    description="Trigger one heartbeat execution immediately",
)
async def run_heartbeat_now(request: Request) -> Any:
    """Trigger one heartbeat run in background for quick testing."""
    from ..agent_context import get_agent_for_request
    from ..crons.heartbeat import run_heartbeat_once
    import asyncio
    import logging

    agent = await get_agent_for_request(request)

    async def _run_once_bg() -> None:
        try:
            workspace_dir = getattr(agent.runner, "workspace_dir", None)
            await run_heartbeat_once(
                runner=agent.runner,
                channel_manager=agent.channel_manager,
                agent_id=agent.agent_id,
                workspace_dir=workspace_dir,
            )
        except Exception as e:  # pylint: disable=broad-except
            logging.getLogger(__name__).exception(
                "manual heartbeat run failed: %s",
                e,
            )

    asyncio.create_task(_run_once_bg())
    return {"started": True}


@router.get(
    "/agents/llm-routing",
    response_model=AgentsLLMRoutingConfig,
    summary="Get agent LLM routing settings",
)
async def get_agents_llm_routing() -> AgentsLLMRoutingConfig:
    config = load_config()
    return config.agents.llm_routing


@router.put(
    "/agents/llm-routing",
    response_model=AgentsLLMRoutingConfig,
    summary="Update agent LLM routing settings",
)
async def put_agents_llm_routing(
    body: AgentsLLMRoutingConfig = Body(...),
) -> AgentsLLMRoutingConfig:
    config = load_config()
    config.agents.llm_routing = body
    save_config(config)
    return body


# ── User Timezone ────────────────────────────────────────────────────


@router.get(
    "/user-timezone",
    summary="Get user timezone",
    description="Return the configured user IANA timezone",
)
async def get_user_timezone() -> dict:
    config = load_config()
    return {"timezone": config.user_timezone}


@router.put(
    "/user-timezone",
    summary="Update user timezone",
    description="Set the user IANA timezone",
)
async def put_user_timezone(
    body: dict = Body(..., description="Body with 'timezone' key"),
) -> dict:
    tz = body.get("timezone", "").strip()
    if not tz:
        raise HTTPException(status_code=400, detail="timezone is required")
    resolved = normalize_tz(tz)
    if resolved is None:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid IANA timezone: {tz!r}",
        )
    config = load_config()
    config.user_timezone = resolved
    save_config(config)
    return {"timezone": resolved}


# ── ACPX Provider (claude-acpx) ──────────────────────────────────────


@router.get(
    "/acpx-provider",
    summary="Get acpx provider operational tuning",
    description=(
        "Return the per-turn timeout and terminal wait_for_exit timeout "
        "used by the claude-acpx provider. Live-applied: changes take "
        "effect on the next turn without a service restart."
    ),
)
async def get_acpx_provider_config() -> dict:
    from qwenpaw.config.config import AcpxProviderConfig

    config = load_config()
    return config.acpx_provider.model_dump()


@router.put(
    "/acpx-provider",
    summary="Update acpx provider operational tuning",
)
async def put_acpx_provider_config(
    body: dict = Body(
        ...,
        description=(
            "Body with optional 'turn_timeout_seconds' and "
            "'terminal_wait_seconds' keys. Either or both may be "
            "supplied; missing keys keep their current values."
        ),
    ),
) -> dict:
    from qwenpaw.config.config import AcpxProviderConfig

    config = load_config()
    current = config.acpx_provider.model_dump()
    merged = {**current, **{
        k: v for k, v in body.items()
        if k in ("turn_timeout_seconds", "terminal_wait_seconds")
    }}
    try:
        config.acpx_provider = AcpxProviderConfig(**merged)
    except Exception as e:  # noqa: BLE001 — pydantic ValidationError
        raise HTTPException(
            status_code=400,
            detail=f"Invalid acpx_provider config: {e}",
        ) from e
    save_config(config)
    return config.acpx_provider.model_dump()


# ── Security / Tool Guard ────────────────────────────────────────────


@router.get(
    "/security/tool-guard",
    response_model=ToolGuardConfig,
    summary="Get tool guard settings",
)
async def get_tool_guard() -> ToolGuardConfig:
    config = load_config()
    return config.security.tool_guard


@router.put(
    "/security/tool-guard",
    response_model=ToolGuardConfig,
    summary="Update tool guard settings",
)
async def put_tool_guard(
    body: ToolGuardConfig = Body(...),
) -> ToolGuardConfig:
    config = load_config()
    config.security.tool_guard = body
    save_config(config)

    from ...security.tool_guard.engine import get_guard_engine

    engine = get_guard_engine()
    engine.enabled = body.enabled
    engine.reload_rules()

    return body


@router.get(
    "/security/tool-guard/builtin-rules",
    response_model=List[ToolGuardRuleConfig],
    summary="List built-in guard rules from YAML files",
)
async def get_builtin_rules() -> List[ToolGuardRuleConfig]:
    from ...security.tool_guard.guardians.rule_guardian import (
        load_rules_from_directory,
    )

    rules = load_rules_from_directory()
    return [
        ToolGuardRuleConfig(
            id=r.id,
            tools=r.tools,
            params=r.params,
            category=r.category.value,
            severity=r.severity.value,
            patterns=r.patterns,
            exclude_patterns=r.exclude_patterns,
            description=r.description,
            remediation=r.remediation,
        )
        for r in rules
    ]


# ── Security / File Guard ────────────────────────────────────────────


class FileGuardResponse(BaseModel):
    enabled: bool = True
    paths: List[str] = []


class FileGuardUpdateBody(BaseModel):
    enabled: Optional[bool] = None
    paths: Optional[List[str]] = None


@router.get(
    "/security/file-guard",
    response_model=FileGuardResponse,
    summary="Get file guard settings",
)
async def get_file_guard() -> FileGuardResponse:
    config = load_config()
    fg = config.security.file_guard
    from ...security.tool_guard.guardians.file_guardian import (
        ensure_file_guard_paths,
    )

    paths = ensure_file_guard_paths(fg.sensitive_files or [])
    return FileGuardResponse(enabled=fg.enabled, paths=paths)


@router.put(
    "/security/file-guard",
    response_model=FileGuardResponse,
    summary="Update file guard settings",
)
async def put_file_guard(
    body: FileGuardUpdateBody,
) -> FileGuardResponse:
    config = load_config()
    fg = config.security.file_guard

    if body.enabled is not None:
        fg.enabled = body.enabled
    if body.paths is not None:
        from ...security.tool_guard.guardians.file_guardian import (
            ensure_file_guard_paths,
        )

        fg.sensitive_files = ensure_file_guard_paths(body.paths)

    save_config(config)

    from ...security.tool_guard.engine import get_guard_engine

    engine = get_guard_engine()
    engine.reload_rules()

    return FileGuardResponse(
        enabled=fg.enabled,
        paths=fg.sensitive_files,
    )


# ── Security / Skill Scanner ────────────────────────────────────────


@router.get(
    "/security/skill-scanner",
    response_model=SkillScannerConfig,
    summary="Get skill scanner settings",
)
async def get_skill_scanner() -> SkillScannerConfig:
    config = load_config()
    return config.security.skill_scanner


@router.put(
    "/security/skill-scanner",
    response_model=SkillScannerConfig,
    summary="Update skill scanner settings",
)
async def put_skill_scanner(
    body: SkillScannerConfig = Body(...),
) -> SkillScannerConfig:
    config = load_config()
    config.security.skill_scanner = body
    save_config(config)
    return body


@router.get(
    "/security/skill-scanner/blocked-history",
    summary="Get blocked skills history",
)
async def get_blocked_history() -> list:
    from ...security.skill_scanner import get_blocked_history as _get_history

    records = _get_history()
    return [r.to_dict() for r in records]


@router.delete(
    "/security/skill-scanner/blocked-history",
    summary="Clear all blocked skills history",
)
async def delete_blocked_history() -> dict:
    from ...security.skill_scanner import clear_blocked_history

    clear_blocked_history()
    return {"cleared": True}


@router.delete(
    "/security/skill-scanner/blocked-history/{index}",
    summary="Remove a single blocked history entry",
)
async def delete_blocked_entry(
    index: int = Path(..., ge=0),
) -> dict:
    from ...security.skill_scanner import remove_blocked_entry

    ok = remove_blocked_entry(index)
    if not ok:
        raise HTTPException(status_code=404, detail="Entry not found")
    return {"removed": True}


class WhitelistAddRequest(BaseModel):
    skill_name: str
    content_hash: str = ""


@router.post(
    "/security/skill-scanner/whitelist",
    summary="Add a skill to the whitelist",
)
async def add_to_whitelist(
    body: WhitelistAddRequest = Body(...),
) -> dict:
    skill_name = body.skill_name.strip()
    content_hash = body.content_hash
    if not skill_name:
        raise HTTPException(status_code=400, detail="skill_name is required")

    config = load_config()
    scanner_cfg = config.security.skill_scanner

    for entry in scanner_cfg.whitelist:
        if entry.skill_name == skill_name:
            raise HTTPException(
                status_code=409,
                detail=f"Skill '{skill_name}' is already whitelisted",
            )

    scanner_cfg.whitelist.append(
        SkillScannerWhitelistEntry(
            skill_name=skill_name,
            content_hash=content_hash,
            added_at=datetime.now(timezone.utc).isoformat(),
        ),
    )
    save_config(config)
    return {"whitelisted": True, "skill_name": skill_name}


@router.delete(
    "/security/skill-scanner/whitelist/{skill_name}",
    summary="Remove a skill from the whitelist",
)
async def remove_from_whitelist(
    skill_name: str = Path(..., min_length=1),
) -> dict:
    config = load_config()
    scanner_cfg = config.security.skill_scanner
    original_len = len(scanner_cfg.whitelist)
    scanner_cfg.whitelist = [
        e for e in scanner_cfg.whitelist if e.skill_name != skill_name
    ]
    if len(scanner_cfg.whitelist) == original_len:
        raise HTTPException(
            status_code=404,
            detail=f"Skill '{skill_name}' not found in whitelist",
        )
    save_config(config)
    return {"removed": True, "skill_name": skill_name}


# ── WhatsApp auth (QR / pair code) ────────────────────────────
# Per-agent pairing state keyed by agent_id.
# QwenPaw runs as a single-process server; concurrent pairing for
# different agents is safe because each gets its own state dict.
_whatsapp_pair_states: dict[str, dict] = {}


def _get_wa_pair_state(agent_id: str) -> dict:
    """Get or create per-agent WhatsApp pairing state."""
    if agent_id not in _whatsapp_pair_states:
        _whatsapp_pair_states[agent_id] = {
            "client": None,
            "code": None,
            "status": "idle",
            "qr_data": None,
            "task": None,
        }
    return _whatsapp_pair_states[agent_id]


async def _teardown_whatsapp_pair_client(agent_id: str) -> None:
    """Disconnect + cancel the temporary WhatsApp pairing client.

    The Console QR / pair endpoints (``/channels/whatsapp/qrcode`` and
    ``/channels/whatsapp/pair``) open a short-lived ``NewAClient`` against
    the same ``neonize.db`` the live channel uses, purely to drive the
    QR / pair-code handshake.  Once pairing succeeds that client stays
    connected, holding WhatsApp's single-socket-per-device slot and the
    SQLite lock.  Call this before restarting the channel so the restart
    doesn't fight it.  Best-effort and idempotent: never raises, and is a
    no-op when no pairing client is registered.
    """
    import asyncio

    state = _get_wa_pair_state(agent_id)
    client = state.get("client")
    if client is not None:
        try:
            await client.disconnect()
        except Exception:
            pass
    task = state.get("task")
    if task is not None and not task.done():
        task.cancel()
        try:
            await asyncio.wait_for(task, timeout=2.0)
        except (asyncio.TimeoutError, asyncio.CancelledError, Exception):
            pass
    state.update(
        {
            "client": None,
            "code": None,
            "status": "idle",
            "qr_data": None,
            "task": None,
        },
    )


def _get_wa_auth_dir(agent) -> str:
    """Resolve WhatsApp auth directory from agent config.

    Priority (matches ``WhatsAppChannel._resolve_wa_auth_dir``):
      1. Explicit ``auth_dir`` in the agent's channel config
      2. ``agent.workspace_dir/credentials/whatsapp/default`` (per-agent)
      3. ``WORKING_DIR/credentials/whatsapp/default`` (install-wide fallback)

    Router and channel class must agree on this ordering — otherwise the
    pair/QR endpoints write to a different ``neonize.db`` than the live
    channel reads from.
    """
    from pathlib import Path
    from ...constant import WORKING_DIR

    wa_cfg = getattr(agent.config.channels, "whatsapp", None)
    explicit = (getattr(wa_cfg, "auth_dir", "") if wa_cfg else "") or ""
    if explicit:
        return explicit
    ws = getattr(agent, "workspace_dir", None)
    if ws:
        return str(
            Path(ws).expanduser() / "credentials" / "whatsapp" / "default",
        )
    return str(WORKING_DIR / "credentials" / "whatsapp" / "default")


@router.post(
    "/channels/whatsapp/pair",
    summary="Start WhatsApp pairing",
    description="Start WhatsApp pairing. Returns a pair code to enter on your phone.",
)
async def start_whatsapp_pair(request: Request, phone: str = "") -> dict:
    """Start WhatsApp pair code auth. Requires E.164 phone number."""
    import re

    E164_RE = re.compile(r"^\+[1-9]\d{4,14}$")
    if not phone or not E164_RE.match(phone):
        raise HTTPException(
            status_code=400,
            detail="Phone number required in E.164 format "
            "(^\\+[1-9]\\d{4,14}$, e.g. +85212345678)",
        )
    import asyncio

    try:
        from neonize.aioze.client import NewAClient
        from neonize.events import ConnectedEv
    except ImportError:
        raise HTTPException(
            status_code=500,
            detail="neonize-qwenpaw not installed. "
            "Install: pip install qwenpaw[whatsapp]",
        )

    from ..agent_context import get_agent_for_request

    agent = await get_agent_for_request(request)
    auth_dir = _get_wa_auth_dir(agent)
    state = _get_wa_pair_state(agent.agent_id)

    from pathlib import Path

    db_path = str(Path(auth_dir).expanduser() / "neonize.db")
    Path(auth_dir).expanduser().mkdir(parents=True, exist_ok=True)

    # Disconnect any existing pairing client + cancel its task for this agent
    # (two NewAClients against the same neonize.db fight over the SQLite lock
    # / websocket and one of them drops inbound messages silently).
    old_client = state.get("client")
    if old_client:
        try:
            await old_client.disconnect()
        except Exception:
            pass
    old_task = state.get("task")
    if old_task and not old_task.done():
        old_task.cancel()
        try:
            await asyncio.wait_for(old_task, timeout=2.0)
        except (asyncio.TimeoutError, asyncio.CancelledError, Exception):
            pass

    state["status"] = "pairing"
    state["code"] = None
    state["qr_data"] = None
    state["task"] = None

    client = NewAClient(name=db_path)
    state["client"] = client

    @client.event(ConnectedEv)
    async def on_connected(c, evt):
        state["status"] = "connected"

    @client.qr
    async def on_qr(c, qr_bytes):
        import base64

        try:
            import segno
            import io

            qr = segno.make_qr(qr_bytes)
            buf = io.BytesIO()
            qr.save(buf, kind="png", scale=5, border=2)
            state["qr_data"] = base64.b64encode(buf.getvalue()).decode()
        except Exception:
            state["qr_data"] = base64.b64encode(qr_bytes).decode()

    state["task"] = await client.connect()
    await asyncio.sleep(3)

    try:
        code = await client.PairPhone(phone, True)
        state["code"] = code
        state["status"] = "waiting_pair"
        return {"status": "waiting_pair", "pair_code": code, "phone": phone}
    except Exception as e:
        await asyncio.sleep(2)
        if state["qr_data"]:
            state["status"] = "waiting_qr"
            return {"status": "waiting_qr", "qr_image": state["qr_data"]}
        state["status"] = "error"
        # Surface as HTTP 502 so the Console's fetch() drops into .catch()
        # instead of treating the 200/error-payload as a successful pair.
        raise HTTPException(
            status_code=502,
            detail=f"WhatsApp pairing failed: {e}",
        ) from e


@router.get(
    "/channels/whatsapp/pair/status",
    summary="Check WhatsApp pairing status",
)
async def check_whatsapp_pair_status(request: Request) -> dict:
    """Check current WhatsApp pairing status."""
    from ..agent_context import get_agent_for_request

    agent = await get_agent_for_request(request)
    state = _get_wa_pair_state(agent.agent_id)
    result = {"status": state["status"]}
    if state["code"]:
        result["pair_code"] = state["code"]
    if state["qr_data"]:
        result["qr_image"] = state["qr_data"]
    return result


@router.post(
    "/channels/whatsapp/pair/stop",
    summary="Stop WhatsApp pairing",
)
async def stop_whatsapp_pair(request: Request) -> dict:
    """Stop the WhatsApp pairing process."""
    from ..agent_context import get_agent_for_request

    agent = await get_agent_for_request(request)
    state = _get_wa_pair_state(agent.agent_id)
    client = state.get("client")
    if client:
        try:
            await client.disconnect()
        except Exception:
            pass
    state.update(
        {
            "client": None,
            "code": None,
            "status": "idle",
            "qr_data": None,
            "task": None,
        },
    )
    return {"status": "stopped"}


@router.post(
    "/channels/whatsapp/qrcode",
    summary="Get WhatsApp QR code for linking",
)
async def get_whatsapp_qrcode(request: Request) -> dict:
    """Start WhatsApp QR auth. Returns QR code image for scanning."""
    import asyncio
    import base64
    import io

    try:
        from neonize.aioze.client import NewAClient
        from neonize.events import ConnectedEv
        import segno
    except ImportError:
        raise HTTPException(
            status_code=500,
            detail="neonize-qwenpaw or segno not installed. "
            "Install: pip install qwenpaw[whatsapp]",
        )

    from ..agent_context import get_agent_for_request

    agent = await get_agent_for_request(request)
    auth_dir = _get_wa_auth_dir(agent)
    state = _get_wa_pair_state(agent.agent_id)

    from pathlib import Path

    db_path = str(Path(auth_dir).expanduser() / "neonize.db")
    Path(auth_dir).expanduser().mkdir(parents=True, exist_ok=True)

    # Disconnect any existing pairing client for this agent (mirror of
    # start_whatsapp_pair()): two NewAClients against the same neonize.db
    # cause SQLite lock / websocket collisions.
    old_client = state.get("client")
    if old_client:
        try:
            await old_client.disconnect()
        except Exception:
            pass
    old_task = state.get("task")
    if old_task and not old_task.done():
        old_task.cancel()
        try:
            await asyncio.wait_for(old_task, timeout=2.0)
        except (asyncio.TimeoutError, asyncio.CancelledError, Exception):
            pass

    qr_ready = asyncio.Event()
    qr_result = {"image": None}

    client = NewAClient(name=db_path)
    state["client"] = client
    state["status"] = "waiting_qr"

    @client.event(ConnectedEv)
    async def on_connected(c, evt):
        state["status"] = "connected"

    @client.qr
    async def on_qr(c, qr_bytes):
        try:
            qr = segno.make_qr(qr_bytes)
            buf = io.BytesIO()
            qr.save(buf, kind="png", scale=6, border=2)
            qr_result["image"] = base64.b64encode(buf.getvalue()).decode()
            qr_ready.set()
        except Exception:
            qr_result["image"] = None
            qr_ready.set()

    state["task"] = await client.connect()

    try:
        await asyncio.wait_for(qr_ready.wait(), timeout=15)
    except asyncio.TimeoutError:
        pass

    if qr_result["image"]:
        state["qr_data"] = qr_result["image"]
        return {"status": "waiting_qr", "qr_image": qr_result["image"]}
    state["status"] = "error"
    # Non-2xx so the Console's fetch() drops into .catch() rather than
    # silently succeeding with an error payload.
    raise HTTPException(
        status_code=502,
        detail="QR code not generated (WhatsApp client did not emit a QR within the timeout)",
    )


@router.post(
    "/channels/whatsapp/unbind",
    summary="Unbind WhatsApp session",
    description="Delete the WhatsApp session database so the next connection requires re-pairing.",
)
async def unbind_whatsapp(request: Request) -> dict:
    """Delete neonize.db to force re-authentication on next start.

    Disconnects any in-memory pairing client and cancels its connect task
    BEFORE deleting the sqlite file — otherwise the running client would
    still hold the file open (Windows) or try to write to a now-missing
    database (all platforms).
    """
    import asyncio
    from pathlib import Path as _P

    from ..agent_context import get_agent_for_request

    agent = await get_agent_for_request(request)
    auth_dir = _get_wa_auth_dir(agent)
    state = _get_wa_pair_state(agent.agent_id)

    # Stop any in-memory pairing client + task for this agent first.
    old_client = state.get("client")
    if old_client:
        try:
            await old_client.disconnect()
        except Exception:
            pass
    old_task = state.get("task")
    if old_task and not old_task.done():
        old_task.cancel()
        try:
            await asyncio.wait_for(old_task, timeout=2.0)
        except (asyncio.TimeoutError, asyncio.CancelledError, Exception):
            pass

    db_path = _P(auth_dir).expanduser() / "neonize.db"
    if db_path.exists():
        db_path.unlink()
        state.update(
            {
                "client": None,
                "code": None,
                "status": "idle",
                "qr_data": None,
                "task": None,
            },
        )
        return {
            "status": "unbound",
            "detail": "Session deleted. Restart QwenPaw to re-pair.",
        }
    return {"status": "idle", "detail": "No session found."}


@router.get(
    "/channels/whatsapp/status",
    summary="Get WhatsApp connection status",
)
async def get_whatsapp_status(request: Request) -> dict:
    """Check if WhatsApp is linked."""
    try:
        from pathlib import Path
        from ..agent_context import get_agent_for_request

        agent = await get_agent_for_request(request)
        auth_dir = _get_wa_auth_dir(agent)
        db_path = Path(auth_dir).expanduser() / "neonize.db"
        if not db_path.exists():
            return {"linked": False, "phone": None}
        import asyncio
        import sqlite3

        def _check_linked():
            conn = sqlite3.connect(str(db_path))
            try:
                # whatsmeow_device.jid is the bot's own JID, e.g.
                # "85212345678.0:0@s.whatsapp.net". We return the phone
                # part so the UI can display a real E.164 number instead
                # of a literal "linked" placeholder.
                row = conn.execute(
                    "SELECT jid FROM whatsmeow_device LIMIT 1",
                ).fetchone()
                if not row or not row[0]:
                    return (False, None)
                jid = str(row[0])
                # strip "@s.whatsapp.net" and device suffix (".0:0")
                user = jid.split("@", 1)[0].split(".", 1)[0].split(":", 1)[0]
                phone = f"+{user}" if user.isdigit() else user
                return (True, phone)
            except Exception:
                return (False, None)
            finally:
                conn.close()

        linked, phone = await asyncio.to_thread(_check_linked)
        if linked:
            return {"linked": True, "phone": phone}
        return {"linked": False, "phone": None}
    except Exception as e:
        return {"linked": False, "error": str(e)}


# ── Global Media Server ────────────────────────────────────────


@router.get(
    "/media-server",
    summary="Get global media server config",
)
async def get_media_server_config() -> dict:
    """Return global media server configuration."""
    config = load_config()
    return config.media_server.model_dump()


@router.put(
    "/media-server",
    summary="Update global media server config",
)
async def put_media_server_config(
    request: Request,
    body: MediaServerConfig = Body(...),
) -> dict:
    """Update global media server config and save to config.json."""
    config = load_config()
    config.media_server = body
    save_config(config)

    # Reconcile live MediaServer instance
    ms = getattr(request.app.state, "media_server", None)
    body_dict = body.model_dump()

    if body.enabled and ms is None:
        # Start new server. The MediaServer constructor accepts max_size_mb
        # (MB), not raw bytes — keep the same unit here to avoid doubling.
        from ..media_server import MediaServer
        from urllib.parse import urlparse

        parsed = urlparse(body.server_url or "")
        server = MediaServer(
            host=parsed.hostname or "127.0.0.1",
            port=parsed.port or 8089,
            secret=body_dict.get("media_secret", ""),
            allowed_dirs=body_dict.get("allowed_dirs", []),
            max_size_mb=body_dict.get("max_size_mb", 100),
            tunnel_domain=body_dict.get("tunnel_domain", ""),
            tunnel_mode=body.tunnel_mode,
            named_tunnel_name=body.named_tunnel_name,
            named_tunnel_hostname=body.named_tunnel_hostname,
            named_tunnel_config_file=body.named_tunnel_config_file,
        )
        await server.start()
        request.app.state.media_server = server
    elif not body.enabled and ms is not None:
        # Stop running server
        await ms.stop()
        request.app.state.media_server = None
    elif ms is not None:
        # Update running server config in-place
        ms.secret = body_dict.get("media_secret") or ms.secret
        ms.allowed_dirs = body_dict.get("allowed_dirs", ms.allowed_dirs)
        ms.max_size = body_dict.get("max_size_mb", 100) * 1024 * 1024
        ms.user_tunnel_domain = body_dict.get("tunnel_domain", "")
        # Only overwrite the effective tunnel_domain from user config when
        # no managed tunnel is currently owning it.
        if ms._tunnel_driver is None:
            ms.tunnel_domain = ms.user_tunnel_domain
        await ms.reconcile_tunnel(
            tunnel_mode=body.tunnel_mode,
            named_tunnel_name=body.named_tunnel_name,
            named_tunnel_hostname=body.named_tunnel_hostname,
            named_tunnel_config_file=body.named_tunnel_config_file,
        )

    return body.model_dump()


@router.get(
    "/media-server/status",
    summary="Get media server running status",
)
async def get_media_server_status(request: Request) -> dict:
    """Check if the global media server is running and healthy."""
    ms = getattr(request.app.state, "media_server", None)
    running = bool(
        ms is not None
        and ms._server_task is not None
        and not ms._server_task.done(),
    )
    tunnel_url = ms.get_tunnel_url() if ms is not None else ""
    return {
        "running": running,
        "port": ms.port if ms else None,
        "tunnel_mode": ms.tunnel_mode if ms is not None else "manual",
        "tunnel_url": tunnel_url,
        "tunnel_running": bool(tunnel_url),
    }


# -- MemPalace Config --


@router.get("/mempalace", tags=["config"])
async def get_mempalace_config(request: Request):
    from ..agent_context import get_current_agent_id
    from qwenpaw.config.config import load_agent_config, MemPalaceHooksConfig

    agent_id = get_current_agent_id() or "default"
    try:
        cfg = load_agent_config(agent_id)
        mp = getattr(cfg, "mempalace", None)
        return mp.model_dump() if mp else {"enabled": False}
    except Exception:
        return {"enabled": False}


@router.put("/mempalace", tags=["config"])
async def update_mempalace_config(request: Request, body: dict = Body(...)):
    from ..agent_context import get_current_agent_id
    from qwenpaw.config.config import (
        load_agent_config,
        save_agent_config,
        MemPalaceHooksConfig,
    )

    agent_id = get_current_agent_id() or "default"
    try:
        new_cfg = MemPalaceHooksConfig(**body)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    cfg = load_agent_config(agent_id)
    cfg.mempalace = new_cfg
    save_agent_config(agent_id, cfg)
    schedule_agent_reload(request, agent_id)
    return new_cfg.model_dump()


@router.get("/skillclaw-capture", tags=["config"])
async def get_skillclaw_capture_config(request: Request):
    """Return per-agent SkillClaw capture-hook configuration."""
    from ..agent_context import get_current_agent_id
    from qwenpaw.config.config import load_agent_config

    agent_id = get_current_agent_id() or "default"
    try:
        cfg = load_agent_config(agent_id)
        sc = getattr(cfg, "skillclaw_capture", None)
        return (
            sc.model_dump()
            if sc
            else {
                "enabled": False,
                "records_dir": "",
                "session_id_prefix": "",
            }
        )
    except Exception:
        return {
            "enabled": False,
            "records_dir": "",
            "session_id_prefix": "",
        }


@router.put("/skillclaw-capture", tags=["config"])
async def update_skillclaw_capture_config(
    request: Request,
    body: dict = Body(...),
):
    """Update SkillClaw capture-hook configuration for the active agent.
    Triggers an async hot-reload so the running agent registers (or
    drops) the hook on its next instantiation."""
    from ..agent_context import get_current_agent_id
    from qwenpaw.config.config import (
        load_agent_config,
        save_agent_config,
        SkillClawCaptureConfig,
    )

    agent_id = get_current_agent_id() or "default"
    try:
        new_cfg = SkillClawCaptureConfig(**body)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    cfg = load_agent_config(agent_id)
    cfg.skillclaw_capture = new_cfg
    save_agent_config(agent_id, cfg)
    schedule_agent_reload(request, agent_id)
    return new_cfg.model_dump()


# ── Signal link flow (signal-cli subprocess pairing) ──────────────────────
# Ported from PR #3508 (feat/signal-channel) for testing on dev.

# server; concurrent linking for different agents is safe because each
# gets its own state dict. This mirrors WhatsApp's ``_whatsapp_pair_states``.
_signal_link_states: dict[str, dict] = {}


def _get_signal_link_state(agent_id: str) -> dict:
    """Get or create per-agent Signal link state."""
    if agent_id not in _signal_link_states:
        _signal_link_states[agent_id] = {
            "proc": None,
            "task": None,
            "status": "idle",
            "qr_image": None,
            "link_url": None,
            "device_name": None,
            "phone": None,
            "uuid": None,
            "error": None,
        }
    return _signal_link_states[agent_id]


def _get_signal_data_dir(agent) -> "Path":
    """Resolve signal-cli data directory from agent config.

    Priority (must match ``_resolve_signal_data_dir`` in channel.py):
      1. Explicit ``data_dir`` in the agent's signal channel config
      2. ``agent.workspace_dir/credentials/signal/default`` (per-agent)
      3. ``WORKING_DIR/credentials/signal/default`` (install-wide fallback)
    """
    from ..channels.signal.channel import _resolve_signal_data_dir
    from pathlib import Path

    sig_cfg = getattr(agent.config.channels, "signal", None)
    explicit = (getattr(sig_cfg, "data_dir", "") if sig_cfg else "") or ""
    ws = getattr(agent, "workspace_dir", None)
    ws_path = Path(ws).expanduser() if ws else None
    return _resolve_signal_data_dir(explicit, ws_path)


def _get_signal_cli_path(agent) -> str:
    """Resolve signal-cli binary path from agent config.

    Defaults to ``signal-cli`` (PATH lookup) when no explicit path set.
    """
    sig_cfg = getattr(agent.config.channels, "signal", None)
    if sig_cfg and getattr(sig_cfg, "signal_cli_path", ""):
        return sig_cfg.signal_cli_path
    return "signal-cli"


def _read_signal_accounts(data_dir: "Path") -> dict:
    """Read ``<data_dir>/data/accounts.json`` and return a normalized dict.

    Always returns a ``dict`` whose ``accounts`` key is a ``list``. If the
    file is missing, malformed, or doesn't match the expected shape (e.g.
    top-level is an array, or ``accounts`` is absent / non-list), fall back
    to ``{"accounts": []}`` so callers can uniformly call
    ``accounts.get("accounts", [])`` without exception handling.
    """
    import json

    accounts_file = data_dir / "data" / "accounts.json"
    if not accounts_file.exists():
        return {"accounts": []}
    try:
        parsed = json.loads(accounts_file.read_text())
    except Exception:
        return {"accounts": []}
    if not isinstance(parsed, dict):
        return {"accounts": []}
    accounts = parsed.get("accounts")
    if not isinstance(accounts, list):
        return {"accounts": []}
    # Filter entries to dicts, dropping anything malformed (non-dict entries
    # would break downstream ``.get("number")`` calls).
    parsed["accounts"] = [a for a in accounts if isinstance(a, dict)]
    return parsed


class SignalLinkBody(BaseModel):
    """Request body for POST /channels/signal/link."""

    device_name: str = "QwenPaw"


async def _run_signal_link(
    agent_id: str,
    signal_cli_path: str,
    data_dir: "Path",
    device_name: str,
) -> None:
    """Background task: keep the ``signal-cli link`` subprocess alive until
    the user scans the QR on their phone, then capture the phone/uuid from
    its ``Associated with: +phone`` final line.

    State-dict shape is locked by the ``/channels/signal/link`` endpoint:
    we only flip ``status`` from ``waiting_qr`` → ``linked`` / ``error``
    and fill in ``phone`` / ``error``. The URL + QR PNG are captured by
    the endpoint before this task starts, so reads of ``link_url`` /
    ``qr_image`` remain valid throughout.
    """
    import asyncio
    import re as _re

    state = _get_signal_link_states_raw(agent_id)
    proc = state.get("proc")
    if not proc:
        return
    try:
        assert proc.stdout is not None
        # Read remaining stdout until EOF; signal-cli prints a final
        # "Associated with: +<phone>" on success. Don't hold state
        # open forever — wait for the subprocess itself to exit.
        tail: list[str] = []
        while True:
            try:
                raw = await asyncio.wait_for(
                    proc.stdout.readline(),
                    timeout=1.0,
                )
            except asyncio.TimeoutError:
                if proc.returncode is not None:
                    break
                continue
            if not raw:
                break
            line = raw.decode(errors="replace").strip()
            if not line:
                continue
            tail.append(line)
            m = _re.search(r"Associated with:\s*(\+\d+)", line)
            if m:
                phone = m.group(1)
                state["phone"] = phone
                # Pull UUID from accounts.json (authoritative source —
                # signal-cli doesn't print it on stdout).
                accounts = _read_signal_accounts(data_dir)
                for acc in accounts.get("accounts", []):
                    if acc.get("number") == phone:
                        state["uuid"] = acc.get("uuid") or ""
                        break
                state["status"] = "linked"
                break
        # Always drain so the pipe doesn't back up if the phone cancels.
        try:
            await asyncio.wait_for(proc.wait(), timeout=5.0)
        except asyncio.TimeoutError:
            pass
        if state["status"] != "linked":
            rc = proc.returncode
            # Non-linked exit: surface stderr + exit code for the user.
            stderr_bytes = b""
            if proc.stderr is not None:
                try:
                    stderr_bytes = await proc.stderr.read()
                except Exception:
                    pass
            stderr = stderr_bytes.decode(errors="replace").strip()
            detail = stderr or ("\n".join(tail[-5:]) if tail else "")
            state["status"] = "error"
            state["error"] = f"signal-cli link exited with code {rc}" + (
                f": {detail}" if detail else ""
            )
    except asyncio.CancelledError:
        raise
    except Exception as e:  # pylint: disable=broad-except
        state["status"] = "error"
        state["error"] = f"link task failed: {e}"


def _get_signal_link_states_raw(agent_id: str) -> dict:
    """Raw alias used by the background task (avoids double-init in tests)."""
    return _signal_link_states.setdefault(agent_id, {})


@router.post(
    "/channels/signal/link",
    summary="Start Signal device-link flow",
    description=(
        "Spawns ``signal-cli link -n <device_name>`` as a short-lived child "
        "process and returns a base64 PNG of the device-link URL. The user "
        "scans this QR in Signal → Settings → Linked Devices → Link new "
        "device; the backend watches the subprocess stdout for the "
        "'Associated with: +<phone>' confirmation line."
    ),
)
async def start_signal_link(
    request: Request,
    body: SignalLinkBody = Body(default_factory=SignalLinkBody),
) -> dict:
    """Start the signal-cli link flow and return a QR image for the user."""
    import asyncio
    import base64
    import io
    import re as _re

    from ..agent_context import get_agent_for_request

    agent = await get_agent_for_request(request)
    state = _get_signal_link_state(agent.agent_id)

    # Block linking when an account is already linked — keeps the flow
    # single-purpose and prevents accidentally overwriting session state.
    data_dir = _get_signal_data_dir(agent)
    accounts = _read_signal_accounts(data_dir)
    if accounts.get("accounts"):
        raise HTTPException(
            status_code=409,
            detail=(
                "signal-cli data dir already holds a linked account. "
                "Call POST /channels/signal/unbind first, or clear "
                f"{data_dir}/data/ manually."
            ),
        )

    # Kill any previous link subprocess for this agent — two
    # signal-cli processes against the same data dir would fight over
    # the sqlite lock (identical pattern to WhatsApp's pair endpoint).
    old_proc = state.get("proc")
    if old_proc is not None and old_proc.returncode is None:
        try:
            old_proc.terminate()
            await asyncio.wait_for(old_proc.wait(), timeout=2.0)
        except (asyncio.TimeoutError, ProcessLookupError, Exception):
            try:
                old_proc.kill()
            except Exception:
                pass
    old_task = state.get("task")
    if old_task is not None and not old_task.done():
        old_task.cancel()
        try:
            await asyncio.wait_for(old_task, timeout=1.0)
        except (asyncio.TimeoutError, asyncio.CancelledError, Exception):
            pass

    state.update(
        {
            "proc": None,
            "task": None,
            "status": "starting",
            "qr_image": None,
            "link_url": None,
            "device_name": body.device_name,
            "phone": None,
            "uuid": None,
            "error": None,
        },
    )

    signal_cli_path = _get_signal_cli_path(agent)
    try:
        data_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        raise HTTPException(
            status_code=500,
            detail=f"Cannot create signal data_dir {data_dir}: {e}",
        ) from e

    cmd = [
        signal_cli_path,
        "-c",
        str(data_dir),
        "link",
        "-n",
        body.device_name or "QwenPaw",
    ]
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except FileNotFoundError as e:
        state["status"] = "error"
        state["error"] = f"signal-cli binary not found: {signal_cli_path}"
        raise HTTPException(
            status_code=500,
            detail=state["error"],
        ) from e
    state["proc"] = proc

    # Read the link URL line from stdout. signal-cli prints the link as
    # a standalone line; on newer releases it's sgnl://linkdevice?..., on
    # older ones it's tsdevice:/?... — accept both.
    link_url: Optional[str] = None
    assert proc.stdout is not None
    link_re = _re.compile(r"(?:sgnl://linkdevice\?|tsdevice:/\??)\S+")
    try:
        for _ in range(50):  # at most ~50 lines before giving up
            try:
                raw = await asyncio.wait_for(
                    proc.stdout.readline(),
                    timeout=15.0,
                )
            except asyncio.TimeoutError:
                break
            if not raw:
                break
            line = raw.decode(errors="replace").strip()
            if not line:
                continue
            m = link_re.search(line)
            if m:
                link_url = m.group(0)
                break
    except Exception as e:
        try:
            proc.terminate()
        except Exception:
            pass
        state["status"] = "error"
        state["error"] = f"link stdout read failed: {e}"
        raise HTTPException(
            status_code=502,
            detail=state["error"],
        ) from e

    if not link_url:
        # No URL captured — subprocess may have died or is hung. Surface
        # stderr and shut it down rather than leave a zombie.
        stderr_bytes = b""
        if proc.stderr is not None:
            try:
                stderr_bytes = await asyncio.wait_for(
                    proc.stderr.read(),
                    timeout=2.0,
                )
            except asyncio.TimeoutError:
                pass
        try:
            proc.terminate()
        except Exception:
            pass
        detail = stderr_bytes.decode(errors="replace").strip() or (
            "signal-cli link did not emit a tsdevice:/ or sgnl:// URL "
            "within 15s"
        )
        state["status"] = "error"
        state["error"] = detail
        raise HTTPException(status_code=502, detail=detail)

    # Encode the link URL as a PNG QR. segno is already a dep of the
    # WhatsApp pair endpoint — keep the import guarded so the missing-dep
    # message is actionable.
    try:
        import segno  # type: ignore
    except ImportError as e:  # pragma: no cover - dev import issue
        try:
            proc.terminate()
        except Exception:
            pass
        raise HTTPException(
            status_code=500,
            detail="segno not installed. Install: pip install segno",
        ) from e
    qr = segno.make_qr(link_url)
    buf = io.BytesIO()
    qr.save(buf, kind="png", scale=5, border=2)
    qr_image = base64.b64encode(buf.getvalue()).decode()

    state.update(
        {
            "status": "waiting_qr",
            "qr_image": qr_image,
            "link_url": link_url,
        },
    )

    # Spawn the background watcher to flip state once the phone scans.
    state["task"] = asyncio.create_task(
        _run_signal_link(
            agent.agent_id,
            signal_cli_path,
            data_dir,
            body.device_name,
        ),
        name=f"signal_link_watcher_{agent.agent_id}",
    )

    return {
        "status": "waiting_qr",
        "qr_image": qr_image,
        "link_url": link_url,
    }


@router.get(
    "/channels/signal/link/status",
    summary="Check Signal link status",
)
async def check_signal_link_status(request: Request) -> dict:
    """Return the current per-agent Signal link state."""
    from ..agent_context import get_agent_for_request

    agent = await get_agent_for_request(request)
    state = _get_signal_link_state(agent.agent_id)
    out: dict[str, Any] = {"status": state["status"]}
    if state.get("qr_image"):
        out["qr_image"] = state["qr_image"]
    if state.get("link_url"):
        out["link_url"] = state["link_url"]
    if state.get("phone"):
        out["phone"] = state["phone"]
    if state.get("uuid"):
        out["uuid"] = state["uuid"]
    if state.get("error"):
        out["error"] = state["error"]
    return out


@router.post(
    "/channels/signal/link/stop",
    summary="Cancel the in-progress Signal link flow",
)
async def stop_signal_link(request: Request) -> dict:
    """Kill the signal-cli link subprocess for this agent."""
    import asyncio
    from ..agent_context import get_agent_for_request

    agent = await get_agent_for_request(request)
    state = _get_signal_link_state(agent.agent_id)
    proc = state.get("proc")
    if proc is not None and proc.returncode is None:
        try:
            proc.terminate()
            await asyncio.wait_for(proc.wait(), timeout=2.0)
        except (asyncio.TimeoutError, ProcessLookupError, Exception):
            try:
                proc.kill()
            except Exception:
                pass
    task = state.get("task")
    if task is not None and not task.done():
        task.cancel()
        try:
            await asyncio.wait_for(task, timeout=1.0)
        except (asyncio.TimeoutError, asyncio.CancelledError, Exception):
            pass
    state.update(
        {
            "proc": None,
            "task": None,
            "status": "idle",
            "qr_image": None,
            "link_url": None,
            "error": None,
        },
    )
    return {"status": "stopped"}


@router.post(
    "/channels/signal/unbind",
    summary="Unlink Signal account",
    description=(
        "Clears signal-cli's SQLite store for the currently linked "
        "account so the next link flow starts fresh. Does NOT touch "
        "the signal-cli binary or user-level config outside the "
        "resolved data_dir."
    ),
)
async def unbind_signal(request: Request) -> dict:
    """Delete the signal-cli data dir contents for this agent.

    Stops the link watcher (if any) before deleting so the subprocess
    doesn't hold open handles on Windows.

    NOTE: This does NOT stop the main signal channel subprocess if it's
    running — the channel owns its own lifecycle via ChannelManager.
    Operators should disable the channel (``enabled: false``) in config
    before unbinding, or restart QwenPaw after. That matches how
    ``unbind_whatsapp`` behaves.
    """
    import asyncio
    import shutil
    from pathlib import Path as _P
    from ..agent_context import get_agent_for_request

    agent = await get_agent_for_request(request)
    state = _get_signal_link_state(agent.agent_id)

    # Stop any running link flow first.
    proc = state.get("proc")
    if proc is not None and proc.returncode is None:
        try:
            proc.terminate()
            await asyncio.wait_for(proc.wait(), timeout=2.0)
        except (asyncio.TimeoutError, ProcessLookupError, Exception):
            try:
                proc.kill()
            except Exception:
                pass
    task = state.get("task")
    if task is not None and not task.done():
        task.cancel()
        try:
            await asyncio.wait_for(task, timeout=1.0)
        except (asyncio.TimeoutError, asyncio.CancelledError, Exception):
            pass

    data_dir: _P = _get_signal_data_dir(agent)
    account_data = data_dir / "data"
    if account_data.exists():
        try:
            shutil.rmtree(account_data)
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to remove {account_data}: {e}",
            ) from e
        state.update(
            {
                "proc": None,
                "task": None,
                "status": "idle",
                "qr_image": None,
                "link_url": None,
                "phone": None,
                "uuid": None,
                "error": None,
            },
        )
        return {
            "status": "unbound",
            "detail": (
                "Signal account data cleared. Disable or restart the "
                "channel before re-linking to release the subprocess lock."
            ),
        }
    return {"status": "idle", "detail": "No signal account data to clear."}


@router.get(
    "/channels/signal/status",
    summary="Get Signal link status",
)
async def get_signal_status(request: Request) -> dict:
    """Return whether a Signal account is linked and, if so, its E.164
    phone + UUID.

    Reads ``<data_dir>/data/accounts.json`` directly — no subprocess
    required.
    """
    from ..agent_context import get_agent_for_request

    agent = await get_agent_for_request(request)
    try:
        data_dir = _get_signal_data_dir(agent)
        accounts = _read_signal_accounts(data_dir)
    except Exception as e:
        # Resolving data_dir / reading the account store failed
        # unexpectedly. Surface a real non-2xx so the Console can
        # distinguish this from "simply not linked".
        logger.exception("signal: get_status failed")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to read Signal account store: {e}",
        ) from e
    entries = accounts.get("accounts") or []
    if not entries:
        return {"linked": False, "phone": None, "uuid": None}
    first = entries[0]
    return {
        "linked": True,
        "phone": first.get("number") or None,
        "uuid": first.get("uuid") or None,
    }


# ── Signal directory endpoints (list known contacts + groups) ─────────────
# Read directly from signal-cli's account.db SQLite (opened in URI read-only
# mode so it doesn't contend with the running jsonRpc daemon's write lock).
# Used by the Console drawer to populate dropdowns for groups /
# allow_from / group_allow_from instead of forcing users to type raw UUIDs
# or base64 group-ids.


def _signal_account_db_path(data_dir: "_P") -> Optional["_P"]:
    """Resolve the SQLite account DB path for the first linked account.

    Returns ``<data_dir>/data/<path>.d/account.db`` where ``<path>`` is the
    opaque directory name signal-cli picked (e.g. ``750890``). Reads
    accounts.json to discover it. Returns None if no linked account.
    """
    accounts = _read_signal_accounts(data_dir)
    entries = accounts.get("accounts") or []
    if not entries:
        return None
    path = entries[0].get("path") or ""
    if not path:
        return None
    db = data_dir / "data" / f"{path}.d" / "account.db"
    if not db.exists():
        return None
    return db


@router.get(
    "/channels/signal/contacts",
    summary="List known Signal contacts (from signal-cli store)",
)
async def list_signal_contacts(request: Request) -> dict:
    """Return ``{contacts: [{number, uuid, name}, ...]}``.

    Read from ``account.db`` in SQLite URI read-only mode so it doesn't
    contend with the running jsonRpc daemon. ``name`` is the first
    non-empty of: user-set ``given_name + family_name``, ``nick_name``, or
    profile ``profile_given_name + profile_family_name``.
    """
    import sqlite3
    from ..agent_context import get_agent_for_request

    agent = await get_agent_for_request(request)
    data_dir = _get_signal_data_dir(agent)
    db = _signal_account_db_path(data_dir)
    if db is None:
        return {"contacts": []}
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        cur = conn.cursor()
        rows = cur.execute(
            """
            SELECT number, aci, given_name, family_name,
                   nick_name, profile_given_name, profile_family_name
            FROM recipient
            WHERE (number IS NOT NULL AND number != '')
               OR (aci IS NOT NULL AND aci != '')
            ORDER BY COALESCE(profile_given_name, given_name, nick_name, number, aci)
            """,
        ).fetchall()
    except Exception as e:
        logger.warning("signal: list_contacts SQLite read failed: %s", e)
        return {"contacts": []}
    finally:
        conn.close()

    contacts = []
    for number, aci, gn, fn, nn, pgn, pfn in rows:
        parts = [p for p in (gn, fn) if p]
        display = " ".join(parts) if parts else ""
        if not display and nn:
            display = nn
        if not display:
            pparts = [p for p in (pgn, pfn) if p]
            display = " ".join(pparts) if pparts else ""
        contacts.append(
            {
                "number": number or "",
                "uuid": aci or "",
                "name": display or "",
            },
        )
    return {"contacts": contacts}


@router.get(
    "/channels/signal/groups",
    summary="List known Signal groups (from signal-cli store)",
)
async def list_signal_groups(request: Request) -> dict:
    """Return ``{groups: [{id, blocked}, ...]}``.

    ``id`` is the **base64 group-id** — the same form used in
    ``channels.signal.groups`` config allowlist. signal-cli stores group
    metadata (name, members) as a protobuf BLOB in ``group_data``, which
    this endpoint does NOT decode; users see raw ids but can match them
    against Signal app's group screen by group-id hash visible in Signal
    Desktop's debug info. Later work may add a protobuf-parsed name.
    """
    import base64
    import sqlite3
    from ..agent_context import get_agent_for_request

    agent = await get_agent_for_request(request)
    data_dir = _get_signal_data_dir(agent)
    db = _signal_account_db_path(data_dir)
    if db is None:
        return {"groups": []}
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        cur = conn.cursor()
        rows = cur.execute(
            "SELECT group_id, blocked FROM group_v2 ORDER BY _id",
        ).fetchall()
    except Exception as e:
        logger.warning("signal: list_groups SQLite read failed: %s", e)
        return {"groups": []}
    finally:
        conn.close()

    groups = []
    for gid_blob, blocked in rows:
        if not gid_blob:
            continue
        gid_b64 = base64.b64encode(gid_blob).decode("ascii")
        groups.append({"id": gid_b64, "blocked": bool(blocked)})
    return {"groups": groups}


# ── Security / Allow No Auth Hosts ────────────────────────────────────


class AllowNoAuthHostsResponse(BaseModel):
    """Response model for allow_no_auth_hosts configuration."""

    hosts: List[str] = Field(
        description="List of IP addresses allowed without authentication",
    )


class AllowNoAuthHostsUpdateBody(BaseModel):
    """Request body for updating allow_no_auth_hosts configuration."""

    hosts: List[str] = Field(
        description="List of IP addresses allowed without authentication",
    )


@router.get(
    "/security/allow-no-auth-hosts",
    response_model=AllowNoAuthHostsResponse,
    summary="Get allow no auth hosts configuration",
)
async def get_allow_no_auth_hosts() -> AllowNoAuthHostsResponse:
    """Get the list of IP addresses allowed without authentication."""
    config = load_config()
    return AllowNoAuthHostsResponse(
        hosts=config.security.allow_no_auth_hosts,
    )


@router.put(
    "/security/allow-no-auth-hosts",
    response_model=AllowNoAuthHostsResponse,
    summary="Update allow no auth hosts configuration",
)
async def put_allow_no_auth_hosts(
    body: AllowNoAuthHostsUpdateBody = Body(...),
) -> AllowNoAuthHostsResponse:
    """Update the list of IP addresses allowed without authentication.

    Validates and normalizes each IP address:
    - Strips whitespace
    - Removes empty strings
    - Deduplicates entries
    - Validates as literal IPv4/IPv6 using ipaddress module
    - Returns 400 on invalid IP addresses
    """
    import ipaddress

    # Normalize and validate IP addresses
    normalized_hosts = []
    seen = set()
    invalid_ips = []

    for host in body.hosts:
        # Strip whitespace
        host = host.strip()

        # Skip empty strings
        if not host:
            continue

        # Validate IP address format
        try:
            # This validates and normalizes the IP address
            ip_obj = ipaddress.ip_address(host)
            # Use the compressed string representation
            normalized_ip = str(ip_obj)

            # Deduplicate
            if normalized_ip not in seen:
                seen.add(normalized_ip)
                normalized_hosts.append(normalized_ip)
        except ValueError:
            invalid_ips.append(host)

    # Return 400 if any invalid IP addresses were provided
    if invalid_ips:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Invalid IP address(es): {', '.join(invalid_ips)}. "
                "Only literal IPv4/IPv6 addresses are allowed."
            ),
        )

    config = load_config()
    config.security.allow_no_auth_hosts = normalized_hosts
    save_config(config)
    return AllowNoAuthHostsResponse(hosts=normalized_hosts)
