# -*- coding: utf-8 -*-
"""QwenPaw Agent - Main agent implementation.

This module provides the main QwenPawAgent class built on ReActAgent,
with integrated tools, skills, and memory management.
"""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from typing import Any, List, Literal, Optional, Type, TYPE_CHECKING

from agentscope.agent import ReActAgent
from agentscope.agent._react_agent import _MemoryMark
from agentscope.memory import InMemoryMemory
from agentscope.message import Msg
from agentscope.tool import Toolkit
from anyio import ClosedResourceError
from pydantic import BaseModel

from ..app.mcp import HttpStatefulClient, StdIOStatefulClient
from .chat_log import (
    append_to_log as chat_log_append,
    collect_unpersisted as chat_log_collect_unpersisted,
)
from .command_handler import CommandHandler
from .hooks import BootstrapHook
from .model_factory import create_model_and_formatter
from .prompt import (
    build_multimodal_hint,
    build_system_prompt_from_working_dir,
    get_active_model_supports_multimodal,
)
from .skill_system import (
    apply_skill_config_env_overrides,
    ensure_skills_initialized,
    get_workspace_skills_dir,
    resolve_effective_skills,
)
from .tool_guard_mixin import ToolGuardMixin
from .tools import (
    browser_use,
    delegate_external_agent,
    chat_with_agent,
    check_agent_task,
    submit_to_agent,
    desktop_screenshot,
    edit_file,
    execute_shell_command,
    get_current_time,
    get_token_usage,
    glob_search,
    grep_search,
    list_agents,
    read_file,
    send_file_to_user,
    set_user_timezone,
    signal_add_stickers_to_pack,
    signal_create_sticker_pack,
    signal_install_sticker_pack,
    signal_list_sticker_packs,
    signal_prepare_sticker_webp,
    signal_preview_sticker,
    signal_send_sticker,
    view_image,
    view_video,
    write_file,
    create_memory_search_tool,
    mempalace_diary_write,
)
from .utils import process_file_and_media_blocks_in_message
from ..constant import (
    MEDIA_UNSUPPORTED_PLACEHOLDER,
    WORKING_DIR,
)
from ..exceptions import ModelContextLengthExceededException
from ..providers.model_capability_cache import get_capability_cache

if TYPE_CHECKING:
    from ..agents.memory import BaseMemoryManager
    from ..agents.context import BaseContextManager
    from ..config.config import AgentProfileConfig
    from .context import AgentContext

logger = logging.getLogger(__name__)

# Valid namesake strategies for tool registration
NamesakeStrategy = Literal["override", "skip", "raise", "rename"]


class QwenPawAgent(ToolGuardMixin, ReActAgent):
    """QwenPaw Agent with integrated tools, skills, and memory management.

    This agent extends ReActAgent with:
    - Built-in tools (shell, file operations, browser, etc.)
    - Dynamic skill loading from working directory
    - Memory management with auto-compaction
    - Bootstrap guidance for first-time setup
    - System command handling (/compact, /new, etc.)
    - Tool-guard security interception (via ToolGuardMixin)

    MRO note
    ~~~~~~~~
    ``ToolGuardMixin`` overrides ``_acting`` and ``_reasoning`` via
    Python's MRO: QwenPawAgent → ToolGuardMixin → ReActAgent.  If you
    add a ``_acting`` or ``_reasoning`` override in this class, you
    **must** call ``super()._acting(...)`` / ``super()._reasoning(...)``
    so the guard interception remains active.
    """

    def __init__(
        self,
        agent_config: "AgentProfileConfig",
        env_context: Optional[str] = None,
        mcp_clients: Optional[List[Any]] = None,
        memory_manager: BaseMemoryManager | None = None,
        context_manager: BaseContextManager | None = None,
        request_context: Optional[dict[str, str]] = None,
        namesake_strategy: NamesakeStrategy = "skip",
        workspace_dir: Path | None = None,
        task_tracker: Any | None = None,
        plan_notebook: Any | None = None,
    ):
        """Initialize QwenPawAgent.

        Args:
            agent_config: Agent profile configuration containing all settings
                including running config (max_iters, max_input_length,
                memory_compact_threshold, etc.) and language setting.
            env_context: Optional environment context to prepend to
                system prompt
            mcp_clients: Optional list of MCP clients for tool
                integration
            memory_manager: Optional memory manager instance. Pass ``None``
                to disable the memory manager entirely.
            context_manager: Optional context manager instance
            request_context: Optional request context with session_id,
                user_id, channel, agent_id
            namesake_strategy: Strategy to handle namesake tool functions.
                Options: "override", "skip", "raise", "rename"
                (default: "skip")
            workspace_dir: Workspace directory for reading prompt files
                (if None, uses global WORKING_DIR)
        """
        self._agent_config = agent_config
        self._env_context = env_context
        self._request_context = dict(request_context or {})
        self._mcp_clients = mcp_clients or []
        self._namesake_strategy = namesake_strategy
        self._workspace_dir = workspace_dir
        self._task_tracker = task_tracker

        # Extract configuration from agent_config
        running_config = agent_config.running
        self._language = agent_config.language

        # Initialize toolkit with built-in tools
        toolkit = self._create_toolkit(namesake_strategy=namesake_strategy)

        # Load and register skills
        self._register_skills(toolkit)

        # Initialize memory_manager and context_manager for use
        # in _build_sys_prompt
        self.memory_manager = memory_manager
        self.context_manager = context_manager

        # Build system prompt
        sys_prompt = self._build_sys_prompt()

        # Create model and formatter using factory method
        model, formatter = create_model_and_formatter(agent_id=agent_config.id)
        model_info = (
            f"{agent_config.active_model.provider_id}/"
            f"{agent_config.active_model.model}"
            if agent_config.active_model
            else "global-fallback"
        )
        logger.info(
            f"Agent '{agent_config.id}' initialized with model: "
            f"{model_info} (class: {model.__class__.__name__})",
        )
        # Initialize parent ReActAgent
        init_kwargs: dict[str, Any] = {
            "name": agent_config.name or "QwenPaw",
            "model": model,
            "sys_prompt": sys_prompt,
            "toolkit": toolkit,
            "memory": InMemoryMemory(),
            "formatter": formatter,
            "max_iters": running_config.max_iters,
        }
        if plan_notebook is not None:
            init_kwargs["plan_notebook"] = plan_notebook
        super().__init__(**init_kwargs)

        # Register memory tools provided by the memory manager
        if self.memory_manager is not None:
            memory_tools = self.memory_manager.list_memory_tools()
            for tool_fn in memory_tools:
                self.toolkit.register_tool_function(
                    tool_fn,
                    namesake_strategy=self._namesake_strategy,
                )
            logger.debug(
                "Registered memory tools: %s",
                [fn.__name__ for fn in memory_tools],
            )

        # Configure context manager memory if available
        if self.context_manager is not None:
            self.memory: "AgentContext" = (
                self.context_manager.get_agent_context()
            )
            logger.debug("Context manager configured")

        # Wrap memory.add so every non-transient call also lands in the
        # per-chat append-only log (see ``chat_log.py``).  MUST run
        # AFTER the context_manager swap above (which replaces
        # ``self.memory`` with an AgentContext) — wrapping the
        # pre-swap InMemoryMemory would attach to a soon-discarded
        # instance.
        self._install_chat_log_wrapper()

        # Setup command handler
        self.command_handler = CommandHandler(
            agent_name=self.name,
            memory=self.memory,
            memory_manager=self.memory_manager,
            context_manager=self.context_manager,
        )

        # Register hooks
        self._register_hooks()

    def _create_toolkit(
        self,
        namesake_strategy: NamesakeStrategy = "skip",
    ) -> Toolkit:
        """Create and populate toolkit with built-in tools.

        Args:
            namesake_strategy: Strategy to handle namesake tool functions.
                Options: "override", "skip", "raise", "rename"
                (default: "skip")

        Returns:
            Configured toolkit instance
        """
        toolkit = Toolkit()

        # Check which tools are enabled from agent config
        enabled_tools = {}
        async_execution_tools = {}
        try:
            if hasattr(self._agent_config, "tools") and hasattr(
                self._agent_config.tools,
                "builtin_tools",
            ):
                builtin_tools = self._agent_config.tools.builtin_tools
                enabled_tools = {
                    name: tool.enabled for name, tool in builtin_tools.items()
                }
                # Only selected long-running tools support async_execution.
                async_capable_tool_names = {
                    "execute_shell_command",
                    "delegate_external_agent",
                }
                async_execution_tools = {
                    name: builtin_tools.get(name).async_execution
                    if name in builtin_tools
                    else False
                    for name in async_capable_tool_names
                }
        except Exception as e:
            logger.warning(
                f"Failed to load agent tools config: {e}, "
                "all tools will be disabled",
            )

        # Map of tool functions (hardcoded builtin tools)
        tool_functions = {
            "execute_shell_command": execute_shell_command,
            "read_file": read_file,
            "write_file": write_file,
            "edit_file": edit_file,
            "grep_search": grep_search,
            "glob_search": glob_search,
            "browser_use": browser_use,
            "desktop_screenshot": desktop_screenshot,
            "view_image": view_image,
            "view_video": view_video,
            "send_file_to_user": send_file_to_user,
            "get_current_time": get_current_time,
            "set_user_timezone": set_user_timezone,
            "get_token_usage": get_token_usage,
            "delegate_external_agent": delegate_external_agent,
            "list_agents": list_agents,
            "chat_with_agent": chat_with_agent,
            "submit_to_agent": submit_to_agent,
            "check_agent_task": check_agent_task,
            "signal_list_sticker_packs": signal_list_sticker_packs,
            "signal_preview_sticker": signal_preview_sticker,
            "signal_install_sticker_pack": signal_install_sticker_pack,
            "signal_create_sticker_pack": signal_create_sticker_pack,
            "signal_add_stickers_to_pack": signal_add_stickers_to_pack,
            "signal_prepare_sticker_webp": signal_prepare_sticker_webp,
            "signal_send_sticker": signal_send_sticker,
        }

        # Per-request channel gate for cross-channel tools.
        # Observed 2026-04-24: adding 7 signal_sticker tools bloated the
        # toolkit; claude-opus-4-7 in a WhatsApp reply dropped tool_use
        # entirely (seemingly-irrelevant signal_* options made the model
        # play safe and reply with text only).  Rule: only expose
        # signal_* when the inbound channel is signal OR the request
        # originates from a UI surface where the user might be testing
        # them (console).  Agents pointed at whatsapp/discord/etc
        # don't see them at all, keeping their tool list focused.
        req_ctx = getattr(self, "_request_context", {}) or {}
        active_channel = (req_ctx.get("channel") or "").lower()
        # "console" and empty string = UI/debug context → allow all.
        # Any other explicit channel = scope to that channel only.
        # Per-agent override: ``allow_cross_channel_signal_tools`` lifts the
        # gate entirely (use case: migrate a WhatsApp sticker pack to Signal,
        # where the inbound channel is whatsapp but the agent legitimately
        # needs signal_create_sticker_pack et al).  Default off so routine
        # WhatsApp agents keep the focused toolkit that avoids the 2026-04-24
        # claude-opus tool_use drop regression.
        allow_cross_channel = bool(
            getattr(
                getattr(self._agent_config, "running", None),
                "allow_cross_channel_signal_tools",
                False,
            ),
        )
        signal_tools_allowed = allow_cross_channel or active_channel in (
            "",
            "console",
            "signal",
        )

        # Track hardcoded built-in tools for backward compatibility
        hardcoded_builtin_tools = set(tool_functions.keys())

        # Dynamically load plugin-registered tools
        from . import tools as tools_module

        plugin_tools = set()
        for tool_name in getattr(tools_module, "__all__", []):
            if tool_name not in tool_functions:
                tool_func = getattr(tools_module, tool_name, None)
                if callable(tool_func):
                    tool_functions[tool_name] = tool_func
                    plugin_tools.add(tool_name)
                    logger.debug(
                        "Discovered plugin tool: %s",
                        tool_name,
                    )

        # Register tools with appropriate defaults
        for tool_name, tool_func in tool_functions.items():
            # For plugin tools: skip if not in config (security)
            # For hardcoded tools: default to enabled (backward compatibility)
            if tool_name in plugin_tools:
                if tool_name not in enabled_tools:
                    logger.debug(
                        "Skipped unconfigured plugin tool: %s",
                        tool_name,
                    )
                    continue
            else:
                # Hardcoded built-in tool: use default-to-enabled
                pass

            # Check if tool is enabled
            if not enabled_tools.get(
                tool_name,
                tool_name in hardcoded_builtin_tools,
            ):
                logger.debug("Skipped disabled tool: %s", tool_name)
                continue

            # Skip signal_* when inbound is a different chat channel.
            if tool_name.startswith("signal_") and not signal_tools_allowed:
                logger.debug(
                    "Skipped %s: active channel=%s (signal tools hidden)",
                    tool_name,
                    active_channel,
                )
                continue

            # Get async_execution setting (default to False for backward
            # compatibility)
            async_exec = async_execution_tools.get(tool_name, False)

            toolkit.register_tool_function(
                tool_func,
                namesake_strategy=namesake_strategy,
                async_execution=async_exec,
            )
            logger.debug(
                "Registered tool: %s (async_execution=%s)",
                tool_name,
                async_exec,
            )

        # Auto-register background task management tools if any *enabled*
        # tool has async_execution set
        has_async_tools = any(
            async_execution_tools.get(name, False)
            for name in tool_functions
            if enabled_tools.get(name, True)
        )
        if has_async_tools:
            try:
                toolkit.register_tool_function(
                    toolkit.view_task,
                    namesake_strategy=namesake_strategy,
                )
                toolkit.register_tool_function(
                    toolkit.wait_task,
                    namesake_strategy=namesake_strategy,
                )
                toolkit.register_tool_function(
                    toolkit.cancel_task,
                    namesake_strategy=namesake_strategy,
                )
                logger.debug(
                    "Registered background task management tools "
                    "(view_task, wait_task, cancel_task)",
                )
            except Exception as e:
                logger.warning(
                    f"Failed to register task management tools: {e}",
                )

        return toolkit

    def _register_skills(self, toolkit: Toolkit) -> None:
        """Load and register skills from workspace directory.

        Uses the registry-backed skill resolver to determine effective
        skills for the current channel.

        Args:
            toolkit: Toolkit to register skills to
        """
        sc_cfg = getattr(self._agent_config, "skillclaw_capture", None)
        if (
            sc_cfg
            and getattr(sc_cfg, "enabled", False)
            and getattr(sc_cfg, "inject_catalog", True)
            and getattr(sc_cfg, "disable_native_skill_prompt", True)
        ):
            logger.info(
                "Skipping native skill prompt registration; SkillClaw "
                "capture hook will inject the SkillClaw catalog",
            )
            return

        workspace_dir = self._workspace_dir or WORKING_DIR

        ensure_skills_initialized(workspace_dir)

        request_context = getattr(self, "_request_context", {})
        channel_name = request_context.get("channel", "console")

        effective_skills = resolve_effective_skills(
            workspace_dir,
            channel_name,
        )

        working_skills_dir = get_workspace_skills_dir(Path(workspace_dir))

        for skill_name in effective_skills:
            skill_dir = working_skills_dir / skill_name
            if skill_dir.exists():
                try:
                    toolkit.register_agent_skill(str(skill_dir))
                    logger.debug("Registered skill: %s", skill_name)
                except Exception as e:
                    logger.error(
                        "Failed to register skill '%s': %s",
                        skill_name,
                        e,
                    )

    def _build_sys_prompt(self) -> str:
        """Build system prompt from working dir files and env context.

        Returns:
            Complete system prompt string
        """
        # Get agent_id from request_context
        agent_id = (
            self._request_context.get("agent_id")
            if self._request_context
            else None
        )

        # Check if heartbeat is enabled in agent config
        heartbeat_enabled = False
        if (
            hasattr(self._agent_config, "heartbeat")
            and self._agent_config.heartbeat is not None
        ):
            heartbeat_enabled = self._agent_config.heartbeat.enabled

        sys_prompt = build_system_prompt_from_working_dir(
            working_dir=self._workspace_dir,
            agent_id=agent_id,
            heartbeat_enabled=heartbeat_enabled,
            language=self._language,
            memory_manager=self.memory_manager,
        )
        logger.debug("System prompt:\n%s...", sys_prompt[:100])

        # Inject multimodal capability awareness
        multimodal_hint = build_multimodal_hint()
        if multimodal_hint:
            sys_prompt = sys_prompt + "\n\n" + multimodal_hint

        if self._env_context is not None:
            sys_prompt = sys_prompt + "\n\n" + self._env_context

        return sys_prompt

    def _register_hooks(self) -> None:
        """Register pre-reasoning and pre-acting hooks."""
        # Bootstrap hook - checks BOOTSTRAP.md on first interaction
        # Use workspace_dir if available, else fallback to WORKING_DIR
        working_dir = (
            self._workspace_dir if self._workspace_dir else WORKING_DIR
        )
        bootstrap_hook = BootstrapHook(
            working_dir=working_dir,
            language=self._language,
        )
        self.register_instance_hook(
            hook_type="pre_reasoning",
            hook_name="bootstrap_hook",
            hook=bootstrap_hook.__call__,
        )
        logger.debug("Registered bootstrap hook")

        # Context manager hooks - delegate compaction / tool-result pruning
        # to the context manager's lifecycle methods
        if self.context_manager is not None:
            self.register_instance_hook(
                hook_type="pre_reply",
                hook_name="context_pre_reply",
                hook=self.context_manager.pre_reply,
            )
            self.register_instance_hook(
                hook_type="pre_reasoning",
                hook_name="context_pre_reasoning",
                hook=self.context_manager.pre_reasoning,
            )
            self.register_instance_hook(
                hook_type="post_acting",
                hook_name="context_post_acting",
                hook=self.context_manager.post_acting,
            )
            self.register_instance_hook(
                hook_type="post_reply",
                hook_name="context_post_reply",
                hook=self.context_manager.post_reply,
            )
            logger.debug("Registered context manager hooks")

        # MemPalace hooks — config-driven registration
        mp_cfg = getattr(self._agent_config, "mempalace", None)
        if mp_cfg and mp_cfg.enabled:
            try:
                from .hooks.mempalace_diary import (
                    MemPalacePreCompactHook,
                    MemPalaceIntervalHook,
                    MemPalacePreReplyHook,
                )

                registered = []
                _session_id = self._request_context.get(
                    "session_id",
                    "default",
                )
                if mp_cfg.precompact_save.enabled:
                    hook = MemPalacePreCompactHook(
                        compact_threshold=mp_cfg.precompact_save.threshold,
                    )
                    self.register_instance_hook(
                        hook_type="pre_reasoning",
                        hook_name="mempalace_precompact",
                        hook=hook.__call__,
                    )
                    registered.append("precompact")
                if mp_cfg.interval_save.enabled:
                    hook = MemPalaceIntervalHook(
                        working_dir=working_dir,
                        write_interval=mp_cfg.interval_save.write_interval,
                        session_id=_session_id,
                    )
                    self.register_instance_hook(
                        hook_type="post_reasoning",
                        hook_name="mempalace_interval",
                        hook=hook.__call__,
                    )
                    registered.append("interval")
                if mp_cfg.pre_reply_save:
                    hook = MemPalacePreReplyHook(
                        working_dir=working_dir,
                        session_id=_session_id,
                    )
                    self.register_instance_hook(
                        hook_type="pre_reply",
                        hook_name="mempalace_prereply",
                        hook=hook.__call__,
                    )
                    registered.append("pre_reply")
                # L2 Room Recall — auto-inject relevant context
                if getattr(mp_cfg, "l2_recall", True):
                    try:
                        from .hooks.mempalace_recall import MemPalaceRecallHook

                        recall_hook = MemPalaceRecallHook()
                        self.register_instance_hook(
                            hook_type="pre_reasoning",
                            hook_name="mempalace_l2_recall",
                            hook=recall_hook.__call__,
                        )
                        registered.append("l2_recall")
                    except ImportError as e:
                        logger.warning("L2 Recall hook not available: %s", e)
                logger.info(
                    "MemPalace hooks registered: %s",
                    ", ".join(registered),
                )
            except ImportError as e:
                logger.warning("MemPalace hooks not available: %s", e)
        else:
            logger.debug("MemPalace hooks disabled")

        # Session WAL — write-ahead log for crash recovery.  Scoped by
        # session_id so a crashed tool on one channel/chat can't
        # trigger recovery on an unrelated channel/chat for the same
        # agent (see the docstring in ``tool_wal.py`` for the failure
        # mode this prevents).
        wal_enabled = (
            mp_cfg.session_wal if (mp_cfg and mp_cfg.enabled) else False
        )
        if wal_enabled:
            try:
                from .hooks.tool_wal import (
                    SessionWAL,
                    ToolWALPreActingHook,
                    ToolWALPostActingHook,
                    ReasoningWALHook,
                )

                wal = SessionWAL(
                    working_dir=working_dir,
                    session_id=_session_id,
                )
                self.register_instance_hook(
                    hook_type="pre_acting",
                    hook_name="wal_pre_acting",
                    hook=ToolWALPreActingHook(wal).__call__,
                )
                self.register_instance_hook(
                    hook_type="post_acting",
                    hook_name="wal_post_acting",
                    hook=ToolWALPostActingHook(wal).__call__,
                )
                self.register_instance_hook(
                    hook_type="post_reasoning",
                    hook_name="wal_post_reasoning",
                    hook=ReasoningWALHook(wal).__call__,
                )
                logger.info(
                    "Session WAL hooks registered (session=%s)",
                    _session_id,
                )
            except ImportError as e:
                logger.warning(f"Session WAL hooks not available: {e}")

        # SkillClaw session capture — port of the upstream proxy's
        # record-writer into a ``pre_reasoning`` hook so evolve_server
        # can feed on CoPaw sessions without us running the proxy.
        sc_cfg = getattr(self._agent_config, "skillclaw_capture", None)
        if sc_cfg and sc_cfg.enabled:
            try:
                from .hooks.skillclaw_capture import SkillClawCaptureHook

                _sc_session_id = self._request_context.get(
                    "session_id",
                    "default",
                )
                _sc_channel = self._request_context.get(
                    "channel_name",
                    "all",
                )
                sc_hook = SkillClawCaptureHook(
                    records_dir=sc_cfg.records_dir,
                    session_id=_sc_session_id,
                    session_id_prefix=sc_cfg.session_id_prefix,
                    mode=sc_cfg.mode,
                    ingest_url=sc_cfg.ingest_url,
                    ingest_api_key=sc_cfg.ingest_api_key,
                    workspace_dir=working_dir,
                    channel_name=_sc_channel,
                    inject_catalog=sc_cfg.inject_catalog,
                    skills_dir=sc_cfg.skills_dir,
                    skills_public_root=sc_cfg.skills_public_root,
                    max_skills_prompt_chars=sc_cfg.max_skills_prompt_chars,
                    read_tool_name=sc_cfg.read_tool_name,
                )
                self.register_instance_hook(
                    hook_type="pre_reasoning",
                    hook_name="skillclaw_capture",
                    hook=sc_hook.pre_reasoning,
                )
                self.register_instance_hook(
                    hook_type="post_reasoning",
                    hook_name="skillclaw_capture",
                    hook=sc_hook.post_reasoning,
                )
                self.register_instance_hook(
                    hook_type="post_acting",
                    hook_name="skillclaw_capture",
                    hook=sc_hook.post_acting,
                )
                logger.info(
                    "SkillClaw capture hook registered (session=%s, "
                    "records=%s)",
                    sc_hook._session_id,
                    sc_hook._path,
                )
            except ImportError as e:
                logger.warning("SkillClaw capture hook not available: %s", e)

    def rebuild_sys_prompt(self) -> None:
        """Rebuild and replace the system prompt.

        Useful after load_session_state to ensure the prompt reflects
        the latest AGENTS.md / SOUL.md / PROFILE.md on disk.

        Updates both self._sys_prompt and the first system-role
        message stored in self.memory.content (if one exists).
        """
        self._sys_prompt = self._build_sys_prompt()

        if self.memory is None:
            logger.warning(
                "rebuild_sys_prompt: self.memory is None, "
                "skipping in-memory system prompt update.",
            )
            return

        for msg, _marks in self.memory.content:
            if msg.role == "system":
                msg.content = self.sys_prompt
            break

    async def register_mcp_clients(
        self,
        namesake_strategy: NamesakeStrategy = "skip",
    ) -> None:
        """Register MCP clients on this agent's toolkit after construction.

        Args:
            namesake_strategy: Strategy to handle namesake tool functions.
                Options: "override", "skip", "raise", "rename"
                (default: "skip")
        """
        for i, client in enumerate(self._mcp_clients):
            client_name = getattr(client, "name", repr(client))
            try:
                await self.toolkit.register_mcp_client(
                    client,
                    namesake_strategy=namesake_strategy,
                    execution_timeout=client.read_timeout_seconds,
                )
            except (ClosedResourceError, asyncio.CancelledError) as error:
                if self._should_propagate_cancelled_error(error):
                    raise
                logger.warning(
                    "MCP client '%s' session interrupted while listing tools; "
                    "trying recovery",
                    client_name,
                )
                recovered_client = await self._recover_mcp_client(client)
                if recovered_client is not None:
                    self._mcp_clients[i] = recovered_client
                    try:
                        await self.toolkit.register_mcp_client(
                            recovered_client,
                            namesake_strategy=namesake_strategy,
                            execution_timeout=client.read_timeout_seconds,
                        )
                        continue
                    except asyncio.CancelledError as recover_error:
                        if self._should_propagate_cancelled_error(
                            recover_error,
                        ):
                            raise
                        logger.warning(
                            "MCP client '%s' registration cancelled after "
                            "recovery, skipping",
                            client_name,
                        )
                    except Exception as e:  # pylint: disable=broad-except
                        logger.warning(
                            "MCP client '%s' still unavailable after "
                            "recovery, skipping: %s",
                            client_name,
                            e,
                        )
                else:
                    logger.warning(
                        "MCP client '%s' recovery failed, skipping",
                        client_name,
                    )
            except Exception as e:  # pylint: disable=broad-except
                logger.warning(
                    "Failed to register MCP client '%s', skipping: %s",
                    client_name,
                    e,
                    exc_info=True,
                )

    async def _recover_mcp_client(self, client: Any) -> Any | None:
        """Recover MCP client from broken session and return healthy client."""
        if await self._reconnect_mcp_client(client):
            return client

        rebuilt_client = self._rebuild_mcp_client(client)
        if rebuilt_client is None:
            return None

        if await self._reconnect_mcp_client(rebuilt_client):
            return self._reuse_shared_client_reference(
                original_client=client,
                rebuilt_client=rebuilt_client,
            )

        return None

    @staticmethod
    def _reuse_shared_client_reference(
        original_client: Any,
        rebuilt_client: Any,
    ) -> Any:
        """Keep manager-shared client reference stable after rebuild."""
        original_dict = getattr(original_client, "__dict__", None)
        rebuilt_dict = getattr(rebuilt_client, "__dict__", None)
        if isinstance(original_dict, dict) and isinstance(rebuilt_dict, dict):
            original_dict.update(rebuilt_dict)
            return original_client
        return rebuilt_client

    @staticmethod
    def _should_propagate_cancelled_error(error: BaseException) -> bool:
        """Only swallow MCP-internal cancellations, not task cancellation."""
        if not isinstance(error, asyncio.CancelledError):
            return False

        task = asyncio.current_task()
        if task is None:
            return False

        cancelling = getattr(task, "cancelling", None)
        if callable(cancelling):
            return cancelling() > 0

        # Python < 3.11: Task.cancelling() is unavailable.
        # Fall back to propagating CancelledError to avoid swallowing
        # genuine task cancellations when we cannot inspect the state.
        return True

    @staticmethod
    async def _reconnect_mcp_client(
        client: Any,
        timeout: float = 60.0,
    ) -> bool:
        """Best-effort reconnect for stateful MCP clients."""
        close_fn = getattr(client, "close", None)
        if callable(close_fn):
            try:
                await close_fn()
            except asyncio.CancelledError:  # pylint: disable=try-except-raise
                raise
            except Exception:  # pylint: disable=broad-except
                pass

        connect_fn = getattr(client, "connect", None)
        if not callable(connect_fn):
            return False

        try:
            await asyncio.wait_for(connect_fn(), timeout=timeout)
            return True
        except asyncio.CancelledError:  # pylint: disable=try-except-raise
            raise
        except asyncio.TimeoutError:
            return False
        except Exception:  # pylint: disable=broad-except
            return False

    @staticmethod
    def _rebuild_mcp_client(client: Any) -> Any | None:
        """Rebuild a fresh MCP client instance from stored config metadata."""
        rebuild_info = getattr(client, "_qwenpaw_rebuild_info", None)
        if not isinstance(rebuild_info, dict):
            return None

        transport = rebuild_info.get("transport")
        name = rebuild_info.get("name")

        try:
            if transport == "stdio":
                rebuilt_client = StdIOStatefulClient(
                    name=name,
                    command=rebuild_info.get("command"),
                    args=rebuild_info.get("args", []),
                    env=rebuild_info.get("env", {}),
                    cwd=rebuild_info.get("cwd"),
                    tool_call_timeout=rebuild_info.get("tool_call_timeout"),
                )
                setattr(rebuilt_client, "_qwenpaw_rebuild_info", rebuild_info)
                return rebuilt_client

            raw_headers = rebuild_info.get("headers") or {}
            headers = (
                {k: os.path.expandvars(v) for k, v in raw_headers.items()}
                if raw_headers
                else None
            )
            rebuilt_client = HttpStatefulClient(
                name=name,
                transport=transport,
                url=rebuild_info.get("url"),
                headers=headers,
                tool_call_timeout=rebuild_info.get("tool_call_timeout"),
            )
            setattr(rebuilt_client, "_qwenpaw_rebuild_info", rebuild_info)
            return rebuilt_client
        except Exception:  # pylint: disable=broad-except
            return None

    # ------------------------------------------------------------------
    # Media-block fallback: strip unsupported media blocks (image, audio,
    # video) from memory and retry when the model rejects them.
    # ------------------------------------------------------------------

    _MEDIA_BLOCK_TYPES = {"image", "audio", "video"}

    # ------------------------------------------------------------------
    # Plan gate: block non-create_plan tools when /plan gate is active
    # ------------------------------------------------------------------

    _PLAN_TOOLS_WITH_JSON_ARGS = frozenset(
        {
            "create_plan",
            "revise_current_plan",
        },
    )
    _PLAN_JSON_KEYS = ("subtask", "subtasks")

    @staticmethod
    def _fix_stringified_json_args(tool_call) -> None:
        """Parse JSON-string arguments that models sometimes produce for
        nested objects (e.g. ``subtask``).  Modifies *tool_call* in place."""
        import json as _json

        inp = tool_call.get("input")
        if not isinstance(inp, dict):
            return
        for key in QwenPawAgent._PLAN_JSON_KEYS:
            val = inp.get(key)
            if isinstance(val, str):
                try:
                    inp[key] = _json.loads(val)
                except (ValueError, TypeError):
                    pass
            elif isinstance(val, list):
                for i, item in enumerate(val):
                    if isinstance(item, str):
                        try:
                            val[i] = _json.loads(item)
                        except (ValueError, TypeError):
                            pass

    async def _acting(self, tool_call) -> dict | None:
        """Check plan tool gate before delegating to ToolGuardMixin."""
        from ..plan.hints import check_plan_tool_gate

        tool_name = str(tool_call.get("name", ""))

        if tool_name in self._PLAN_TOOLS_WITH_JSON_ARGS:
            self._fix_stringified_json_args(tool_call)

        nb = getattr(self, "plan_notebook", None)

        # Pre-lock BEFORE executing create_plan / revise_current_plan so that
        # parallel tool calls (asyncio.gather) cannot slip an execution
        # tool past the gate before the lock is set.
        # pylint: disable=protected-access
        if nb is not None and tool_name in {
            "create_plan",
            "revise_current_plan",
        }:
            nb._plan_awaiting_user_confirm = True

        if nb is not None:
            err = check_plan_tool_gate(nb, tool_name)
            if err:
                from agentscope.message import ToolResultBlock

                tool_res_msg = Msg(
                    "system",
                    [
                        ToolResultBlock(
                            type="tool_result",
                            id=tool_call["id"],
                            name=tool_name,
                            output=[{"type": "text", "text": err}],
                        ),
                    ],
                    "system",
                )
                await self.print(tool_res_msg, True)
                await self.memory.add(tool_res_msg)
                return None

        result = await super()._acting(tool_call)

        if nb is not None and tool_name in {
            "create_plan",
            "revise_current_plan",
        }:
            # Force the next post-plan reasoning pass to be text-only.  This
            # prevents models from emitting other tools in the same turn
            # run before the user has confirmed the plan or modified it.
            # pylint: disable=protected-access
            nb._plan_text_only_after_mutation = True

        return result

    _AUTO_CONTINUE_MAX_EXTRA = 2
    _AUTO_CONTINUE_TAIL_CHARS = 600

    _AUTO_CONTINUE_HINT_EN = (
        "<system-hint>"
        "Your previous assistant turn had text only (no tool calls). "
        "Use the trailing excerpt in <previous-assistant-tail> (if present) "
        "plus the conversation to decide in this **reasoning** step: if the "
        "user's task still needs tools, emit tool_use now; if it is fully "
        "done, reply with a short text only (no tools). "
        "Do not stop with plans or code fences alone when tools are still "
        "needed."
        "</system-hint>"
    )
    _AUTO_CONTINUE_HINT_ZH = (
        "<system-hint>"
        "上轮助手仅文字、未调工具。请结合上下文与 <previous-assistant-tail> "
        "（若有）在本轮推理中判断：仍需执行则立刻 tool；已完结则简短收尾。"
        "需要操作时勿只输出计划或代码块。"
        "</system-hint>"
    )

    def _auto_continue_system_hint(self) -> str:
        """Pick hint by agent language (zh vs others)."""
        raw_lang = getattr(self._agent_config, "language", None)
        lang = (raw_lang or "").strip().lower()
        if lang == "zh":
            return self._AUTO_CONTINUE_HINT_ZH
        return self._AUTO_CONTINUE_HINT_EN

    @staticmethod
    def _auto_continue_tail_context(msg: Msg, max_chars: int) -> str:
        """Assistant text suffix for hint (fixed cut, not sentence NLP)."""
        raw = msg.get_text_content() if msg is not None else ""
        text = (raw or "").strip()
        if not text:
            return ""
        if len(text) <= max_chars:
            return text
        return text[-max_chars:].lstrip()

    async def _drain_pending_steer_messages(self) -> list[Msg]:
        """Insert console steer messages queued for this active chat."""
        tracker = self._task_tracker
        request_context = getattr(self, "_request_context", {}) or {}
        chat_id = request_context.get("chat_id")
        if tracker is None or not chat_id:
            return []

        drain = getattr(tracker, "drain_pending_input", None)
        if not callable(drain):
            return []

        pending_items = await drain(chat_id)  # pylint: disable=not-callable
        if not pending_items:
            return []

        pending_msgs: list[Msg] = []
        for item in pending_items:
            if isinstance(item, Msg):
                pending_msgs.append(item)
            elif isinstance(item, list):
                pending_msgs.extend(
                    [msg for msg in item if isinstance(msg, Msg)],
                )
            elif item is not None:
                pending_msgs.append(Msg("user", str(item), "user"))

        if not pending_msgs:
            return []

        await process_file_and_media_blocks_in_message(pending_msgs)
        await self.memory.add(pending_msgs)
        logger.info(
            "Inserted %d pending steer message(s) into active chat %s",
            len(pending_msgs),
            str(chat_id)[:12],
        )
        return pending_msgs

    async def _auto_continue_if_text_only(
        self,
        msg: Msg,
        tool_choice: Literal["auto", "none", "required"] | None,
    ) -> Msg:
        """Nudge the model when it returns text-only mid-task.

        Injects a language-matched hint (with a trailing excerpt of the
        assistant text for self-review) and runs up to
        ``_AUTO_CONTINUE_MAX_EXTRA`` extra ``_reasoning`` passes until a
        tool_use appears or the cap is
        hit.  Uses the original ``tool_choice`` unchanged (no switching).
        If an extra pass still returns text-only, keep the prior response to
        avoid repeated duplicated answers.
        """
        from ..plan.hints import should_skip_auto_continue

        if msg is not None and not msg.has_content_blocks("tool_use"):
            pending_msgs = await self._drain_pending_steer_messages()
            if pending_msgs:
                logger.info(
                    "Steer mode: continuing current turn after %d "
                    "pending message(s)",
                    len(pending_msgs),
                )
                return await self._reasoning(tool_choice=tool_choice)

        nb = getattr(self, "plan_notebook", None)
        if should_skip_auto_continue(nb):
            return msg

        running = self._agent_config.running
        if not running.auto_continue_on_text_only:
            return msg
        if msg is None or msg.has_content_blocks("tool_use"):
            return msg

        extra = 0
        while extra < self._AUTO_CONTINUE_MAX_EXTRA:
            if msg.has_content_blocks("tool_use"):
                break
            extra += 1
            tail = self._auto_continue_tail_context(
                msg,
                self._AUTO_CONTINUE_TAIL_CHARS,
            )
            hint_body = self._auto_continue_system_hint()
            if tail:
                hint_body += (
                    "\n\n<previous-assistant-tail>\n"
                    f"{tail}\n"
                    "</previous-assistant-tail>"
                )
            logger.info(
                "Auto-continue: text-only (%d/%d); hint + _reasoning "
                "tool_choice=%r",
                extra,
                self._AUTO_CONTINUE_MAX_EXTRA,
                tool_choice,
            )
            hint_msg = Msg("user", hint_body, "user")
            await self.memory.add(hint_msg, marks=_MemoryMark.HINT)
            try:
                next_msg = await super()._reasoning(tool_choice=tool_choice)
            except Exception:
                logger.warning(
                    "Auto-continue extra _reasoning failed; "
                    "keeping prior response",
                    exc_info=True,
                )
                break
            if next_msg.has_content_blocks("tool_use"):
                msg = next_msg
                continue
            logger.info(
                "Auto-continue extra _reasoning still text-only; "
                "keeping prior response",
            )
            break

        return msg

    def _get_model_key(self) -> str | None:
        """Return the capability-cache key for the active model."""
        model = getattr(self, "model", None)
        return getattr(model, "model_key", None)

    def _model_rejects_media(self) -> bool:
        """Check the capability cache for a learned ``rejects_media`` flag."""
        key = self._get_model_key()
        if key is None:
            return False
        return get_capability_cache().get(key, "rejects_media", False)

    def _proactive_strip_media_blocks(self) -> int:
        """Proactively strip media blocks from memory before model call.

        Only called when the active model does not support multimodal.
        Returns the number of blocks stripped.
        """
        return self._strip_media_blocks_from_memory()

    def _uses_request_time_media_normalization(self) -> bool:
        """Return True when request-time normalization can handle media."""
        return getattr(self, "formatter", None) is not None

    def _set_formatter_media_strip(self, enabled: bool) -> None:
        """Toggle request-time media stripping on the active formatter."""
        formatter = getattr(self, "formatter", None)
        if formatter is None:
            return
        setattr(formatter, "_qwenpaw_force_strip_media", enabled)

    def _install_chat_log_wrapper(self) -> None:
        """Wrap ``self.memory.add`` so every non-transient call also
        appends to ``<workspace>/chats/<chat_id>.jsonl``.

        Idempotent — calling twice is a no-op.  Stores the unwrapped
        original on ``self._original_memory_add`` so reconcile-time
        re-injection can bypass the wrapper (otherwise replaying log
        entries would re-append them and the log would grow without
        bound).

        ``chat_id`` is read from ``self._request_context`` lazily at
        each call — same instance can serve multiple chat_ids across
        successive ``reply()`` invocations as the runner rotates the
        request context.  No chat_id ⇒ no append (stand-alone /
        warm-up / smoke calls don't pollute logs).
        """
        if getattr(self, "_chat_log_wrapped", False):
            return
        self._chat_log_wrapped = True
        self._original_memory_add = self.memory.add  # type: ignore[attr-defined]

        async def logged_add(memories, marks=None, **kwargs):
            result = await self._original_memory_add(
                memories,
                marks=marks,
                **kwargs,
            )
            try:
                req_ctx = getattr(self, "_request_context", None) or {}
                session_id = req_ctx.get("session_id") or ""
                user_id = req_ctx.get("user_id") or ""
                if not session_id:
                    return result
                # HINT-marked entries are agentscope's transient nudges
                # — skip them so the log only carries real conversation.
                mark_str = (
                    str(marks).lower()
                    if marks is not None
                    and not isinstance(
                        marks,
                        (list, tuple, set),
                    )
                    else " ".join(str(m).lower() for m in (marks or []))
                )
                if "hint" in mark_str:
                    return result
                workspace = self._workspace_dir or WORKING_DIR
                chat_log_append(
                    workspace,
                    session_id,
                    memories,
                    marks,
                    user_id=user_id,
                )
            except Exception as e:  # pylint: disable=broad-exception-caught
                logger.debug(
                    "chat_log: append wrapper swallowed error: %s",
                    e,
                )
            return result

        self.memory.add = logged_add  # type: ignore[attr-defined]

    @staticmethod
    def _resolve_watermark_path(
        workspace: Path | str,
        session_id: str,
        user_id: str,
        channel: str,
    ) -> Path:
        """Pick the session.json path whose mtime is the right watermark.

        Mirrors ``SafeJSONSession._get_save_path``: when ``channel`` is
        set, sessions are written to ``sessions/{channel}/{fname}``; a
        legacy flat copy may also exist at ``sessions/{fname}`` from
        the pre-namespacing era but is no longer updated.  Reading
        mtime from that stale flat copy makes the watermark stale and
        causes every chat_log entry written since to be re-injected as
        "unpersisted" — the bug that surfaced as "context recovers
        full history after /compact".

        Pure function so the regression test can exercise it without
        constructing a real ReActAgent.
        """
        from ..app.runner.session import sanitize_filename

        safe_sid = sanitize_filename(session_id)
        safe_uid = sanitize_filename(user_id) if user_id else ""
        fname = (
            f"{safe_uid}_{safe_sid}.json" if safe_uid else f"{safe_sid}.json"
        )
        sessions_dir = Path(workspace) / "sessions"
        if channel:
            namespaced = sessions_dir / sanitize_filename(channel) / fname
            if namespaced.exists():
                return namespaced
            # First-turn fallback before migration copies the legacy file.
            legacy = sessions_dir / fname
            if legacy.exists():
                return legacy
            return namespaced  # let caller see "doesn't exist"
        return sessions_dir / fname

    async def _reconcile_chat_log_into_memory(self) -> None:
        """Inject log entries that were captured between the last
        successful ``save_session_state`` and the current process
        start.

        Called from ``reply()`` before ``super().reply()`` runs, so
        the agent sees the unpersisted prior turn in its prompt
        without us having to extend the boot path.

        No-op when:
        * No ``chat_id`` in request context (no log to read).
        * No log file yet (first turn for this chat).
        * Log entries all carry ``ts <= session.json mtime`` (nothing
          was lost — last save covered everything).
        * Every unpersisted entry's ``msg.id`` already happens to be
          in current memory (rare, but possible if reload + reconcile
          cross paths).

        Failures are logged and swallowed.  A reconcile that errors
        must NOT block the next reply — losing reconciliation is
        recoverable, blocking the agent is not.
        """
        try:
            req_ctx = getattr(self, "_request_context", None) or {}
            session_id = req_ctx.get("session_id") or ""
            user_id = req_ctx.get("user_id") or ""
            if not session_id:
                return
            workspace = self._workspace_dir or WORKING_DIR

            channel = req_ctx.get("channel") or ""
            session_path = self._resolve_watermark_path(
                workspace,
                session_id,
                user_id,
                channel,
            )

            memory_msg_ids: set[str] = set()
            for pair in self.memory.content or []:
                # ``content`` is list[(Msg, marks)] — we only care about
                # the Msg's id field for dedup.
                m = pair[0] if isinstance(pair, tuple) else pair
                if hasattr(m, "id") and m.id:
                    memory_msg_ids.add(m.id)

            unpersisted = chat_log_collect_unpersisted(
                workspace,
                session_id,
                session_path,
                memory_msg_ids,
                user_id=user_id,
            )
            if not unpersisted:
                return

            # Inject via the UNWRAPPED add — going through the wrapper
            # would re-append these entries to the log we just read.
            original_add = getattr(self, "_original_memory_add", None)
            if original_add is None:
                logger.debug(
                    "chat_log: reconcile skipped — no _original_memory_add",
                )
                return
            await original_add(unpersisted)
            logger.info(
                "chat_log: reconciled %d unpersisted entr%s into memory "
                "(session=%s)",
                len(unpersisted),
                "y" if len(unpersisted) == 1 else "ies",
                session_id,
            )
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.warning(
                "chat_log: reconcile swallowed error: %s",
                e,
            )

    @staticmethod
    def _filter_plan_tools(msg: Msg, nb: Any) -> Msg:
        """Arm `_plan_awaiting_user_confirm` before any tool runs.

        Race-prevention: when the assistant message carries `create_plan` /
        `revise_current_plan` alongside other ``tool_use`` blocks, callers
        of ``asyncio.gather`` may hit `_acting()`` on sibling tools before
        the mutation tool executes. Setting the lock here (before tools run)
        makes `check_plan_tool_gate` refuse non-plan-management tools while
        still returning a readable tool_result instead of stripping blocks.
        """
        if nb is None or not isinstance(msg.content, list):
            return msg
        mut = ("create_plan", "revise_current_plan")
        if any(
            isinstance(b, dict)
            and b.get("type") == "tool_use"
            and b.get("name", "") in mut
            for b in msg.content
        ):
            # pylint: disable-next=protected-access
            nb._plan_awaiting_user_confirm = True
        return msg

    # pylint: disable=too-many-branches
    async def _reasoning(
        self,
        tool_choice: Literal["auto", "none", "required"] | None = None,
    ) -> Msg:
        """Override reasoning with proactive media filtering and a
        durable error tombstone.

        Layers, outside-in:

        1. Outer wrapper: if reasoning raises (after retries below
           are exhausted), record an error stub in memory before
           re-raising.  Without this, an upstream HTTP error during
           the very first SSE event leaves only an empty assistant
           ``Msg(content=[])`` in memory (parent's ``finally`` adds
           it, but it's contentless) — the next turn has no record
           of what went wrong, so retries are blind.  Skips
           ``CancelledError`` to leave the interrupt path's
           fake-tool-result writes alone.

        2. Proactive media filter: strip media before calling when
           the model is not flagged multimodal, or the capability cache
           records a previous ``rejects_media`` finding for this model.

        3. Passive bad-request / media fallback: if the model call
           still 400s on media, strip remaining blocks and retry, then
           record the finding in the capability cache. If the model is
           marked as multimodal but still errors on media, log a warning
           about the possibly inaccurate capability flag.

        4. Plan gate: ``_filter_plan_tools`` pre-locks when the assistant
           schedules plan mutation tools; ``_plan_text_only_after_mutation``
           forces ``tool_choice="none"`` once so the model cannot issue
           execution tools immediately after ``create_plan`` / revise.

        Calls ``super()._reasoning`` to keep the ToolGuardMixin
        interception active.
        """
        try:
            await self._drain_pending_steer_messages()
            return await self._reasoning_with_media_fallback(tool_choice)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            await self._record_reasoning_failure(e, tool_choice)
            raise

    # pylint: disable=too-many-branches
    async def _reasoning_with_media_fallback(
        self,
        tool_choice: Literal["auto", "none", "required"] | None = None,
    ) -> Msg:
        """Inner reasoning path: media filtering + bad-request retry.

        Extracted so the outer ``_reasoning`` can wrap it cleanly with
        the error-tombstone path without entangling retry control flow.
        """
        # Plan gate (from upstream): when create_plan / revise_current_plan
        # ran in the previous turn, force tool_choice="none" once so the
        # model can't immediately issue execution tools without giving the
        # user a chance to review the plan.
        nb = getattr(self, "plan_notebook", None)
        if nb is not None and getattr(
            nb,
            "_plan_text_only_after_mutation",
            False,
        ):
            # pylint: disable=protected-access
            nb._plan_text_only_after_mutation = False
            tool_choice = "none"

        # --- Proactive filtering layer ---
        should_strip = (
            not get_active_model_supports_multimodal()
            or self._model_rejects_media()
        )
        if should_strip:
            if self._uses_request_time_media_normalization():
                self._set_formatter_media_strip(True)
                logger.debug(
                    "Formatter will strip media from copied messages "
                    "before reasoning.",
                )
            else:
                n = self._proactive_strip_media_blocks()
                if n > 0:
                    logger.warning(
                        "Proactively stripped %d media block(s) - "
                        "model does not support multimodal.",
                        n,
                    )

        # --- Passive fallback layer (existing logic) ---
        try:
            msg = await super()._reasoning(tool_choice=tool_choice)
        except ModelContextLengthExceededException as ctx_exc:
            # Z.AI streams a final empty chunk with
            # ``finish_reason="model_context_window_exceeded"`` instead of
            # raising an HTTP error.  The provider wrapper now surfaces
            # this as a typed exception (see openai_chat_model_compat).
            # Convert it into a user-visible apology reply so the channel
            # doesn't go silent; the user can then ``/new`` or ``/compact``.
            return await self._build_context_exceeded_reply(ctx_exc)
        except Exception as e:
            if not self._is_bad_request_or_media_error(e):
                raise

            model_key = self._get_model_key()

            if self._uses_request_time_media_normalization():
                if get_active_model_supports_multimodal():
                    logger.warning(
                        "Model marked multimodal but "
                        "rejected media. "
                        "Capability flag may be wrong.",
                    )
                self._set_formatter_media_strip(True)
                try:
                    logger.warning(
                        "_reasoning failed (%s). "
                        "Retrying with request-time media stripping.",
                        e,
                    )
                    msg = await super()._reasoning(tool_choice=tool_choice)
                    if model_key and not self._is_format_specific_media_error(
                        e,
                    ):
                        get_capability_cache().learn(
                            model_key,
                            "rejects_media",
                            True,
                        )
                    elif model_key:
                        logger.info(
                            "Not learning rejects_media for %s — error "
                            "looks file/format specific (%s)",
                            model_key,
                            str(e)[:160],
                        )
                    return msg
                finally:
                    self._set_formatter_media_strip(False)

            n_stripped = self._strip_media_blocks_from_memory()
            if n_stripped == 0:
                raise

            if get_active_model_supports_multimodal():
                logger.warning(
                    "Model marked multimodal but "
                    "rejected media. "
                    "Capability flag may be wrong.",
                )

            logger.warning(
                "_reasoning failed (%s). "
                "Stripped %d media block(s) from memory, retrying.",
                e,
                n_stripped,
            )
            msg = await super()._reasoning(tool_choice=tool_choice)
            if model_key and not self._is_format_specific_media_error(e):
                get_capability_cache().learn(
                    model_key,
                    "rejects_media",
                    True,
                )
            elif model_key:
                logger.info(
                    "Not learning rejects_media for %s — error "
                    "looks file/format specific (%s)",
                    model_key,
                    str(e)[:160],
                )
        finally:
            if should_strip and self._uses_request_time_media_normalization():
                self._set_formatter_media_strip(False)

        msg = self._filter_plan_tools(msg, nb)

        return await self._auto_continue_if_text_only(msg, tool_choice)

    async def _record_reasoning_failure(
        self,
        exc: Exception,
        tool_choice: Literal["auto", "none", "required"] | None,
    ) -> None:
        """Stamp memory with an audit entry describing the failed turn.

        Two scenarios to handle, distinguished by whether the parent's
        try/finally already wrote a placeholder ``assistant`` Msg:

        * **Handshake or first-chunk error** — parent's ``finally``
          ran with ``msg.content = []`` (no chunks ever yielded), so
          the last memory entry is a contentless assistant Msg.
          Replace its content in place so the agent's next prompt
          carries actual context, not an empty turn.

        * **Mid-stream error after partial yield** — the assistant
          Msg in memory holds whatever chunks 1..N did yield (text,
          thinking, tool_use).  Don't overwrite it; append a
          ``system``-role note instead so the partial work survives
          and the next turn knows the stream was cut.

        * **No assistant placeholder at all** (e.g. tool guard
          rejected before model call) — append a fresh ``system``
          note.

        Best-effort: any internal failure here is logged and
        swallowed — we MUST not mask the original exception by
        raising from a tombstone helper.
        """
        try:
            from agentscope.message import TextBlock

            err_text = (
                f"[reasoning failed: {type(exc).__name__}: "
                f"{str(exc)[:300]}; tool_choice={tool_choice}; "
                "next user turn will retry from this point]"
            )

            history = list(self.memory.content)  # [(msg, marks), ...]
            last_msg = None
            if history:
                last_pair = history[-1]
                if isinstance(last_pair, (list, tuple)) and last_pair:
                    last_msg = last_pair[0]

            is_empty_placeholder = (
                last_msg is not None
                and getattr(last_msg, "role", None) == "assistant"
                and (
                    not last_msg.content
                    or (
                        isinstance(last_msg.content, list)
                        and not last_msg.get_content_blocks("text")
                        and not last_msg.get_content_blocks("tool_use")
                        and not last_msg.get_content_blocks("thinking")
                    )
                )
            )

            if is_empty_placeholder:
                last_msg.content = [
                    TextBlock(type="text", text=err_text),
                ]
                logger.warning(
                    "_reasoning failed (%s); replaced empty assistant "
                    "placeholder in memory with error stub",
                    exc,
                )
                return

            stub = Msg(
                "system",
                [TextBlock(type="text", text=err_text)],
                "system",
            )
            await self.memory.add(stub)
            logger.warning(
                "_reasoning failed (%s); appended error stub to memory "
                "(prior assistant content preserved)",
                exc,
            )
        except Exception as stub_err:  # pylint: disable=broad-exception-caught
            logger.warning(
                "Failed to record reasoning failure to memory: %s "
                "(original error: %s)",
                stub_err,
                exc,
            )

    # User-facing text per agent language for context-window overflow.
    # Kept short and actionable so channels with strict character limits
    # (e.g. SMS-style fallbacks) still render the whole thing.
    _CONTEXT_EXCEEDED_TEXT = {
        "zh": ("對話歷史已超出模型上下文窗口，無法繼續推理。" "請輸入 /new 開新對話，或 /compact 壓縮現有歷史後再試。"),
        "en": (
            "The conversation has exceeded the model's context window. "
            "Please send /new to start a fresh chat, or /compact to "
            "compress the current history and retry."
        ),
        "ru": (
            "История диалога превысила контекстное окно модели. "
            "Отправьте /new для новой беседы или /compact для сжатия "
            "истории, затем повторите."
        ),
    }

    async def _build_context_exceeded_reply(
        self,
        exc: ModelContextLengthExceededException,
    ) -> Msg:
        """Convert a context-overflow exception into a user-visible reply.

        Z.AI's overflow signal is silent at the wire level; the upstream
        parser now raises ``ModelContextLengthExceededException`` (see
        ``openai_chat_model_compat``).  Without this handler the agent
        run would propagate the error and the channel would see an empty
        reply.  Instead, we:

        1. Replace the empty assistant placeholder that the parent's
           ``_reasoning`` finally-block added (when the stream raised
           mid-iteration) with a real TextBlock so the next turn's
           history doesn't carry an empty entry.
        2. Print the message with ``last=True`` so streaming channels
           emit it as the final chunk.
        3. Return it for the ReAct loop to use as ``reply_msg`` — no
           tool_use → the loop exits naturally.
        """
        from agentscope.message import (
            TextBlock,
        )  # local: avoid cycle on import

        lang = (self._language or "en").lower()
        text = self._CONTEXT_EXCEEDED_TEXT.get(
            lang,
            self._CONTEXT_EXCEEDED_TEXT["en"],
        )

        logger.warning(
            "Surfacing ModelContextLengthExceededException to user "
            "(model=%s, finish_reason=%s)",
            getattr(exc, "details", {}).get("model_name", "unknown")
            if hasattr(exc, "details")
            else "unknown",
            getattr(exc, "details", {}).get("finish_reason", "?")
            if hasattr(exc, "details")
            else "?",
        )

        # Try to mutate the empty placeholder the parent's `_reasoning`
        # finally already inserted, so memory doesn't keep an empty entry.
        msg: Msg | None = None
        try:
            history = list(self.memory.content)
            last_msg = None
            if history:
                last_pair = history[-1]
                if isinstance(last_pair, (list, tuple)) and last_pair:
                    last_msg = last_pair[0]

            is_empty_placeholder = (
                last_msg is not None
                and getattr(last_msg, "role", None) == "assistant"
                and (
                    not last_msg.content
                    or (
                        isinstance(last_msg.content, list)
                        and not last_msg.get_content_blocks("text")
                        and not last_msg.get_content_blocks("tool_use")
                        and not last_msg.get_content_blocks("thinking")
                    )
                )
            )

            if is_empty_placeholder:
                last_msg.content = [TextBlock(type="text", text=text)]
                msg = last_msg
        except Exception:  # pylint: disable=broad-exception-caught
            msg = None

        if msg is None:
            msg = Msg(
                self.name,
                [TextBlock(type="text", text=text)],
                "assistant",
            )
            try:
                await self.memory.add(msg)
            except Exception:  # pylint: disable=broad-exception-caught
                logger.warning(
                    "Failed to append context-exceeded reply to memory",
                    exc_info=True,
                )

        try:
            await self.print(msg, True)
        except Exception:  # pylint: disable=broad-exception-caught
            logger.warning(
                "Failed to print context-exceeded reply",
                exc_info=True,
            )

        return msg

    # pylint: disable=too-many-branches
    async def _summarizing(self) -> Msg:
        """Override summarizing with proactive media filtering,
        passive fallback, and tool_use block filtering.

        1. Proactive layer: if the model does not support multimodal
           **or** the capability cache records ``rejects_media``,
           strip media blocks *before* calling the model.
        2. Passive layer: if the model call still fails with a
           bad-request / media error, strip remaining blocks and retry,
           then record the finding in the capability cache.
        3. If the model IS marked as multimodal but still errors on
           media, log a warning about possibly inaccurate capability flag.

        Some models (e.g. kimi-k2.5) generate tool_use blocks even when
        no tools are provided.  We set ``_in_summarizing`` so that
        ``print`` can strip tool_use blocks from streaming chunks.
        """
        # --- Proactive filtering layer ---
        should_strip = (
            not get_active_model_supports_multimodal()
            or self._model_rejects_media()
        )
        if should_strip:
            if self._uses_request_time_media_normalization():
                self._set_formatter_media_strip(True)
                logger.debug(
                    "Formatter will strip media from copied messages "
                    "before summarizing.",
                )
            else:
                n = self._proactive_strip_media_blocks()
                if n > 0:
                    logger.warning(
                        "Proactively stripped %d media block(s) - "
                        "model does not support multimodal.",
                        n,
                    )

        # --- Passive fallback layer ---
        self._in_summarizing = True
        try:
            try:
                msg = await super()._summarizing()
            except ModelContextLengthExceededException as ctx_exc:
                # Summarizing also blows past the context window when
                # max_iters is reached on an already-bloated session.
                # Same surfacing path as ``_reasoning_with_media_fallback``.
                return await self._build_context_exceeded_reply(ctx_exc)
            except Exception as e:
                if not self._is_bad_request_or_media_error(e):
                    raise

                model_key = self._get_model_key()

                if self._uses_request_time_media_normalization():
                    if get_active_model_supports_multimodal():
                        logger.warning(
                            "Model marked multimodal but "
                            "rejected media. "
                            "Capability flag may be wrong.",
                        )
                    self._set_formatter_media_strip(True)
                    try:
                        logger.warning(
                            "_summarizing failed (%s). "
                            "Retrying with request-time media stripping.",
                            e,
                        )
                        msg = await super()._summarizing()
                        if (
                            model_key
                            and not self._is_format_specific_media_error(
                                e,
                            )
                        ):
                            get_capability_cache().learn(
                                model_key,
                                "rejects_media",
                                True,
                            )
                        elif model_key:
                            logger.info(
                                "Not learning rejects_media for %s — error "
                                "looks file/format specific (%s)",
                                model_key,
                                str(e)[:160],
                            )
                    finally:
                        self._set_formatter_media_strip(False)
                else:
                    n_stripped = self._strip_media_blocks_from_memory()
                    if n_stripped == 0:
                        raise

                    if get_active_model_supports_multimodal():
                        logger.warning(
                            "Model marked multimodal but "
                            "rejected media. "
                            "Capability flag may be wrong.",
                        )

                    logger.warning(
                        "_summarizing failed (%s). "
                        "Stripped %d media block(s) from memory, retrying.",
                        e,
                        n_stripped,
                    )
                    msg = await super()._summarizing()
                    if model_key and not self._is_format_specific_media_error(
                        e,
                    ):
                        get_capability_cache().learn(
                            model_key,
                            "rejects_media",
                            True,
                        )
                    elif model_key:
                        logger.info(
                            "Not learning rejects_media for %s — error "
                            "looks file/format specific (%s)",
                            model_key,
                            str(e)[:160],
                        )
        finally:
            self._in_summarizing = False
            if should_strip and self._uses_request_time_media_normalization():
                self._set_formatter_media_strip(False)

        return self._strip_tool_use_from_msg(msg)

    async def print(
        self,
        msg: Msg,
        last: bool = True,
        speech: Any = None,
    ) -> None:
        """Filter tool_use blocks during _summarizing before they hit the
        message queue, preventing the frontend from briefly rendering
        phantom tool calls that will never be executed.

        On the *final* streaming event (``last=True``), append the
        round-end notice so users see it immediately instead of only
        after a page refresh.  Intermediate events that become empty
        after filtering are silently skipped to avoid blank UI flashes.
        """

        if not getattr(self, "_in_summarizing", False):
            return await super().print(msg, last, speech=speech)

        original = msg.content
        modified = False

        if isinstance(original, list):
            filtered = [
                b
                for b in original
                if not (isinstance(b, dict) and b.get("type") == "tool_use")
            ]
            if not filtered and not last:
                return
            if len(filtered) != len(original) or last:
                msg.content = filtered
                if last:
                    msg.content.append(
                        {"type": "text", "text": self._ROUND_END_NOTICE},
                    )
                modified = True
        elif isinstance(original, str) and last:
            msg.content = original + self._ROUND_END_NOTICE
            modified = True
        if modified:
            try:
                return await super().print(msg, last, speech=speech)
            finally:
                msg.content = original
        return await super().print(msg, last, speech=speech)

    _ROUND_END_NOTICE = (
        "\n\n---\n"
        "本轮调用已达最大次数，回复已终止，请继续输入。\n"
        "Maximum iterations reached for this round. "
        "Please send a new message to continue."
    )

    @staticmethod
    def _strip_tool_use_from_msg(msg: Msg) -> Msg:
        """Remove tool_use blocks from a message and append a user notice.

        When _summarizing is called without tools, some models still
        return tool_use blocks.  Those blocks can never be executed, so
        strip them and append a bilingual notice telling the user this
        round of calls has ended.
        """
        if isinstance(msg.content, str):
            msg.content += QwenPawAgent._ROUND_END_NOTICE
            return msg

        filtered = [
            block
            for block in msg.content
            if not (
                isinstance(block, dict) and block.get("type") == "tool_use"
            )
        ]

        n_removed = len(msg.content) - len(filtered)
        if n_removed:
            logger.debug(
                "Stripped %d tool_use block(s) from _summarizing response",
                n_removed,
            )

        filtered.append(
            {"type": "text", "text": QwenPawAgent._ROUND_END_NOTICE},
        )
        msg.content = filtered
        return msg

    @staticmethod
    def _is_bad_request_or_media_error(exc: Exception) -> bool:
        """Return True for 400-class or media-related model errors.

        Targets bad-request (400) errors because unsupported media
        content typically causes request validation failures.  Keyword
        matching provides an extra safety net for providers that use
        non-standard status codes.
        """
        status = getattr(exc, "status_code", None)
        if status == 400:
            return True

        error_str = str(exc).lower()
        keywords = [
            "image",
            "audio",
            "video",
            "vision",
            "multimodal",
            "image_url",
        ]
        return any(kw in error_str for kw in keywords)

    # Markers that mean "this specific *file/format* tripped the
    # decoder", not "this model cannot handle any media".  Learning
    # ``rejects_media=True`` from these would poison the capability
    # cache for the rest of the process — observed on 2026-05-12 when
    # an animated WebP sticker (z.ai code 1210) caused every later
    # view_image call to be silently stripped before reaching the
    # vision model.  Keep this list lowercase; the matcher casefolds.
    _FORMAT_SPECIFIC_MEDIA_ERROR_MARKERS = (
        # z.ai / Zhipu — image format / parse / decode errors
        "1210",
        "图片输入格式",
        "图片解析",
        "image format",
        "image input format",
        "format/parse",
        "format error",
        "parse error",
        "decode error",
        "decoding error",
        "unsupported image format",
        # Mimo / Qwen-VL video codec / container rejections
        "multimodal data is corrupted",
        "only mp4/wmv/mov/avi",
        "invalid video format",
        "unsupported video format",
        "invalid image format",
        # Generic codec markers
        "codec",
        "container",
    )

    @classmethod
    def _is_format_specific_media_error(cls, exc: Exception) -> bool:
        """Return True iff the model's rejection looks file/format
        specific (so a strip-and-retry success does NOT prove the
        model can't handle media — only that *this* file's format
        was unsupported).  Callers should skip the
        ``rejects_media=True`` learn() in that case.
        """
        text = str(exc).casefold()
        return any(
            marker.casefold() in text
            for marker in cls._FORMAT_SPECIFIC_MEDIA_ERROR_MARKERS
        )

    def _strip_media_blocks_from_memory(self) -> int:
        """Remove media blocks (image/audio/video) from all messages.

        Also strips media blocks nested inside ToolResultBlock outputs.
        Inserts placeholder text when stripping leaves content empty to
        avoid malformed API requests.

        Returns:
            Total number of media blocks removed.
        """
        media_types = self._MEDIA_BLOCK_TYPES
        total_stripped = 0

        for msg, _marks in self.memory.content:
            if not isinstance(msg.content, list):
                continue

            new_content = []
            stripped_this_message = 0
            for block in msg.content:
                if (
                    isinstance(block, dict)
                    and block.get("type") in media_types
                ):
                    total_stripped += 1
                    stripped_this_message += 1
                    continue

                if (
                    isinstance(block, dict)
                    and block.get("type") == "tool_result"
                    and isinstance(block.get("output"), list)
                ):
                    original_len = len(block["output"])
                    block["output"] = [
                        item
                        for item in block["output"]
                        if not (
                            isinstance(item, dict)
                            and item.get("type") in media_types
                        )
                    ]
                    stripped_count = original_len - len(block["output"])
                    total_stripped += stripped_count
                    stripped_this_message += stripped_count
                    if stripped_count > 0 and not block["output"]:
                        block["output"] = MEDIA_UNSUPPORTED_PLACEHOLDER

                new_content.append(block)

            if not new_content and stripped_this_message > 0:
                new_content.append(
                    {
                        "type": "text",
                        "text": MEDIA_UNSUPPORTED_PLACEHOLDER,
                    },
                )

            msg.content = new_content

        return total_stripped

    # pylint: disable=protected-access
    async def reply(
        self,
        msg: Msg | list[Msg] | None = None,
        structured_model: Type[BaseModel] | None = None,
    ) -> Msg:
        """Override reply to process file blocks and handle commands.

        Args:
            msg: Input message(s) from user
            structured_model: Optional pydantic model for structured output

        Returns:
            Response message
        """
        # Set workspace_dir and recent_max_bytes in context for tool functions
        from ..config.context import (
            set_current_workspace_dir,
            set_current_recent_max_bytes,
            set_current_shell_command_timeout,
            set_current_shell_command_executable,
        )

        set_current_workspace_dir(self._workspace_dir)
        light_ctx = self._agent_config.running.light_context_config
        pruning_config = light_ctx.tool_result_pruning_config
        set_current_recent_max_bytes(
            pruning_config.pruning_recent_msg_max_bytes,
        )
        set_current_shell_command_timeout(
            self._agent_config.running.shell_command_timeout,
        )
        set_current_shell_command_executable(
            self._agent_config.running.shell_command_executable or None,
        )

        # Process file and media blocks in messages
        if msg is not None:
            await process_file_and_media_blocks_in_message(msg)

        # Check if message is a system command
        last_msg = msg[-1] if isinstance(msg, list) else msg
        query = (
            last_msg.get_text_content() if isinstance(last_msg, Msg) else None
        )

        if self.command_handler.is_command(query):
            logger.info(f"Received command: {query}")
            msg = await self.command_handler.handle_command(query)
            await self.print(msg)
            return msg

        # Normal message processing
        logger.info("QwenPawAgent.reply: max_iters=%s", self.max_iters)

        request_context = getattr(self, "_request_context", {}) or {}
        channel_name = request_context.get("channel", "console")
        workspace_dir = Path(self._workspace_dir or WORKING_DIR)

        # Pull anything captured between the last save_session_state and
        # now (e.g. SIGKILL casualties or Phase A tombstones whose
        # session.json snapshot didn't make it to disk) back into
        # ``self.memory`` BEFORE agentscope's ``super().reply()``
        # processes the new user input.  This way the upcoming prompt
        # carries the full prior turn even when ``session.json`` is
        # behind the actual conversation.
        await self._reconcile_chat_log_into_memory()

        with apply_skill_config_env_overrides(workspace_dir, channel_name):
            return await super().reply(
                msg=msg,
                structured_model=structured_model,
            )

    async def interrupt(self, msg: Msg | list[Msg] | None = None) -> None:
        """Interrupt the current reply process and wait for cleanup."""
        if self._reply_task and not self._reply_task.done():
            task = self._reply_task
            task.cancel(msg)
            try:
                await task
            except asyncio.CancelledError:
                if not task.cancelled():
                    raise
            except Exception:
                logger.warning(
                    "Exception occurred during interrupt cleanup",
                    exc_info=True,
                )
