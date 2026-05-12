# -*- coding: utf-8 -*-
# pylint: disable=protected-access
# ChannelManager is the framework owner of BaseChannel and must call
# _is_native_payload and _consume_one_request as part of the contract.

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    TYPE_CHECKING,
)

from .base import BaseChannel, ContentType, ProcessHandler, TextContent
from .command_registry import CommandRegistry
from .registry import get_channel_registry
from .unified_queue_manager import UnifiedQueueManager
from ...config import get_available_channels

if TYPE_CHECKING:
    from ...config.config import Config

logger = logging.getLogger(__name__)

# Callback when user reply was sent: (channel, user_id, session_id)
OnLastDispatch = Optional[Callable[[str, str, str], None]]

# Default max size per channel queue
_CHANNEL_QUEUE_MAXSIZE = 1000


async def _process_batch(ch: BaseChannel, batch: List[Any]) -> None:
    """Merge if needed and process one payload (native or request)."""
    if ch.channel == "dingtalk" and batch and ch._is_native_payload(batch[0]):
        first = batch[0] if isinstance(batch[0], dict) else {}
        logger.info(
            "manager _process_batch dingtalk: batch_len=%s first_has_sw=%s",
            len(batch),
            bool(first.get("session_webhook")),
        )
    if len(batch) > 1 and ch._is_native_payload(batch[0]):
        merged = ch.merge_native_items(batch)
        if ch.channel == "dingtalk" and isinstance(merged, dict):
            logger.info(
                "manager _process_batch dingtalk merged: has_sw=%s",
                bool(merged.get("session_webhook")),
            )
        await ch._consume_one_request(merged)
    elif len(batch) > 1:
        merged = ch.merge_requests(batch)
        if merged is not None:
            await ch._consume_one_request(merged)
        else:
            await ch.consume_one(batch[0])
    elif ch._is_native_payload(batch[0]):
        await ch._consume_one_request(batch[0])
    else:
        await ch.consume_one(batch[0])


class ChannelManager:
    """Owns queues and consumer loops; channels define how to consume via
    consume_one(). Enqueue via enqueue(channel_id, payload) (thread-safe).
    """

    def __init__(self, channels: List[BaseChannel]):
        self.channels = channels
        self._lock = asyncio.Lock()
        self._loop: Optional[asyncio.AbstractEventLoop] = None

        # New unified queue system
        self._command_registry = CommandRegistry()
        self._queue_manager: UnifiedQueueManager | None = None
        self._workspace = None

        # Per-channel locks to prevent concurrent restarts
        self._restart_locks: dict[str, asyncio.Lock] = {}

        # Track enqueue tasks for graceful shutdown
        self._enqueue_tasks: set[asyncio.Task] = set()

    @classmethod
    def from_env(
        cls,
        process: ProcessHandler,
        on_last_dispatch: OnLastDispatch = None,
    ) -> "ChannelManager":
        """
        Create channels from env and inject unified process
        (AgentRequest -> Event stream).
        process is typically runner.stream_query, handled by AgentApp's
        process endpoint.
        on_last_dispatch: called when a user send+reply was sent.
        """
        available = get_available_channels()
        registry = get_channel_registry()
        channels: list[BaseChannel] = [
            ch_cls.from_env(process, on_reply_sent=on_last_dispatch)
            for key, ch_cls in registry.items()
            if key in available
        ]
        return cls(channels)

    @classmethod
    def from_config(
        cls,
        process: ProcessHandler,
        config: "Config",
        on_last_dispatch: OnLastDispatch = None,
        workspace_dir: Path | None = None,
    ) -> "ChannelManager":
        """Create channels from config (config.json or agent.json).

        Args:
            process: Process handler for agent communication
            config: Configuration object with channels
            on_last_dispatch: Callback for dispatch events
            workspace_dir: Agent workspace directory for channel state files
        """
        available = get_available_channels()
        ch = config.channels
        show_tool_details = getattr(config, "show_tool_details", True)
        extra = getattr(ch, "__pydantic_extra__", None) or {}

        channels: list[BaseChannel] = []
        for key, ch_cls in get_channel_registry().items():
            if key not in available:
                continue
            ch_cfg = getattr(ch, key, None)
            if ch_cfg is None and key in extra:
                ch_cfg = extra[key]
            new_ch = cls.build_one_channel(
                key=key,
                ch_cls=ch_cls,
                ch_cfg=ch_cfg,
                process=process,
                on_last_dispatch=on_last_dispatch,
                show_tool_details=show_tool_details,
                workspace_dir=workspace_dir,
            )
            if new_ch is not None:
                channels.append(new_ch)

        return cls(channels)

    @classmethod
    # pylint: disable=too-many-branches
    def build_one_channel(
        cls,
        *,
        key: str,
        ch_cls: type[BaseChannel],
        ch_cfg: Any,
        process: ProcessHandler,
        on_last_dispatch: OnLastDispatch,
        show_tool_details: bool,
        workspace_dir: Path | None,
    ) -> Optional[BaseChannel]:
        """Build a single BaseChannel from a per-channel config block.

        Returns None if the config is missing, the channel is not enabled,
        or construction fails. Mirrors the per-channel logic that used to
        live inline in ``from_config`` so that ``reload_channel_service``
        can re-use it for hot-adding a channel that became enabled after
        the manager was first built.
        """
        if ch_cfg is None:
            return None
        if isinstance(ch_cfg, dict):
            from types import SimpleNamespace
            from ...config.config import BaseChannelConfig

            defaults = BaseChannelConfig().model_dump()
            defaults.update(ch_cfg)
            ch_cfg = SimpleNamespace(**defaults)

        # Handle both Pydantic objects (built-in) and dicts (custom channels)
        if isinstance(ch_cfg, dict):
            enabled = ch_cfg.get("enabled", False)
        else:
            enabled = getattr(ch_cfg, "enabled", False)
        if not enabled:
            return None

        if isinstance(ch_cfg, dict):
            filter_tool_messages = ch_cfg.get(
                "filter_tool_messages",
                False,
            )
            filter_thinking = ch_cfg.get("filter_thinking", False)
        else:
            filter_tool_messages = getattr(
                ch_cfg,
                "filter_tool_messages",
                False,
            )
            filter_thinking = getattr(
                ch_cfg,
                "filter_thinking",
                False,
            )

        from_config_kwargs = {
            "process": process,
            "config": ch_cfg,
            "on_reply_sent": on_last_dispatch,
            "show_tool_details": show_tool_details,
            "filter_tool_messages": filter_tool_messages,
            "filter_thinking": filter_thinking,
            "workspace_dir": workspace_dir,
        }

        # Only pass kwargs that the channel's from_config accepts
        import inspect

        sig = inspect.signature(ch_cls.from_config)
        if any(
            p.kind == inspect.Parameter.VAR_KEYWORD
            for p in sig.parameters.values()
        ):
            filtered_kwargs = from_config_kwargs
        else:
            filtered_kwargs = {
                k: v
                for k, v in from_config_kwargs.items()
                if k in sig.parameters
            }

        try:
            return ch_cls.from_config(**filtered_kwargs)
        except Exception as e:
            logger.warning(
                "Failed to initialize channel '%s', skipping: %s",
                key,
                e,
            )
            return None

    def _make_enqueue_cb(self, channel_id: str) -> Callable[[Any], None]:
        """Return a callback that enqueues payload for the given channel."""

        def cb(payload: Any) -> None:
            self.enqueue(channel_id, payload)

        return cb

    def _extract_session_id(
        self,
        ch: BaseChannel,
        payload: Any,
    ) -> str:
        """Extract normalized session_id from payload.

        Args:
            ch: Channel instance
            payload: Native dict or AgentRequest

        Returns:
            Normalized session_id (e.g. "console:user1")

        Note:
            Uses same logic as ch.get_debounce_key for consistency
        """
        # Check if payload already has normalized session_id
        # (e.g. from batch merge or previous processing)
        if isinstance(payload, dict):
            existing_sid = payload.get("session_id")
            if existing_sid:
                return existing_sid

        if hasattr(payload, "session_id"):
            existing_sid = payload.session_id
            if existing_sid:
                return existing_sid

        # Use channel's debounce key (delegates to resolve_session_id)
        return ch.get_debounce_key(payload)

    def _enqueue_one(self, channel_id: str, payload: Any) -> None:
        """Run on event loop: classify priority and route to queue manager.

        Note:
            This is the new routing layer using UnifiedQueueManager
        """
        if self._queue_manager is None:
            logger.warning(
                "enqueue: queue_manager not initialized for channel=%s",
                channel_id,
            )
            return

        # Get channel instance
        ch = next(
            (c for c in self.channels if c.channel == channel_id),
            None,
        )
        if not ch:
            logger.warning(
                "enqueue: channel not found: channel_id=%s",
                channel_id,
            )
            return

        # Extract query text for priority classification
        query = ch._extract_query_from_payload(payload)

        # Get priority level
        priority_level = self._command_registry.get_priority_level(query)

        # Extract normalized session_id
        session_id = self._extract_session_id(ch, payload)

        max_concurrent = ch.get_max_concurrent_runs(
            payload,
            priority_level=priority_level,
            query=query,
        )
        ch.set_payload_max_concurrent_runs(payload, max_concurrent)

        # Route to unified queue manager with task tracking
        task = asyncio.create_task(
            self._enqueue_with_timeout(
                channel_id,
                session_id,
                priority_level,
                payload,
                query,
                max_concurrent,
            ),
        )
        self._enqueue_tasks.add(task)
        task.add_done_callback(self._enqueue_tasks.discard)

    async def _enqueue_with_timeout(
        self,
        channel_id: str,
        session_id: str,
        priority_level: int,
        payload: Any,
        query: str,
        max_concurrent: int = 1,
    ) -> None:
        """Enqueue with timeout protection to prevent unbounded blocking.

        Args:
            channel_id: Channel identifier
            session_id: Normalized session ID
            priority_level: Priority level
            payload: Message payload
            query: Extracted query text for logging
        """
        try:
            await asyncio.wait_for(
                self._queue_manager.enqueue(
                    channel_id,
                    session_id,
                    priority_level,
                    payload,
                    max_concurrent=max_concurrent,
                ),
                timeout=30.0,
            )
            logger.debug(
                f"Enqueued: channel={channel_id} "
                f"session={session_id[:30]} "
                f"priority={priority_level} "
                f"max_concurrent={max_concurrent} "
                f"query={query[:40] if query else '(empty)'}",
            )
        except asyncio.TimeoutError:
            pass
        except asyncio.CancelledError:
            logger.debug(
                f"Enqueue cancelled: channel={channel_id} "
                f"session={session_id[:30]}",
            )
            raise
        except Exception as e:
            logger.exception(
                f"Enqueue failed: channel={channel_id} "
                f"session={session_id[:30]} error={e}",
            )

    def enqueue(self, channel_id: str, payload: Any) -> None:
        """Enqueue a payload for the channel. Thread-safe (e.g. from sync
        WebSocket or polling thread). Call after start_all().
        """
        if self._loop is None:
            logger.warning("enqueue: loop not set for channel=%s", channel_id)
            return
        self._loop.call_soon_threadsafe(
            self._enqueue_one,
            channel_id,
            payload,
        )

    async def _consume_queue(
        self,
        queue: asyncio.Queue,
        channel_id: str,
        session_id: str,
        priority_level: int,
        max_concurrent: int,
    ) -> None:
        """Consumer function for UnifiedQueueManager.

        This implements the per-queue consumer loop with batch merging.

        Args:
            queue: The queue to consume from
            channel_id: Channel identifier
            session_id: Normalized session ID
            priority_level: Priority level

        Note:
            Preserves original batch merging logic (drain + merge)
        """
        logger.info(
            f"Consumer started: channel={channel_id} "
            f"session={session_id[:30]} "
            f"priority={priority_level}",
        )

        while True:
            try:
                # Get first payload
                payload = await queue.get()

                # Re-fetch channel each iteration so replace_channel()
                # swaps are picked up automatically.
                ch = await self.get_channel(channel_id)
                if not ch:
                    # Channel may be temporarily absent during a
                    # replace_channel() swap.  Retry a few times before
                    # giving up so we don't silently drop the payload.
                    for _retry in range(3):
                        await asyncio.sleep(0.5)
                        ch = await self.get_channel(channel_id)
                        if ch:
                            break
                    if not ch:
                        logger.error(
                            "Consumer: channel not found after"
                            " retries: channel_id=%s",
                            channel_id,
                        )
                        return

                # Drain queue for same-key payloads (batch merge logic)
                # Note: In new architecture, same-key means same QueueKey,
                # so all payloads in this queue already have same
                # (channel_id, session_id, priority_level).
                # We still drain to merge rapid-fire messages (e.g. images)
                # when the queue is serial. Parallel group queues process one
                # payload per worker so separate messages can run together.
                batch = [payload]
                if max_concurrent <= 1:
                    while True:
                        try:
                            next_payload = queue.get_nowait()
                            batch.append(next_payload)
                        except asyncio.QueueEmpty:
                            break

                # Process batch (with merge logic)
                await _process_batch(ch, batch)

                # Update processed count
                if self._queue_manager is not None:
                    await self._queue_manager.increment_processed(
                        channel_id,
                        session_id,
                        priority_level,
                        count=len(batch),
                    )

                logger.debug(
                    f"Processed batch: channel={channel_id} "
                    f"session={session_id[:30]} "
                    f"priority={priority_level} "
                    f"max_concurrent={max_concurrent} "
                    f"batch_size={len(batch)}",
                )

            except asyncio.CancelledError:
                logger.debug(
                    f"Consumer cancelled: channel={channel_id} "
                    f"session={session_id[:30]} "
                    f"priority={priority_level}",
                )
                break
            except Exception:
                logger.exception(
                    f"Consumer failed: channel={channel_id} "
                    f"session={session_id[:30]} "
                    f"priority={priority_level}",
                )

    async def start_all(self) -> None:
        """Start all channels and queue manager."""
        self._loop = asyncio.get_running_loop()

        # Initialize UnifiedQueueManager with consumer function
        self._queue_manager = UnifiedQueueManager(
            consumer_fn=self._consume_queue,
            queue_maxsize=_CHANNEL_QUEUE_MAXSIZE,
        )

        # Start cleanup loop
        self._queue_manager.start_cleanup_loop()

        # Set enqueue callback for each channel
        async with self._lock:
            snapshot = list(self.channels)

        for ch in snapshot:
            if getattr(ch, "uses_manager_queue", True):
                ch.set_enqueue(self._make_enqueue_cb(ch.channel))

        logger.debug(
            f"Starting channels: {[g.channel for g in snapshot]}",
        )

        # Start each channel
        for g in snapshot:
            try:
                await g.start()
            except Exception:
                logger.exception(f"failed to start channels={g.channel}")

    async def stop_all(self) -> None:
        """Stop all channels and queue manager."""
        # Cancel all pending enqueue tasks
        if self._enqueue_tasks:
            logger.info(
                f"Cancelling {len(self._enqueue_tasks)} pending enqueue tasks",
            )
            for task in self._enqueue_tasks:
                task.cancel()

            # Wait for tasks to finish cancellation
            if self._enqueue_tasks:
                _, pending = await asyncio.wait(
                    self._enqueue_tasks,
                    timeout=2.0,
                    return_when=asyncio.ALL_COMPLETED,
                )
                if pending:
                    logger.warning(
                        f"stop_all: {len(pending)} enqueue task(s) "
                        f"still pending after 2s",
                    )
            self._enqueue_tasks.clear()

        # Stop queue manager (stops all consumers and cleanup task)
        if self._queue_manager is not None:
            await self._queue_manager.stop_all()
            self._queue_manager = None

        # Stop channels
        async with self._lock:
            snapshot = list(self.channels)

        for ch in snapshot:
            ch.set_enqueue(None)

        async def _stop(ch):
            try:
                await ch.stop()
            except asyncio.CancelledError:
                pass
            except Exception:
                logger.exception(f"failed to stop channels={ch.channel}")

        await asyncio.gather(*[_stop(g) for g in reversed(snapshot)])

        logger.info("ChannelManager stopped")

    async def get_channel(self, channel: str) -> Optional[BaseChannel]:
        async with self._lock:
            for ch in self.channels:
                if ch.channel == channel:
                    return ch
            return None

    async def get_channel_health(
        self,
        channel_name: str,
    ) -> Dict[str, Any]:
        """Get health status for a specific channel.

        Args:
            channel_name: Channel identifier (e.g. "dingtalk", "telegram")

        Returns:
            Health status dict from the channel's health_check() method.

        Raises:
            KeyError: If channel is not found in this manager.
        """
        channel_instance = await self.get_channel(channel_name)
        if channel_instance is None:
            raise KeyError(f"Channel not found: {channel_name}")
        try:
            return await channel_instance.health_check()
        except Exception as exc:
            logger.exception(
                "health_check failed for channel=%s",
                channel_name,
            )
            return {
                "channel": channel_name,
                "status": "unhealthy",
                "detail": str(exc),
            }

    async def restart_channel(self, channel_name: str) -> Dict[str, Any]:
        """Restart a single channel by stopping and re-starting it.

        The channel is stopped, then a fresh instance is created via
        clone() with the current config, and started via replace_channel().

        Args:
            channel_name: Channel identifier (e.g. "dingtalk", "telegram")

        Returns:
            Dict with restart result: channel, status, detail.

        Raises:
            KeyError: If channel is not found in this manager.
        """
        # Per-channel lock prevents concurrent restarts from
        # leaking resources (two clones started, one discarded).
        lock = self._restart_locks.setdefault(
            channel_name,
            asyncio.Lock(),
        )
        async with lock:
            channel_instance = await self.get_channel(channel_name)
            if channel_instance is None:
                raise KeyError(
                    f"Channel not found: {channel_name}",
                )

            logger.info("Restarting channel: %s", channel_name)

            # Load the latest config for this channel
            from ...config.config import load_agent_config

            agent_id = self._workspace.agent_id if self._workspace else None
            if agent_id is None:
                raise RuntimeError(
                    "Cannot restart channel: workspace not set"
                    " on ChannelManager",
                )

            agent_config = load_agent_config(agent_id)
            channels_cfg = agent_config.channels
            if channels_cfg is None:
                raise RuntimeError(
                    f"No channels config found for agent" f" {agent_id}",
                )

            # Get channel-specific config
            channel_cfg = getattr(
                channels_cfg,
                channel_name,
                None,
            )
            if channel_cfg is None:
                extra = (
                    getattr(
                        channels_cfg,
                        "__pydantic_extra__",
                        None,
                    )
                    or {}
                )
                channel_cfg = extra.get(channel_name)
            if channel_cfg is None:
                raise RuntimeError(
                    f"No config found for channel" f" '{channel_name}'",
                )

            # Clone a fresh instance and replace
            new_channel = channel_instance.clone(channel_cfg)
            if self._workspace is not None:
                new_channel.set_workspace(
                    self._workspace,
                    self._command_registry,
                )
            await self.replace_channel(new_channel)

            logger.info(
                "Channel restarted successfully: %s",
                channel_name,
            )
            return {
                "channel": channel_name,
                "status": "restarted",
                "detail": (f"Channel '{channel_name}'" " has been restarted."),
            }

    def set_workspace(self, workspace) -> None:
        """Set workspace and inject to all channels.

        Args:
            workspace: Workspace instance with task_tracker and chat_manager
        """
        self._workspace = workspace
        for ch in self.channels:
            ch.set_workspace(workspace, self._command_registry)
        logger.info(
            f"Injected workspace into {len(self.channels)} channels",
        )

    async def clear_queue(
        self,
        channel_id: str,
        session_id: str,
        priority_level: int,
    ) -> int:
        """Clear a specific queue.

        Args:
            channel_id: Channel identifier
            session_id: Session identifier
            priority_level: Priority level

        Returns:
            Number of messages cleared
        """
        if self._queue_manager is None:
            return 0
        return await self._queue_manager.clear_queue(
            channel_id,
            session_id,
            priority_level,
        )

    async def replace_channel(
        self,
        new_channel: BaseChannel,
    ) -> None:
        """Replace a single channel by name.

        For channels with requires_sequential_restart=True (e.g. WhatsApp),
        the old channel is stopped BEFORE the new one starts to avoid
        resource conflicts (exclusive SQLite locks, websocket collisions).

        For other channels, zero-downtime flow is preserved:
        start new -> swap + stop old.
        """
        new_channel_name = new_channel.channel
        sequential = getattr(new_channel, "requires_sequential_restart", False)

        if sequential:
            # -- Sequential path: STOP OLD first, then START NEW --
            async with self._lock:
                old_channel = None
                old_index = None
                for i, ch in enumerate(self.channels):
                    if ch.channel == new_channel_name:
                        old_channel = ch
                        old_index = i
                        break

                if old_channel is not None:
                    logger.info(
                        f"Sequential restart: stopping old channel "
                        f"{old_channel.channel} first",
                    )
                    try:
                        await old_channel.stop()
                    except asyncio.CancelledError:
                        pass
                    except Exception:
                        logger.exception(
                            f"Failed to stop old channel: "
                            f"{old_channel.channel}",
                        )
                    if old_index is not None:
                        self.channels.pop(old_index)

            if getattr(new_channel, "uses_manager_queue", True):
                new_channel.set_enqueue(
                    self._make_enqueue_cb(new_channel_name),
                )

            logger.info(
                f"Sequential restart: starting new channel "
                f"{new_channel_name}",
            )
            try:
                await new_channel.start()
            except Exception:
                logger.exception(
                    f"Failed to start new channel: {new_channel_name}",
                )
                try:
                    await new_channel.stop()
                except Exception:
                    pass
                raise

            async with self._lock:
                self.channels.append(new_channel)

        else:
            # -- Zero-downtime path: START NEW, then STOP OLD --
            if getattr(new_channel, "uses_manager_queue", True):
                new_channel.set_enqueue(
                    self._make_enqueue_cb(new_channel_name),
                )

            logger.info(f"Pre-starting new channel: {new_channel_name}")
            try:
                await new_channel.start()
            except Exception:
                logger.exception(
                    f"Failed to start new channel: {new_channel_name}",
                )
                try:
                    await new_channel.stop()
                except Exception:
                    pass
                raise

            async with self._lock:
                old_channel = None
                for i, ch in enumerate(self.channels):
                    if ch.channel == new_channel_name:
                        old_channel = ch
                        self.channels[i] = new_channel
                        break

                if old_channel is None:
                    logger.info(f"Adding new channel: {new_channel_name}")
                    self.channels.append(new_channel)
                else:
                    logger.info(
                        f"Stopping old channel: {old_channel.channel}",
                    )
                    try:
                        await old_channel.stop()
                    except asyncio.CancelledError:
                        pass
                    except Exception:
                        logger.exception(
                            f"Failed to stop old channel: "
                            f"{old_channel.channel}",
                        )

    async def remove_channel(self, channel_name: str) -> bool:
        """Stop and remove a channel by name.

        Used by ``reload_channel_service`` when a hot-reload turns
        ``enabled=true`` into ``enabled=false`` for a previously-running
        channel. Pops first under the lock so further enqueue routing
        cannot find the channel, then stops it outside the lock to avoid
        holding the lock across the (potentially slow) shutdown.

        Returns True if the channel was found and removed, False if no
        channel by that name was registered.
        """
        async with self._lock:
            target = None
            target_index = None
            for i, ch in enumerate(self.channels):
                if ch.channel == channel_name:
                    target = ch
                    target_index = i
                    break
            if target is None:
                return False
            self.channels.pop(target_index)

        target.set_enqueue(None)
        try:
            await target.stop()
        except asyncio.CancelledError:
            pass
        except Exception:
            logger.exception(
                "Failed to stop channel %s during removal",
                channel_name,
            )
        return True

    async def send_event(
        self,
        *,
        channel: str,
        user_id: str,
        session_id: str,
        event: Any,
        meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        ch = await self.get_channel(channel)
        if not ch:
            raise KeyError(f"channel not found: {channel}")
        merged_meta = dict(meta or {})
        merged_meta["session_id"] = session_id
        merged_meta["user_id"] = user_id
        bot_prefix = getattr(ch, "bot_prefix", None) or getattr(
            ch,
            "_bot_prefix",
            None,
        )
        if bot_prefix and "bot_prefix" not in merged_meta:
            merged_meta["bot_prefix"] = bot_prefix
        await ch.send_event(
            user_id=user_id,
            session_id=session_id,
            event=event,
            meta=merged_meta,
        )

    async def send_text(
        self,
        *,
        channel: str,
        user_id: str,
        session_id: str,
        text: str,
        meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Send plain text to a specific channel
        (used for scheduled jobs like task_type='text').
        """
        ch = await self.get_channel(channel.lower())
        if not ch:
            raise KeyError(f"channel not found: {channel}")

        # Convert (user_id, session_id) into the channel-specific target handle
        to_handle = ch.to_handle_from_target(
            user_id=user_id,
            session_id=session_id,
        )
        ch_name = getattr(ch, "channel", channel)
        logger.info(
            "channel send_text: channel=%s user_id=%s session_id=%s "
            "to_handle=%s",
            ch_name,
            (user_id or "")[:40],
            (session_id or "")[:40],
            (to_handle or "")[:60],
        )

        # Keep the same behavior as the agent pipeline:
        # if the channel has a fixed bot prefix, merge it into meta so
        # send_content_parts can use it.
        merged_meta = dict(meta or {})
        bot_prefix = getattr(ch, "bot_prefix", None) or getattr(
            ch,
            "_bot_prefix",
            None,
        )
        if bot_prefix and "bot_prefix" not in merged_meta:
            merged_meta["bot_prefix"] = bot_prefix
        merged_meta["session_id"] = session_id
        merged_meta["user_id"] = user_id

        # Send as content parts (single text part; use TextContent so channel
        # getattr(p, "type") / getattr(p, "text") work)
        await ch.send_content_parts(
            to_handle,
            [TextContent(type=ContentType.TEXT, text=text)],
            merged_meta,
        )
