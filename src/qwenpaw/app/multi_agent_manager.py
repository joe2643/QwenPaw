# -*- coding: utf-8 -*-
"""MultiAgentManager: Manages multiple agent workspaces with lazy loading.

Provides centralized management for multiple Workspace objects,
including lazy loading, lifecycle management, and hot reloading.
"""
import asyncio
import enum
import logging
import os
import time
from typing import Any, Dict, Set

from agentscope_runtime.engine.schemas.exception import (
    ConfigurationException,
)

from .workspace import Workspace
from ..config.utils import load_config

logger = logging.getLogger(__name__)


class ReloadResult(str, enum.Enum):
    """Outcome of :meth:`MultiAgentManager.reload_agent`.

    Subclasses ``str`` so legacy truthiness checks keep working:
    ``RELOADED`` → truthy, ``NOT_RUNNING`` / ``SKIPPED_COOLDOWN`` → falsy.
    """

    RELOADED = "reloaded"
    NOT_RUNNING = "not_running"
    SKIPPED_COOLDOWN = "skipped_cooldown"

    def __bool__(self) -> bool:
        return self is ReloadResult.RELOADED


def _cooldown_seconds_from_env(default: float = 30.0) -> float:
    """Read reload cooldown from ``COPAW_RELOAD_COOLDOWN_SECONDS`` env var."""
    raw = os.environ.get("COPAW_RELOAD_COOLDOWN_SECONDS", "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        logger.warning(
            "Invalid COPAW_RELOAD_COOLDOWN_SECONDS=%r; using default %.1fs",
            raw,
            default,
        )
        return default
    return max(0.0, value)


class MultiAgentManager:
    """Manages multiple agent workspaces.

    Features:
    - Lazy loading: Workspaces are created only when first requested
    - Lifecycle management: Start, stop, reload workspaces
    - Thread-safe: Uses async lock for concurrent access
    - Hot reload: Reload individual workspaces without affecting others
    - Parallel startup: Multiple agents start concurrently via
      fine-grained locking (lock released during slow workspace init)

    Reload cooldown
    ---------------
    ``reload_agent`` skips a reload when another one completed within the
    last ``RELOAD_COOLDOWN_SECONDS`` seconds for the same agent. This is a
    back-stop against silent ``agent.json`` writer storms (e.g. from MCP
    tools, skills, or misbehaving routes). The skipped write is NOT lost —
    it will be picked up by the next eligible reload or by the per-agent
    ``AgentConfigWatcher`` which reloads channels/heartbeat/memory jobs
    independently on a 2 s poll.

    Override via env var ``COPAW_RELOAD_COOLDOWN_SECONDS`` (e.g. ``0`` to
    disable the guard during local development).
    """

    #: Minimum seconds between successive zero-downtime reloads of the same
    #: agent. Overridable via the ``COPAW_RELOAD_COOLDOWN_SECONDS`` env var.
    RELOAD_COOLDOWN_SECONDS: float = _cooldown_seconds_from_env()

    def __init__(self):
        """Initialize multi-agent manager."""
        self.agents: Dict[str, Workspace] = {}
        self._lock = asyncio.Lock()
        self._pending_starts: Dict[str, asyncio.Event] = {}
        self._cleanup_tasks: Set[asyncio.Task] = set()
        #: Wall-clock time of the last successful reload per agent_id.
        self._last_reload_at: Dict[str, float] = {}
        #: Counter: how many times the cooldown guard skipped a reload.
        #: Useful for detecting ongoing storms even when individual log
        #: lines get lost in the noise.
        self._reload_skip_count: Dict[str, int] = {}
        logger.debug("MultiAgentManager initialized")

    async def get_agent(self, agent_id: str) -> Workspace:
        """Get agent workspace by ID (lazy loading with dedup).

        If workspace doesn't exist in memory, it will be created and started.
        Multiple concurrent callers for the same agent_id are coordinated:
        the first caller creates the workspace while others wait.

        The lock is only held briefly for dict checks/mutations, not during
        the slow workspace startup, allowing parallel agent initialization.

        Args:
            agent_id: Agent ID to retrieve

        Returns:
            Workspace: The requested workspace instance

        Raises:
            ConfigurationException: If agent ID not found in configuration
        """
        # Fast path: already loaded (no lock)
        if agent_id in self.agents:
            logger.debug(f"Returning cached agent: {agent_id}")
            return self.agents[agent_id]

        should_start = False
        event = None
        agent_ref = None

        async with self._lock:
            # Re-check under lock
            if agent_id in self.agents:
                logger.debug(f"Returning cached agent: {agent_id}")
                return self.agents[agent_id]

            if agent_id in self._pending_starts:
                # Another task is already starting this agent; wait for it
                event = self._pending_starts[agent_id]
            else:
                # We are the first caller — validate config and claim startup
                config = load_config()
                if agent_id not in config.agents.profiles:
                    raise ConfigurationException(
                        config_key="agent",
                        message=(
                            f"Agent '{agent_id}' not found in configuration. "
                            f"Available agents: "
                            f"{list(config.agents.profiles.keys())}"
                        ),
                    )
                agent_ref = config.agents.profiles[agent_id]
                event = asyncio.Event()
                self._pending_starts[agent_id] = event
                should_start = True

        if not should_start:
            # Wait for the in-progress startup to finish
            await event.wait()
            if agent_id in self.agents:
                logger.debug(f"Returning cached agent: {agent_id}")
                return self.agents[agent_id]
            raise ConfigurationException(
                config_key="agent",
                message=f"Agent '{agent_id}' failed to initialize",
            )

        # We are the starter — create outside the lock for parallelism
        t0 = time.perf_counter()
        logger.debug(f"Creating new workspace: {agent_id}")
        instance = Workspace(
            agent_id=agent_id,
            workspace_dir=agent_ref.workspace_dir,
        )

        try:
            await instance.start()
            instance.set_manager(self)

            async with self._lock:
                self.agents[agent_id] = instance

            elapsed = time.perf_counter() - t0
            logger.debug(
                f"Workspace created and started: {agent_id} "
                f"({elapsed:.3f}s)",
            )
            return instance
        except Exception as e:
            logger.error(f"Failed to start workspace {agent_id}: {e}")
            raise
        finally:
            # Always clean up pending state and signal waiters
            # This handles cancellation (CancelledError) and all other cases
            async with self._lock:
                self._pending_starts.pop(agent_id, None)
            event.set()

    async def _graceful_stop_old_instance(
        self,
        old_instance: Workspace,
        agent_id: str,
    ) -> None:
        """Gracefully stop old instance after checking for active tasks.

        If active tasks exist, schedule delayed cleanup in background.
        Otherwise, stop immediately.

        Args:
            old_instance: The old workspace instance to stop
            agent_id: Agent ID for logging
        """
        has_active = await old_instance.task_tracker.has_active_tasks()

        if has_active:
            # Active tasks - schedule delayed cleanup in background
            active_tasks = await old_instance.task_tracker.list_active_tasks()
            logger.info(
                f"Old workspace instance has {len(active_tasks)} active "
                f"task(s): {active_tasks}. Scheduling delayed cleanup for "
                f"{agent_id}.",
            )

            async def delayed_cleanup():
                """Wait for tasks to complete, then stop old instance."""
                try:
                    # Wait up to 1 minutes for tasks to complete
                    completed = await old_instance.task_tracker.wait_all_done(
                        timeout=60.0,
                    )
                    if completed:
                        logger.info(
                            f"All tasks completed for old instance "
                            f"{agent_id}. Stopping now.",
                        )
                    else:
                        logger.warning(
                            f"Timeout waiting for tasks to complete for "
                            f"{agent_id}. Forcing stop after 5 minutes.",
                        )

                    await old_instance.stop(final=False)
                    logger.info(
                        f"Old workspace instance stopped: {agent_id}. "
                        f"Delayed cleanup completed.",
                    )
                except Exception as e:
                    logger.warning(
                        f"Error during delayed cleanup for {agent_id}: {e}. "
                        f"New instance is serving requests.",
                    )

            # Create background task for delayed cleanup and track it
            cleanup_task = asyncio.create_task(delayed_cleanup())
            self._cleanup_tasks.add(cleanup_task)

            def _on_cleanup_done(task: asyncio.Task) -> None:
                """Remove task from tracking set and log errors."""
                self._cleanup_tasks.discard(task)
                if task.cancelled():
                    logger.info(
                        f"Delayed cleanup task for {agent_id} was cancelled.",
                    )
                    return
                exc = task.exception()
                if exc is not None:
                    logger.warning(
                        f"Error in delayed cleanup task for {agent_id}: "
                        f"{exc}.",
                    )

            cleanup_task.add_done_callback(_on_cleanup_done)
            logger.info(
                f"Zero-downtime reload completed: {agent_id}. "
                f"Old instance cleanup scheduled in background.",
            )
        else:
            # No active tasks - stop immediately
            logger.debug(
                f"No active tasks in old instance {agent_id}. "
                f"Stopping immediately.",
            )
            try:
                await old_instance.stop(final=False)
                logger.info(
                    f"Old workspace instance stopped: {agent_id}. "
                    f"Zero-downtime reload completed.",
                )
            except Exception as e:
                logger.warning(
                    f"Failed to stop old workspace instance for "
                    f"{agent_id}: {e}. "
                    f"New instance is active and serving requests.",
                )

    async def stop_agent(self, agent_id: str) -> bool:
        """Stop a specific agent instance.

        Args:
            agent_id: Agent ID to stop

        Returns:
            bool: True if agent was stopped, False if not running
        """
        async with self._lock:
            if agent_id not in self.agents:
                logger.warning(f"Agent not running: {agent_id}")
                return False

            instance = self.agents[agent_id]
            await instance.stop()
            del self.agents[agent_id]
            logger.info(f"Agent stopped and removed: {agent_id}")
            return True

    async def reload_agent(self, agent_id: str) -> ReloadResult:
        """Reload a specific agent instance with zero-downtime.

        This method performs a seamless reload by:
        1. Creating and fully starting a new workspace instance (no lock)
        2. Atomically replacing the old instance with the new one (with lock)
        3. Gracefully stopping the old instance (no lock):
           - If active tasks exist: schedule delayed cleanup in background
           - If no active tasks: stop immediately

        The lock is only held during the atomic swap to minimize blocking
        time for other agent operations.

        This ensures that:
        - New requests are immediately handled by the new instance
        - Ongoing SSE/streaming tasks continue uninterrupted
        - Other agents remain accessible during reload
        - The manager returns quickly without waiting for old tasks
        - Old instance is automatically cleaned up after tasks complete

        Args:
            agent_id: Agent ID to reload

        Returns:
            :class:`ReloadResult`. The enum subclasses ``str`` and its
            ``__bool__`` is ``True`` only for ``RELOADED``, so existing
            truthiness checks keep working. Possible values:

            * ``RELOADED`` — the swap completed successfully.
            * ``NOT_RUNNING`` — agent is not registered or the new
              workspace failed to start; the caller should prompt the
              user to trigger a lazy-load via a normal request.
            * ``SKIPPED_COOLDOWN`` — another reload completed within the
              last :attr:`RELOAD_COOLDOWN_SECONDS`; the change will be
              picked up on the next eligible reload.
        """
        # Step 1: Check if agent exists (quick check with lock)
        async with self._lock:
            if agent_id not in self.agents:
                logger.debug(
                    f"Agent not running, will be loaded on next "
                    f"request: {agent_id}",
                )
                return ReloadResult.NOT_RUNNING
            old_instance = self.agents[agent_id]

            # Cooldown guard: skip reload if another one started very
            # recently for this agent. A silent writer storm (MCP tool,
            # skill script, runaway loop) can otherwise trigger a reload
            # every few seconds, which in turn re-fires the agent's last
            # session message and produces duplicate channel sends.
            #
            # We claim the cooldown slot *optimistically* — the timestamp
            # is bumped the moment we decide to proceed, NOT after the
            # swap completes. This closes a race where many concurrent
            # callers all read a stale timestamp and slip past the guard
            # in parallel. A failed reload still holds the slot for the
            # full cooldown window, which is the desired behaviour: rapid
            # retries of a failing reload don't help either.
            last_at = self._last_reload_at.get(agent_id, 0.0)
            elapsed = time.monotonic() - last_at
            if (
                self.RELOAD_COOLDOWN_SECONDS > 0
                and elapsed < self.RELOAD_COOLDOWN_SECONDS
            ):
                skip_count = self._reload_skip_count.get(agent_id, 0) + 1
                self._reload_skip_count[agent_id] = skip_count
                log_fn = (
                    logger.error if skip_count >= 10 else logger.warning
                )
                log_fn(
                    "reload_agent skipped for %s: only %.1fs since last "
                    "reload (cooldown=%.0fs, skip_count=%d). Likely a "
                    "silent agent.json writer triggering a storm — check "
                    "save_agent_config callers.",
                    agent_id,
                    elapsed,
                    self.RELOAD_COOLDOWN_SECONDS,
                    skip_count,
                )
                return ReloadResult.SKIPPED_COOLDOWN

            # Cooldown passed — claim the slot and reset the skip counter.
            self._last_reload_at[agent_id] = time.monotonic()
            self._reload_skip_count.pop(agent_id, None)

        logger.info(f"Reloading agent (zero-downtime): {agent_id}")

        # Step 2: Load configuration (outside lock)
        config = load_config()
        if agent_id not in config.agents.profiles:
            logger.error(
                f"Agent '{agent_id}' not found in configuration "
                f"during reload",
            )
            return ReloadResult.NOT_RUNNING

        agent_ref = config.agents.profiles[agent_id]

        # Step 3: Create and start new workspace instance (outside lock)
        # This is the slow part, but doesn't block other agents
        logger.info(f"Creating new workspace instance: {agent_id}")
        new_instance = Workspace(
            agent_id=agent_id,
            workspace_dir=agent_ref.workspace_dir,
        )

        # Step 3.5: Set reusable components from old instance (if any)
        async with self._lock:
            old_instance = self.agents.get(agent_id)

        if old_instance:
            # Get all reusable services from old instance's ServiceManager
            # pylint: disable=protected-access
            reusable = old_instance._service_manager.get_reusable_services()
            # pylint: enable=protected-access

            if reusable:
                await new_instance.set_reusable_components(reusable)
                logger.info(
                    f"Set reusable components for {agent_id}: "
                    f"{list(reusable.keys())}",
                )

        try:
            await new_instance.start()
            new_instance.set_manager(self)  # Set manager reference
            logger.info(f"New workspace instance started: {agent_id}")
        except Exception as e:
            logger.exception(
                f"Failed to start new workspace instance for {agent_id}: {e}",
            )
            # Try to clean up the failed new instance
            try:
                await new_instance.stop()
            except Exception:
                pass  # Best effort cleanup
            # Old instance is still running and serving requests
            return ReloadResult.NOT_RUNNING

        # Step 4: Atomic swap (minimal lock time)
        # From this point, reload is considered successful
        async with self._lock:
            # Double-check agent still exists
            if agent_id not in self.agents:
                logger.warning(
                    f"Agent {agent_id} was removed during reload, "
                    f"stopping new instance",
                )
                await new_instance.stop()
                return ReloadResult.NOT_RUNNING

            # Swap instances atomically. The cooldown timestamp was
            # already claimed when we passed the guard in Step 1.
            old_instance = self.agents[agent_id]
            self.agents[agent_id] = new_instance
            logger.info(f"Workspace instance replaced: {agent_id}")

        # Step 5: Gracefully stop old instance (outside lock)
        # Delegates to helper method to avoid too-many-statements
        await self._graceful_stop_old_instance(old_instance, agent_id)

        return ReloadResult.RELOADED

    async def cancel_all_cleanup_tasks(self) -> None:
        """Cancel and await all pending delayed cleanup tasks.

        This ensures that any in-progress background cleanups are either
        completed or cleanly cancelled before the manager is torn down.
        Called by stop_all() during shutdown.
        """
        if not self._cleanup_tasks:
            return

        logger.info(
            f"Cancelling {len(self._cleanup_tasks)} pending cleanup "
            f"task(s)...",
        )
        tasks = list(self._cleanup_tasks)
        self._cleanup_tasks.clear()

        for task in tasks:
            if not task.done():
                task.cancel()

        # Await completion of all tasks, collecting exceptions
        await asyncio.gather(*tasks, return_exceptions=True)
        logger.info("All cleanup tasks cancelled/completed")

    async def shutdown_all_runners(self, timeout: float = 30.0) -> None:
        """Drain in-flight ``query_handler`` tasks across every
        workspace's runner so their ``finally`` blocks run and the
        latest session state lands on disk.

        Must be called BEFORE :meth:`stop_all` — workspaces tearing
        down yank the ``SafeJSONSession`` out from under any query
        still trying to save.  Each runner is drained in parallel
        (bounded by ``timeout`` total, not ``N*timeout``) so a slow
        workspace doesn't starve others.

        Args:
            timeout: Per-runner deadline.  Matches uvicorn's default
                ``graceful_timeout`` (30s) so the systemd stop
                sequence doesn't escalate to SIGKILL mid-flush.
        """
        if not self.agents:
            logger.debug("shutdown_all_runners: no agents loaded")
            return

        runners: list[tuple[str, Any]] = []
        for agent_id, workspace in self.agents.items():
            runner = getattr(workspace, "runner", None)
            if runner is None:
                continue
            handler = getattr(runner, "shutdown_handler", None)
            if handler is None:
                continue
            runners.append((agent_id, runner))

        if not runners:
            logger.debug(
                "shutdown_all_runners: no runners with shutdown handler",
            )
            return

        logger.info(
            "Draining in-flight queries across %d runner(s) "
            "(per-runner timeout=%.1fs)...",
            len(runners),
            timeout,
        )

        async def _drain(agent_id: str, runner: Any) -> tuple[str, bool]:
            try:
                ok = await runner.shutdown_handler(timeout=timeout)
                return agent_id, bool(ok)
            except Exception as e:
                logger.error(
                    "Error draining runner for %s: %s",
                    agent_id, e, exc_info=True,
                )
                return agent_id, False

        results = await asyncio.gather(
            *(_drain(aid, r) for aid, r in runners),
            return_exceptions=False,
        )
        incomplete = [aid for aid, ok in results if not ok]
        if incomplete:
            logger.warning(
                "shutdown_all_runners: %d runner(s) did not fully drain "
                "within timeout — some session state may be lost: %s",
                len(incomplete), incomplete,
            )
        else:
            logger.info("shutdown_all_runners: all runners drained")

    async def stop_all(self):
        """Stop all agent instances.

        Called during application shutdown to clean up resources.
        Cancels any pending delayed cleanup tasks and stops all agents.
        """
        logger.info(f"Stopping all agents ({len(self.agents)} running)...")

        # First, cancel pending cleanup tasks to avoid orphaned instances
        await self.cancel_all_cleanup_tasks()

        # Create list of agent IDs to avoid modifying dict during iteration
        agent_ids = list(self.agents.keys())

        for agent_id in agent_ids:
            try:
                instance = self.agents[agent_id]
                await instance.stop()
                logger.debug(f"Agent stopped: {agent_id}")
            except Exception as e:
                logger.error(f"Error stopping agent {agent_id}: {e}")

        self.agents.clear()
        logger.info("All agents stopped")

    def list_loaded_agents(self) -> list[str]:
        """List currently loaded agent IDs.

        Returns:
            list[str]: List of loaded agent IDs
        """
        return list(self.agents.keys())

    def is_agent_loaded(self, agent_id: str) -> bool:
        """Check if agent is currently loaded.

        Args:
            agent_id: Agent ID to check

        Returns:
            bool: True if agent is loaded and running
        """
        return agent_id in self.agents

    async def preload_agent(self, agent_id: str) -> bool:
        """Preload an agent instance during startup.

        Args:
            agent_id: Agent ID to preload

        Returns:
            bool: True if successfully preloaded, False if failed
        """
        try:
            await self.get_agent(agent_id)
            logger.info(f"Successfully preloaded agent: {agent_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to preload agent {agent_id}: {e}")
            return False

    async def start_all_configured_agents(self) -> dict[str, bool]:
        """Start all enabled agents defined in configuration concurrently.

        Only agents with enabled=True will be started.
        Disabled agents are skipped to save resources.

        Agents are started truly in parallel: get_agent() only holds the
        manager lock briefly for dict checks, releasing it during the slow
        workspace initialization.

        Returns:
            dict[str, bool]: Mapping of agent_id to success status
        """
        config = load_config()
        # Filter only enabled agents
        enabled_agents = {
            agent_id: ref
            for agent_id, ref in config.agents.profiles.items()
            if getattr(ref, "enabled", True)
        }
        agent_ids = list(enabled_agents.keys())

        if not agent_ids:
            logger.warning("No enabled agents configured in config")
            return {}

        total_agents = len(config.agents.profiles)
        disabled_count = total_agents - len(agent_ids)
        logger.debug(
            f"Starting {len(agent_ids)} enabled agent(s) "
            f"({disabled_count} disabled)",
        )

        async def start_single_agent(agent_id: str) -> tuple[str, bool]:
            """Start a single agent with error handling."""
            try:
                logger.debug(f"Starting agent: {agent_id}")
                await self.get_agent(agent_id)
                logger.debug(f"Agent started successfully: {agent_id}")
                return (agent_id, True)
            except Exception as e:
                logger.error(
                    f"Failed to start agent {agent_id}: {e}. "
                    f"Continuing with other agents...",
                )
                return (agent_id, False)

        # Truly parallel: get_agent releases lock during workspace startup
        results = await asyncio.gather(
            *[start_single_agent(agent_id) for agent_id in agent_ids],
            return_exceptions=False,
        )

        # Build result mapping
        result_map = dict(results)
        success_count = sum(1 for success in result_map.values() if success)
        logger.info(
            f"Agent startup complete: {success_count}/{len(agent_ids)} "
            f"agents started successfully, {disabled_count} disabled",
        )

        return result_map

    def __repr__(self) -> str:
        """String representation of manager."""
        loaded = list(self.agents.keys())
        return f"MultiAgentManager(loaded_agents={loaded})"
