# -*- coding: utf-8 -*-
"""Unified queue manager with per-session priority queues.

This module implements the core queue management system using a three-tuple
QueueKey: (channel_id, session_id, priority_level). It enables:

1. Concurrent processing for different sessions
2. Concurrent processing for different priority levels within the same session
3. Strict serialization for messages with the same QueueKey by default
   (optionally bounded concurrency for group sessions)
4. On-demand consumer creation (no fixed worker pools)
5. Automatic cleanup of idle queues

Architecture:
    - QueueKey = (channel_id, session_id, priority_level)
    - Each QueueKey has its own asyncio.Queue and consumer task set
    - Consumers are created on-demand when first message arrives
    - Idle queues are automatically cleaned up after timeout
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine, Dict, Tuple

logger = logging.getLogger(__name__)

# Type alias for queue key
QueueKey = Tuple[str, str, int]  # (channel_id, session_id, priority_level)

# Consumer function signature: (queue, channel_id, session_id, priority)
# Must be an async function (coroutine)
ConsumerFn = Callable[
    [asyncio.Queue, str, str, int, int],
    Coroutine[Any, Any, None],
]


@dataclass
class QueueState:
    """State for a single queue.

    Attributes:
        queue: The asyncio.Queue for this QueueKey
        consumer_tasks: The asyncio.Tasks running the consumers
        max_concurrent: Maximum concurrent consumers for this queue
        created_at: Timestamp when queue was created
        last_activity: Timestamp of last message (enqueue or dequeue)
        processed_count: Number of messages processed
    """

    queue: asyncio.Queue
    consumer_tasks: set[asyncio.Task[None]] = field(default_factory=set)
    max_concurrent: int = 1
    created_at: float = field(default_factory=time.time)
    last_activity: float = field(default_factory=time.time)
    processed_count: int = 0


class UnifiedQueueManager:
    """Unified queue manager for all channels and priorities.

    Features:
    - Dynamic consumer creation: no fixed worker pools
    - Per-session + per-priority isolation
    - Automatic idle queue cleanup
    - Metrics and monitoring

    Example:
        >>> async def consumer(q, ch_id, sess_id, priority):
        ...     while True:
        ...         payload = await q.get()
        ...         # Process payload
        ...
        >>> manager = UnifiedQueueManager(consumer_fn=consumer)
        >>> manager.start_cleanup_loop()
        >>> manager.enqueue("console", "console:user1", 0, payload)
    """

    def __init__(
        self,
        consumer_fn: ConsumerFn,
        queue_maxsize: int = 1000,
        idle_timeout: float = 600.0,  # 10 minutes
        cleanup_interval: float = 60.0,  # 1 minute
    ):
        """Initialize queue manager.

        Args:
            consumer_fn: Consumer function (queue, ch_id, sess_id, priority)
            queue_maxsize: Max size per queue (0 = unbounded)
            idle_timeout: Cleanup idle queues after N seconds
            cleanup_interval: Run cleanup every N seconds
        """
        self._consumer_fn = consumer_fn
        self._queue_maxsize = queue_maxsize
        self._idle_timeout = idle_timeout
        self._cleanup_interval = cleanup_interval

        # QueueKey → QueueState
        self._queues: Dict[QueueKey, QueueState] = {}

        # Lock for thread-safe access to _queues dict
        self._lock = asyncio.Lock()

        # Cleanup task
        self._cleanup_task: asyncio.Task[None] | None = None

        # Running flag
        self._running = False

        logger.debug(
            f"UnifiedQueueManager initialized: "
            f"maxsize={queue_maxsize}, "
            f"idle_timeout={idle_timeout}s, "
            f"cleanup_interval={cleanup_interval}s",
        )

    async def enqueue(
        self,
        channel_id: str,
        session_id: str,
        priority_level: int,
        payload: Any,
        max_concurrent: int = 1,
    ) -> None:
        """Enqueue a message for processing.

        Args:
            channel_id: Channel identifier (e.g. "console", "feishu")
            session_id: Normalized session ID (e.g. "console:user1")
            priority_level: Priority level (0 = critical, 20 = normal)
            payload: Message payload
            max_concurrent: Bounded number of concurrent consumers for this
                QueueKey. Values below 1 are treated as 1.

        Note:
            Creates queue and consumer on-demand if not exists
        """
        queue_key = (channel_id, session_id, priority_level)

        # Get or create queue and consumer
        state = await self._get_or_create_queue(queue_key, max_concurrent)

        # Update activity timestamp
        state.last_activity = time.time()

        # Enqueue payload with bounded wait to avoid indefinite blocking
        try:
            await asyncio.wait_for(state.queue.put(payload), timeout=30.0)
        except asyncio.TimeoutError:
            logger.warning(
                "Queue full timeout (30s): "
                f"channel={channel_id} "
                f"session={session_id[:30]} "
                f"priority={priority_level} "
                f"qsize={state.queue.qsize()}",
            )
            raise

        logger.debug(
            f"Enqueued: channel={channel_id} "
            f"session={session_id[:30]} "
            f"priority={priority_level} "
            f"qsize={state.queue.qsize()}",
        )

    async def _get_or_create_queue(
        self,
        queue_key: QueueKey,
        max_concurrent: int = 1,
    ) -> QueueState:
        """Get or create queue and consumer for the given key.

        Args:
            queue_key: (channel_id, session_id, priority_level)

        Returns:
            QueueState with queue and consumer task
        """
        async with self._lock:
            # Check if already exists
            max_concurrent = max(1, int(max_concurrent or 1))
            if queue_key in self._queues:
                state = self._queues[queue_key]
                if max_concurrent > state.max_concurrent:
                    state.max_concurrent = max_concurrent
                    self._ensure_worker_count_locked(queue_key, state)
                return state

            # Create new queue
            queue = asyncio.Queue(maxsize=self._queue_maxsize)

            # Create state
            state = QueueState(
                queue=queue,
                max_concurrent=max_concurrent,
            )

            # Store state
            self._queues[queue_key] = state
            self._ensure_worker_count_locked(queue_key, state)

            channel_id, session_id, priority_level = queue_key
            logger.info(
                f"Created queue: channel={channel_id} "
                f"session={session_id[:30]} "
                f"priority={priority_level} "
                f"max_concurrent={max_concurrent}",
            )

            return state

    def _ensure_worker_count_locked(
        self,
        queue_key: QueueKey,
        state: QueueState,
    ) -> None:
        """Start consumers until ``state.max_concurrent`` is satisfied.

        Caller holds ``self._lock``.
        """
        channel_id, session_id, priority_level = queue_key
        live_tasks = {t for t in state.consumer_tasks if not t.done()}
        state.consumer_tasks = live_tasks
        while len(state.consumer_tasks) < state.max_concurrent:
            worker_idx = len(state.consumer_tasks) + 1
            consumer_task = asyncio.create_task(
                self._run_consumer(
                    state.queue,
                    channel_id,
                    session_id,
                    priority_level,
                    worker_idx,
                    state.max_concurrent,
                ),
                name=(
                    f"consumer_{channel_id}_"
                    f"{session_id[:20]}_{priority_level}_{worker_idx}"
                ),
            )
            state.consumer_tasks.add(consumer_task)

    async def _run_consumer(
        self,
        queue: asyncio.Queue,
        channel_id: str,
        session_id: str,
        priority_level: int,
        worker_idx: int = 1,
        max_concurrent: int = 1,
    ) -> None:
        """Run consumer loop for a single queue.

        Args:
            queue: The queue to consume from
            channel_id: Channel identifier
            session_id: Normalized session ID
            priority_level: Priority level

        Note:
            This loop runs until cancelled or queue is idle for cleanup
        """
        queue_key = (channel_id, session_id, priority_level)

        logger.info(
            f"Consumer started: channel={channel_id} "
            f"session={session_id[:30]} "
            f"priority={priority_level} "
            f"worker={worker_idx}/{max_concurrent}",
        )

        try:
            # Delegate to the consumer function
            await self._consumer_fn(
                queue,
                channel_id,
                session_id,
                priority_level,
                max_concurrent,
            )
        except asyncio.CancelledError:
            logger.debug(
                f"Consumer cancelled: "
                f"channel={channel_id} "
                f"session={session_id[:30]} "
                f"priority={priority_level} "
                f"worker={worker_idx}",
            )
            raise
        except Exception:
            logger.exception(
                f"Consumer failed: "
                f"channel={channel_id} "
                f"session={session_id[:30]} "
                f"priority={priority_level} "
                f"worker={worker_idx}",
            )
        finally:
            # Remove this consumer.  The queue is dropped only when the last
            # worker exits; cleanup may already have popped it.
            current = asyncio.current_task()
            async with self._lock:
                state = self._queues.get(queue_key)
                if state is not None and current is not None:
                    state.consumer_tasks.discard(current)
                    if not state.consumer_tasks:
                        self._queues.pop(queue_key, None)

            logger.info(
                f"Consumer stopped: channel={channel_id} "
                f"session={session_id[:30]} "
                f"priority={priority_level} "
                f"worker={worker_idx}",
            )

    def start_cleanup_loop(self) -> None:
        """Start the background cleanup task.

        Note:
            Call this after manager initialization
        """
        if self._cleanup_task is not None:
            logger.warning("Cleanup loop already running")
            return

        self._running = True
        self._cleanup_task = asyncio.create_task(
            self._cleanup_idle_queues(),
            name="unified_queue_cleanup",
        )
        logger.info("Cleanup loop started")

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
        queue_key = (channel_id, session_id, priority_level)
        cleared_count = 0

        async with self._lock:
            state = self._queues.get(queue_key)
            if state:
                while not state.queue.empty():
                    try:
                        state.queue.get_nowait()
                        cleared_count += 1
                    except asyncio.QueueEmpty:
                        break

        if cleared_count > 0:
            logger.info(
                f"Cleared {cleared_count} messages from queue: "
                f"channel={channel_id} session={session_id[:30]} "
                f"priority={priority_level}",
            )

        return cleared_count

    async def stop_all(self) -> None:
        """Stop all consumers and cleanup task gracefully.

        Note:
            Call this during app shutdown
        """
        self._running = False

        # Cancel cleanup task
        if self._cleanup_task is not None:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
            self._cleanup_task = None

        # Cancel all consumer tasks
        async with self._lock:
            consumer_tasks = [
                task
                for state in self._queues.values()
                for task in state.consumer_tasks
            ]
            queue_count = len(self._queues)

        if consumer_tasks:
            for task in consumer_tasks:
                task.cancel()

            # Wait for all consumers to finish
            _, pending = await asyncio.wait(
                consumer_tasks,
                timeout=5.0,
                return_when=asyncio.ALL_COMPLETED,
            )

            if pending:
                logger.warning(
                    f"stop_all: {len(pending)} consumer(s) "
                    f"still pending after 5s",
                )

        # Clear queues dict
        async with self._lock:
            self._queues.clear()

        logger.info(f"Stopped all queues: total={queue_count}")

    async def _cleanup_idle_queues(self) -> None:
        """Background task to cleanup idle queues.

        Runs every cleanup_interval seconds and removes queues
        that have been empty and idle for longer than idle_timeout.
        """
        logger.info("Cleanup loop running")

        while self._running:
            try:
                await asyncio.sleep(self._cleanup_interval)

                now = time.time()
                to_cleanup: list[QueueKey] = []

                # Find idle queues
                async with self._lock:
                    for key, state in self._queues.items():
                        # Check if queue is empty and idle
                        if state.queue.empty():
                            idle_time = now - state.last_activity
                            if idle_time > self._idle_timeout:
                                to_cleanup.append(key)

                # Cancel idle consumers (outside lock)
                for key in to_cleanup:
                    cleanup_state: QueueState | None = None
                    async with self._lock:
                        if key in self._queues:
                            cleanup_state = self._queues.pop(key)

                    if cleanup_state is not None:
                        for task in cleanup_state.consumer_tasks:
                            task.cancel()
                        try:
                            await asyncio.gather(
                                *cleanup_state.consumer_tasks,
                                return_exceptions=True,
                            )
                        except asyncio.CancelledError:
                            pass

                        channel_id, session_id, priority_level = key
                        logger.info(
                            f"Cleaned up idle queue: "
                            f"channel={channel_id} "
                            f"session={session_id[:30]} "
                            f"priority={priority_level} "
                            f"processed={cleanup_state.processed_count}",
                        )

            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("Cleanup loop error")

        logger.info("Cleanup loop stopped")

    async def get_metrics(self) -> Dict[str, Any]:
        """Get queue metrics for monitoring.

        Returns:
            {
                "total_queues": int,
                "queues": [
                    {
                        "channel_id": str,
                        "session_id": str,
                        "priority_level": int,
                        "qsize": int,
                        "processed_count": int,
                        "age_seconds": float,
                        "idle_seconds": float,
                    },
                    ...
                ]
            }
        """
        now = time.time()
        queues_info = []

        async with self._lock:
            for key, state in self._queues.items():
                channel_id, session_id, priority_level = key
                queues_info.append(
                    {
                        "channel_id": channel_id,
                        "session_id": session_id,
                        "priority_level": priority_level,
                        "qsize": state.queue.qsize(),
                        "processed_count": state.processed_count,
                        "max_concurrent": state.max_concurrent,
                        "age_seconds": now - state.created_at,
                        "idle_seconds": now - state.last_activity,
                    },
                )

        return {
            "total_queues": len(queues_info),
            "queues": queues_info,
        }

    async def increment_processed(
        self,
        channel_id: str,
        session_id: str,
        priority_level: int,
        count: int = 1,
    ) -> None:
        """Increment processed count for a queue.

        Args:
            channel_id: Channel identifier
            session_id: Normalized session ID
            priority_level: Priority level
            count: Number to increment by

        Note:
            Called by consumer after processing messages
        """
        queue_key = (channel_id, session_id, priority_level)

        async with self._lock:
            state = self._queues.get(queue_key)
            if state is not None:
                state.processed_count += count
                state.last_activity = time.time()
