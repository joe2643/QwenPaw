# -*- coding: utf-8 -*-
"""Task tracker for background runs: streaming, reconnect, multi-subscriber.

For internal streaming runs, ``run_key`` is typically ``ChatSpec.id``
(chat_id). For externally-managed tasks (registered via
:meth:`TaskTracker.register_external_task`), ``run_key`` is an opaque
identifier chosen by the caller (e.g. a UUID prefixed with ``"ext-"``).
Per run: task, queues, event buffer. Reconnects get buffer replay + new
events. Cleanup when task completes.
"""
from __future__ import annotations

import asyncio
import json
import logging
import uuid
import weakref
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, AsyncGenerator, Callable, Coroutine, Optional

logger = logging.getLogger(__name__)

_SENTINEL = None


@dataclass
class _RunState:
    """Per-run state (task, queues, buffer), guarded by tracker lock."""

    task: asyncio.Future
    queues: list[asyncio.Queue] = field(default_factory=list)
    buffer: list[str] = field(default_factory=list)
    pending_inputs: list[Any] = field(default_factory=list)
    parent_key: str | None = None
    # Set once the producer task emits its first SSE event. Steer input
    # that arrives before this is technically pre-stream and will be
    # picked up by the very first ``_drain_pending_steer_messages`` at
    # the top of ``_reasoning``.
    streaming_started: bool = False
    # Set true while the agent is inside a compaction LLM call (driven
    # by ``LightContextManager.pre_reasoning``). Steer input is still
    # accepted but is logged so timing is observable in logs.
    compacting: bool = False
    start_time: Optional[datetime] = None
    finish_time: Optional[datetime] = None


class TaskTracker:
    """Per-workspace tracker: run_key -> RunState.

    All mutations to _runs under _lock. Producer broadcasts under lock.
    Subscribers use unbounded per-connection queues; disconnect removes them
    via :meth:`detach_subscriber`.
    """

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._runs: dict[str, _RunState] = {}
        self._child_runs: dict[str, set[str]] = {}
        self._queue_run_keys = weakref.WeakKeyDictionary()
        self._global_last_run_at: Optional[datetime] = None
        self._global_last_finish_at: Optional[datetime] = None

    @property
    def lock(self) -> asyncio.Lock:
        return self._lock

    async def get_status(self, run_key: str) -> str:
        """Return ``'idle'`` or ``'running'``."""
        async with self._lock:
            state = self._runs.get(run_key)
            child_keys = self._active_child_keys_locked(run_key)
            if child_keys:
                return "running"
        if state is None or state.task.done():
            return "idle"
        return "running"

    async def get_global_status(self) -> dict:
        """Get global agent status summary.

        Returns:
            dict with keys:
                - status: 'idle' | 'running'
                - running_task_count: int
                - last_run_at: Optional[datetime]
                - last_finish_at: Optional[datetime]
        """
        async with self._lock:
            running_count = sum(
                1 for state in self._runs.values() if not state.task.done()
            )
            status = "running" if running_count > 0 else "idle"

            return {
                "status": status,
                "running_task_count": running_count,
                "last_run_at": self._global_last_run_at,
                "last_finish_at": self._global_last_finish_at,
            }

    async def has_active_tasks(self) -> bool:
        """Check if any tasks are currently running.

        Returns:
            bool: True if any tasks are active, False otherwise
        """
        async with self._lock:
            for state in self._runs.values():
                if not state.task.done():
                    return True
            return False

    async def list_active_tasks(self) -> list[str]:
        """List all currently running task keys.

        Returns:
            list[str]: List of active run_keys
        """
        async with self._lock:
            return [
                run_key
                for run_key, state in self._runs.items()
                if not state.task.done()
            ]

    async def wait_all_done(self, timeout: float = 300.0) -> bool:
        """Wait for all active tasks to complete.

        Args:
            timeout: Maximum time to wait in seconds (default: 300s = 5min)

        Returns:
            bool: True if all tasks completed, False if timeout occurred
        """

        async def _wait_loop() -> None:
            while await self.has_active_tasks():
                await asyncio.sleep(0.5)

        try:
            await asyncio.wait_for(_wait_loop(), timeout=timeout)
            return True
        except asyncio.TimeoutError:
            return False

    async def register_external_task(self, run_key: str) -> None:
        """Register an externally-managed task so it is visible to
        :meth:`has_active_tasks` and :meth:`wait_all_done`.

        This is used for tasks managed outside of QwenPaw's own streaming
        pipeline (e.g. background tasks dispatched through
        ``agentscope_runtime``'s ``AgentApp``).  The caller **must** call
        :meth:`unregister_external_task` when the task completes.

        Args:
            run_key: Unique identifier for the external task.
        """
        start_time = datetime.now(timezone.utc)
        async with self._lock:
            if run_key in self._runs and not self._runs[run_key].task.done():
                logger.debug(
                    "External task already registered: %s",
                    run_key,
                )
                return
            # Use an unresolved Future as the "task" — it stays not-done
            # until unregister_external_task resolves it.
            future: asyncio.Future = asyncio.get_running_loop().create_future()
            self._runs[run_key] = _RunState(
                task=future,
                queues=[],
                buffer=[],
                start_time=start_time,
            )
            self._global_last_run_at = start_time
            logger.debug("Registered external task: %s", run_key)

    async def unregister_external_task(self, run_key: str) -> None:
        """Mark an externally-managed task as done and remove it.

        Sends the sentinel value to any subscriber queues so their
        :meth:`stream_from_queue` consumers terminate cleanly instead of
        hanging. Idempotent — safe to call even if *run_key* was never
        registered or was already unregistered.

        Args:
            run_key: Unique identifier previously passed to
                :meth:`register_external_task`.
        """
        finish_time = datetime.now(timezone.utc)
        async with self._lock:
            state = self._runs.pop(run_key, None)
            if state is None:
                return
            self._remove_child_locked(run_key, state)
            # Notify any subscriber queues so consumers exit cleanly.
            for q in state.queues:
                q.put_nowait(_SENTINEL)
            if not state.task.done():
                state.task.set_result(None)
            state.finish_time = finish_time
            self._global_last_finish_at = finish_time
            logger.debug("Unregistered external task: %s", run_key)

    async def attach(self, run_key: str) -> asyncio.Queue | None:
        """Attach to an existing run.

        Returns a new queue pre-filled with the event buffer, or ``None``
        if no run is active for *run_key*.
        """
        async with self._lock:
            concrete_key, state = self._select_attach_state_locked(run_key)
            if state is None or state.task.done():
                return None
            q: asyncio.Queue = asyncio.Queue()
            for sse in state.buffer:
                q.put_nowait(sse)
            state.queues.append(q)
            self._queue_run_keys[q] = concrete_key
            return q

    async def detach_subscriber(
        self,
        run_key: str,
        queue: asyncio.Queue,
    ) -> None:
        """Remove *queue* from *run_key*'s subscriber list.

        Idempotent if the run ended or *queue* was already removed.
        """
        async with self._lock:
            concrete_key = self._queue_run_keys.get(queue, run_key)
            state = self._runs.get(concrete_key)
            if state is None:
                return
            try:
                state.queues.remove(queue)
            except ValueError:
                pass
            self._queue_run_keys.pop(queue, None)

    async def request_stop(self, run_key: str) -> bool:
        """Cancel the run. Returns ``True`` if it was running."""
        logger.debug("[STOP] request_stop called for run_key=%s", run_key)
        async with self._lock:
            target_keys = [run_key]
            target_keys.extend(sorted(self._child_runs.get(run_key, set())))
            states = [
                self._runs[key]
                for key in target_keys
                if key in self._runs and not self._runs[key].task.done()
            ]
            logger.debug(
                "[STOP] run_key=%s running_count=%s",
                run_key,
                len(states),
            )
            if not states:
                logger.debug(
                    "[STOP] Cannot stop run_key=%s (not running)",
                    run_key,
                )
                return False
            logger.debug(
                "[STOP] Calling task.cancel() for run_key=%s",
                run_key,
            )
            for state in states:
                state.task.cancel()
            logger.debug("[STOP] task.cancel() called for run_key=%s", run_key)
            return True

    async def attach_or_start(
        self,
        run_key: str,
        payload: Any,
        stream_fn: Callable[..., Coroutine],
        max_concurrent_runs: int = 1,
    ) -> tuple[asyncio.Queue, bool]:
        """Attach to an existing run or start a new one.

        Returns ``(queue, is_new_run)``.
        """
        async with self._lock:
            max_concurrent_runs = max(1, int(max_concurrent_runs or 1))
            concrete_run_key = run_key
            parent_key: str | None = None

            if max_concurrent_runs <= 1:
                state = self._runs.get(run_key)
            else:
                active_children = self._active_child_keys_locked(run_key)
                if len(active_children) >= max_concurrent_runs:
                    concrete_run_key, state = self._select_attach_state_locked(
                        run_key,
                    )
                else:
                    state = None
                    parent_key = run_key
                    concrete_run_key = (
                        f"{run_key}::run:{uuid.uuid4().hex[:12]}"
                    )

            if state is not None and not state.task.done():
                q: asyncio.Queue = asyncio.Queue()
                for sse in state.buffer:
                    q.put_nowait(sse)
                state.queues.append(q)
                self._queue_run_keys[q] = concrete_run_key
                return q, False

            my_queue: asyncio.Queue = asyncio.Queue()
            run = _RunState(
                task=asyncio.Future(),  # placeholder, replaced below
                queues=[my_queue],
                buffer=[],
                parent_key=parent_key,
            )
            self._runs[concrete_run_key] = run
            self._queue_run_keys[my_queue] = concrete_run_key
            if parent_key is not None:
                self._child_runs.setdefault(parent_key, set()).add(
                    concrete_run_key,
                )

            tracker_ref = weakref.ref(self)

            async def _producer() -> None:
                start_time = datetime.now(timezone.utc)

                try:
                    tracker = tracker_ref()
                    if tracker is not None:
                        async with tracker.lock:
                            run.start_time = start_time
                            # pylint: disable=protected-access
                            tracker._global_last_run_at = start_time

                    async for sse in stream_fn(payload):
                        tracker = tracker_ref()
                        if tracker is None:
                            return
                        async with tracker.lock:
                            if not run.streaming_started:
                                run.streaming_started = True
                            run.buffer.append(sse)
                            for q in run.queues:
                                q.put_nowait(sse)
                except asyncio.CancelledError:
                    logger.debug("run cancelled run_key=%s", concrete_run_key)
                except Exception:
                    logger.exception("run error run_key=%s", concrete_run_key)
                    err_sse = (
                        "data: "
                        f"{json.dumps({'error': 'internal server error'})}\n\n"
                    )
                    tracker = tracker_ref()
                    if tracker is not None:
                        async with tracker.lock:
                            run.buffer.append(err_sse)
                            for q in run.queues:
                                q.put_nowait(err_sse)
                finally:
                    finish_time = datetime.now(timezone.utc)
                    tracker = tracker_ref()
                    if tracker is not None:
                        async with tracker.lock:
                            run.finish_time = finish_time
                            # pylint: disable=protected-access
                            tracker._global_last_finish_at = finish_time
                            for q in run.queues:
                                q.put_nowait(_SENTINEL)
                            # pylint: disable=protected-access
                            tracker._runs.pop(
                                concrete_run_key,
                                None,
                            )
                            tracker._remove_child_locked(
                                concrete_run_key,
                                run,
                            )

            run.task = asyncio.create_task(_producer())
            return my_queue, True

    async def enqueue_pending_input(
        self,
        run_key: str,
        pending_input: Any | list[Any],
    ) -> asyncio.Queue | None:
        """Queue same-turn input for an active run and attach to its stream.

        Returns a subscriber queue for the active run, or ``None`` when there
        is no live run for *run_key*.

        Steer that arrives before the producer streams its first event, or
        while the agent is mid-compaction, is still accepted (drained at
        the next ``_reasoning`` boundary) but is logged so timing can be
        traced if a steer feels "lost".
        """
        async with self._lock:
            concrete_key, state = self._select_attach_state_locked(run_key)
            if state is None or state.task.done():
                return None

            if isinstance(pending_input, list):
                state.pending_inputs.extend(pending_input)
                count = len(pending_input)
            else:
                state.pending_inputs.append(pending_input)
                count = 1

            if state.compacting:
                logger.info(
                    "Steer queued during compaction run_key=%s count=%d "
                    "(will inject after compaction LLM call returns)",
                    concrete_key,
                    count,
                )
            elif not state.streaming_started:
                logger.info(
                    "Steer queued before streaming started run_key=%s count=%d "
                    "(will inject at first reasoning step)",
                    concrete_key,
                    count,
                )

            q: asyncio.Queue = asyncio.Queue()
            for sse in state.buffer:
                q.put_nowait(sse)
            state.queues.append(q)
            self._queue_run_keys[q] = concrete_key
            return q

    async def mark_compacting(self, run_key: str, flag: bool) -> bool:
        """Toggle the compacting flag for a run.

        Called by the agent's pre_reasoning compaction hook around the
        compactor LLM call so :meth:`enqueue_pending_input` can log the
        timing of any steer that arrives during compaction.

        Returns ``True`` if the flag was applied, ``False`` if no live
        run is found for *run_key*.
        """
        async with self._lock:
            _concrete_key, state = self._select_attach_state_locked(run_key)
            if state is None or state.task.done():
                return False
            state.compacting = bool(flag)
            return True

    async def is_compacting(self, run_key: str) -> bool:
        """Return whether the run is currently inside a compaction LLM call."""
        async with self._lock:
            _concrete_key, state = self._select_attach_state_locked(run_key)
            if state is None or state.task.done():
                return False
            return state.compacting

    async def is_streaming(self, run_key: str) -> bool:
        """Return whether the producer has emitted its first SSE event."""
        async with self._lock:
            _concrete_key, state = self._select_attach_state_locked(run_key)
            if state is None or state.task.done():
                return False
            return state.streaming_started

    async def drain_pending_input(self, run_key: str) -> list[Any]:
        """Drain pending same-turn input for an active run."""
        async with self._lock:
            _concrete_key, state = self._select_attach_state_locked(run_key)
            if state is None or state.task.done():
                return []
            pending = list(state.pending_inputs)
            state.pending_inputs.clear()
            return pending

    def _active_child_keys_locked(self, parent_key: str) -> list[str]:
        """Return live concrete run keys for a parent chat key."""
        keys = list(self._child_runs.get(parent_key, set()))
        live: list[str] = []
        for key in keys:
            state = self._runs.get(key)
            if state is not None and not state.task.done():
                live.append(key)
        if live:
            self._child_runs[parent_key] = set(live)
        else:
            self._child_runs.pop(parent_key, None)
        return live

    def _select_attach_state_locked(
        self,
        run_key: str,
    ) -> tuple[str, _RunState | None]:
        """Pick an active concrete state for reconnect/overflow attach."""
        state = self._runs.get(run_key)
        if state is not None and not state.task.done():
            return run_key, state
        for child_key in sorted(self._active_child_keys_locked(run_key)):
            child = self._runs.get(child_key)
            if child is not None and not child.task.done():
                return child_key, child
        return run_key, None

    def _remove_child_locked(
        self,
        concrete_run_key: str,
        state: _RunState,
    ) -> None:
        """Detach a completed child from its parent mapping."""
        parent_key = state.parent_key
        if not parent_key:
            return
        children = self._child_runs.get(parent_key)
        if not children:
            return
        children.discard(concrete_run_key)
        if not children:
            self._child_runs.pop(parent_key, None)

    async def stream_from_queue(
        self,
        queue: asyncio.Queue,
        run_key: str,
    ) -> AsyncGenerator[str, None]:
        """Yield SSE strings from *queue* until the sentinel ``None``.

        Always detaches *queue* from *run_key* when this stream ends or is
        closed (including client disconnect), so reconnects do not leak queues.
        """
        try:
            while True:
                try:
                    event = await queue.get()
                    if event is _SENTINEL:
                        break
                    yield event
                except asyncio.CancelledError:
                    break
        finally:
            await self.detach_subscriber(run_key, queue)
