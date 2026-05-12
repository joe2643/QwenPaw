#!/usr/bin/env python3
"""End-to-end probe of the steer pipeline using the real production code.

What this exercises:
  1. console router decision: same_session_mode == "steer" → tracker.enqueue_pending_input
  2. tracker.enqueue_pending_input -> state.pending_inputs
  3. tracker.drain_pending_input -> consumed list
  4. agent._drain_pending_steer_messages -> memory.add(pending_msgs)

It uses a fake stream task (gate-controlled) and a stub agent so no LLM is called.
"""
from __future__ import annotations

import asyncio
import sys
from types import SimpleNamespace
from typing import Any

from agentscope.message import Msg

from qwenpaw.app.runner.task_tracker import TaskTracker


# ── helpers ────────────────────────────────────────────────────────────


class _FakeMemory:
    def __init__(self):
        self.added: list[Msg] = []

    async def add(self, msgs):
        if isinstance(msgs, list):
            self.added.extend(msgs)
        else:
            self.added.append(msgs)


class _FakeAgent:
    """Drives ``_drain_pending_steer_messages`` against the real method."""

    def __init__(self, tracker: TaskTracker, chat_id: str):
        self._task_tracker = tracker
        self._request_context = {"chat_id": chat_id}
        self.memory = _FakeMemory()

    # Bind the real method off QwenPawAgent so we test the production logic.
    from qwenpaw.agents.react_agent import QwenPawAgent  # noqa: E402

    _drain_pending_steer_messages = (
        QwenPawAgent._drain_pending_steer_messages
    )


def _print(label: str, **fields: Any) -> None:
    body = " ".join(f"{k}={v!r}" for k, v in fields.items())
    print(f"[{label}] {body}", flush=True)


# ── steps ──────────────────────────────────────────────────────────────


async def main() -> int:
    failures: list[str] = []
    chat_id = "probe-chat-1"
    tracker = TaskTracker()
    gate = asyncio.Event()
    tick_count = {"n": 0}

    async def fake_stream(_payload):
        tick_count["n"] += 1
        yield "data: started\n\n"
        await gate.wait()
        yield "data: done\n\n"

    # 1) Start a run (simulates first user message hitting attach_or_start)
    queue, is_new = await tracker.attach_or_start(
        chat_id, {"text": "first message"}, fake_stream,
    )
    assert is_new is True, "attach_or_start should report a new run"
    _print(
        "step1.start",
        is_new=is_new,
        queue_attached=queue is not None,
    )

    # 2) Enqueue a steer message while the run is still alive
    pending_msgs = [Msg("user", "second message", "user")]
    attach_q = await tracker.enqueue_pending_input(chat_id, pending_msgs)
    if attach_q is None:
        failures.append(
            "enqueue_pending_input returned None — run wasn't found",
        )
    else:
        _print("step2.enqueue", attached=True)

    # 3) Drive the agent's drain logic against the live tracker
    agent = _FakeAgent(tracker, chat_id)
    drained = await agent._drain_pending_steer_messages()
    _print(
        "step3.drain",
        drained_count=len(drained),
        memory_added_count=len(agent.memory.added),
        first_text=(
            drained[0].get_text_content() if drained else "<empty>"
        ),
    )
    if len(drained) != 1:
        failures.append(
            f"expected 1 drained msg, got {len(drained)}",
        )
    if len(agent.memory.added) != 1:
        failures.append(
            f"expected 1 memory.add call, got {len(agent.memory.added)}",
        )
    if drained and drained[0].get_text_content() != "second message":
        failures.append(
            f"drained text mismatch: {drained[0].get_text_content()!r}",
        )

    # 4) Drain a second time should return nothing (queue cleared)
    drained2 = await agent._drain_pending_steer_messages()
    _print("step4.drain_again", drained_count=len(drained2))
    if drained2:
        failures.append(
            f"second drain should be empty, got {len(drained2)}",
        )

    # 5) Stop the run cleanly so we exit
    gate.set()
    async for _ in tracker.stream_from_queue(queue, chat_id):
        pass
    done = await tracker.wait_all_done(timeout=2.0)
    _print("step5.shutdown", all_done=done)

    # 6) After run finishes, enqueue should refuse (no active run)
    refused_q = await tracker.enqueue_pending_input(
        chat_id, Msg("user", "too late", "user"),
    )
    _print("step6.no_run_refusal", returned_none=refused_q is None)
    if refused_q is not None:
        failures.append(
            "enqueue_pending_input should return None when run is done",
        )

    print()
    if failures:
        print("FAIL:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("PASS — steer queue + drain pipeline behaves as expected.")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
