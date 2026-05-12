# -*- coding: utf-8 -*-
"""Tests for UnifiedQueueManager bounded same-key concurrency."""

import asyncio

import pytest

from qwenpaw.app.channels.unified_queue_manager import UnifiedQueueManager


@pytest.mark.asyncio
async def test_same_key_can_process_in_parallel_when_configured():
    active = 0
    max_seen = 0
    started = 0
    both_started = asyncio.Event()

    async def consumer(queue, channel_id, session_id, priority_level, max_concurrent):
        nonlocal active, max_seen, started
        del channel_id, session_id, priority_level, max_concurrent
        while True:
            gate = await queue.get()
            active += 1
            started += 1
            max_seen = max(max_seen, active)
            if started == 2:
                both_started.set()
            await gate.wait()
            active -= 1

    manager = UnifiedQueueManager(
        consumer_fn=consumer,
        queue_maxsize=10,
        idle_timeout=60,
        cleanup_interval=60,
    )
    first = asyncio.Event()
    second = asyncio.Event()

    await manager.enqueue("whatsapp", "whatsapp:group:g", 20, first, max_concurrent=2)
    await manager.enqueue("whatsapp", "whatsapp:group:g", 20, second, max_concurrent=2)

    await asyncio.wait_for(both_started.wait(), timeout=1.0)
    assert max_seen == 2

    first.set()
    second.set()
    await asyncio.sleep(0)
    await manager.stop_all()


@pytest.mark.asyncio
async def test_same_key_stays_serial_by_default():
    active = 0
    max_seen = 0
    started = 0
    first_started = asyncio.Event()

    async def consumer(queue, channel_id, session_id, priority_level, max_concurrent):
        nonlocal active, max_seen, started
        del channel_id, session_id, priority_level, max_concurrent
        while True:
            gate = await queue.get()
            active += 1
            started += 1
            max_seen = max(max_seen, active)
            if started == 1:
                first_started.set()
            await gate.wait()
            active -= 1

    manager = UnifiedQueueManager(
        consumer_fn=consumer,
        queue_maxsize=10,
        idle_timeout=60,
        cleanup_interval=60,
    )
    first = asyncio.Event()
    second = asyncio.Event()

    await manager.enqueue("whatsapp", "whatsapp:group:g", 20, first)
    await manager.enqueue("whatsapp", "whatsapp:group:g", 20, second)

    await asyncio.wait_for(first_started.wait(), timeout=1.0)
    await asyncio.sleep(0.05)
    assert started == 1
    assert max_seen == 1

    first.set()
    await asyncio.sleep(0)
    second.set()
    await manager.stop_all()
