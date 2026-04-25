# -*- coding: utf-8 -*-
"""Tests for ``MultiAgentManager.reload_agent`` cooldown guard.

Background
----------
A silent writer storm on ``agent.json`` (e.g. an MCP tool or skill that
repeatedly flips ``active_model``) used to trigger a reload cascade: each
write → file watcher diff → ``schedule_agent_reload`` → full workspace
reload → replay of the last session message → channel send → repeat.

Fix: ``MultiAgentManager.reload_agent`` now enforces a cooldown so
successive reloads of the same agent within ``RELOAD_COOLDOWN_SECONDS``
are skipped with a warning. The return value distinguishes the three
outcomes via :class:`ReloadResult` so callers can report them accurately.
"""
# pylint: disable=protected-access
from __future__ import annotations

import asyncio
import logging
import time

import pytest

from qwenpaw.app.multi_agent_manager import (
    MultiAgentManager,
    ReloadResult,
)


class _FakeWorkspace:
    """Bare workspace stub that satisfies reload_agent's cooldown path.

    The cooldown-skip branches short-circuit before any real workspace
    operation, so an empty object is enough.
    """

    def __init__(self, agent_id: str):
        self.agent_id = agent_id


def _attach_module_logger(caplog):
    """Ensure records from the module logger reach caplog."""
    logger = logging.getLogger("qwenpaw.app.multi_agent_manager")
    logger.propagate = True
    logger.addHandler(caplog.handler)
    return logger


# ---------------------------------------------------------------------------
# ReloadResult bool semantics (backward compat)
# ---------------------------------------------------------------------------


def test_reload_result_bool_only_reloaded_is_truthy():
    """``if result:`` stays correct for legacy callers."""
    assert bool(ReloadResult.RELOADED) is True
    assert bool(ReloadResult.NOT_RUNNING) is False
    assert bool(ReloadResult.SKIPPED_COOLDOWN) is False


# ---------------------------------------------------------------------------
# Cooldown skip path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cooldown_skips_rapid_second_reload(caplog):
    """A second reload within the cooldown window returns SKIPPED_COOLDOWN."""
    mgr = MultiAgentManager()
    mgr.RELOAD_COOLDOWN_SECONDS = 5.0

    agent_id = "default"
    mgr.agents[agent_id] = _FakeWorkspace(agent_id)
    mgr._last_reload_at[agent_id] = time.monotonic()

    caplog.set_level(logging.WARNING)
    _attach_module_logger(caplog)

    result = await mgr.reload_agent(agent_id)

    assert result is ReloadResult.SKIPPED_COOLDOWN
    assert mgr._reload_skip_count[agent_id] == 1


@pytest.mark.asyncio
async def test_cooldown_skip_count_accumulates(caplog):
    """Repeat skips bump the counter; once it crosses 10, log level escalates."""
    mgr = MultiAgentManager()
    mgr.RELOAD_COOLDOWN_SECONDS = 60.0

    agent_id = "default"
    mgr.agents[agent_id] = _FakeWorkspace(agent_id)
    mgr._last_reload_at[agent_id] = time.monotonic()

    caplog.set_level(logging.WARNING)
    _attach_module_logger(caplog)

    for _ in range(11):
        assert (
            await mgr.reload_agent(agent_id) is ReloadResult.SKIPPED_COOLDOWN
        )

    assert mgr._reload_skip_count[agent_id] == 11
    levels = {
        r.levelno for r in caplog.records if "skip_count" in r.getMessage()
    }
    assert (
        logging.ERROR in levels
    ), "storm escalation (≥10 skips) should emit at least one ERROR record"


@pytest.mark.asyncio
async def test_cooldown_warning_explains_why(caplog):
    """The skip warning includes 'cooldown', 'skip_count', and the reason hint."""
    mgr = MultiAgentManager()
    mgr.RELOAD_COOLDOWN_SECONDS = 30.0

    agent_id = "default"
    mgr.agents[agent_id] = _FakeWorkspace(agent_id)
    mgr._last_reload_at[agent_id] = time.monotonic() - 5.0

    caplog.set_level(logging.WARNING)
    _attach_module_logger(caplog)

    result = await mgr.reload_agent(agent_id)
    assert result is ReloadResult.SKIPPED_COOLDOWN

    messages = [r.getMessage() for r in caplog.records]
    assert any(
        "reload_agent skipped" in m
        and "cooldown" in m
        and "skip_count" in m
        and "save_agent_config callers" in m
        for m in messages
    ), f"Expected diagnostic-rich cooldown warning; got: {messages}"


# ---------------------------------------------------------------------------
# Cooldown-not-triggered paths
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cooldown_is_per_agent(caplog):
    """An agent in cooldown must not block a different agent's guard check."""
    mgr = MultiAgentManager()
    mgr.RELOAD_COOLDOWN_SECONDS = 60.0

    mgr.agents["A"] = _FakeWorkspace("A")
    mgr.agents["B"] = _FakeWorkspace("B")
    mgr._last_reload_at["A"] = time.monotonic()

    caplog.set_level(logging.WARNING)
    _attach_module_logger(caplog)

    # A: in cooldown
    assert await mgr.reload_agent("A") is ReloadResult.SKIPPED_COOLDOWN

    a_skip_warnings = [
        r
        for r in caplog.records
        if "reload_agent skipped" in r.getMessage()
        and "for A" in r.getMessage()
    ]
    assert len(a_skip_warnings) == 1

    # B: different agent — should PASS the cooldown guard.
    # We can't run the full reload path without Workspace fixtures, so we
    # assert at the log level: calling reload for B must not produce an
    # additional "skipped for B" warning. (Downstream errors are fine.)
    try:
        await mgr.reload_agent("B")
    except Exception:
        pass

    b_skip_warnings = [
        r
        for r in caplog.records
        if "reload_agent skipped" in r.getMessage()
        and "for B" in r.getMessage()
    ]
    assert not b_skip_warnings, (
        "B should have passed the cooldown guard; "
        f"got skip warnings: {[r.getMessage() for r in b_skip_warnings]}"
    )


@pytest.mark.asyncio
async def test_not_running_returns_not_running_enum():
    """Unknown agent returns NOT_RUNNING (not SKIPPED_COOLDOWN)."""
    mgr = MultiAgentManager()
    result = await mgr.reload_agent("ghost")
    assert result is ReloadResult.NOT_RUNNING


@pytest.mark.asyncio
async def test_cooldown_zero_disables_guard(caplog):
    """Setting cooldown to 0 disables the guard entirely (for dev use)."""
    mgr = MultiAgentManager()
    mgr.RELOAD_COOLDOWN_SECONDS = 0.0

    agent_id = "default"
    mgr.agents[agent_id] = _FakeWorkspace(agent_id)
    mgr._last_reload_at[agent_id] = time.monotonic()

    caplog.set_level(logging.WARNING)
    _attach_module_logger(caplog)

    try:
        await mgr.reload_agent(agent_id)
    except Exception:
        pass

    skip_records = [
        r for r in caplog.records if "reload_agent skipped" in r.getMessage()
    ]
    assert (
        not skip_records
    ), "cooldown=0 must never short-circuit the reload path"


# ---------------------------------------------------------------------------
# Concurrent reloads
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_concurrent_reloads_respect_cooldown(caplog):
    """Ten simultaneous reloads → at most one runs; the rest are skipped.

    Regression guard for the race between the lock release (after swap) and
    a second caller that entered the lock acquire queue while the first was
    still inside the critical section. The cooldown read + update sit in
    the same critical section, so all but the "winner" must observe a
    freshly-set timestamp and return SKIPPED_COOLDOWN.
    """
    mgr = MultiAgentManager()
    mgr.RELOAD_COOLDOWN_SECONDS = 60.0

    agent_id = "default"
    mgr.agents[agent_id] = _FakeWorkspace(agent_id)
    # Seed with an old timestamp so the first caller passes the guard.
    mgr._last_reload_at[agent_id] = time.monotonic() - 1000.0

    caplog.set_level(logging.WARNING)
    _attach_module_logger(caplog)

    # Fire 10 reloads in parallel; swallow downstream errors from missing
    # workspace fixtures — we only care about cooldown-vs-pass semantics.
    async def one_reload():
        try:
            return await mgr.reload_agent(agent_id)
        except Exception:
            # The first task (which passes the guard) will fail in the
            # workspace init pipeline; that's expected with fake fixtures.
            return None

    results = await asyncio.gather(*(one_reload() for _ in range(10)))
    skipped = sum(1 for r in results if r is ReloadResult.SKIPPED_COOLDOWN)

    # Exactly one task should have passed the guard; the rest must be
    # reported as cooldown-skipped. Because the passing task errors out
    # before reaching the atomic-swap timestamp update, later callers may
    # not see a fresh timestamp — so we accept either ≥9 or 10 skipped as
    # long as no fewer than 9 were guarded.
    assert (
        skipped >= 9
    ), f"cooldown must serialize concurrent reloads (got {skipped}/10 skipped)"
