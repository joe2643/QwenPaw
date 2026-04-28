# -*- coding: utf-8 -*-
"""Lightweight in-process counters for the claude-acpx provider.

Tracks how often each turn took the seed_full vs ship_tail vs reseed
path.  Reading these counters lets ops verify the cache thesis
(codex C10) post-launch:

  ship_tail / total > ~70%  →  cache prefix is staying stable, plan
                                paid off
  reseed     / total > ~30%  →  drift detection firing too often,
                                investigate

Counters are process-local, not persisted.  Exposed via the same
``token_usage`` dashboard surface in a follow-up; for now caller can
log or expose via an internal endpoint.

Counters are deliberately not asyncio.Lock-protected: increments are
single-statement on CPython integer adds, which are atomic by GIL.
Cost of an occasional missed count under contention is far below the
noise floor of "did the design pay off".
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class _Counters:
    seed_full: int = 0
    ship_tail: int = 0
    reseed: int = 0
    effort_set: int = 0
    tear_down: int = 0
    error: int = 0

    def total_turns(self) -> int:
        return self.seed_full + self.ship_tail + self.reseed

    def hit_ratio(self) -> float:
        """ship_tail share of total turns.  Returns 0.0 when no
        turns recorded yet (avoids ZeroDivision)."""
        t = self.total_turns()
        return (self.ship_tail / t) if t else 0.0

    def as_dict(self) -> dict[str, int | float]:
        return {
            "seed_full": self.seed_full,
            "ship_tail": self.ship_tail,
            "reseed": self.reseed,
            "effort_set": self.effort_set,
            "tear_down": self.tear_down,
            "error": self.error,
            "total_turns": self.total_turns(),
            "hit_ratio": round(self.hit_ratio(), 4),
        }


_GLOBAL = _Counters()


def record_seed_full() -> None:
    _GLOBAL.seed_full += 1


def record_ship_tail() -> None:
    _GLOBAL.ship_tail += 1


def record_reseed() -> None:
    _GLOBAL.reseed += 1


def record_effort_set() -> None:
    _GLOBAL.effort_set += 1


def record_tear_down() -> None:
    _GLOBAL.tear_down += 1


def record_error() -> None:
    _GLOBAL.error += 1


def snapshot() -> dict[str, int | float]:
    """Read-only view of current counter values."""
    return _GLOBAL.as_dict()


def reset_for_test() -> None:
    """Reset all counters to zero — tests only."""
    global _GLOBAL  # noqa: PLW0603
    _GLOBAL = _Counters()
