# -*- coding: utf-8 -*-
"""Shared timestamp-formatting helper used by every chat channel
so the agent gets a consistent envelope shape across surfaces."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from qwenpaw.app.channels._format import format_local_timestamp


def test_short_style_includes_date_and_time():
    out = format_local_timestamp(1777106276, style="short")
    assert "-04-" in out
    assert ":" in out


def test_long_style_uses_chinese_date_format():
    out = format_local_timestamp(1777106276, style="long")
    assert "年" in out and "月" in out and "日" in out
    # H:M:S present
    assert out.count(":") >= 2


def test_handles_milliseconds_same_as_seconds():
    """WhatsApp / Signal sometimes hand us ms-since-epoch."""
    s_secs = format_local_timestamp(1777106276, style="short")
    s_ms = format_local_timestamp(1777106276 * 1000, style="short")
    assert s_secs == s_ms


def test_handles_aware_datetime():
    """An aware datetime in a different zone still renders in the
    host's local zone — single-zone output keeps the agent from
    having to translate."""
    dt_utc = datetime(2026, 4, 25, 8, 37, 56, tzinfo=timezone.utc)
    out = format_local_timestamp(dt_utc, style="long")
    # Whatever the host zone, day is the same as for the epoch case.
    epoch_out = format_local_timestamp(1777106276, style="long")
    assert out == epoch_out


def test_invalid_input_returns_empty_string():
    assert format_local_timestamp(None) == ""
    assert format_local_timestamp("not a number") == ""
    assert format_local_timestamp({}) == ""
