# -*- coding: utf-8 -*-
"""Shared formatting helpers for inbound channel messages.

The timestamp formatter lives here (rather than inside any one
channel) because every chat channel ends up needing the same thing:
render the upstream send timestamp in the host's local timezone so
the agent never has to reason across two zones.  Single source of
truth keeps the rendered shape consistent across WhatsApp / Signal /
Discord / Telegram so the model can rely on a single regex when
extracting times from the envelope.
"""

import datetime


def format_local_timestamp(ts, style: str = "long") -> str:
    """Render ``ts`` (epoch seconds, epoch ms, str, or
    ``datetime.datetime``) in the host's local timezone.

    ``style="long"``  → ``"2026年4月25日 19:40:11 JST"`` (history block)
    ``style="short"`` → ``"2026-04-25 19:40 JST"`` (envelope prefix)

    The trailing label is whatever the system reports via
    ``time.tzname`` for this moment (handles DST transitions
    correctly because we resolve via ``astimezone()`` per call).
    Returns ``""`` on any parse failure so the caller can substitute
    the raw value or omit the prefix without crashing.
    """
    try:
        if isinstance(ts, datetime.datetime):
            dt = ts.astimezone()
        else:
            ts_val = float(ts)
            if ts_val > 1e12:
                ts_val /= 1000  # epoch milliseconds → seconds
            dt = datetime.datetime.fromtimestamp(ts_val).astimezone()
    except (TypeError, ValueError, OverflowError):
        return ""
    tz_label = dt.strftime("%Z") or ""
    if style == "short":
        return (dt.strftime("%Y-%m-%d %H:%M ") + tz_label).rstrip()
    return (
        f"{dt.year}年{dt.month}月{dt.day}日 "
        + dt.strftime("%H:%M:%S ")
        + tz_label
    ).rstrip()
