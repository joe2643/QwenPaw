# -*- coding: utf-8 -*-
"""Unit tests for session-scoped SessionWAL isolation.

Regression coverage for the cross-channel crash-recovery leak:
a pending ``tool_start`` from a Signal session must not surface
as a crash banner in the next WhatsApp session for the same
agent.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from qwenpaw.agents.hooks.tool_wal import (
    SessionWAL,
    _session_wal_filename,
)


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    """Fresh ``working_dir`` — one per test so WAL files don't leak
    between cases."""
    return tmp_path


# ---------------------------------------------------------------- #
# Filename resolution                                              #
# ---------------------------------------------------------------- #


class TestFilenameResolution:
    def test_none_or_default_yields_legacy_filename(self):
        # Kept for tests / tools that predate session scoping —
        # never pass ``None`` in production.
        assert _session_wal_filename(None) == ".session_wal.jsonl"
        assert _session_wal_filename("") == ".session_wal.jsonl"
        assert _session_wal_filename("default") == ".session_wal.jsonl"

    def test_session_id_gets_hashed_into_filename(self):
        fname = _session_wal_filename("signal:group:abc")
        assert fname.startswith(".session_wal.")
        assert fname.endswith(".jsonl")
        # 12-hex-char digest between the dots.
        parts = fname.split(".")
        assert len(parts[2]) == 12
        assert all(c in "0123456789abcdef" for c in parts[2])

    def test_same_session_id_yields_same_filename(self):
        # Stable hash — required so a session that crashed can find
        # its own WAL on restart.
        a = _session_wal_filename("signal:group:abc")
        b = _session_wal_filename("signal:group:abc")
        assert a == b

    def test_different_session_ids_yield_different_filenames(self):
        a = _session_wal_filename("signal:group:abc")
        b = _session_wal_filename("whatsapp:group:xyz")
        assert a != b

    def test_special_chars_in_session_id_dont_break_path(self):
        # Real session ids contain ``:`` and ``/`` (``signal:group:...``,
        # ``whatsapp:group:120363421135228220@g.us``).  The hash has
        # to be a valid filename on every OS.
        fname = _session_wal_filename(
            "whatsapp:group:120363421135228220@g.us",
        )
        assert "/" not in fname
        assert ":" not in fname
        assert "@" not in fname


# ---------------------------------------------------------------- #
# WAL instance uses the right file                                 #
# ---------------------------------------------------------------- #


class TestSessionWALInstance:
    def test_session_id_routes_writes_to_scoped_file(
        self,
        workspace: Path,
    ):
        wal = SessionWAL(workspace, session_id="signal:group:abc")
        wal.log_tool_start("bash", "ls -la")
        # The scoped file exists; the legacy file does not.
        expected_name = _session_wal_filename("signal:group:abc")
        assert (workspace / expected_name).exists()
        assert not (workspace / ".session_wal.jsonl").exists()

    def test_no_session_id_writes_to_legacy_file(
        self,
        workspace: Path,
    ):
        # Back-compat path — keep working for tests that predate
        # session scoping.
        wal = SessionWAL(workspace)
        wal.log_tool_start("bash", "ls")
        assert (workspace / ".session_wal.jsonl").exists()

    def test_wal_file_override_wins_over_session_id(
        self,
        workspace: Path,
    ):
        # Tests sometimes want to poke a specific filename.
        wal = SessionWAL(
            workspace,
            session_id="signal:group:abc",
            wal_file=".custom.jsonl",
        )
        wal.log_tool_start("bash", "x")
        assert (workspace / ".custom.jsonl").exists()

    def test_rotation_still_works_per_scoped_file(
        self,
        workspace: Path,
    ):
        wal = SessionWAL(workspace, session_id="signal:group:rot")
        # Push enough entries to trip rotation (_WAL_MAX_LINES = 200).
        for i in range(250):
            wal.log_tool_start(f"tool_{i}", f"args_{i}")
        # After rotation: file shrinks to the second half
        # (_maybe_rotate keeps ``len//2:``).
        line_count = len(wal.wal_path.read_text().strip().split("\n"))
        assert line_count <= 200
        assert line_count > 50  # sanity — rotation didn't nuke everything


# ---------------------------------------------------------------- #
# Crash-report isolation (THE regression guard)                    #
# ---------------------------------------------------------------- #


class TestCrashReportIsolation:
    def test_pending_on_session_a_does_not_leak_to_session_b(
        self,
        workspace: Path,
    ):
        # Session A starts a tool and crashes without log_tool_done.
        wal_a = SessionWAL(workspace, session_id="signal:group:abc")
        wal_a.log_tool_start("image_gen", "prompt=莊方宜")
        # No matching log_tool_done → status stays "pending".

        # Session B is a fresh channel/chat for the same agent.
        report_b = SessionWAL.get_crash_report(
            workspace,
            session_id="whatsapp:group:xyz",
        )
        # The load-bearing assertion — prior to session scoping this
        # was the leak that rebroadcast the Signal image-gen task
        # into a WhatsApp group.
        assert report_b is None

    def test_session_a_sees_its_own_crash(
        self,
        workspace: Path,
    ):
        # Positive case: same session id, crash visible.
        wal = SessionWAL(workspace, session_id="signal:group:abc")
        wal.log_tool_start("image_gen", "prompt=x")
        report = SessionWAL.get_crash_report(
            workspace,
            session_id="signal:group:abc",
        )
        assert report is not None
        assert "CRASH RECOVERY" in report
        assert "image_gen" in report

    def test_completed_tool_does_not_trigger_crash(
        self,
        workspace: Path,
    ):
        wal = SessionWAL(workspace, session_id="signal:group:abc")
        wal.log_tool_start("bash", "ls")
        wal.log_tool_done("bash")
        report = SessionWAL.get_crash_report(
            workspace,
            session_id="signal:group:abc",
        )
        assert report is None

    def test_legacy_unscoped_wal_does_not_contaminate_scoped_session(
        self,
        workspace: Path,
    ):
        # A pre-upgrade installation might still have
        # ``.session_wal.jsonl`` on disk from before this change.
        # A new *scoped* session must not pick it up.
        legacy = SessionWAL(workspace)  # no session_id → legacy file
        legacy.log_tool_start("old_task", "stuff")
        # Fresh session after upgrade.
        report = SessionWAL.get_crash_report(
            workspace,
            session_id="signal:group:abc",
        )
        assert report is None

    def test_crash_detection_marks_pending_as_crashed(
        self,
        workspace: Path,
    ):
        # After get_crash_report finds a pending entry and reports
        # it, it rewrites the entry as ``status: "crashed"`` so the
        # next call is a no-op (recovery fires once per crash).
        sid = "signal:group:abc"
        wal = SessionWAL(workspace, session_id=sid)
        wal.log_tool_start("image_gen", "x")
        first = SessionWAL.get_crash_report(workspace, session_id=sid)
        second = SessionWAL.get_crash_report(workspace, session_id=sid)
        assert first is not None
        assert second is None

    def test_reports_list_most_recent_pending(
        self,
        workspace: Path,
    ):
        # When multiple tool_starts are pending in the same session
        # (nested / interrupted), the reported one is the MOST
        # RECENT pending entry.
        wal = SessionWAL(workspace, session_id="signal:group:abc")
        wal.log_tool_start("older_tool", "x")
        wal.log_tool_start("newer_tool", "y")
        report = SessionWAL.get_crash_report(
            workspace,
            session_id="signal:group:abc",
        )
        assert "newer_tool" in report


# ---------------------------------------------------------------- #
# get_recent mirrors the scoping                                   #
# ---------------------------------------------------------------- #


def test_get_recent_is_scoped(workspace: Path) -> None:
    wal_a = SessionWAL(workspace, session_id="sig:a")
    wal_a.log_tool_start("tool_a", "x")
    wal_b = SessionWAL(workspace, session_id="wa:b")
    wal_b.log_tool_start("tool_b", "y")
    recent_a = SessionWAL.get_recent(workspace, session_id="sig:a")
    recent_b = SessionWAL.get_recent(workspace, session_id="wa:b")
    # Each side sees only its own event.
    assert [e["tool"] for e in recent_a] == ["tool_a"]
    assert [e["tool"] for e in recent_b] == ["tool_b"]
