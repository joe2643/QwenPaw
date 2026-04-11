# -*- coding: utf-8 -*-
"""Unit/smoke tests for copaw.skill_review."""
import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# WAL sample fixture
# ---------------------------------------------------------------------------

WAL_SAMPLE = [
    {
        "ts": "2026-04-11T20:37:37.818003",
        "type": "tool_start",
        "tool": "tavily_search",
        "args": '{"query": "UA cinema ticket booking Hong Kong"}',
        "status": "done",
    },
    {
        "ts": "2026-04-11T20:37:45.782219",
        "type": "reasoning",
        "content": (
            "Found UA Cinemas booking flow: go to uacinemas.com.hk, "
            "select movie/date/seat, pay with Visa/Mastercard/AlipayHK. "
            "No login required. Official app recommended for exclusive deals."
        ),
    },
    {
        "ts": "2026-04-11T20:38:00.000000",
        "type": "sent",
        "content": (
            "直接去 UA Cinemas 官網 uacinemas.com.hk 揀戲院、場次、座位，"
            "再用 Visa/Mastercard 或者 AlipayHK 付款就得。"
        ),
    },
    {
        "ts": "2026-04-11T20:38:15.000000",
        "type": "reasoning",
        "content": (
            "User asked about buying cinema tickets for Detective Conan 2026. "
            "I searched for showtimes and found SCREEN 8 Dolby Cinema evening slots. "
            "Recommended 19:10 and 21:50 sessions. Explained purchasing via official app."
        ),
    },
    {
        "ts": "2026-04-11T20:38:30.000000",
        "type": "sent",
        "content": (
            "係，香港主要院線（UA、MCL、Broadway、Emperor）都係官網直購或者 App。"
            "熱門新片首週末通常係開售後幾分鐘就爆，要快。"
        ),
    },
]


def _write_wal(tmp: Path) -> None:
    wal_path = tmp / ".session_wal.jsonl"
    wal_path.write_text(
        "\n".join(json.dumps(e) for e in WAL_SAMPLE),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# _read_wal
# ---------------------------------------------------------------------------

class TestReadWal:
    def test_parses_all_event_types(self, tmp_path):
        from copaw.skill_review.review import _read_wal
        _write_wal(tmp_path)
        result = _read_wal(tmp_path)
        assert "[reasoning]" in result
        assert "[sent]" in result
        assert "[tool:tavily_search]" in result

    def test_empty_workspace_returns_empty_string(self, tmp_path):
        from copaw.skill_review.review import _read_wal
        result = _read_wal(tmp_path)
        assert result == ""

    def test_respects_max_entries(self, tmp_path):
        from copaw.skill_review.review import _read_wal
        # Write 300 entries with content long enough to pass the >20 char filter
        wal_path = tmp_path / ".session_wal.jsonl"
        lines = [
            json.dumps({"type": "reasoning", "content": f"message number {i} with enough length"})
            for i in range(300)
        ]
        wal_path.write_text("\n".join(lines), encoding="utf-8")
        result = _read_wal(tmp_path, max_entries=50)
        # Should only have entries from the tail; "message number 299" should be present
        assert "message number 299" in result


# ---------------------------------------------------------------------------
# State file multi-session isolation (Step 1 regression)
# ---------------------------------------------------------------------------

class TestStateFileIsolation:
    """Verify that two hook instances with different session_ids use separate state files."""

    def test_separate_state_files(self, tmp_path):
        from copaw.agents.hooks.mempalace_diary import MemPalaceIntervalHook

        hook_a = MemPalaceIntervalHook(
            working_dir=tmp_path, write_interval=15, session_id="session-aaa"
        )
        hook_b = MemPalaceIntervalHook(
            working_dir=tmp_path, write_interval=15, session_id="session-bbb"
        )

        assert hook_a.state_file != hook_b.state_file
        assert "session-aaa" in str(hook_a.state_file)
        assert "session-bbb" in str(hook_b.state_file)

    def test_session_a_write_does_not_affect_session_b(self, tmp_path):
        from copaw.agents.hooks.mempalace_diary import MemPalaceIntervalHook

        hook_a = MemPalaceIntervalHook(
            working_dir=tmp_path, write_interval=15, session_id="session-aaa"
        )
        hook_b = MemPalaceIntervalHook(
            working_dir=tmp_path, write_interval=15, session_id="session-bbb"
        )

        # Simulate session A having saved at count=15
        hook_a.last_write_count = 15
        hook_a._save_state()

        # session B should still see its own initial count (0)
        assert hook_b.last_write_count == 0

        # Reload hook B from disk — should not pick up session A's count
        hook_b2 = MemPalaceIntervalHook(
            working_dir=tmp_path, write_interval=15, session_id="session-bbb"
        )
        assert hook_b2.last_write_count == 0


# ---------------------------------------------------------------------------
# run_once — mocked LLM
# ---------------------------------------------------------------------------

class TestRunOnceMocked:
    """Test run_once() with mocked LLM to avoid live API calls."""

    def _make_llm_response(self, propose: bool, name: str = "test_skill") -> str:
        if not propose:
            return json.dumps({"propose": False})
        return json.dumps({
            "propose": True,
            "name": name,
            "description": "A test skill",
            "skill_md": "## Purpose\nTest.\n## Steps\n1. Do it.",
        })

    def test_no_proposal_when_wal_empty(self, tmp_path):
        from copaw.skill_review.review import run_once
        proposals = run_once("test", tmp_path, dry_run=True)
        assert proposals == []

    def test_proposal_parsed_correctly(self, tmp_path):
        from copaw.skill_review.review import run_once
        _write_wal(tmp_path)

        fake_response = self._make_llm_response(propose=True, name="hk_cinema_booking")
        with (
            patch("copaw.skill_review.review._load_api_config", return_value=("fake-key", "https://fake.api")),
            patch("copaw.skill_review.review._call_llm", return_value=fake_response),
            patch("copaw.skill_review.review._get_existing_skills", return_value="(none)"),
        ):
            proposals = run_once("test", tmp_path, dry_run=True)

        assert len(proposals) == 1
        assert proposals[0].name == "hk_cinema_booking"
        assert proposals[0].description == "A test skill"
        assert "## Purpose" in proposals[0].skill_md

    def test_no_proposal_when_llm_declines(self, tmp_path):
        from copaw.skill_review.review import run_once
        _write_wal(tmp_path)

        fake_response = self._make_llm_response(propose=False)
        with (
            patch("copaw.skill_review.review._load_api_config", return_value=("fake-key", "https://fake.api")),
            patch("copaw.skill_review.review._call_llm", return_value=fake_response),
            patch("copaw.skill_review.review._get_existing_skills", return_value="(none)"),
        ):
            proposals = run_once("test", tmp_path, dry_run=True)

        assert proposals == []

    def test_create_skill_called_when_not_dry_run(self, tmp_path):
        from copaw.skill_review.review import run_once
        _write_wal(tmp_path)

        fake_response = self._make_llm_response(propose=True, name="my_new_skill")
        mock_svc = MagicMock()
        mock_svc.create_skill.return_value = "my_new_skill"

        with (
            patch("copaw.skill_review.review._load_api_config", return_value=("fake-key", "https://fake.api")),
            patch("copaw.skill_review.review._call_llm", return_value=fake_response),
            patch("copaw.skill_review.review._get_existing_skills", return_value="(none)"),
            # SkillService is lazily imported inside run_once; patch at the source module
            patch("copaw.agents.skills_manager.SkillService", return_value=mock_svc),
        ):
            proposals = run_once("test", tmp_path, dry_run=False)

        assert len(proposals) == 1
        mock_svc.create_skill.assert_called_once_with(
            name="my_new_skill",
            content=proposals[0].skill_md,
            overwrite=False,
            enable=False,
            authored_by="skill_review",
        )

    def test_api_config_failure_returns_empty(self, tmp_path):
        from copaw.skill_review.review import run_once
        _write_wal(tmp_path)

        with patch(
            "copaw.skill_review.review._load_api_config",
            side_effect=FileNotFoundError("bailian.json not found"),
        ):
            proposals = run_once("test", tmp_path, dry_run=True)

        assert proposals == []


# ---------------------------------------------------------------------------
# Live smoke test (skipped unless credentials present)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not (Path.home() / ".copaw.secret" / "providers" / "custom" / "bailian.json").exists(),
    reason="bailian.json not found — live API test skipped",
)
class TestRunOnceLive:
    def test_live_dry_run(self, tmp_path):
        """End-to-end with real LLM (thinking=True). Skipped if no credentials."""
        from copaw.skill_review.review import run_once
        _write_wal(tmp_path)

        proposals = run_once("smoke_test", tmp_path, dry_run=True)

        # Both outcomes (propose / no-propose) are valid — just must not crash
        assert isinstance(proposals, list)
        for p in proposals:
            assert p.name
            assert "## " in p.skill_md  # Must have at least one section
