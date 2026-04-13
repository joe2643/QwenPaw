# -*- coding: utf-8 -*-
"""Unit/smoke tests for qwenpaw.skill_review."""
import json
import subprocess
import tempfile
import unittest.mock
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
        from qwenpaw.skill_review.review import _read_wal
        _write_wal(tmp_path)
        result = _read_wal(tmp_path)
        assert "[reasoning]" in result
        assert "[sent]" in result
        assert "[tool:tavily_search]" in result

    def test_empty_workspace_returns_empty_string(self, tmp_path):
        from qwenpaw.skill_review.review import _read_wal
        result = _read_wal(tmp_path)
        assert result == ""

    def test_respects_max_entries(self, tmp_path):
        from qwenpaw.skill_review.review import _read_wal
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
        from qwenpaw.agents.hooks.mempalace_diary import MemPalaceIntervalHook

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
        from qwenpaw.agents.hooks.mempalace_diary import MemPalaceIntervalHook

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

    def _make_llm_response(
        self, propose: bool, name: str = "test_skill", action: str = "create"
    ) -> str:
        if not propose:
            return json.dumps({"propose": False})
        return json.dumps({
            "propose": True,
            "action": action,
            "name": name,
            "description": "A test skill",
            "skill_md": "## Purpose\nTest.\n## Steps\n1. Do it.",
        })

    def test_no_proposal_when_wal_empty(self, tmp_path):
        from qwenpaw.skill_review.review import run_once
        proposals = run_once("test", tmp_path, dry_run=True)
        assert proposals == []

    def test_proposal_parsed_correctly(self, tmp_path):
        from qwenpaw.skill_review.review import run_once
        _write_wal(tmp_path)

        fake_response = self._make_llm_response(propose=True, name="hk_cinema_booking")
        with (
            patch("qwenpaw.skill_review.review._load_api_config", return_value=("fake-key", "https://fake.api")),
            patch("qwenpaw.skill_review.review._call_llm", return_value=fake_response),
            patch("qwenpaw.skill_review.review._get_existing_skills", return_value="(none)"),
        ):
            proposals = run_once("test", tmp_path, dry_run=True)

        assert len(proposals) == 1
        assert proposals[0].name == "hk_cinema_booking"
        assert proposals[0].description == "A test skill"
        assert "## Purpose" in proposals[0].skill_md

    def test_no_proposal_when_llm_declines(self, tmp_path):
        from qwenpaw.skill_review.review import run_once
        _write_wal(tmp_path)

        fake_response = self._make_llm_response(propose=False)
        with (
            patch("qwenpaw.skill_review.review._load_api_config", return_value=("fake-key", "https://fake.api")),
            patch("qwenpaw.skill_review.review._call_llm", return_value=fake_response),
            patch("qwenpaw.skill_review.review._get_existing_skills", return_value="(none)"),
        ):
            proposals = run_once("test", tmp_path, dry_run=True)

        assert proposals == []

    def test_create_skill_called_when_not_dry_run(self, tmp_path):
        from qwenpaw.skill_review.review import run_once
        _write_wal(tmp_path)

        fake_response = self._make_llm_response(propose=True, name="my_new_skill")
        mock_svc = MagicMock()
        mock_svc.create_skill.return_value = "my_new_skill"

        with (
            patch("qwenpaw.skill_review.review._load_api_config", return_value=("fake-key", "https://fake.api")),
            patch("qwenpaw.skill_review.review._call_llm", return_value=fake_response),
            patch("qwenpaw.skill_review.review._get_existing_skills", return_value="(none)"),
            # SkillService is lazily imported inside run_once; patch at the source module
            patch("qwenpaw.agents.skills_manager.SkillService", return_value=mock_svc),
        ):
            proposals = run_once("test", tmp_path, dry_run=False)

        assert len(proposals) == 1
        mock_svc.create_skill.assert_called_once_with(
            name="my_new_skill",
            content=proposals[0].skill_md,
            overwrite=False,
            enable=True,
            authored_by="skill_review",
        )

    def test_api_config_failure_returns_empty(self, tmp_path):
        from qwenpaw.skill_review.review import run_once
        _write_wal(tmp_path)

        with patch(
            "qwenpaw.skill_review.review._load_api_config",
            side_effect=FileNotFoundError("bailian.json not found"),
        ):
            proposals = run_once("test", tmp_path, dry_run=True)

        assert proposals == []


# ---------------------------------------------------------------------------
# _send_notification
# ---------------------------------------------------------------------------

class TestSendNotification:
    """Test _send_notification without network calls."""

    def _expected_cmd(self, agent: str = "default") -> list:
        from qwenpaw.skill_review.review import (
            NOTIFICATION_CHANNEL,
            NOTIFICATION_TARGET_SESSION,
            NOTIFICATION_TARGET_USER,
        )
        return [
            "qwenpaw", "channels", "send",
            "--agent-id", agent,
            "--channel", NOTIFICATION_CHANNEL,
            "--target-user", NOTIFICATION_TARGET_USER,
            "--target-session", NOTIFICATION_TARGET_SESSION,
            "--text", unittest.mock.ANY,  # checked separately
        ]

    def test_success_returns_true(self):
        from qwenpaw.skill_review.review import _send_notification
        mock_result = MagicMock()
        mock_result.returncode = 0
        with patch("subprocess.run", return_value=mock_result) as mock_sub:
            result = _send_notification("my_skill", "A test skill", "default")
        assert result is True
        mock_sub.assert_called_once()
        call_args = mock_sub.call_args
        cmd = call_args[0][0]
        assert cmd[:2] == ["qwenpaw", "channels"]
        assert "--agent-id" in cmd
        assert "my_skill" in call_args[0][0][-1]  # skill name in --text
        assert call_args.kwargs["timeout"] == 30

    def test_nonzero_exit_returns_false(self):
        from qwenpaw.skill_review.review import _send_notification
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stderr = "connection refused"
        with patch("subprocess.run", return_value=mock_result):
            result = _send_notification("bad_skill", "desc", "default")
        assert result is False

    def test_timeout_returns_false(self):
        from qwenpaw.skill_review.review import _send_notification
        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired(cmd="qwenpaw", timeout=30)):
            result = _send_notification("slow_skill", "desc", "default")
        assert result is False

    def test_exception_returns_false_does_not_raise(self):
        from qwenpaw.skill_review.review import _send_notification
        with patch("subprocess.run", side_effect=FileNotFoundError("copaw not found")):
            result = _send_notification("any_skill", "desc", "default")
        assert result is False

    def test_text_contains_skill_name_and_description(self):
        from qwenpaw.skill_review.review import _send_notification
        mock_result = MagicMock()
        mock_result.returncode = 0
        with patch("subprocess.run", return_value=mock_result) as mock_sub:
            _send_notification("hk_cinema", "Book HK cinema tickets", "default")
        cmd = mock_sub.call_args[0][0]
        text_idx = cmd.index("--text") + 1
        text = cmd[text_idx]
        assert "hk_cinema" in text
        assert "Book HK cinema tickets" in text
        assert "default" in text  # agent name in enable hint


# ---------------------------------------------------------------------------
# run_once — notification integration
# ---------------------------------------------------------------------------

class TestRunOnceNotification:
    """Verify notification is called/suppressed correctly in run_once."""

    def _make_llm_response(self, name: str = "notif_skill", action: str = "create") -> str:
        return json.dumps({
            "propose": True,
            "action": action,
            "name": name,
            "description": "Notification test skill",
            "skill_md": "## Purpose\nTest.\n## Steps\n1. Done.",
        })

    def test_dry_run_does_not_fire_notification(self, tmp_path):
        from qwenpaw.skill_review.review import run_once
        _write_wal(tmp_path)
        with (
            patch("qwenpaw.skill_review.review._load_api_config", return_value=("k", "https://x")),
            patch("qwenpaw.skill_review.review._call_llm", return_value=self._make_llm_response()),
            patch("qwenpaw.skill_review.review._get_existing_skills", return_value="(none)"),
            patch("qwenpaw.skill_review.review._send_notification") as mock_notif,
        ):
            run_once("test", tmp_path, dry_run=True)
        mock_notif.assert_not_called()

    def test_notification_false_does_not_fire(self, tmp_path):
        from qwenpaw.skill_review.review import run_once
        _write_wal(tmp_path)
        mock_svc = MagicMock()
        mock_svc.create_skill.return_value = "notif_skill"
        with (
            patch("qwenpaw.skill_review.review._load_api_config", return_value=("k", "https://x")),
            patch("qwenpaw.skill_review.review._call_llm", return_value=self._make_llm_response()),
            patch("qwenpaw.skill_review.review._get_existing_skills", return_value="(none)"),
            patch("qwenpaw.agents.skills_manager.SkillService", return_value=mock_svc),
            patch("qwenpaw.skill_review.review._send_notification") as mock_notif,
        ):
            run_once("test", tmp_path, dry_run=False, notification=False)
        mock_notif.assert_not_called()

    def test_normal_path_fires_notification_once(self, tmp_path):
        from qwenpaw.skill_review.review import run_once
        _write_wal(tmp_path)
        mock_svc = MagicMock()
        mock_svc.create_skill.return_value = "notif_skill"
        with (
            patch("qwenpaw.skill_review.review._load_api_config", return_value=("k", "https://x")),
            patch("qwenpaw.skill_review.review._call_llm", return_value=self._make_llm_response()),
            patch("qwenpaw.skill_review.review._get_existing_skills", return_value="(none)"),
            patch("qwenpaw.agents.skills_manager.SkillService", return_value=mock_svc),
            patch("qwenpaw.skill_review.review._send_notification") as mock_notif,
        ):
            proposals = run_once("test", tmp_path, dry_run=False, notification=True)
        assert len(proposals) == 1
        mock_notif.assert_called_once_with(
            skill_name="notif_skill",
            description="Notification test skill",
            agent="test",
            action="create",
        )

    def test_notification_not_fired_when_skill_already_exists(self, tmp_path):
        """create_skill returns None when skill exists — no notification."""
        from qwenpaw.skill_review.review import run_once
        _write_wal(tmp_path)
        mock_svc = MagicMock()
        mock_svc.create_skill.return_value = None  # already exists
        with (
            patch("qwenpaw.skill_review.review._load_api_config", return_value=("k", "https://x")),
            patch("qwenpaw.skill_review.review._call_llm", return_value=self._make_llm_response()),
            patch("qwenpaw.skill_review.review._get_existing_skills", return_value="(none)"),
            patch("qwenpaw.agents.skills_manager.SkillService", return_value=mock_svc),
            patch("qwenpaw.skill_review.review._send_notification") as mock_notif,
        ):
            run_once("test", tmp_path, dry_run=False, notification=True)
        mock_notif.assert_not_called()


# ---------------------------------------------------------------------------
# Q3: Auto-enable + Q2: Update path
# ---------------------------------------------------------------------------

class TestAutoEnableAndUpdatePath:
    """Verify Hermes-style auto-enable and update path behaviour."""

    def _make_create_response(self, name: str = "auto_skill") -> str:
        return json.dumps({
            "propose": True,
            "action": "create",
            "name": name,
            "description": "Auto-enabled skill",
            "skill_md": "## Purpose\nTest.\n## Steps\n1. Done.",
        })

    def _make_update_response(self, name: str = "existing_skill") -> str:
        return json.dumps({
            "propose": True,
            "action": "update",
            "name": name,
            "description": "Updated skill",
            "skill_md": "## Purpose\nUpdated.\n## Steps\n1. Better.",
        })

    def test_new_skill_auto_enabled(self, tmp_path):
        """create path must pass enable=True (Hermes-style auto-enable)."""
        from qwenpaw.skill_review.review import run_once
        _write_wal(tmp_path)
        mock_svc = MagicMock()
        mock_svc.create_skill.return_value = "auto_skill"

        with (
            patch("qwenpaw.skill_review.review._load_api_config", return_value=("k", "https://x")),
            patch("qwenpaw.skill_review.review._call_llm", return_value=self._make_create_response()),
            patch("qwenpaw.skill_review.review._get_existing_skills", return_value="(none)"),
            patch("qwenpaw.agents.skills_manager.SkillService", return_value=mock_svc),
            patch("qwenpaw.skill_review.review._send_notification"),
        ):
            proposals = run_once("test", tmp_path, dry_run=False)

        assert len(proposals) == 1
        mock_svc.create_skill.assert_called_once_with(
            name="auto_skill",
            content=proposals[0].skill_md,
            overwrite=False,
            enable=True,
            authored_by="skill_review",
        )

    def test_update_path_calls_overwrite(self, tmp_path):
        """update path must pass overwrite=True so existing skill is replaced."""
        from qwenpaw.skill_review.review import run_once
        _write_wal(tmp_path)
        mock_svc = MagicMock()
        mock_svc.create_skill.return_value = "existing_skill"

        with (
            patch("qwenpaw.skill_review.review._load_api_config", return_value=("k", "https://x")),
            patch("qwenpaw.skill_review.review._call_llm", return_value=self._make_update_response()),
            patch("qwenpaw.skill_review.review._get_existing_skills", return_value="- existing_skill: old desc"),
            patch("qwenpaw.agents.skills_manager.SkillService", return_value=mock_svc),
            patch("qwenpaw.skill_review.review._send_notification"),
        ):
            proposals = run_once("test", tmp_path, dry_run=False)

        assert len(proposals) == 1
        assert proposals[0].action == "update"
        mock_svc.create_skill.assert_called_once_with(
            name="existing_skill",
            content=proposals[0].skill_md,
            overwrite=True,
            enable=True,
            authored_by="skill_review",
        )

    def test_notification_create_template_contains_auto_enabled(self):
        """Create notification must mention that skill is already live."""
        from qwenpaw.skill_review.review import _send_notification
        mock_result = MagicMock()
        mock_result.returncode = 0
        with patch("subprocess.run", return_value=mock_result) as mock_sub:
            _send_notification("new_skill", "A brand new skill", "default", action="create")
        cmd = mock_sub.call_args[0][0]
        text = cmd[cmd.index("--text") + 1]
        assert "已啟用" in text
        assert "new_skill" in text
        assert "A brand new skill" in text

    def test_notification_update_template_contains_update_wording(self):
        """Update notification must mention skill was updated and state preserved."""
        from qwenpaw.skill_review.review import _send_notification
        mock_result = MagicMock()
        mock_result.returncode = 0
        with patch("subprocess.run", return_value=mock_result) as mock_sub:
            _send_notification("old_skill", "Improved skill", "default", action="update")
        cmd = mock_sub.call_args[0][0]
        text = cmd[cmd.index("--text") + 1]
        assert "更新咗" in text
        assert "old_skill" in text
        assert "原有 enabled state 保留" in text


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
        from qwenpaw.skill_review.review import run_once
        _write_wal(tmp_path)

        proposals = run_once("smoke_test", tmp_path, dry_run=True)

        # Both outcomes (propose / no-propose) are valid — just must not crash
        assert isinstance(proposals, list)
        for p in proposals:
            assert p.name
            assert "## " in p.skill_md  # Must have at least one section


# ---------------------------------------------------------------------------
# User context loading tests
# ---------------------------------------------------------------------------

class TestLoadUserContext:
    """Tests for _load_user_context() — PROFILE.md loading with identity stripping."""

    def test_user_context_loaded(self, tmp_path):
        """PROFILE.md exists → user profile sections returned, identity stripped."""
        from qwenpaw.skill_review.review import _load_user_context

        profile = (
            "## 身份\n\n"
            "- **名字：** 夕慶 (Yūkei) / Vesper\n"
            "- **定位：** OpenClaw AI\n\n"
            "## 用户资料\n\n"
            "- **名字：** joe\n"
            "- **時區：** UTC+9\n"
        )
        (tmp_path / "PROFILE.md").write_text(profile)

        result = _load_user_context(tmp_path)

        assert "joe" in result
        assert "UTC+9" in result
        assert "Vesper" not in result
        assert "夕慶" not in result
        assert "身份" not in result

    def test_user_context_missing_file(self, tmp_path):
        """PROFILE.md doesn't exist → returns empty string, no crash."""
        from qwenpaw.skill_review.review import _load_user_context

        result = _load_user_context(tmp_path)

        assert result == ""

    def test_user_context_strips_identity(self, tmp_path):
        """## Identity (English heading) is also stripped."""
        from qwenpaw.skill_review.review import _load_user_context

        profile = (
            "## Identity\n\n"
            "- Name: Vesper\n"
            "- Role: AI Agent\n\n"
            "## User Profile\n\n"
            "- Name: joe\n"
            "- Timezone: UTC+9\n"
        )
        (tmp_path / "PROFILE.md").write_text(profile)

        result = _load_user_context(tmp_path)

        assert "joe" in result
        assert "Vesper" not in result
        assert "AI Agent" not in result

    def test_user_context_truncation(self, tmp_path):
        """Content exceeding max_chars is truncated."""
        from qwenpaw.skill_review.review import _load_user_context

        profile = "## User Profile\n\n" + ("x" * 3000)
        (tmp_path / "PROFILE.md").write_text(profile)

        result = _load_user_context(tmp_path, max_chars=500)

        assert len(result) < 600  # 500 + "[...truncated]" + some header
        assert "[...truncated]" in result

    def test_user_context_with_frontmatter(self, tmp_path):
        """YAML frontmatter is stripped before processing."""
        from qwenpaw.skill_review.review import _load_user_context

        profile = (
            "---\n"
            "title: Agent Profile\n"
            "---\n\n"
            "## 用户资料\n\n"
            "- **名字：** joe\n"
        )
        (tmp_path / "PROFILE.md").write_text(profile)

        result = _load_user_context(tmp_path)

        assert "joe" in result
        assert "title:" not in result
