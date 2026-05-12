# -*- coding: utf-8 -*-
"""Console router helper tests."""

from qwenpaw.app.routers.console import (
    _console_parallel_max_concurrent_runs,
    _is_immediate_console_control_command,
)
from qwenpaw.config import AgentsRunningConfig


def test_console_stop_command_bypasses_parallel_run_attachment():
    """Typed /stop must not consume a parallel child-run slot."""
    payload = {
        "content_parts": [
            {
                "type": "text",
                "text": "/stop",
            },
        ],
    }

    assert _is_immediate_console_control_command(payload) is True


def test_console_normal_message_uses_same_session_mode():
    payload = {
        "content_parts": [
            {
                "type": "text",
                "text": "hello",
            },
        ],
    }

    assert _is_immediate_console_control_command(payload) is False


def test_console_parallel_max_runs_uses_running_config():
    config = AgentsRunningConfig(same_session_parallel_max_runs=5)

    assert _console_parallel_max_concurrent_runs(config) == 5


def test_console_parallel_max_runs_defaults_for_legacy_config():
    class LegacyRunningConfig:
        pass

    assert _console_parallel_max_concurrent_runs(LegacyRunningConfig()) == 3
