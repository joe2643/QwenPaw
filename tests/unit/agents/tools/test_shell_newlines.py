# -*- coding: utf-8 -*-
"""Regression tests for ``_collapse_embedded_newlines`` on Unix.

On 2026-04-24 the collapse-outside-quotes path silently broke every
shell heredoc and every multi-line command.  LLMs kept complaining
"heredoc 壞咗 / 無法寫 multi-line script" because their tool calls
were getting mangled before ``sh -c`` even saw them.  These tests
lock in the fix (pass-through on Unix) so it can't regress.
"""
from __future__ import annotations

import sys

import pytest

from qwenpaw.agents.tools.shell import _collapse_embedded_newlines


@pytest.mark.skipif(sys.platform == "win32", reason="Unix-only fix")
class TestUnixPassThrough:
    def test_heredoc_body_preserved(self):
        # Heredocs need the newline AFTER ``<<TOKEN`` to open the
        # body and the one BEFORE ``TOKEN`` on its own line to close
        # it.  Collapsing either to a space turns the whole thing
        # into a line that makes ``sh`` hang reading stdin.
        cmd = "cat <<'EOF'\nline1\nline2\nEOF"
        result = _collapse_embedded_newlines(cmd)
        assert result == cmd, (
            "heredoc newlines must be preserved — collapsing them "
            "breaks every heredoc the LLM writes."
        )

    def test_heredoc_with_dash_strip_tabs(self):
        # ``<<-EOF`` variant strips leading tabs; the newlines still
        # need to be preserved or the stripping happens on a
        # single-line string and produces garbage.
        cmd = "cat <<-'EOF'\n\tindented\nEOF"
        assert _collapse_embedded_newlines(cmd) == cmd

    def test_multi_command_separator_preserved(self):
        # ``git status\ngit diff`` must stay two commands.  The old
        # code collapsed the newline to a space → ``git status git diff``
        # which parses as one bogus ``git`` invocation.
        cmd = "git status\ngit diff"
        assert _collapse_embedded_newlines(cmd) == cmd

    def test_quoted_newline_preserved(self):
        cmd = 'echo "line1\nline2"'
        assert _collapse_embedded_newlines(cmd) == cmd

    def test_no_newline_returns_input(self):
        cmd = "ls -la /tmp"
        assert _collapse_embedded_newlines(cmd) is cmd


@pytest.mark.skipif(sys.platform != "win32", reason="Windows-only behaviour")
class TestWindowsCollapse:
    def test_windows_still_collapses_all_newlines(self):
        # cmd.exe truncates at the first newline regardless of
        # quoting, so on Windows we still collapse everything to
        # spaces.  Otherwise the command wouldn't even execute.
        cmd = "git status\ngit diff"
        result = _collapse_embedded_newlines(cmd)
        assert "\n" not in result
        assert result == "git status git diff"
