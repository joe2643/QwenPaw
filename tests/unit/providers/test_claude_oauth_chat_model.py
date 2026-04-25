# -*- coding: utf-8 -*-
"""Unit tests for the Claude Code OAuth wrapper around AnthropicChatModel.

These tests cover the pure helpers — tool-name prefix transform,
identity-preamble injection, haiku kwarg stripping — plus the subclass
MRO resolution that CoPaw's ``model_factory`` depends on.  We stay
away from exercising a real ``client.messages.create`` call here; the
live OAuth round-trip is already covered by the smoke test in
``claude_auth.py``'s ``_smoke``.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from qwenpaw.providers.anthropic_provider import (
    ClaudeOAuthChatModel,
    MCP_TOOL_PREFIX,
    _TOOL_NAME_REVERSE_MAP,
    _inject_identity_system,
    _prefix_tool_name,
    _record_tool_name_mapping,
    _rewrite_history_tool_names_outbound,
    _strip_haiku_incompatible_kwargs,
    _strip_tool_use_names_inplace,
    _unprefix_tool_name_heuristic,
)
from qwenpaw.providers.claude_auth import CLAUDE_CODE_IDENTITY


# ---------------------------------------------------------------- #
# Identity preamble injection                                      #
# ---------------------------------------------------------------- #


class TestInjectIdentitySystem:
    def test_empty_system_yields_identity_only(self):
        result = _inject_identity_system(None, CLAUDE_CODE_IDENTITY)
        assert result == [{"type": "text", "text": CLAUDE_CODE_IDENTITY}]

    def test_string_system_is_prepended_not_merged(self):
        result = _inject_identity_system("be terse", CLAUDE_CODE_IDENTITY)
        # Must be two blocks — Anthropic validates byte-equality on
        # block 0 and rejects the request if the identity is
        # concatenated with the caller's system.
        assert len(result) == 2
        assert result[0] == {"type": "text", "text": CLAUDE_CODE_IDENTITY}
        assert result[1] == {"type": "text", "text": "be terse"}

    def test_list_system_is_prepended(self):
        caller = [
            {"type": "text", "text": "hello"},
            {"type": "text", "text": "world"},
        ]
        result = _inject_identity_system(caller, CLAUDE_CODE_IDENTITY)
        assert len(result) == 3
        assert result[0]["text"] == CLAUDE_CODE_IDENTITY
        assert result[1:] == caller

    def test_identity_injection_is_idempotent(self):
        once = _inject_identity_system("hello", CLAUDE_CODE_IDENTITY)
        twice = _inject_identity_system(once, CLAUDE_CODE_IDENTITY)
        # Already starts with identity — don't double it.
        assert once == twice


# ---------------------------------------------------------------- #
# Tool-name mcp_ prefix / strip transform                          #
# ---------------------------------------------------------------- #


class TestToolNamePrefix:
    @pytest.mark.parametrize(
        "original, expected_prefixed",
        [
            ("bash", "mcp_Bash"),
            ("read_file", "mcp_Read_file"),  # first char only uppercased
            ("fooBar", "mcp_FooBar"),
            ("Edit", "mcp_Edit"),  # already upper; no change to tail
            ("1tool", "mcp_1tool"),  # digit passes through
            ("", ""),  # empty stays empty
        ],
    )
    def test_prefix_rule_matches_opencode_convention(
        self,
        original: str,
        expected_prefixed: str,
    ):
        assert _prefix_tool_name(original) == expected_prefixed

    def test_heuristic_strip_is_lossless_for_lowercase_first(self):
        # The bug-compatible heuristic from opencode-claude-auth —
        # round-trips correctly ONLY for lowercase-first names.
        for name in ("bash", "read_file", "fooBar"):
            roundtrip = _unprefix_tool_name_heuristic(
                _prefix_tool_name(name),
            )
            assert roundtrip == name

    def test_heuristic_strip_is_lossy_for_pascalcase_first(self):
        # ``"Edit"`` → ``"mcp_Edit"`` heuristic-strip → ``"edit"``.
        # This is the reason we maintain a per-call reverse map.
        assert _unprefix_tool_name_heuristic("mcp_Edit") == "edit"

    def test_reverse_map_recovers_pascalcase_first(self):
        reverse: dict[str, str] = {}
        token = _TOOL_NAME_REVERSE_MAP.set(reverse)
        try:
            for orig in ("bash", "Edit", "fooBar", "list_files"):
                prefixed = _prefix_tool_name(orig)
                _record_tool_name_mapping(orig, prefixed)
            # Simulate a ChatResponse with prefixed names on the way back.
            resp = SimpleNamespace(
                content=[
                    {
                        "type": "tool_use",
                        "id": "1",
                        "name": "mcp_Bash",
                        "input": {},
                    },
                    {
                        "type": "tool_use",
                        "id": "2",
                        "name": "mcp_Edit",
                        "input": {},
                    },
                    {
                        "type": "tool_use",
                        "id": "3",
                        "name": "mcp_FooBar",
                        "input": {},
                    },
                    {"type": "text", "text": "ok"},
                ],
            )
            _strip_tool_use_names_inplace(resp, reverse)
            names = [
                b["name"] for b in resp.content if b.get("type") == "tool_use"
            ]
            # Lossless — "Edit" stays "Edit", "fooBar" recovers camelCase.
            assert names == ["bash", "Edit", "fooBar"]
        finally:
            _TOOL_NAME_REVERSE_MAP.reset(token)


class TestHistoryRewrite:
    def test_only_tool_use_blocks_are_renamed(self):
        messages = [
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "calling"},
                    {
                        "type": "tool_use",
                        "id": "t1",
                        "name": "list_files",
                        "input": {"dir": "/"},
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "t1",
                        "content": "[]",
                    },
                ],
            },
        ]
        rewritten = _rewrite_history_tool_names_outbound(messages)
        # User text untouched.
        assert rewritten[0] == {"role": "user", "content": "hi"}
        # Assistant tool_use name prefixed; non-tool_use blocks left alone.
        new_assistant = rewritten[1]["content"]
        assert new_assistant[0] == {"type": "text", "text": "calling"}
        assert new_assistant[1]["name"] == "mcp_List_files"
        assert new_assistant[1]["input"] == {"dir": "/"}
        # tool_result block has no ``name`` — untouched.
        assert rewritten[2] == messages[2]

    def test_already_prefixed_names_skipped(self):
        messages = [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "t1",
                        "name": "mcp_Already",
                        "input": {},
                    },
                ],
            },
        ]
        rewritten = _rewrite_history_tool_names_outbound(messages)
        # Idempotent; not double-prefixed.
        assert rewritten[0]["content"][0]["name"] == "mcp_Already"

    def test_input_doesnt_get_clobbered(self):
        # Regression guard — the transform touches ``name`` only; the
        # ``input`` dict (which may contain user-typed PascalCase tool
        # names inside its values, unrelated to tool-call identifiers)
        # must pass through unchanged.
        messages = [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "t1",
                        "name": "exec",
                        "input": {
                            "cmd": "grep -r mcp_Foo src/",
                            "env": {"PATH": "/usr/bin"},
                        },
                    },
                ],
            },
        ]
        rewritten = _rewrite_history_tool_names_outbound(messages)
        block = rewritten[0]["content"][0]
        assert block["name"] == "mcp_Exec"
        assert block["input"] == {
            "cmd": "grep -r mcp_Foo src/",
            "env": {"PATH": "/usr/bin"},
        }


# ---------------------------------------------------------------- #
# Haiku kwarg stripping                                            #
# ---------------------------------------------------------------- #


class TestStripHaikuIncompatibleKwargs:
    def test_no_op_for_non_haiku_models(self):
        kwargs = {
            "model": "claude-opus-4-7",
            "thinking": {"type": "adaptive"},
            "output_config": {"effort": "max"},
        }
        snapshot = {
            **kwargs,
            "thinking": dict(kwargs["thinking"]),
            "output_config": dict(kwargs["output_config"]),
        }
        _strip_haiku_incompatible_kwargs(kwargs)
        assert kwargs == snapshot

    def test_adaptive_thinking_dropped_on_haiku(self):
        kwargs: dict[str, Any] = {
            "model": "claude-haiku-4-5",
            "thinking": {"type": "adaptive", "display": "summarized"},
        }
        _strip_haiku_incompatible_kwargs(kwargs)
        assert "thinking" not in kwargs

    def test_effort_stripped_from_output_config_on_haiku(self):
        kwargs: dict[str, Any] = {
            "model": "claude-haiku-4-5",
            "output_config": {"effort": "high", "other": "keep"},
        }
        _strip_haiku_incompatible_kwargs(kwargs)
        # Other keys preserved; effort dropped.
        assert kwargs["output_config"] == {"other": "keep"}

    def test_output_config_dropped_when_effort_was_only_key(self):
        kwargs: dict[str, Any] = {
            "model": "claude-haiku-4-5",
            "output_config": {"effort": "high"},
        }
        _strip_haiku_incompatible_kwargs(kwargs)
        assert "output_config" not in kwargs

    def test_manual_thinking_with_budget_preserved_on_haiku(self):
        # Haiku supports manual extended thinking — only adaptive is
        # rejected.  We preserve ``type=enabled`` with budget_tokens.
        kwargs: dict[str, Any] = {
            "model": "claude-haiku-4-5",
            "thinking": {"type": "enabled", "budget_tokens": 4000},
        }
        _strip_haiku_incompatible_kwargs(kwargs)
        assert kwargs["thinking"] == {
            "type": "enabled",
            "budget_tokens": 4000,
        }

    def test_thinking_effort_stripped_on_haiku(self):
        # A hypothetical future caller might put effort inside the
        # thinking block; strip it so haiku doesn't 400.
        kwargs: dict[str, Any] = {
            "model": "claude-haiku-4-5",
            "thinking": {
                "type": "enabled",
                "budget_tokens": 1000,
                "effort": "low",
            },
        }
        _strip_haiku_incompatible_kwargs(kwargs)
        assert kwargs["thinking"] == {
            "type": "enabled",
            "budget_tokens": 1000,
        }

    def test_model_field_missing_is_no_op(self):
        # Defensive: if for some reason model isn't set, the wrapper
        # should not crash.
        kwargs = {"thinking": {"type": "adaptive"}}
        _strip_haiku_incompatible_kwargs(kwargs)
        assert kwargs == {"thinking": {"type": "adaptive"}}


# ---------------------------------------------------------------- #
# Formatter MRO resolution (bug we fixed in model_factory.py)      #
# ---------------------------------------------------------------- #


class TestFormatterMROResolution:
    def test_subclass_inherits_anthropic_formatter(self):
        # Without the MRO fix, the dict ``.get`` returns OpenAI default
        # for any AnthropicChatModel subclass — which silently broke
        # tool-calling agents under claude-oauth.
        from agentscope.model import (
            AnthropicChatModel,
            OpenAIChatModel,
        )
        from agentscope.formatter import (
            AnthropicChatFormatter,
            OpenAIChatFormatter,
        )

        from qwenpaw.agents.model_factory import (
            _get_formatter_for_chat_model,
        )

        assert (
            _get_formatter_for_chat_model(OpenAIChatModel)
            is OpenAIChatFormatter
        )
        assert (
            _get_formatter_for_chat_model(AnthropicChatModel)
            is AnthropicChatFormatter
        )
        # The load-bearing assertion — without the MRO walk this
        # would return OpenAIChatFormatter and break message shape.
        assert (
            _get_formatter_for_chat_model(ClaudeOAuthChatModel)
            is AnthropicChatFormatter
        )


# ---------------------------------------------------------------- #
# MCP prefix constant sanity                                       #
# ---------------------------------------------------------------- #


def test_mcp_prefix_value():
    # Anthropic's OAuth billing validator rejects lowercase-first tool
    # names when multiple tools are sent.  The convention every public
    # adapter (openclaw, opencode-claude-auth) uses is ``mcp_`` +
    # upper-first.  Don't drift from that.
    assert MCP_TOOL_PREFIX == "mcp_"
