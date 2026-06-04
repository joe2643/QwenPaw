# -*- coding: utf-8 -*-
"""Tests for the Anthropic-style ``<invoke>`` fallback parser path in
``tag_parser`` and ``anthropic_provider._recover_text_tool_calls_inplace``.
"""
from __future__ import annotations

from types import SimpleNamespace

from qwenpaw.local_models.tag_parser import (
    parse_tool_calls_from_text,
    text_contains_tool_call_tag,
)
from qwenpaw.providers.anthropic_provider import (
    _recover_text_tool_calls_inplace,
)


SAMPLE_INVOKE = """\
<invoke name="mcp_Execute_shell_command">
<parameter name="command">echo hi
echo bye</parameter>
<parameter name="timeout">15</parameter>
</invoke>"""


def test_text_contains_tool_call_tag_detects_invoke() -> None:
    assert text_contains_tool_call_tag(SAMPLE_INVOKE)
    assert text_contains_tool_call_tag("prefix " + SAMPLE_INVOKE)
    assert not text_contains_tool_call_tag("nothing relevant")


def test_parse_tool_calls_from_text_invoke_basic() -> None:
    result = parse_tool_calls_from_text(SAMPLE_INVOKE)
    assert len(result.tool_calls) == 1
    call = result.tool_calls[0]
    assert call.name == "mcp_Execute_shell_command"
    # Multi-line value preserves its inner newline; surrounding pretty-
    # printing newlines are trimmed.
    assert call.arguments == {
        "command": "echo hi\necho bye",
        "timeout": "15",
    }
    assert result.text_before == ""
    assert result.text_after == ""


def test_parse_tool_calls_from_text_invoke_with_surrounding_text() -> None:
    text = f"Sure, running command now.\n\n{SAMPLE_INVOKE}\n\nDone."
    result = parse_tool_calls_from_text(text)
    assert len(result.tool_calls) == 1
    assert result.text_before == "Sure, running command now."
    assert result.text_after == "Done."


def test_parse_tool_calls_from_text_invoke_single_quotes() -> None:
    text = (
        "<invoke name='do_thing'>"
        "<parameter name='x'>1</parameter>"
        "</invoke>"
    )
    result = parse_tool_calls_from_text(text)
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].name == "do_thing"
    assert result.tool_calls[0].arguments == {"x": "1"}


def test_parse_tool_calls_from_text_tool_call_takes_precedence() -> None:
    # When both formats coexist the original <tool_call> path wins; the
    # invoke fallback only fires when no <tool_call> blocks are present.
    text = (
        '<tool_call>{"name": "a", "arguments": {}}</tool_call>'
        '<invoke name="b"><parameter name="x">1</parameter></invoke>'
    )
    result = parse_tool_calls_from_text(text)
    names = [c.name for c in result.tool_calls]
    assert names == ["a"]


def test_recover_text_tool_calls_inplace_injects_block() -> None:
    resp = SimpleNamespace(
        content=[
            {
                "type": "text",
                "text": "Working on it.\n" + SAMPLE_INVOKE,
            },
        ],
    )
    _recover_text_tool_calls_inplace(resp)

    types = [b["type"] for b in resp.content]
    assert types == ["text", "tool_use"]
    assert resp.content[0]["text"] == "Working on it."
    tu = resp.content[1]
    assert tu["name"] == "mcp_Execute_shell_command"
    assert tu["input"]["command"] == "echo hi\necho bye"
    assert tu["input"]["timeout"] == "15"
    assert isinstance(tu["id"], str) and tu["id"]


def test_recover_text_tool_calls_inplace_drops_empty_text_block() -> None:
    resp = SimpleNamespace(content=[{"type": "text", "text": SAMPLE_INVOKE}])
    _recover_text_tool_calls_inplace(resp)
    types = [b["type"] for b in resp.content]
    assert types == ["tool_use"]


def test_recover_text_tool_calls_inplace_skips_when_native_tool_use_present(
) -> None:
    # If the response already has a structured tool_use block we leave
    # the response untouched to avoid double-dispatching.
    resp = SimpleNamespace(
        content=[
            {"type": "text", "text": "trailing chatter " + SAMPLE_INVOKE},
            {
                "type": "tool_use",
                "id": "real_id",
                "name": "real_tool",
                "input": {},
            },
        ],
    )
    original = list(resp.content)
    _recover_text_tool_calls_inplace(resp)
    assert resp.content == original


def test_recover_text_tool_calls_inplace_noop_when_no_tag() -> None:
    resp = SimpleNamespace(
        content=[{"type": "text", "text": "just a plain reply"}],
    )
    original_id = id(resp.content)
    _recover_text_tool_calls_inplace(resp)
    # No tag → list reference must stay the same (we only mutate when we
    # actually recover something).
    assert id(resp.content) == original_id
