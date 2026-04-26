# -*- coding: utf-8 -*-
"""Unit tests for the ACP ↔ chat-completions translator.  Mirrors
``test_codex_translate.py`` in shape: synthetic JSON-line iter →
assert chunk shape; both stream + collect paths; reasoning_content
gate; new ship_tail mode.
"""

from __future__ import annotations

import asyncio
import json
from typing import AsyncIterator

import pytest

from qwenpaw.providers.acpx_translate import (
    DEFAULT_ACPX_CMD,
    StreamState,
    collect_as_chat_completion,
    content_to_acp_blocks,
    extract_tail_from_history,
    render_history_for_seed,
    stateful_acpx_cmd,
    translate_acp_updates_to_chat_chunks,
)


# ---------------------------------------------------------------- #
# content_to_acp_blocks                                            #
# ---------------------------------------------------------------- #


class TestContentToAcpBlocks:
    def test_string_content(self) -> None:
        out = content_to_acp_blocks("hello")
        assert out == [{"type": "text", "text": "hello"}]

    def test_text_block(self) -> None:
        out = content_to_acp_blocks([{"type": "text", "text": "hi"}])
        assert out == [{"type": "text", "text": "hi"}]

    def test_image_data_url_b64(self) -> None:
        out = content_to_acp_blocks([
            {"type": "input_image", "image_url": "data:image/png;base64,AAAA"},
        ])
        assert len(out) == 1
        assert out[0] == {"type": "image", "mimeType": "image/png", "data": "AAAA"}

    def test_image_url(self) -> None:
        out = content_to_acp_blocks([
            {"type": "input_image", "image_url": "https://example/img.png"},
        ])
        assert out == [{"type": "resource_link", "uri": "https://example/img.png"}]

    def test_anthropic_image_b64(self) -> None:
        out = content_to_acp_blocks([
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": "ZZZZ",
                },
            },
        ])
        assert out == [{"type": "image", "mimeType": "image/jpeg", "data": "ZZZZ"}]

    def test_empty_list_yields_empty_text(self) -> None:
        out = content_to_acp_blocks([])
        assert out == [{"type": "text", "text": ""}]


# ---------------------------------------------------------------- #
# render_history_for_seed                                          #
# ---------------------------------------------------------------- #


class TestRenderHistoryForSeed:
    def test_user_only(self) -> None:
        out = render_history_for_seed([{"role": "user", "content": "hi"}])
        assert out == [{"type": "text", "text": "hi"}]

    def test_system_concatenated_into_text(self) -> None:
        out = render_history_for_seed([
            {"role": "system", "content": "you are helpful"},
            {"role": "user", "content": "hi"},
        ])
        # System rides as leading text block; user follows.
        assert out[0]["type"] == "text"
        assert "you are helpful" in out[0]["text"]
        assert out[-1]["text"] == "hi"

    def test_assistant_prefixed(self) -> None:
        out = render_history_for_seed([
            {"role": "user", "content": "u1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "u2"},
        ])
        # Assistant turn becomes "Assistant: a1" in transcript text.
        full_text = " ".join(b.get("text", "") for b in out)
        assert "Assistant: a1" in full_text


# ---------------------------------------------------------------- #
# extract_tail_from_history                                        #
# ---------------------------------------------------------------- #


class TestExtractTail:
    def test_only_new_user_msg(self) -> None:
        msgs = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "u1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "u2"},
        ]
        # Pretend we already shipped the first 3.
        out = extract_tail_from_history(msgs, from_idx=3)
        assert out == [{"type": "text", "text": "u2"}]

    def test_tool_result_inlined(self) -> None:
        msgs = [
            {"role": "user", "content": "u1"},
            {"role": "tool", "tool_call_id": "call_1", "content": "the result"},
        ]
        out = extract_tail_from_history(msgs, from_idx=1)
        assert len(out) == 1
        assert out[0]["type"] == "text"
        assert "tool_call_id=call_1" in out[0]["text"]
        assert "the result" in out[0]["text"]

    def test_empty_tail_returns_empty_block(self) -> None:
        # Caller bug — but defensively returns a single empty text
        # block; spawn layer substitutes "(empty prompt)".
        out = extract_tail_from_history(
            [{"role": "user", "content": "x"}],
            from_idx=5,
        )
        assert out == [{"type": "text", "text": ""}]


# ---------------------------------------------------------------- #
# CLI cmd builders                                                 #
# ---------------------------------------------------------------- #


class TestCmdBuilders:
    def test_default_cmd_uses_exec(self) -> None:
        assert "exec" in DEFAULT_ACPX_CMD
        assert "--format" in DEFAULT_ACPX_CMD
        assert "--json-strict" in DEFAULT_ACPX_CMD

    def test_stateful_cmd_uses_dash_s(self) -> None:
        cmd = stateful_acpx_cmd("copaw-foo")
        assert "-s" in cmd
        assert "copaw-foo" in cmd
        assert "exec" not in cmd  # stateful, not one-shot
        assert "--format" in cmd
        assert "--json-strict" in cmd


# ---------------------------------------------------------------- #
# translate_acp_updates_to_chat_chunks — streaming                 #
# ---------------------------------------------------------------- #


async def _line_iter(lines: list[str]) -> AsyncIterator[str]:
    for line in lines:
        yield line


def _msg_update(kind: str, **extra) -> dict:
    """Build one ACP session/update notification."""
    update = {"sessionUpdate": kind}
    update.update(extra)
    return {
        "jsonrpc": "2.0",
        "method": "session/update",
        "params": {"sessionId": "s1", "update": update},
    }


def _final(stop_reason: str = "end_turn") -> dict:
    return {
        "jsonrpc": "2.0",
        "id": "1",
        "result": {"stopReason": stop_reason},
    }


class TestTranslateAcpUpdates:
    @pytest.mark.asyncio
    async def test_agent_message_chunk_to_content(self) -> None:
        lines = [
            json.dumps(_msg_update(
                "agent_message_chunk",
                content={"type": "text", "text": "Hello"},
            )),
            json.dumps(_final()),
        ]
        state = StreamState(model="claude-acpx")
        chunks = []
        async for c in translate_acp_updates_to_chat_chunks(_line_iter(lines), state):
            chunks.append(c)

        # First chunk: role=assistant.  Then content.  Then final.
        assert chunks[0]["choices"][0]["delta"] == {"role": "assistant"}
        assert chunks[1]["choices"][0]["delta"] == {"content": "Hello"}
        assert chunks[-1]["choices"][0]["finish_reason"] == "stop"

    @pytest.mark.asyncio
    async def test_agent_thought_chunk_to_reasoning_content(self) -> None:
        # Codex-style filtering: thought goes to reasoning_content,
        # NOT content (channels suppress reasoning_content for
        # user-facing send via existing filter).
        lines = [
            json.dumps(_msg_update(
                "agent_thought_chunk",
                content={"type": "text", "text": "thinking..."},
            )),
            json.dumps(_msg_update(
                "agent_message_chunk",
                content={"type": "text", "text": "Done."},
            )),
            json.dumps(_final()),
        ]
        state = StreamState(model="claude-acpx")
        chunks = []
        async for c in translate_acp_updates_to_chat_chunks(_line_iter(lines), state):
            chunks.append(c)

        # Find the reasoning chunk.
        reasoning_chunks = [
            c for c in chunks if "reasoning_content" in c["choices"][0]["delta"]
        ]
        content_chunks = [
            c for c in chunks if c["choices"][0]["delta"].get("content")
        ]
        assert len(reasoning_chunks) == 1
        assert reasoning_chunks[0]["choices"][0]["delta"]["reasoning_content"] == "thinking..."
        assert len(content_chunks) == 1
        assert content_chunks[0]["choices"][0]["delta"]["content"] == "Done."

    @pytest.mark.asyncio
    async def test_tool_call_to_tool_calls_delta(self) -> None:
        lines = [
            json.dumps(_msg_update(
                "tool_call",
                toolCallId="tc_1",
                title="read_file",
                rawInput={"path": "x.py"},
            )),
            json.dumps(_final()),
        ]
        state = StreamState(model="claude-acpx")
        chunks = []
        async for c in translate_acp_updates_to_chat_chunks(_line_iter(lines), state):
            chunks.append(c)

        tool_chunks = [
            c for c in chunks if "tool_calls" in c["choices"][0]["delta"]
        ]
        assert len(tool_chunks) == 1
        tc = tool_chunks[0]["choices"][0]["delta"]["tool_calls"][0]
        assert tc["id"] == "tc_1"
        assert tc["function"]["name"] == "read_file"
        assert tc["function"]["arguments"] == '{"path": "x.py"}'

    @pytest.mark.asyncio
    async def test_finish_reason_tool_calls_when_tool_emitted(self) -> None:
        lines = [
            json.dumps(_msg_update(
                "tool_call",
                toolCallId="tc_1",
                title="read_file",
                rawInput={},
            )),
            json.dumps(_final("end_turn")),
        ]
        state = StreamState(model="claude-acpx")
        chunks = []
        async for c in translate_acp_updates_to_chat_chunks(_line_iter(lines), state):
            chunks.append(c)
        # Tool call wins finish_reason regardless of stop reason.
        assert chunks[-1]["choices"][0]["finish_reason"] == "tool_calls"

    @pytest.mark.asyncio
    async def test_stop_reason_max_tokens_to_length(self) -> None:
        lines = [
            json.dumps(_msg_update(
                "agent_message_chunk",
                content={"type": "text", "text": "..."},
            )),
            json.dumps(_final("max_tokens")),
        ]
        state = StreamState(model="claude-acpx")
        chunks = []
        async for c in translate_acp_updates_to_chat_chunks(_line_iter(lines), state):
            chunks.append(c)
        assert chunks[-1]["choices"][0]["finish_reason"] == "length"

    @pytest.mark.asyncio
    async def test_stop_reason_cancelled_to_stop(self) -> None:
        lines = [json.dumps(_final("cancelled"))]
        state = StreamState(model="claude-acpx")
        chunks = []
        async for c in translate_acp_updates_to_chat_chunks(_line_iter(lines), state):
            chunks.append(c)
        assert chunks[-1]["choices"][0]["finish_reason"] == "stop"

    @pytest.mark.asyncio
    async def test_error_response_raises(self) -> None:
        lines = [
            json.dumps({
                "jsonrpc": "2.0",
                "id": "1",
                "error": {"code": -32000, "message": "boom"},
            }),
        ]
        state = StreamState(model="claude-acpx")
        with pytest.raises(RuntimeError, match="boom"):
            async for _ in translate_acp_updates_to_chat_chunks(_line_iter(lines), state):
                pass

    @pytest.mark.asyncio
    async def test_malformed_json_line_skipped(self) -> None:
        lines = [
            "not json at all",
            "",
            json.dumps(_msg_update(
                "agent_message_chunk",
                content={"type": "text", "text": "ok"},
            )),
            json.dumps(_final()),
        ]
        state = StreamState(model="claude-acpx")
        chunks = []
        async for c in translate_acp_updates_to_chat_chunks(_line_iter(lines), state):
            chunks.append(c)
        # Survives malformed lines and yields the real content.
        contents = [
            c["choices"][0]["delta"].get("content")
            for c in chunks
            if c["choices"][0]["delta"].get("content")
        ]
        assert contents == ["ok"]

    @pytest.mark.asyncio
    async def test_unknown_session_update_ignored(self) -> None:
        # plan / available_commands_update / etc. are metadata; we
        # don't translate them, but they MUST NOT crash the iter.
        lines = [
            json.dumps(_msg_update("plan", entries=[])),
            json.dumps(_msg_update("available_commands_update", availableCommands=[])),
            json.dumps(_msg_update("current_mode_update", currentModeId="auto")),
            json.dumps(_msg_update(
                "agent_message_chunk",
                content={"type": "text", "text": "ok"},
            )),
            json.dumps(_final()),
        ]
        state = StreamState(model="claude-acpx")
        chunks = [c async for c in translate_acp_updates_to_chat_chunks(_line_iter(lines), state)]
        # Should still produce the agent_message content.
        contents = [
            c["choices"][0]["delta"].get("content")
            for c in chunks
            if c["choices"][0]["delta"].get("content")
        ]
        assert contents == ["ok"]


# ---------------------------------------------------------------- #
# collect_as_chat_completion — non-streaming                       #
# ---------------------------------------------------------------- #


class TestCollectAsChatCompletion:
    @pytest.mark.asyncio
    async def test_text_only(self) -> None:
        lines = [
            json.dumps(_msg_update(
                "agent_message_chunk",
                content={"type": "text", "text": "Hello "},
            )),
            json.dumps(_msg_update(
                "agent_message_chunk",
                content={"type": "text", "text": "world"},
            )),
            json.dumps(_final()),
        ]
        state = StreamState(model="claude-acpx")
        out = await collect_as_chat_completion(_line_iter(lines), state)
        assert out["choices"][0]["message"]["content"] == "Hello world"
        assert out["choices"][0]["finish_reason"] == "stop"

    @pytest.mark.asyncio
    async def test_reasoning_content_accumulated(self) -> None:
        lines = [
            json.dumps(_msg_update(
                "agent_thought_chunk",
                content={"type": "text", "text": "think A"},
            )),
            json.dumps(_msg_update(
                "agent_thought_chunk",
                content={"type": "text", "text": " think B"},
            )),
            json.dumps(_msg_update(
                "agent_message_chunk",
                content={"type": "text", "text": "answer"},
            )),
            json.dumps(_final()),
        ]
        state = StreamState(model="claude-acpx")
        out = await collect_as_chat_completion(_line_iter(lines), state)
        msg = out["choices"][0]["message"]
        assert msg["content"] == "answer"
        assert msg["reasoning_content"] == "think A think B"

    @pytest.mark.asyncio
    async def test_tool_calls_force_finish_reason(self) -> None:
        lines = [
            json.dumps(_msg_update(
                "tool_call",
                toolCallId="tc_x",
                title="run_shell",
                rawInput={"cmd": "ls"},
            )),
            json.dumps(_final("end_turn")),  # not tool_calls!
        ]
        state = StreamState(model="claude-acpx")
        out = await collect_as_chat_completion(_line_iter(lines), state)
        assert out["choices"][0]["finish_reason"] == "tool_calls"
        assert len(out["choices"][0]["message"]["tool_calls"]) == 1
