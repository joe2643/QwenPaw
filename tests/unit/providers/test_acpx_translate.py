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
    async def test_tool_call_emits_reasoning_not_tool_calls(self) -> None:
        """In ACPX hybrid mode Claude Code executes its own tools via
        ACP reverse-RPC; the ``tool_call`` notifications are
        informational. Translator must NOT surface them as OpenAI
        ``tool_calls`` (would cause agentscope ReAct to dispatch a
        non-existent function and raise ``FunctionNotFoundError:
        Cannot find the function named Terminal``)."""
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
        async for c in translate_acp_updates_to_chat_chunks(
            _line_iter(lines), state,
        ):
            chunks.append(c)

        # No chunk should carry ``tool_calls``.
        tool_chunks = [
            c for c in chunks if "tool_calls" in c["choices"][0]["delta"]
        ]
        assert not tool_chunks, (
            "tool_call notifications must not become OpenAI tool_calls"
        )
        # Tool call surfaces as reasoning_content with the call shape.
        reasoning_chunks = [
            c for c in chunks
            if "reasoning_content" in c["choices"][0]["delta"]
        ]
        assert reasoning_chunks
        body = reasoning_chunks[0]["choices"][0]["delta"]["reasoning_content"]
        assert "claude-code" in body
        assert "read_file" in body
        assert "x.py" in body

    @pytest.mark.asyncio
    async def test_tool_call_preview_deferred_until_rawinput_filled(
        self,
    ) -> None:
        """claude-agent-acp emits the first ``tool_call`` notification
        on ``content_block_start`` with ``rawInput={}`` — actual args
        stream in via ``input_json_delta`` and only land in a later
        ``tool_call_update`` with the full input.  Without a defer the
        preview locks in ``Terminal({})`` and the user never sees what
        Claude actually ran.  Regression test for the empty-args
        complaint observed 2026-05-02 in WhatsApp output."""
        lines = [
            json.dumps(_msg_update(
                "tool_call",
                toolCallId="tc_42",
                title="Terminal",
                rawInput={},
                status="pending",
            )),
            json.dumps(_msg_update(
                "tool_call_update",
                toolCallId="tc_42",
                rawInput={"command": "ls -la /tmp"},
                status="in_progress",
            )),
            json.dumps(_msg_update(
                "tool_call_update",
                toolCallId="tc_42",
                status="completed",
            )),
            json.dumps(_final()),
        ]
        state = StreamState(model="claude-acpx")
        chunks = []
        async for c in translate_acp_updates_to_chat_chunks(
            _line_iter(lines), state,
        ):
            chunks.append(c)

        previews = [
            c["choices"][0]["delta"].get("reasoning_content")
            for c in chunks
            if "reasoning_content" in c["choices"][0]["delta"]
        ]
        # Exactly one preview — emitted on the update that filled in
        # rawInput, NOT on the initial empty tool_call notification.
        assert len(previews) == 1, (
            f"expected exactly one preview, got: {previews!r}"
        )
        assert "Terminal" in previews[0]
        assert "ls -la /tmp" in previews[0]
        assert "({})" not in previews[0], (
            "empty-args preview leaked through despite later update"
        )

    @pytest.mark.asyncio
    async def test_tool_call_preview_fallback_when_rawinput_never_arrives(
        self,
    ) -> None:
        """If a tool call reaches a terminal status without ever
        surfacing meaningful rawInput, flush a fallback preview so the
        call is still visible in the trail (instead of vanishing
        entirely)."""
        lines = [
            json.dumps(_msg_update(
                "tool_call",
                toolCallId="tc_silent",
                title="MysteryTool",
                rawInput={},
                status="pending",
            )),
            json.dumps(_msg_update(
                "tool_call_update",
                toolCallId="tc_silent",
                status="failed",
            )),
            json.dumps(_final()),
        ]
        state = StreamState(model="claude-acpx")
        chunks = []
        async for c in translate_acp_updates_to_chat_chunks(
            _line_iter(lines), state,
        ):
            chunks.append(c)

        previews = [
            c["choices"][0]["delta"].get("reasoning_content")
            for c in chunks
            if "reasoning_content" in c["choices"][0]["delta"]
        ]
        assert len(previews) == 1
        assert "MysteryTool" in previews[0]
        assert "({})" in previews[0]

    @pytest.mark.asyncio
    async def test_finish_reason_maps_from_stop_reason_only(self) -> None:
        """Even when a ``tool_call`` was observed, finish_reason must
        come from ``stopReason`` — not ``tool_calls`` (the work has
        already executed via reverse-RPC, agentscope has nothing to
        dispatch)."""
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
        async for c in translate_acp_updates_to_chat_chunks(
            _line_iter(lines), state,
        ):
            chunks.append(c)
        assert chunks[-1]["choices"][0]["finish_reason"] == "stop"

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
    async def test_tool_call_does_not_emit_openai_tool_calls(self) -> None:
        """Non-streaming path: ACPX ``tool_call`` notifications must
        surface only as ``reasoning_content`` and never as OpenAI
        ``tool_calls``. Otherwise agentscope's ReAct loop would try
        to dispatch a function named ``run_shell``/``Terminal``/etc.
        through CoPaw's toolkit and raise ``FunctionNotFoundError``.
        finish_reason maps from ``stopReason`` only."""
        lines = [
            json.dumps(_msg_update(
                "tool_call",
                toolCallId="tc_x",
                title="run_shell",
                rawInput={"cmd": "ls"},
            )),
            json.dumps(_final("end_turn")),
        ]
        state = StreamState(model="claude-acpx")
        out = await collect_as_chat_completion(_line_iter(lines), state)
        assert out["choices"][0]["finish_reason"] == "stop"
        msg = out["choices"][0]["message"]
        assert "tool_calls" not in msg
        assert "claude-code" in (msg.get("reasoning_content") or "")
        assert "run_shell" in (msg.get("reasoning_content") or "")


class TestImageMarker:
    """Tests for ``_image_attach_marker`` + ``_spill_base64_image``."""

    def test_marker_sanitizes_unsafe_chars(self) -> None:
        """A path containing the marker terminator ``]`` or a literal
        newline cannot break the marker boundary — those characters
        are mapped to safe placeholders so Claude Code's parser
        cannot be tricked into mis-reading the marker."""
        from qwenpaw.providers.acpx_translate import _image_attach_marker

        marker = _image_attach_marker(
            "/tmp/weird]name\nwith\\junk.png",
            label="image",
        )
        assert "]name" not in marker  # ``]`` was replaced
        assert "\n" not in marker      # newline was replaced
        # The marker is still well-formed: opens with ``[`` and ends
        # with the SINGLE closing ``]``.
        assert marker.startswith("[")
        assert marker.count("]") == 1
        assert marker.endswith("]")

    def test_spill_refuses_oversized_payload(self, monkeypatch) -> None:
        """Inline base64 images above the byte cap must be refused
        rather than spilling a 1 GB file to disk."""
        from qwenpaw.providers import acpx_translate as mod

        # Lower the cap to 1 KB so we can exercise the guard cheaply.
        monkeypatch.setattr(mod, "_MAX_INLINE_IMAGE_BYTES", 1024)

        # 4 KB raw bytes encoded ⇒ ~5.5 KB base64 ⇒ above the 1 KB cap.
        import base64

        b64 = base64.b64encode(b"x" * 4096).decode("ascii")
        path = mod._spill_base64_image(b64, "image/png")
        assert path == "", (
            "oversized base64 should be refused (empty path returned)"
        )

    def test_eviction_unlinks_evicted_file(self, monkeypatch) -> None:
        """When the FIFO cache evicts an entry, the underlying file
        must be unlinked too — earlier the dict entry was popped but
        the file leaked, so the spill directory grew without bound."""
        from pathlib import Path as _Path
        import base64

        from qwenpaw.providers import acpx_translate as mod

        # Lower cap to 2 so we evict deterministically with 3 distinct
        # images.
        monkeypatch.setattr(mod, "_IMAGE_SPILL_CACHE_CAP", 2)
        monkeypatch.setattr(mod, "_IMAGE_SPILL_CACHE", {})

        paths: list[str] = []
        for i in range(3):
            b64 = base64.b64encode(b"img-" + bytes(str(i), "ascii")).decode(
                "ascii",
            )
            path = mod._spill_base64_image(b64, "image/png")
            assert path
            paths.append(path)

        # First entry must be evicted (cap=2, three insertions).
        assert not _Path(paths[0]).exists(), (
            "first cache entry's file should be unlinked on FIFO eviction"
        )
        # Second and third entries still present.
        assert _Path(paths[1]).exists()
        assert _Path(paths[2]).exists()

        # Cleanup files we created so the test doesn't leave artefacts.
        for p in paths[1:]:
            try:
                _Path(p).unlink()
            except FileNotFoundError:
                pass
