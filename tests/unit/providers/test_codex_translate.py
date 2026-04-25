# -*- coding: utf-8 -*-
"""Unit tests for the OpenAI chat/completions ↔ Codex Responses API
translation used by ``CodexOAuthChatModel``.
"""

from __future__ import annotations

import json
from typing import AsyncIterator

import pytest

from qwenpaw.providers.codex_translate import (
    DEFAULT_MODEL,
    StreamState,
    build_responses_body,
    collect_as_chat_completion,
    content_to_plain_text,
    content_to_responses_items,
    convert_messages_to_responses_input,
    convert_tools,
    translate_responses_events_to_chat_chunks,
)


# ---------------------------------------------------------------- #
# content_to_plain_text                                            #
# ---------------------------------------------------------------- #


class TestContentToPlainText:
    def test_string_passthrough(self):
        assert content_to_plain_text("hello") == "hello"

    def test_list_of_text_blocks(self):
        assert (
            content_to_plain_text(
                [
                    {"type": "text", "text": "a"},
                    {"type": "text", "text": "b"},
                ],
            )
            == "ab"
        )

    def test_image_blocks_become_placeholder(self):
        # Non-text items become "[image attached]"; lossy but
        # ChatGPT-backend only takes strings for system/assistant/tool
        # slots.
        assert (
            content_to_plain_text(
                [
                    {"type": "text", "text": "see "},
                    {"type": "image_url", "image_url": "data:..."},
                ],
            )
            == "see [image attached]"
        )

    @pytest.mark.parametrize("val", [None, 42, {"x": 1}])
    def test_fallback_to_empty_string(self, val):
        assert content_to_plain_text(val) == ""


# ---------------------------------------------------------------- #
# content_to_responses_items (user message shape)                  #
# ---------------------------------------------------------------- #


class TestContentToResponsesItems:
    def test_string_becomes_input_text_block(self):
        assert content_to_responses_items("hi") == [
            {"type": "input_text", "text": "hi"},
        ]

    def test_image_url_dict_form(self):
        out = content_to_responses_items(
            [
                {
                    "type": "image_url",
                    "image_url": {"url": "data:img/png;base64,xxx"},
                },
            ],
        )
        assert out == [
            {"type": "input_image", "image_url": "data:img/png;base64,xxx"},
        ]

    def test_image_url_string_form(self):
        out = content_to_responses_items(
            [
                {"type": "image_url", "image_url": "https://ex.com/x.png"},
            ],
        )
        assert out == [
            {"type": "input_image", "image_url": "https://ex.com/x.png"},
        ]

    def test_anthropic_style_base64_image(self):
        out = content_to_responses_items(
            [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": "ABC",
                    },
                },
            ],
        )
        assert out == [
            {
                "type": "input_image",
                "image_url": "data:image/jpeg;base64,ABC",
            },
        ]

    def test_empty_list_yields_empty_text_placeholder(self):
        # Responses API rejects empty content arrays — we emit a
        # placeholder to keep the request valid.
        assert content_to_responses_items([]) == [
            {"type": "input_text", "text": ""},
        ]


# ---------------------------------------------------------------- #
# convert_messages_to_responses_input                              #
# ---------------------------------------------------------------- #


class TestConvertMessagesToResponsesInput:
    def test_system_joins_as_instructions(self):
        instructions, items = convert_messages_to_responses_input(
            [
                {"role": "system", "content": "one"},
                {"role": "system", "content": "two"},
                {"role": "user", "content": "hi"},
            ],
        )
        assert instructions == "one\n\ntwo"
        assert items[0] == {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": "hi"}],
        }

    def test_assistant_text_becomes_output_text(self):
        _, items = convert_messages_to_responses_input(
            [
                {"role": "assistant", "content": "ok"},
            ],
        )
        assert items == [
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "ok"}],
            },
        ]

    def test_assistant_with_tool_calls_splits_into_siblings(self):
        # The chat shape packs tool_calls *inside* an assistant
        # message; the Responses shape wants them as sibling
        # ``function_call`` items in order.
        _, items = convert_messages_to_responses_input(
            [
                {
                    "role": "assistant",
                    "content": "thinking...",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "get_time",
                                "arguments": '{"tz":"UTC"}',
                            },
                        },
                    ],
                },
            ],
        )
        assert len(items) == 2
        assert items[0]["role"] == "assistant"
        assert items[1] == {
            "type": "function_call",
            "call_id": "call_1",
            "name": "get_time",
            "arguments": '{"tz":"UTC"}',
        }

    def test_tool_role_becomes_function_call_output(self):
        _, items = convert_messages_to_responses_input(
            [
                {
                    "role": "tool",
                    "tool_call_id": "call_1",
                    "content": "12:00",
                },
            ],
        )
        assert items == [
            {
                "type": "function_call_output",
                "call_id": "call_1",
                "output": "12:00",
            },
        ]

    def test_assistant_text_missing_still_emits_tool_calls(self):
        _, items = convert_messages_to_responses_input(
            [
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "ls", "arguments": "{}"},
                        },
                    ],
                },
            ],
        )
        # Only the function_call item — no message item for empty text.
        assert len(items) == 1
        assert items[0]["type"] == "function_call"

    def test_assistant_missing_call_id_gets_synthesized(self):
        _, items = convert_messages_to_responses_input(
            [
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            # No ``id`` provided.
                            "function": {"name": "ls", "arguments": "{}"},
                        },
                    ],
                },
            ],
        )
        assert items[0]["call_id"].startswith("call_")


# ---------------------------------------------------------------- #
# convert_tools                                                    #
# ---------------------------------------------------------------- #


class TestConvertTools:
    def test_none_passthrough(self):
        assert convert_tools(None) is None
        assert convert_tools([]) is None

    def test_function_tool_flattened(self):
        out = convert_tools(
            [
                {
                    "type": "function",
                    "function": {
                        "name": "search",
                        "description": "do search",
                        "parameters": {"type": "object", "properties": {}},
                    },
                },
            ],
        )
        assert out == [
            {
                "type": "function",
                "name": "search",
                "description": "do search",
                "parameters": {"type": "object", "properties": {}},
                "strict": False,
            },
        ]

    def test_unknown_tool_type_passthrough(self):
        # Non-function tool shapes get forwarded verbatim so the
        # caller can experiment with newer types.
        weird = [{"type": "retrieval", "retrieval": {}}]
        assert convert_tools(weird) == weird

    def test_missing_parameters_defaults_to_empty_object(self):
        out = convert_tools(
            [
                {"type": "function", "function": {"name": "noop"}},
            ],
        )
        assert out[0]["parameters"] == {"type": "object", "properties": {}}


# ---------------------------------------------------------------- #
# build_responses_body                                             #
# ---------------------------------------------------------------- #


class TestBuildResponsesBody:
    def test_model_inside_allowed_set_preserved(self):
        body = build_responses_body({"model": "gpt-5.2", "messages": []})
        assert body["model"] == "gpt-5.2"

    def test_unknown_model_passes_through_verbatim(self):
        # Historically this coerced to DEFAULT_MODEL, which silently
        # greenlit the UI's Test Connection button for unsupported
        # slugs (e.g. ``gpt-5.5`` on a ChatGPT-account token).  The
        # backend is now the source of truth — we forward what the
        # caller asked and let its 400 ("not supported with a
        # ChatGPT account") reach the user honestly.
        body = build_responses_body({"model": "gpt-5.5", "messages": []})
        assert body["model"] == "gpt-5.5"

    def test_empty_model_defaults_to_default(self):
        # Empty / missing model is still a programming error worth
        # defaulting for — better than a 400 for an empty string.
        for model in ("", None):
            body = build_responses_body(
                {"model": model, "messages": []}
                if model is not None
                else {"messages": []},
            )
            assert body["model"] == DEFAULT_MODEL

    def test_empty_instructions_filled_with_placeholder(self):
        # ChatGPT backend rejects empty ``instructions`` with 400.
        body = build_responses_body(
            {
                "model": "gpt-5.4",
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
        assert body["instructions"] == "You are a helpful assistant."

    def test_upstream_always_streams(self):
        body = build_responses_body(
            {
                "model": "gpt-5.4",
                "messages": [],
                "stream": False,
            },
        )
        # We stream upstream regardless of client preference; the
        # non-streaming caller drains the stream into a single body.
        assert body["stream"] is True

    def test_reasoning_effort_defaults_to_low(self):
        body = build_responses_body({"model": "gpt-5.4", "messages": []})
        assert body["reasoning"] == {"effort": "low"}

    def test_reasoning_effort_passthrough(self):
        body = build_responses_body(
            {
                "model": "gpt-5.4",
                "messages": [],
                "reasoning_effort": "high",
            },
        )
        assert body["reasoning"] == {"effort": "high"}

    def test_store_is_false(self):
        # ``store=true`` requires an API-key account; ChatGPT OAuth
        # rejects it.
        body = build_responses_body({"model": "gpt-5.4", "messages": []})
        assert body["store"] is False

    def test_tools_translated(self):
        body = build_responses_body(
            {
                "model": "gpt-5.4",
                "messages": [],
                "tools": [
                    {
                        "type": "function",
                        "function": {"name": "f", "parameters": {}},
                    },
                ],
            },
        )
        assert body["tools"][0]["name"] == "f"

    def test_sampling_knobs_dropped(self):
        # ChatGPT backend rejects max_output_tokens / temperature /
        # top_p etc. when called with a ChatGPT-account OAuth token;
        # we drop them rather than 400 the user.
        body = build_responses_body(
            {
                "model": "gpt-5.4",
                "messages": [],
                "max_tokens": 500,
                "temperature": 0.7,
                "top_p": 0.9,
            },
        )
        assert "max_tokens" not in body
        assert "temperature" not in body
        assert "top_p" not in body


# ---------------------------------------------------------------- #
# Streaming translation — state machine                            #
# ---------------------------------------------------------------- #


class _FakeUpstream:
    """httpx.Response-lookalike that yields a pre-canned SSE stream."""

    def __init__(self, lines: list[str]) -> None:
        self._lines = lines

    async def aiter_lines(self) -> AsyncIterator[str]:
        for line in self._lines:
            yield line


def _sse(event: dict) -> str:
    return f"data: {json.dumps(event)}"


class TestTranslateResponsesEvents:
    @pytest.mark.asyncio
    async def test_text_delta_then_completed(self):
        state = StreamState(model="gpt-5.4")
        upstream = _FakeUpstream(
            [
                _sse({"type": "response.output_text.delta", "delta": "hel"}),
                _sse({"type": "response.output_text.delta", "delta": "lo"}),
                _sse(
                    {
                        "type": "response.completed",
                        "response": {
                            "output": [],
                            "usage": {
                                "input_tokens": 10,
                                "output_tokens": 2,
                                "total_tokens": 12,
                            },
                        },
                    },
                ),
            ],
        )
        chunks = []
        async for c in translate_responses_events_to_chat_chunks(
            upstream,
            state,
        ):
            chunks.append(c)
        # First chunk: role, next 2: content deltas, final: finish.
        assert chunks[0]["choices"][0]["delta"] == {"role": "assistant"}
        assert chunks[1]["choices"][0]["delta"] == {"content": "hel"}
        assert chunks[2]["choices"][0]["delta"] == {"content": "lo"}
        final = chunks[-1]
        assert final["choices"][0]["finish_reason"] == "stop"
        assert final["usage"] == {
            "prompt_tokens": 10,
            "completion_tokens": 2,
            "total_tokens": 12,
        }

    @pytest.mark.asyncio
    async def test_tool_call_flow_yields_openai_shape(self):
        state = StreamState(model="gpt-5.4")
        upstream = _FakeUpstream(
            [
                _sse(
                    {
                        "type": "response.output_item.added",
                        "item": {
                            "type": "function_call",
                            "id": "fc_1",
                            "call_id": "call_abc",
                            "name": "search",
                        },
                    },
                ),
                _sse(
                    {
                        "type": "response.function_call_arguments.delta",
                        "item_id": "fc_1",
                        "delta": '{"q":"hi',
                    },
                ),
                _sse(
                    {
                        "type": "response.function_call_arguments.delta",
                        "item_id": "fc_1",
                        "delta": '"}',
                    },
                ),
                _sse({"type": "response.function_call_arguments.done"}),
                _sse(
                    {
                        "type": "response.completed",
                        "response": {
                            "output": [
                                {"type": "function_call", "name": "search"},
                            ],
                        },
                    },
                ),
            ],
        )
        chunks = []
        async for c in translate_responses_events_to_chat_chunks(
            upstream,
            state,
        ):
            chunks.append(c)

        # The first chunk with a tool_call should announce name + id.
        tc_announce = next(
            c for c in chunks if "tool_calls" in c["choices"][0]["delta"]
        )
        tool_call = tc_announce["choices"][0]["delta"]["tool_calls"][0]
        assert tool_call["id"] == "call_abc"
        assert tool_call["function"]["name"] == "search"

        # Final chunk carries tool_calls finish_reason.
        assert chunks[-1]["choices"][0]["finish_reason"] == "tool_calls"

    @pytest.mark.asyncio
    async def test_response_failed_raises(self):
        state = StreamState(model="gpt-5.4")
        upstream = _FakeUpstream(
            [
                _sse(
                    {
                        "type": "response.failed",
                        "response": {"error": {"message": "rate limited"}},
                    },
                ),
            ],
        )
        with pytest.raises(RuntimeError, match="rate limited"):
            async for _ in translate_responses_events_to_chat_chunks(
                upstream,
                state,
            ):
                pass

    @pytest.mark.asyncio
    async def test_ignores_unknown_event_types(self):
        # Future-proofing: upstream adds a new event → we should
        # ignore it, not crash.
        state = StreamState(model="gpt-5.4")
        upstream = _FakeUpstream(
            [
                _sse({"type": "response.output_text.delta", "delta": "x"}),
                _sse({"type": "response.invented", "extra": 123}),
                _sse(
                    {"type": "response.completed", "response": {"output": []}},
                ),
            ],
        )
        chunks = []
        async for c in translate_responses_events_to_chat_chunks(
            upstream,
            state,
        ):
            chunks.append(c)
        assert chunks[-1]["choices"][0]["finish_reason"] == "stop"

    @pytest.mark.asyncio
    async def test_ignores_non_data_lines_and_empty(self):
        state = StreamState(model="gpt-5.4")
        upstream = _FakeUpstream(
            [
                "",
                "event: ping",
                "data: ",
                "data: [DONE]",
                _sse({"type": "response.output_text.delta", "delta": "hi"}),
                _sse(
                    {"type": "response.completed", "response": {"output": []}},
                ),
            ],
        )
        chunks = []
        async for c in translate_responses_events_to_chat_chunks(
            upstream,
            state,
        ):
            chunks.append(c)
        # Must include the text delta; must not crash on empty/malformed lines.
        contents = [c["choices"][0]["delta"].get("content") for c in chunks]
        assert "hi" in contents


class TestReasoningTextLeak:
    """Codex/gpt-5.x for OAuth users emits the reasoning summary
    inside the same ``response.output_text.delta`` event stream as
    the user-facing reply.  Without an active-item gate those
    fragments leaked straight into the assistant ``content`` and
    surfaced in WhatsApp/Signal as scratch-style text.  These tests
    pin down the gate."""

    @pytest.mark.asyncio
    async def test_reasoning_item_text_is_dropped_in_stream(self):
        state = StreamState(model="gpt-5.5")
        upstream = _FakeUpstream(
            [
                # Reasoning item announced first, its text deltas must
                # not surface as ``content``.
                _sse(
                    {
                        "type": "response.output_item.added",
                        "item": {"id": "rs_1", "type": "reasoning"},
                    },
                ),
                _sse(
                    {
                        "type": "response.output_text.delta",
                        "delta": "Need analyze new video.",
                    },
                ),
                _sse(
                    {
                        "type": "response.output_text.delta",
                        "delta": " Let's try maybe MIMO.",
                    },
                ),
                _sse({"type": "response.output_item.done"}),
                # Then the real assistant message.
                _sse(
                    {
                        "type": "response.output_item.added",
                        "item": {"id": "msg_1", "type": "message"},
                    },
                ),
                _sse(
                    {
                        "type": "response.output_text.delta",
                        "delta": "Hello user!",
                    },
                ),
                _sse({"type": "response.output_item.done"}),
                _sse(
                    {
                        "type": "response.completed",
                        "response": {"output": []},
                    },
                ),
            ],
        )
        chunks = []
        async for c in translate_responses_events_to_chat_chunks(
            upstream,
            state,
        ):
            chunks.append(c)

        # Concatenate every ``content`` delta — only the message
        # text should be present.
        text = "".join(
            c["choices"][0]["delta"].get("content", "") for c in chunks
        )
        assert text == "Hello user!"
        assert "Need analyze" not in text
        assert "MIMO" not in text

    @pytest.mark.asyncio
    async def test_reasoning_item_text_is_dropped_in_collect(self):
        state = StreamState(model="gpt-5.5")
        upstream = _FakeUpstream(
            [
                _sse(
                    {
                        "type": "response.output_item.added",
                        "item": {"id": "rs_1", "type": "reasoning"},
                    },
                ),
                _sse(
                    {
                        "type": "response.output_text.delta",
                        "delta": "scratch reasoning",
                    },
                ),
                _sse({"type": "response.output_item.done"}),
                _sse(
                    {
                        "type": "response.output_item.added",
                        "item": {"id": "msg_1", "type": "message"},
                    },
                ),
                _sse(
                    {
                        "type": "response.output_text.delta",
                        "delta": "user-facing",
                    },
                ),
                _sse({"type": "response.output_item.done"}),
                _sse(
                    {
                        "type": "response.completed",
                        "response": {"output": []},
                    },
                ),
            ],
        )
        body = await collect_as_chat_completion(upstream, state)
        msg = body["choices"][0]["message"]
        assert msg["content"] == "user-facing"

    @pytest.mark.asyncio
    async def test_text_without_item_envelope_still_forwarded(self):
        # Older models that omit ``output_item.added`` should keep
        # working — the gate falls back to None == passthrough.
        state = StreamState(model="gpt-5.4")
        upstream = _FakeUpstream(
            [
                _sse({"type": "response.output_text.delta", "delta": "ok"}),
                _sse(
                    {
                        "type": "response.completed",
                        "response": {"output": []},
                    },
                ),
            ],
        )
        chunks = []
        async for c in translate_responses_events_to_chat_chunks(
            upstream,
            state,
        ):
            chunks.append(c)
        text = "".join(
            c["choices"][0]["delta"].get("content", "") for c in chunks
        )
        assert text == "ok"


# ---------------------------------------------------------------- #
# Non-streaming drain                                              #
# ---------------------------------------------------------------- #


class TestCollectAsChatCompletion:
    @pytest.mark.asyncio
    async def test_assembles_text_response(self):
        state = StreamState(model="gpt-5.4")
        upstream = _FakeUpstream(
            [
                _sse({"type": "response.output_text.delta", "delta": "he"}),
                _sse({"type": "response.output_text.delta", "delta": "llo"}),
                _sse(
                    {
                        "type": "response.completed",
                        "response": {
                            "output": [],
                            "usage": {
                                "input_tokens": 3,
                                "output_tokens": 2,
                                "total_tokens": 5,
                            },
                        },
                    },
                ),
            ],
        )
        body = await collect_as_chat_completion(upstream, state)
        assert body["object"] == "chat.completion"
        assert body["choices"][0]["message"]["content"] == "hello"
        assert body["choices"][0]["finish_reason"] == "stop"
        assert body["usage"] == {
            "prompt_tokens": 3,
            "completion_tokens": 2,
            "total_tokens": 5,
        }

    @pytest.mark.asyncio
    async def test_tool_call_finish_reason_overrides_stop(self):
        # Even if the response.completed heuristic wouldn't say
        # tool_calls, the presence of tool_calls in our assembled
        # message forces finish_reason="tool_calls" per the OpenAI
        # chat/completions contract.
        state = StreamState(model="gpt-5.4")
        upstream = _FakeUpstream(
            [
                _sse(
                    {
                        "type": "response.output_item.added",
                        "item": {
                            "type": "function_call",
                            "id": "fc_1",
                            "call_id": "call_1",
                            "name": "ls",
                        },
                    },
                ),
                _sse(
                    {
                        "type": "response.function_call_arguments.delta",
                        "item_id": "fc_1",
                        "delta": "{}",
                    },
                ),
                _sse(
                    {
                        "type": "response.completed",
                        "response": {
                            # Heuristic would otherwise say "stop" (no
                            # function_call in output) — but tool_calls forces it.
                            "output": [],
                        },
                    },
                ),
            ],
        )
        body = await collect_as_chat_completion(upstream, state)
        assert body["choices"][0]["finish_reason"] == "tool_calls"
        msg = body["choices"][0]["message"]
        assert msg["tool_calls"][0]["function"]["name"] == "ls"
        assert msg["tool_calls"][0]["function"]["arguments"] == "{}"
        # Index field stripped before returning (chat contract).
        assert "index" not in msg["tool_calls"][0]

    @pytest.mark.asyncio
    async def test_response_failed_raises(self):
        state = StreamState(model="gpt-5.4")
        upstream = _FakeUpstream(
            [
                _sse(
                    {
                        "type": "response.failed",
                        "response": {"error": {"message": "boom"}},
                    },
                ),
            ],
        )
        with pytest.raises(RuntimeError, match="boom"):
            await collect_as_chat_completion(upstream, state)


class TestPhaseBasedCommentaryDrop:
    """ChatGPT Responses API tags assistant message items with
    ``phase``: ``commentary`` for scratch preamble, ``final_answer``
    for the user-facing reply.  Both share the same
    ``response.output_text.delta`` event stream — only the parent
    ``output_item.added.item.phase`` distinguishes them.
    Reference: OpenClaw's metadata-based detection
    (``OpenAIResponsesAssistantPhase`` in their codebase).
    """

    @pytest.mark.asyncio
    async def test_commentary_phase_text_dropped_in_stream(self):
        """Commentary-phase deltas (Codex scratch like 'Need view_video')
        must NOT reach the chat-completion content."""
        state = StreamState(model="gpt-5.5")
        upstream = _FakeUpstream([
            _sse({
                "type": "response.output_item.added",
                "item": {
                    "id": "msg_1",
                    "type": "message",
                    "phase": "commentary",
                },
            }),
            _sse({"type": "response.output_text.delta",
                  "delta": "Need view_video maybe"}),
            _sse({"type": "response.output_text.delta",
                  "delta": " returns note?"}),
            _sse({"type": "response.output_item.done"}),
            _sse({
                "type": "response.output_item.added",
                "item": {
                    "id": "msg_2",
                    "type": "message",
                    "phase": "final_answer",
                },
            }),
            _sse({"type": "response.output_text.delta",
                  "delta": "Hello user!"}),
            _sse({"type": "response.output_item.done"}),
            _sse({"type": "response.completed",
                  "response": {"output": []}}),
        ])
        chunks = []
        async for c in translate_responses_events_to_chat_chunks(
            upstream, state,
        ):
            chunks.append(c)
        text = "".join(
            c["choices"][0]["delta"].get("content", "")
            for c in chunks
        )
        # Only the final_answer text reaches the channel.
        assert text == "Hello user!"
        assert "Need view_video" not in text

    @pytest.mark.asyncio
    async def test_final_answer_passes_through(self):
        """Sanity: final_answer phase text is forwarded as before."""
        state = StreamState(model="gpt-5.5")
        upstream = _FakeUpstream([
            _sse({
                "type": "response.output_item.added",
                "item": {
                    "id": "msg_1",
                    "type": "message",
                    "phase": "final_answer",
                },
            }),
            _sse({"type": "response.output_text.delta",
                  "delta": "哼，收到。我先睇下。"}),
            _sse({"type": "response.output_item.done"}),
            _sse({"type": "response.completed",
                  "response": {"output": []}}),
        ])
        chunks = []
        async for c in translate_responses_events_to_chat_chunks(
            upstream, state,
        ):
            chunks.append(c)
        text = "".join(
            c["choices"][0]["delta"].get("content", "")
            for c in chunks
        )
        assert text == "哼，收到。我先睇下。"

    @pytest.mark.asyncio
    async def test_missing_phase_treated_as_final_answer(self):
        """Backwards compat: when the older API doesn't expose
        ``phase`` on message items, default behaviour is to
        forward the text (assume final_answer).  Prevents
        regression for non-Codex providers that share this
        translator and never set the field."""
        state = StreamState(model="gpt-5.4")
        upstream = _FakeUpstream([
            _sse({
                "type": "response.output_item.added",
                "item": {"id": "msg_1", "type": "message"},
            }),
            _sse({"type": "response.output_text.delta",
                  "delta": "ok"}),
            _sse({"type": "response.output_item.done"}),
            _sse({"type": "response.completed",
                  "response": {"output": []}}),
        ])
        chunks = []
        async for c in translate_responses_events_to_chat_chunks(
            upstream, state,
        ):
            chunks.append(c)
        text = "".join(
            c["choices"][0]["delta"].get("content", "")
            for c in chunks
        )
        assert text == "ok"

    @pytest.mark.asyncio
    async def test_commentary_drop_in_collect(self):
        """Same gate works on the non-streaming drain path."""
        state = StreamState(model="gpt-5.5")
        upstream = _FakeUpstream([
            _sse({
                "type": "response.output_item.added",
                "item": {
                    "id": "msg_1",
                    "type": "message",
                    "phase": "commentary",
                },
            }),
            _sse({"type": "response.output_text.delta",
                  "delta": "scratch text"}),
            _sse({"type": "response.output_item.done"}),
            _sse({
                "type": "response.output_item.added",
                "item": {
                    "id": "msg_2",
                    "type": "message",
                    "phase": "final_answer",
                },
            }),
            _sse({"type": "response.output_text.delta",
                  "delta": "user-facing"}),
            _sse({"type": "response.output_item.done"}),
            _sse({"type": "response.completed",
                  "response": {"output": []}}),
        ])
        body = await collect_as_chat_completion(upstream, state)
        assert body["choices"][0]["message"]["content"] == "user-facing"
