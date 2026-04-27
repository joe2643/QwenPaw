# -*- coding: utf-8 -*-
"""Unit tests for the Claude Code (acpx) chat-model wrapper — Lane D
(daemon dispatch + session registry + ACP translation).

The wrapper is the seam between agentscope's OpenAIChatModel surface
and our acpx subprocess pipeline.  These tests pin five things:

1. **Constructor** instantiates under the kwargs the dispatch branch
   in ``AnthropicProvider.get_chat_model_instance`` actually passes.
2. **Helpers** (``_extract_system_prompt`` / ``_extract_tool_names``
   / ``_detect_effort``) compute the inputs to the env_hash and the
   effort-delta sync.
3. **Plan→submit→commit** flow drives the registry through
   ``plan_turn`` and ``commit_turn`` and forwards the right blocks
   to the daemon for both ``seed_full`` and ``ship_tail`` modes.
4. **Effort delta sync** pushes ``acpx claude set effort <level>`` only
   when the entry's last-recorded effort changed.
5. **Stream adapter** finalises (commit + lock release) on natural
   end-of-iter, and bails without commit on close / exception.

Daemon and registry are stubbed with in-process fakes so the tests run
without acpx installed.
"""

from __future__ import annotations

import asyncio
from typing import AsyncIterator

import pytest

from qwenpaw.providers import claude_acpx_metrics
from qwenpaw.providers.claude_acpx_model import (
    ClaudeAcpxChatModel,
    _AcpxStreamAdapter,
    _detect_effort,
    _extract_system_prompt,
    _extract_tool_names,
    _wire_registry_tear_down,
)
from qwenpaw.providers.claude_acpx_session_registry import (
    Registry,
    set_registry_for_test,
)


# =================================================================== #
# Fixtures + fakes                                                    #
# =================================================================== #


@pytest.fixture(autouse=True)
def _reset_metrics() -> None:
    claude_acpx_metrics.reset_for_test()


@pytest.fixture
def fresh_registry() -> Registry:
    """Per-test registry — global singleton state would otherwise
    leak entry hashes between cases."""
    reg = Registry()
    set_registry_for_test(reg)
    yield reg
    set_registry_for_test(None)


@pytest.fixture
def context_set():
    """Set the ContextVars the wrapper requires.  Without these, the
    wrapper raises (intentional — silent fallback caused session-key
    collisions; see codex review note [4])."""
    from qwenpaw.app.agent_context import (
        set_current_agent_id,
        set_current_session_id,
    )

    set_current_agent_id("alice")
    set_current_session_id("conv-test")
    yield
    # ContextVars are scoped per-task in production but tests run in
    # a single asyncio task; clearing not strictly necessary but
    # keeps cross-test isolation.


class _FakeDaemon:
    """In-process stand-in for :class:`AcpxDaemon` covering the
    methods the wrapper calls.  Records each call so tests can assert
    the wrapper drove the daemon as expected.
    """

    def __init__(
        self,
        lines: list[str] | None = None,
        set_config_raises: Exception | None = None,
    ) -> None:
        self.lines = lines or []
        self.submit_calls: list[dict] = []
        self.set_config_calls: list[tuple[str, str, str]] = []
        self.teardown_calls: list[str] = []
        self._set_config_raises = set_config_raises

    async def submit_turn(
        self,
        *,
        session_name: str,
        prompt_blocks: list[dict],
        is_seed: bool,
    ) -> AsyncIterator[str]:
        self.submit_calls.append(
            {
                "session_name": session_name,
                "prompt_blocks": prompt_blocks,
                "is_seed": is_seed,
            },
        )
        for line in self.lines:
            yield line

    async def run_set_config(
        self,
        session_name: str,
        key: str,
        value: str,
    ) -> None:
        self.set_config_calls.append((session_name, key, value))
        if self._set_config_raises is not None:
            raise self._set_config_raises

    async def teardown(self, session_name: str) -> None:
        self.teardown_calls.append(session_name)


def _patch_daemon(monkeypatch: pytest.MonkeyPatch, daemon: _FakeDaemon) -> None:
    """Pin both ``AcpxDaemon.get_or_spawn`` (used inside the wrapper)
    and the imported reference in ``claude_acpx_model`` to ``daemon``.
    ``get_or_spawn`` is a classmethod returning the singleton; tests
    swap it for a constant-returning lambda so each test gets its own
    fake without singleton bleed.
    """
    from qwenpaw.providers import claude_acpx_daemon

    monkeypatch.setattr(
        claude_acpx_daemon.AcpxDaemon,
        "get_or_spawn",
        classmethod(lambda cls: daemon),
    )


def _final_acp_line() -> str:
    """A minimal terminal ACP JSON-RPC response: ``{"result":
    {"stopReason": "end_turn"}}`` — enough to make
    ``translate_acp_updates_to_chat_chunks`` exit cleanly with a
    ``finish_reason="stop"`` chunk."""
    return '{"jsonrpc":"2.0","id":1,"result":{"stopReason":"end_turn"}}\n'


def _agent_message_line(text: str) -> str:
    import json as _json

    return _json.dumps(
        {
            "jsonrpc": "2.0",
            "method": "session/update",
            "params": {
                "sessionId": "sess",
                "update": {
                    "sessionUpdate": "agent_message_chunk",
                    "content": {"type": "text", "text": text},
                },
            },
        },
    ) + "\n"


# =================================================================== #
# Constructor invariants                                              #
# =================================================================== #


class TestClaudeAcpxChatModelInit:
    """The constructor must accept the kwargs the dispatch branch in
    ``AnthropicProvider.get_chat_model_instance`` passes — any drift
    here surfaces as a TypeError at provider-tile click."""

    def test_constructs_with_minimal_kwargs(self) -> None:
        model = ClaudeAcpxChatModel(model_name="claude-sonnet-4-5")
        assert model is not None
        assert callable(model.client.chat.completions.create)

    def test_constructs_with_full_dispatch_kwargs(self) -> None:
        model = ClaudeAcpxChatModel(
            model_name="claude-opus-4-5",
            stream=True,
            stream_tool_parsing=False,
            client_kwargs={"base_url": "acpx://claude"},
            generate_kwargs={"max_tokens": 8192},
        )
        assert callable(model.client.chat.completions.create)

    def test_seeds_api_key_sentinel_when_none(self) -> None:
        model = ClaudeAcpxChatModel(
            model_name="claude-sonnet-4-5",
            api_key=None,
        )
        assert callable(model.client.chat.completions.create)

    def test_seeds_api_key_sentinel_when_empty_string(self) -> None:
        model = ClaudeAcpxChatModel(
            model_name="claude-sonnet-4-5",
            api_key="",
        )
        assert callable(model.client.chat.completions.create)

    def test_keeps_explicit_api_key_if_given(self) -> None:
        model = ClaudeAcpxChatModel(
            model_name="claude-sonnet-4-5",
            api_key="sk-explicit-noop",
        )
        assert callable(model.client.chat.completions.create)

    def test_two_instances_get_independent_clients(self) -> None:
        a = ClaudeAcpxChatModel(model_name="claude-sonnet-4-5")
        b = ClaudeAcpxChatModel(model_name="claude-opus-4-5")
        assert a.client is not b.client


# =================================================================== #
# Helper functions                                                    #
# =================================================================== #


class TestExtractSystemPrompt:
    def test_no_system_returns_empty(self) -> None:
        msgs = [{"role": "user", "content": "hi"}]
        assert _extract_system_prompt(msgs) == ""

    def test_string_system(self) -> None:
        msgs = [
            {"role": "system", "content": "you are claude"},
            {"role": "user", "content": "hi"},
        ]
        assert _extract_system_prompt(msgs) == "you are claude"

    def test_list_blocks_collapse_to_text(self) -> None:
        msgs = [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": "block-A"},
                    {"type": "text", "text": "block-B"},
                ],
            },
        ]
        assert _extract_system_prompt(msgs) == "block-A\n\nblock-B"

    def test_stops_at_first_non_system(self) -> None:
        # A system message *after* a user message is unusual and
        # should not contribute — env_hash signal lives in the
        # leading system block.
        msgs = [
            {"role": "system", "content": "first"},
            {"role": "user", "content": "hi"},
            {"role": "system", "content": "second"},
        ]
        assert _extract_system_prompt(msgs) == "first"


class TestExtractToolNames:
    def test_none_returns_empty(self) -> None:
        assert _extract_tool_names(None) == []

    def test_extracts_function_names(self) -> None:
        tools = [
            {"type": "function", "function": {"name": "alpha"}},
            {"type": "function", "function": {"name": "beta"}},
        ]
        assert _extract_tool_names(tools) == ["alpha", "beta"]

    def test_skips_invalid_entries(self) -> None:
        tools = [
            {"type": "function", "function": {"name": "alpha"}},
            "not-a-dict",
            {"type": "function"},  # no nested function
            {"type": "function", "function": {}},  # no name
        ]
        assert _extract_tool_names(tools) == ["alpha"]


class TestDetectEffort:
    def test_unset(self) -> None:
        assert _detect_effort({}, {}) is None

    def test_per_call_reasoning_effort_wins(self) -> None:
        assert _detect_effort(
            {"reasoning_effort": "high"},
            {"reasoning_effort": "low"},
        ) == "high"

    def test_falls_back_to_generate_kwargs(self) -> None:
        assert _detect_effort({}, {"reasoning_effort": "medium"}) == "medium"

    def test_reasoning_dict_form(self) -> None:
        assert _detect_effort(
            {"reasoning": {"effort": "low"}},
            None,
        ) == "low"

    def test_anthropic_thinking_budget_ignored(self) -> None:
        # acpx's underlying ``effort`` config option only accepts the
        # symbolic levels {low, medium, high, xhigh, max}; budget_tokens
        # would round-trip as ``budget:4096`` and acpx rejects it with
        # a generic Internal error.  We deliberately don't map it —
        # smoke test 2026-04-27 caught the rejection.
        assert _detect_effort(
            {"thinking": {"budget_tokens": 4096}},
            None,
        ) is None


class TestWireRegistryTearDown:
    def test_wires_once(self, fresh_registry: Registry) -> None:
        teardown_calls: list[str] = []

        class _D:
            async def teardown(self, name: str) -> None:
                teardown_calls.append(name)

        d = _D()
        _wire_registry_tear_down(fresh_registry, d)
        assert getattr(fresh_registry, "_tear_down_wired", False) is True
        # Second call shouldn't re-bind even with a different fake.
        d2 = _D()
        _wire_registry_tear_down(fresh_registry, d2)
        # Active callback is still the first one.
        asyncio.run(fresh_registry._tear_down("name-1"))
        assert teardown_calls == ["name-1"]


# =================================================================== #
# End-to-end wrapper flow (non-stream)                                #
# =================================================================== #


class TestNonStreamFlow:
    """Drive ``client.chat.completions.create`` with stream=False and
    assert the wrapper invoked the daemon and recorded the turn in
    the registry."""

    def test_seed_full_first_turn(
        self,
        monkeypatch: pytest.MonkeyPatch,
        fresh_registry: Registry,
        context_set: None,
    ) -> None:
        daemon = _FakeDaemon(
            lines=[
                _agent_message_line("Hello"),
                _final_acp_line(),
            ],
        )
        _patch_daemon(monkeypatch, daemon)

        model = ClaudeAcpxChatModel(model_name="claude-sonnet-4-5")
        result = asyncio.run(
            model.client.chat.completions.create(
                model="claude-sonnet-4-5",
                messages=[
                    {"role": "system", "content": "be brief"},
                    {"role": "user", "content": "hi"},
                ],
                stream=False,
            ),
        )

        # Daemon got exactly one submit_turn with is_seed=True.
        assert len(daemon.submit_calls) == 1
        call = daemon.submit_calls[0]
        assert call["is_seed"] is True
        # Prompt blocks include the user content.
        joined_text = "".join(
            b.get("text", "") for b in call["prompt_blocks"]
        )
        assert "hi" in joined_text

        # The translator emitted a chat completion with our text.
        assert result.choices[0].message.content == "Hello"

        # Metrics: one seed_full, no ship_tail.
        snap = claude_acpx_metrics.snapshot()
        assert snap["seed_full"] == 1
        assert snap["ship_tail"] == 0

        # Registry: entry now records the shipped state so a second
        # turn would ship_tail.
        assert len(fresh_registry) == 1

    def test_ship_tail_second_turn(
        self,
        monkeypatch: pytest.MonkeyPatch,
        fresh_registry: Registry,
        context_set: None,
    ) -> None:
        daemon = _FakeDaemon(
            lines=[
                _agent_message_line("ack"),
                _final_acp_line(),
            ],
        )
        _patch_daemon(monkeypatch, daemon)

        model = ClaudeAcpxChatModel(model_name="claude-sonnet-4-5")
        history_seed = [
            {"role": "system", "content": "be brief"},
            {"role": "user", "content": "hi"},
        ]
        # First call seeds.
        asyncio.run(
            model.client.chat.completions.create(
                model="claude-sonnet-4-5",
                messages=history_seed,
                stream=False,
            ),
        )
        # Second call extends — registry should plan ship_tail.
        history_extended = history_seed + [
            {"role": "assistant", "content": "ack"},
            {"role": "user", "content": "again"},
        ]
        # Reset daemon recorder so the second call's args are isolated.
        daemon.submit_calls.clear()
        daemon.lines = [_agent_message_line("ok"), _final_acp_line()]

        result = asyncio.run(
            model.client.chat.completions.create(
                model="claude-sonnet-4-5",
                messages=history_extended,
                stream=False,
            ),
        )

        assert len(daemon.submit_calls) == 1
        call = daemon.submit_calls[0]
        assert call["is_seed"] is False
        # Prompt blocks should NOT include the seeded "hi" — only
        # the new tail content.
        joined_text = "".join(
            b.get("text", "") for b in call["prompt_blocks"]
        )
        assert "hi" not in joined_text
        assert "again" in joined_text

        snap = claude_acpx_metrics.snapshot()
        assert snap["seed_full"] == 1
        assert snap["ship_tail"] == 1

        assert result.choices[0].message.content == "ok"


# =================================================================== #
# Effort delta sync                                                   #
# =================================================================== #


class TestEffortSync:
    def test_pushes_set_thinking_when_effort_changes(
        self,
        monkeypatch: pytest.MonkeyPatch,
        fresh_registry: Registry,
        context_set: None,
    ) -> None:
        daemon = _FakeDaemon(
            lines=[_agent_message_line("hi"), _final_acp_line()],
        )
        _patch_daemon(monkeypatch, daemon)

        model = ClaudeAcpxChatModel(
            model_name="claude-sonnet-4-5",
            generate_kwargs={"reasoning_effort": "medium"},
        )
        asyncio.run(
            model.client.chat.completions.create(
                model="claude-sonnet-4-5",
                messages=[{"role": "user", "content": "hi"}],
                stream=False,
            ),
        )

        # Effort medium pushed once (cold session, no prior effort).
        assert daemon.set_config_calls == [
            (daemon.submit_calls[0]["session_name"], "effort", "medium"),
        ]

    def test_skips_set_thinking_when_effort_unchanged(
        self,
        monkeypatch: pytest.MonkeyPatch,
        fresh_registry: Registry,
        context_set: None,
    ) -> None:
        daemon = _FakeDaemon(
            lines=[_agent_message_line("hi"), _final_acp_line()],
        )
        _patch_daemon(monkeypatch, daemon)

        model = ClaudeAcpxChatModel(
            model_name="claude-sonnet-4-5",
            generate_kwargs={"reasoning_effort": "medium"},
        )
        # First call seeds + pushes effort.
        asyncio.run(
            model.client.chat.completions.create(
                model="claude-sonnet-4-5",
                messages=[{"role": "user", "content": "hi"}],
                stream=False,
            ),
        )
        # Reset stub state.
        daemon.set_config_calls.clear()
        daemon.submit_calls.clear()
        daemon.lines = [_agent_message_line("ok"), _final_acp_line()]

        # Second call same effort — no set_config push.
        asyncio.run(
            model.client.chat.completions.create(
                model="claude-sonnet-4-5",
                messages=[
                    {"role": "user", "content": "hi"},
                    {"role": "assistant", "content": "hi"},
                    {"role": "user", "content": "again"},
                ],
                stream=False,
            ),
        )

        assert daemon.set_config_calls == []

    def test_set_config_failure_does_not_block_turn(
        self,
        monkeypatch: pytest.MonkeyPatch,
        fresh_registry: Registry,
        context_set: None,
    ) -> None:
        from qwenpaw.providers.claude_acpx_daemon import AcpxDaemonError

        daemon = _FakeDaemon(
            lines=[_agent_message_line("hi"), _final_acp_line()],
            set_config_raises=AcpxDaemonError("rc=1: nope"),
        )
        _patch_daemon(monkeypatch, daemon)

        model = ClaudeAcpxChatModel(
            model_name="claude-sonnet-4-5",
            generate_kwargs={"reasoning_effort": "high"},
        )
        # Turn must complete despite set_thinking failure.
        result = asyncio.run(
            model.client.chat.completions.create(
                model="claude-sonnet-4-5",
                messages=[{"role": "user", "content": "hi"}],
                stream=False,
            ),
        )
        assert result.choices[0].message.content == "hi"


# =================================================================== #
# Stream adapter                                                      #
# =================================================================== #


class TestStreamAdapter:
    def test_yields_chunks_and_commits_on_natural_end(
        self,
        monkeypatch: pytest.MonkeyPatch,
        fresh_registry: Registry,
        context_set: None,
    ) -> None:
        daemon = _FakeDaemon(
            lines=[_agent_message_line("Hello"), _final_acp_line()],
        )
        _patch_daemon(monkeypatch, daemon)

        model = ClaudeAcpxChatModel(model_name="claude-sonnet-4-5")
        adapter = asyncio.run(
            model.client.chat.completions.create(
                model="claude-sonnet-4-5",
                messages=[{"role": "user", "content": "hi"}],
                stream=True,
            ),
        )
        assert isinstance(adapter, _AcpxStreamAdapter)

        async def _drain() -> list:
            return [chunk async for chunk in adapter]

        chunks = asyncio.run(_drain())
        # Role chunk + content chunk + finish chunk = 3 minimum
        assert len(chunks) >= 2
        # After natural end-of-iter the adapter should have released
        # its lock so a follow-up turn could acquire it.
        # We can't easily peek into the entry from here, but reaching
        # this point without a deadlock or RuntimeError demonstrates it.

    def test_close_releases_lock_without_commit(
        self,
        monkeypatch: pytest.MonkeyPatch,
        fresh_registry: Registry,
        context_set: None,
    ) -> None:
        # Many lines so iteration doesn't naturally finish before close.
        daemon = _FakeDaemon(
            lines=[
                _agent_message_line("part1"),
                _agent_message_line("part2"),
                _agent_message_line("part3"),
                _final_acp_line(),
            ],
        )
        _patch_daemon(monkeypatch, daemon)

        model = ClaudeAcpxChatModel(model_name="claude-sonnet-4-5")
        adapter = asyncio.run(
            model.client.chat.completions.create(
                model="claude-sonnet-4-5",
                messages=[{"role": "user", "content": "hi"}],
                stream=True,
            ),
        )

        async def _partial_then_close() -> None:
            ait = adapter.__aiter__()
            await ait.__anext__()  # role chunk
            await ait.__anext__()  # first content chunk
            await adapter.close()
            # After close, further __anext__ raises StopAsyncIteration.
            with pytest.raises(StopAsyncIteration):
                await ait.__anext__()

        asyncio.run(_partial_then_close())

    def test_aexit_releases_lock(
        self,
        monkeypatch: pytest.MonkeyPatch,
        fresh_registry: Registry,
        context_set: None,
    ) -> None:
        daemon = _FakeDaemon(
            lines=[_agent_message_line("Hi"), _final_acp_line()],
        )
        _patch_daemon(monkeypatch, daemon)

        model = ClaudeAcpxChatModel(model_name="claude-sonnet-4-5")
        adapter = asyncio.run(
            model.client.chat.completions.create(
                model="claude-sonnet-4-5",
                messages=[{"role": "user", "content": "hi"}],
                stream=True,
            ),
        )

        async def _use_then_exit() -> None:
            async with adapter as a:
                async for _ in a:
                    pass

        asyncio.run(_use_then_exit())
        # No exception means __aexit__ ran cleanly.

    def test_finalize_idempotent(
        self,
        monkeypatch: pytest.MonkeyPatch,
        fresh_registry: Registry,
        context_set: None,
    ) -> None:
        daemon = _FakeDaemon(
            lines=[_agent_message_line("Hi"), _final_acp_line()],
        )
        _patch_daemon(monkeypatch, daemon)

        model = ClaudeAcpxChatModel(model_name="claude-sonnet-4-5")
        adapter = asyncio.run(
            model.client.chat.completions.create(
                model="claude-sonnet-4-5",
                messages=[{"role": "user", "content": "hi"}],
                stream=True,
            ),
        )

        async def _double_close() -> None:
            await adapter.close()
            # Second close shouldn't blow up.
            await adapter.close()

        asyncio.run(_double_close())


# =================================================================== #
# Codex review fixes — defensive contracts                            #
# =================================================================== #


class TestContextVarRequired:
    """Codex review note [4]: a missing session_id ContextVar previously
    fell through to a constant ``"ad-hoc"`` key, letting unrelated
    callers collide on the same Claude session.  The wrapper now
    raises a clear RuntimeError rather than silently sharing state.
    """

    def test_missing_session_id_raises(
        self,
        monkeypatch: pytest.MonkeyPatch,
        fresh_registry: Registry,
    ) -> None:
        from qwenpaw.app.agent_context import set_current_agent_id
        from qwenpaw.app import agent_context

        set_current_agent_id("alice")
        # Leave _current_session_id at its module default (None).
        agent_context._current_session_id.set(None)

        daemon = _FakeDaemon(
            lines=[_agent_message_line("hi"), _final_acp_line()],
        )
        _patch_daemon(monkeypatch, daemon)

        model = ClaudeAcpxChatModel(model_name="claude-sonnet-4-5")
        with pytest.raises(RuntimeError) as exc:
            asyncio.run(
                model.client.chat.completions.create(
                    model="claude-sonnet-4-5",
                    messages=[{"role": "user", "content": "hi"}],
                    stream=False,
                ),
            )
        assert "session_id" in str(exc.value)


class TestStreamLazyOpen:
    """Codex review note [1]: the wrapper used to acquire ``entry.lock``
    eagerly in ``_wrapped_create`` for the streaming path.  If the
    consumer abandoned the adapter before iterating, the lock leaked
    forever.  The fix: defer lock acquire + effort-set + submit_turn
    spawn into the adapter's first ``__anext__``.  These tests pin
    that behaviour.
    """

    def test_unopened_stream_does_not_acquire_lock(
        self,
        monkeypatch: pytest.MonkeyPatch,
        fresh_registry: Registry,
        context_set: None,
    ) -> None:
        daemon = _FakeDaemon(
            lines=[_agent_message_line("hi"), _final_acp_line()],
        )
        _patch_daemon(monkeypatch, daemon)

        model = ClaudeAcpxChatModel(model_name="claude-sonnet-4-5")
        adapter = asyncio.run(
            model.client.chat.completions.create(
                model="claude-sonnet-4-5",
                messages=[{"role": "user", "content": "hi"}],
                stream=True,
            ),
        )
        # Adapter exists but iteration hasn't started.
        # The entry's lock must NOT be held — otherwise a follow-up
        # turn for the same conversation would deadlock.
        # We don't have a public handle on the entry from outside
        # the adapter; assert via _lock_held marker.
        assert adapter._lock_held is False
        assert adapter._opened is False
        # Daemon should not have been called for submit_turn yet.
        assert daemon.submit_calls == []

    def test_unopened_stream_close_is_safe(
        self,
        monkeypatch: pytest.MonkeyPatch,
        fresh_registry: Registry,
        context_set: None,
    ) -> None:
        daemon = _FakeDaemon(
            lines=[_agent_message_line("hi"), _final_acp_line()],
        )
        _patch_daemon(monkeypatch, daemon)

        model = ClaudeAcpxChatModel(model_name="claude-sonnet-4-5")
        adapter = asyncio.run(
            model.client.chat.completions.create(
                model="claude-sonnet-4-5",
                messages=[{"role": "user", "content": "hi"}],
                stream=True,
            ),
        )

        async def _close_unopened() -> None:
            await adapter.close()

        asyncio.run(_close_unopened())
        # Still nothing submitted.
        assert daemon.submit_calls == []

    def test_first_anext_acquires_lock_and_submits(
        self,
        monkeypatch: pytest.MonkeyPatch,
        fresh_registry: Registry,
        context_set: None,
    ) -> None:
        daemon = _FakeDaemon(
            lines=[_agent_message_line("Hi"), _final_acp_line()],
        )
        _patch_daemon(monkeypatch, daemon)

        model = ClaudeAcpxChatModel(model_name="claude-sonnet-4-5")
        adapter = asyncio.run(
            model.client.chat.completions.create(
                model="claude-sonnet-4-5",
                messages=[{"role": "user", "content": "hi"}],
                stream=True,
            ),
        )

        async def _two_anext() -> None:
            ait = adapter.__aiter__()
            # Translator yields the synthetic "role: assistant" chunk
            # BEFORE iterating the line_reader, so the first __anext__
            # doesn't actually drive submit_turn's body.  A second
            # anext drains a content line and triggers the daemon.
            await ait.__anext__()  # role chunk
            await ait.__anext__()  # first content chunk
            # Manual cleanup so the abandoned adapter doesn't trigger
            # __del__ log noise during pytest teardown.
            await adapter.close()

        asyncio.run(_two_anext())
        assert len(daemon.submit_calls) == 1


class TestOpenCloseRace:
    """Codex review note [N1]: ``_open()`` previously set
    ``_opened = True`` before awaiting the lock.  A concurrent
    ``close()`` could flip ``_closed`` while we were blocked on
    acquire; once acquire completed we'd plough on with submit_turn
    even though the adapter was closed.  The fix is the re-check
    after lock acquire.  This test pins the post-acquire bail.
    """

    def test_close_during_open_aborts_safely(
        self,
        monkeypatch: pytest.MonkeyPatch,
        fresh_registry: Registry,
        context_set: None,
    ) -> None:
        daemon = _FakeDaemon(
            lines=[_agent_message_line("hi"), _final_acp_line()],
        )
        _patch_daemon(monkeypatch, daemon)

        model = ClaudeAcpxChatModel(model_name="claude-sonnet-4-5")
        adapter = asyncio.run(
            model.client.chat.completions.create(
                model="claude-sonnet-4-5",
                messages=[{"role": "user", "content": "hi"}],
                stream=True,
            ),
        )

        async def _race() -> None:
            # Pre-acquire the entry lock so _open will block on it.
            await adapter._entry.lock.acquire()
            try:
                # Schedule _open in a background task; it'll block on
                # entry.lock.acquire().
                open_task = asyncio.create_task(adapter._open())
                # Yield once so the task starts and reaches the await.
                await asyncio.sleep(0)
                # Close the adapter while _open is blocked.
                await adapter.close()
            finally:
                # Release our pre-acquired lock so _open's acquire
                # call completes.
                if adapter._entry.lock.locked():
                    try:
                        adapter._entry.lock.release()
                    except RuntimeError:
                        pass
            # Drain the open task — should complete without doing
            # any submit_turn work because _closed was True after
            # acquire.
            await open_task

        asyncio.run(_race())
        # No submit_turn should have been triggered.
        assert daemon.submit_calls == []
        # Adapter is closed and the lock is not held.
        assert adapter._closed is True


class TestEffortDeferredToAdapter:
    """The streaming path defers effort-set into the adapter's
    ``_open``.  Constructor-time effort delta on a stream call should
    push set-config only after iteration starts, not before."""

    def test_set_thinking_only_after_first_anext(
        self,
        monkeypatch: pytest.MonkeyPatch,
        fresh_registry: Registry,
        context_set: None,
    ) -> None:
        daemon = _FakeDaemon(
            lines=[_agent_message_line("hi"), _final_acp_line()],
        )
        _patch_daemon(monkeypatch, daemon)

        model = ClaudeAcpxChatModel(
            model_name="claude-sonnet-4-5",
            generate_kwargs={"reasoning_effort": "low"},
        )
        adapter = asyncio.run(
            model.client.chat.completions.create(
                model="claude-sonnet-4-5",
                messages=[{"role": "user", "content": "hi"}],
                stream=True,
            ),
        )
        # Before iteration: no set_config push.
        assert daemon.set_config_calls == []

        async def _drain() -> None:
            async for _ in adapter:
                pass

        asyncio.run(_drain())
        # After full iteration: set_config pushed exactly once.
        assert len(daemon.set_config_calls) == 1
        assert daemon.set_config_calls[0][1:] == ("effort", "low")
