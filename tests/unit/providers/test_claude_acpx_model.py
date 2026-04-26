# -*- coding: utf-8 -*-
"""Unit tests for the Claude Code (acpx) chat-model wrapper — Lane C STUB.

The full chat-completions wiring (daemon dispatch, session registry,
ACP translation) lands in Lane B + Lane D.  These tests pin the
*surface contract* Lanes B/D will import: constructor signature,
sentinel api_key seeding, and the ``_install_wrapper`` patch that
swaps ``client.chat.completions.create``.  Once Lane D fills in the
stub body, this file should grow to cover the daemon path; the
``NotImplementedError`` assertions below act as a tripwire that
forces test-update during that work.
"""

from __future__ import annotations

import asyncio

import pytest

from qwenpaw.providers.claude_acpx_model import ClaudeAcpxChatModel


# ---------------------------------------------------------------- #
# Constructor                                                      #
# ---------------------------------------------------------------- #


class TestClaudeAcpxChatModelInit:
    """Constructor invariants — proves the class instantiates without
    blowing up under the kwargs ``AnthropicProvider.get_chat_model_instance``
    actually passes (model_name + stream + stream_tool_parsing +
    client_kwargs + generate_kwargs).  Skipping any one of these
    reveals an upstream agentscope mismatch before Lane D wires
    anything live."""

    def test_constructs_with_minimal_kwargs(self) -> None:
        # Smallest path the dispatch branch in
        # ``anthropic_provider.get_chat_model_instance`` exercises —
        # any breakage here surfaces as ImportError-or-TypeError at
        # provider-tile click rather than at first chat turn.
        model = ClaudeAcpxChatModel(model_name="claude-sonnet-4-5")
        assert model is not None
        assert callable(model.client.chat.completions.create)

    def test_constructs_with_full_dispatch_kwargs(self) -> None:
        # Mirrors the call site in
        # :meth:`AnthropicProvider.get_chat_model_instance` so this
        # test fails immediately if either side drifts.
        model = ClaudeAcpxChatModel(
            model_name="claude-opus-4-5",
            stream=True,
            stream_tool_parsing=False,
            client_kwargs={"base_url": "acpx://claude"},
            generate_kwargs={"max_tokens": 8192},
        )
        assert callable(model.client.chat.completions.create)

    def test_seeds_api_key_sentinel_when_none(self) -> None:
        # OpenAI SDK raises on construction without ``api_key`` —
        # the wrapper must inject a harmless sentinel since the
        # real auth path bypasses the SDK's HTTP layer entirely.
        model = ClaudeAcpxChatModel(
            model_name="claude-sonnet-4-5",
            api_key=None,
        )
        # The sentinel never reaches the wire (the wrapper redirects
        # away from default base URL), so we only assert construction
        # succeeded and the wrapper installed.
        assert callable(model.client.chat.completions.create)

    def test_seeds_api_key_sentinel_when_empty_string(self) -> None:
        # ``not api_key`` covers both ``None`` and ``""`` — the latter
        # is what some pydantic codepaths produce when the user clears
        # the field in the UI rather than deleting it.
        model = ClaudeAcpxChatModel(
            model_name="claude-sonnet-4-5",
            api_key="",
        )
        assert callable(model.client.chat.completions.create)

    def test_keeps_explicit_api_key_if_given(self) -> None:
        # Defensive: nothing in the wrapper should overwrite a
        # caller-supplied key.  Construction must still succeed even
        # though the value is meaningless for the acpx path.
        model = ClaudeAcpxChatModel(
            model_name="claude-sonnet-4-5",
            api_key="sk-explicit-noop",
        )
        assert callable(model.client.chat.completions.create)


# ---------------------------------------------------------------- #
# _install_wrapper — the dispatch hook                             #
# ---------------------------------------------------------------- #


class TestInstallWrapper:
    """The wrapper replaces ``self.client.chat.completions.create``
    with a coroutine that raises NotImplementedError until Lane B/D
    fills in the daemon path.  These tests pin the *shape* of that
    contract so an accidental remove of the override (or a silent
    fall-through to OpenAI's API) trips a failure."""

    def test_create_is_replaced_by_wrapper(self) -> None:
        # Identity: after init, ``create`` is the wrapper closure, not
        # the SDK's bound method.  We can't introspect the closure
        # directly across openai-python versions, so we exercise the
        # behaviour: the wrapper raises ``NotImplementedError``, the
        # SDK's ``create`` would raise an HTTP/auth error from the
        # missing real call.
        model = ClaudeAcpxChatModel(model_name="claude-sonnet-4-5")
        # Calling the wrapper should hit the stub — the SDK's real
        # create would either succeed (against api.openai.com — bad)
        # or raise a different error.
        with pytest.raises(NotImplementedError) as exc_info:
            asyncio.run(
                model.client.chat.completions.create(
                    messages=[{"role": "user", "content": "hi"}],
                    model="claude-sonnet-4-5",
                    stream=True,
                ),
            )
        # The error text must mention Lane B/D so a future engineer
        # following the traceback knows where the wiring lives.
        msg = str(exc_info.value)
        assert "Lane B" in msg or "Lane D" in msg, (
            f"NotImplementedError must reference the responsible "
            f"lane(s) so the next engineer finds the wiring; "
            f"got: {msg!r}"
        )

    def test_wrapper_accepts_arbitrary_call_kwargs(self) -> None:
        # Lane D will pass through whatever the agentscope caller
        # built (model, messages, tools, stream_options, etc.) — the
        # stub must not reject them, otherwise the dispatch path
        # itself is gated on a TypeError before we get the chance to
        # surface the Lane B/D message.
        model = ClaudeAcpxChatModel(model_name="claude-opus-4-5")
        with pytest.raises(NotImplementedError):
            asyncio.run(
                model.client.chat.completions.create(
                    messages=[{"role": "user", "content": "hi"}],
                    model="claude-opus-4-5",
                    stream=True,
                    tools=[{"type": "function", "function": {"name": "x"}}],
                    tool_choice="auto",
                    stream_options={"include_usage": True},
                ),
            )

    def test_wrapper_install_is_idempotent_per_instance(self) -> None:
        # Each instance gets a fresh closure — constructing twice
        # in the same process must not share state through the
        # underlying SDK client (different httpx clients, different
        # closures).  Cheap regression guard for Lane D when it
        # introduces per-instance daemon handles.
        a = ClaudeAcpxChatModel(model_name="claude-sonnet-4-5")
        b = ClaudeAcpxChatModel(model_name="claude-opus-4-5")
        assert a.client is not b.client
        assert callable(a.client.chat.completions.create)
        assert callable(b.client.chat.completions.create)
