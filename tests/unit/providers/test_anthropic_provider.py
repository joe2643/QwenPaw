# -*- coding: utf-8 -*-
from __future__ import annotations

from types import SimpleNamespace

import qwenpaw.providers.anthropic_provider as anthropic_provider_module
from qwenpaw.providers.anthropic_provider import AnthropicProvider


def _make_provider(is_custom: bool = False) -> AnthropicProvider:
    return AnthropicProvider(
        id="anthropic",
        name="Anthropic",
        base_url="https://mock-anthropic.local",
        api_key="ant-test",
        chat_model="AnthropicChatModel",
        is_custom=is_custom,
    )


def test_get_chat_model_instance_uses_configured_max_tokens(
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}

    class FakeAnthropicChatModel:
        def __init__(self, **kwargs) -> None:
            captured.update(kwargs)

    monkeypatch.setattr(
        "agentscope.model.AnthropicChatModel",
        FakeAnthropicChatModel,
    )

    provider = _make_provider()
    provider.generate_kwargs = {
        "max_tokens": 4096,
        "temperature": 0.2,
    }

    provider.get_chat_model_instance("claude-3-5-sonnet")

    assert captured["model_name"] == "claude-3-5-sonnet"
    assert captured["max_tokens"] == 4096


def test_get_chat_model_instance_uses_default_max_tokens_when_unset(
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}

    class FakeAnthropicChatModel:
        def __init__(self, **kwargs) -> None:
            captured.update(kwargs)

    monkeypatch.setattr(
        "agentscope.model.AnthropicChatModel",
        FakeAnthropicChatModel,
    )

    provider = _make_provider()

    provider.get_chat_model_instance("claude-3-5-sonnet")

    assert captured["model_name"] == "claude-3-5-sonnet"
    assert captured["max_tokens"] == 16384


def test_get_chat_model_instance_does_not_mutate_generate_kwargs(
    monkeypatch,
) -> None:
    captured: list[dict[str, object]] = []

    class FakeAnthropicChatModel:
        def __init__(self, **kwargs) -> None:
            captured.append(kwargs)

    monkeypatch.setattr(
        "agentscope.model.AnthropicChatModel",
        FakeAnthropicChatModel,
    )

    provider = _make_provider()
    provider.generate_kwargs = {
        "max_tokens": 32768,
        "temperature": 0.2,
    }

    provider.get_chat_model_instance("claude-3-5-sonnet")
    provider.get_chat_model_instance("claude-3-5-sonnet")

    assert [call["max_tokens"] for call in captured] == [32768, 32768]
    assert provider.generate_kwargs == {
        "max_tokens": 32768,
        "temperature": 0.2,
    }
    assert captured[0]["generate_kwargs"] == {"temperature": 0.2}
    assert captured[1]["generate_kwargs"] == {"temperature": 0.2}


async def test_check_connection_success(monkeypatch) -> None:
    provider = _make_provider()
    called = {"count": 0}

    class FakeModels:
        async def list(self):
            called["count"] += 1
            return SimpleNamespace(data=[])

    fake_client = SimpleNamespace(models=FakeModels())
    monkeypatch.setattr(provider, "_client", lambda timeout=5: fake_client)

    ok, msg = await provider.check_connection(timeout=2.0)

    assert ok is True
    assert msg == ""
    assert called["count"] == 1


async def test_check_connection_api_error_returns_false(monkeypatch) -> None:
    provider = _make_provider()

    class FakeModels:
        async def list(self):
            raise RuntimeError("boom")

    fake_client = SimpleNamespace(models=FakeModels())
    monkeypatch.setattr(provider, "_client", lambda timeout=5: fake_client)
    monkeypatch.setattr(
        anthropic_provider_module.anthropic,
        "APIError",
        Exception,
    )

    ok, msg = await provider.check_connection(timeout=1.0)

    assert ok is False
    assert msg == "Anthropic API error: boom"


async def test_list_model_normalizes_and_deduplicates(monkeypatch) -> None:
    provider = _make_provider()
    rows = [
        SimpleNamespace(id="claude-3-5-haiku", display_name="Claude Haiku"),
        SimpleNamespace(id="claude-3-5-haiku", display_name=""),
        SimpleNamespace(id="claude-3-5-sonnet", display_name=""),
        SimpleNamespace(id="    ", display_name="invalid"),
    ]

    class FakeModels:
        async def list(self):
            return SimpleNamespace(data=rows)

    fake_client = SimpleNamespace(models=FakeModels())
    monkeypatch.setattr(provider, "_client", lambda timeout=5: fake_client)

    models = await provider.fetch_models(timeout=3.0)

    assert [model.id for model in models] == [
        "claude-3-5-haiku",
        "claude-3-5-sonnet",
    ]
    assert [model.name for model in models] == [
        "Claude Haiku",
        "claude-3-5-sonnet",
    ]
    assert not provider.models


async def test_check_model_connection_success(monkeypatch) -> None:
    provider = _make_provider()
    captured: list[dict] = []

    class FakeStream:
        def __aiter__(self):
            return self

        async def __anext__(self):
            raise StopAsyncIteration

    class FakeMessages:
        async def create(self, **kwargs):
            captured.append(kwargs)
            return FakeStream()

    fake_client = SimpleNamespace(messages=FakeMessages())
    monkeypatch.setattr(provider, "_client", lambda timeout=5: fake_client)

    ok, msg = await provider.check_model_connection(
        "claude-3-5-haiku",
        timeout=4.0,
    )

    assert ok is True
    assert msg == ""
    assert len(captured) == 1
    assert captured[0]["model"] == "claude-3-5-haiku"
    assert captured[0]["max_tokens"] == 1
    assert captured[0]["messages"] == [
        {"role": "user", "content": [{"type": "text", "text": "ping"}]},
    ]
    assert captured[0]["stream"] is True


async def test_check_model_connection_empty_model_id_returns_false() -> None:
    provider = _make_provider()

    ok, msg = await provider.check_model_connection("   ", timeout=4.0)

    assert ok is False
    assert msg == "Empty model ID"


async def test_check_model_connection_api_error_returns_false(
    monkeypatch,
) -> None:
    provider = _make_provider()

    class FakeMessages:
        async def create(self, **kwargs):
            _ = kwargs
            raise RuntimeError("failed")

    fake_client = SimpleNamespace(messages=FakeMessages())
    monkeypatch.setattr(provider, "_client", lambda timeout=5: fake_client)
    monkeypatch.setattr(
        anthropic_provider_module.anthropic,
        "APIError",
        Exception,
    )

    ok, msg = await provider.check_model_connection(
        "claude-3-5-haiku",
        timeout=4.0,
    )

    assert ok is False
    assert msg == "Model 'claude-3-5-haiku' is not reachable or usable"


# ---------------------------------------------------------------- #
# api_key sentinel branches — oauth vs acpx vs anthropic-direct    #
# ---------------------------------------------------------------- #


class TestSentinelBranches:
    """Pins the ``api_key`` sentinel dispatch in
    :class:`AnthropicProvider`.  Three string-valued sentinels share
    the same provider class:

    * ``"oauth"`` -> :class:`ClaudeOAuthChatModel` (claude-oauth tile)
    * ``"acpx"``  -> :class:`ClaudeAcpxChatModel`  (claude-acpx tile)
    * anything else -> ``AnthropicChatModel``       (Anthropic-direct)

    Drift between the sentinel constants and the dispatch branch
    silently routes a user's traffic to the wrong path (e.g. user
    picks claude-acpx tile but every chat hits Anthropic direct
    because the sentinel string disagrees).  These tests guard
    against that.
    """

    def test_acpx_sentinel_constant_value(self) -> None:
        # The literal string is load-bearing: it must match what
        # PROVIDER_CLAUDE_ACPX writes into the persisted provider
        # config in provider_manager.py.
        from qwenpaw.providers.anthropic_provider import (
            ACPX_API_KEY_SENTINEL,
        )

        assert ACPX_API_KEY_SENTINEL == "acpx"

    def test_oauth_sentinel_constant_value(self) -> None:
        # Symmetric guard so a future "let's rename to claude-oauth"
        # refactor doesn't silently break PROVIDER_CLAUDE_OAUTH.
        from qwenpaw.providers.anthropic_provider import (
            OAUTH_API_KEY_SENTINEL,
        )

        assert OAUTH_API_KEY_SENTINEL == "oauth"

    def test_is_acpx_true_when_api_key_is_acpx(self) -> None:
        provider = AnthropicProvider(
            id="claude-acpx",
            name="Claude Code (acpx)",
            base_url="acpx://claude",
            api_key="acpx",
            chat_model="OpenAIChatModel",
        )
        assert provider._is_acpx is True
        # Sentinels are mutually exclusive — same api_key cannot be
        # both at once.
        assert provider._is_oauth is False

    def test_is_acpx_false_when_api_key_is_oauth(self) -> None:
        provider = AnthropicProvider(
            id="claude-oauth",
            name="Claude Code (OAuth)",
            base_url="https://api.anthropic.com",
            api_key="oauth",
            chat_model="AnthropicChatModel",
        )
        assert provider._is_acpx is False
        assert provider._is_oauth is True

    def test_is_acpx_false_for_real_api_key(self) -> None:
        # Vanilla Anthropic-direct: api_key is the user's actual sk-
        # token, neither sentinel branch fires.
        provider = AnthropicProvider(
            id="anthropic",
            name="Anthropic",
            base_url="https://api.anthropic.com",
            api_key="sk-ant-real-token",
            chat_model="AnthropicChatModel",
        )
        assert provider._is_acpx is False
        assert provider._is_oauth is False

    def test_acpx_branch_returns_claude_acpx_chat_model(self) -> None:
        # Dispatch contract: get_chat_model_instance must reach the
        # acpx wrapper class when the sentinel is set.  This is
        # what the UI's Settings > Active Model tile cares about.
        from agentscope.model import AnthropicChatModel
        from qwenpaw.providers.claude_acpx_model import (
            ClaudeAcpxChatModel,
        )

        provider = AnthropicProvider(
            id="claude-acpx",
            name="Claude Code (acpx)",
            base_url="acpx://claude",
            api_key="acpx",
            chat_model="OpenAIChatModel",
        )
        model = provider.get_chat_model_instance("claude-sonnet-4-5")
        assert isinstance(model, ClaudeAcpxChatModel)
        # The wrapper subclasses OpenAIChatModel, NOT
        # AnthropicChatModel — guards against a future "swap base
        # class" refactor that would silently change parser behavior.
        assert not isinstance(model, AnthropicChatModel)

    def test_anthropic_branch_returns_anthropic_chat_model(
        self,
        monkeypatch,
    ) -> None:
        # Vanilla path still works — adding the acpx branch must not
        # have shadowed the default dispatch.
        from agentscope.model import AnthropicChatModel
        from qwenpaw.providers.claude_acpx_model import (
            ClaudeAcpxChatModel,
        )

        provider = AnthropicProvider(
            id="anthropic",
            name="Anthropic",
            base_url="https://api.anthropic.com",
            api_key="sk-ant-real-token",
            chat_model="AnthropicChatModel",
        )
        model = provider.get_chat_model_instance("claude-sonnet-4-5")
        assert isinstance(model, AnthropicChatModel)
        # And not an acpx wrapper masquerading via the OpenAIChatModel
        # MRO — defense in depth against a constructor-arg fall-through
        # that would silently route Anthropic-direct traffic to acpx.
        assert not isinstance(model, ClaudeAcpxChatModel)


async def test_update_config_updates_only_non_none_values() -> None:
    provider = _make_provider(is_custom=True)

    provider.update_config(
        {
            "name": "Anthropic Custom",
            "base_url": "https://new.example",
            "api_key": "sk-ant-new",
            "chat_model": "AnthropicChatModel",
            "api_key_prefix": "sk-ant-",
        },
    )

    assert provider.name == "Anthropic Custom"
    assert provider.base_url == "https://new.example"
    assert provider.api_key == "sk-ant-new"
    assert provider.chat_model == "AnthropicChatModel"
    assert provider.api_key_prefix == "sk-ant-"

    provider_info = await provider.get_info()

    assert provider_info.name == "Anthropic Custom"
    assert provider_info.base_url == "https://new.example"
    assert provider_info.api_key == "sk-ant-******"
    assert provider_info.chat_model == "AnthropicChatModel"
    assert provider_info.api_key_prefix == "sk-ant-"
    assert provider_info.is_custom
    assert not provider_info.support_connection_check
