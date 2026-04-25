# -*- coding: utf-8 -*-
"""Unit tests for the Codex OAuth wrapper around OpenAIChatModel.

Covers the wrapper's init-time invariants (api_key sentinel, create
monkey-patch) and the ``_CodexOAuthAsyncStream`` adapter's lifecycle
(lazy open on first iteration, ChatCompletionChunk translation,
cleanup on exhaustion, upstream HTTP-error surface).

The live OAuth round-trip is covered by the daemon's own smoke tests;
here we keep httpx stubbed so CI never hits chatgpt.com.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx
import pytest
from openai.types.chat import ChatCompletion, ChatCompletionChunk

from qwenpaw.providers.codex_oauth_model import (
    CodexOAuthChatModel,
    _CodexOAuthAsyncStream,
    _extract_unfetchable_url,
    _raise_for_upstream_status,
    _strip_unfetchable_image_from_body,
)
from qwenpaw.providers.codex_translate import StreamState


# ---------------------------------------------------------------- #
# Fakes                                                            #
# ---------------------------------------------------------------- #


class _FakeAuth:
    """Stand-in for ``CodexAuth`` — exposes just the surface
    ``CodexOAuthChatModel`` touches: ``ensure_fresh``,
    ``auth_headers``, ``base_url``.
    """

    def __init__(self, base_url: str = "https://chatgpt.com/backend-api") -> None:
        self.base_url = base_url
        self.ensure_fresh_calls = 0
        self.auth_headers_calls = 0

    async def ensure_fresh(self) -> None:
        self.ensure_fresh_calls += 1

    async def auth_headers(self) -> dict[str, str]:
        self.auth_headers_calls += 1
        return {"Authorization": "Bearer fake-token"}


class _FakeResponse:
    """Stand-in for :class:`httpx.Response` — the bits
    ``_raise_for_upstream_status`` and ``_CodexOAuthAsyncStream`` read.
    """

    def __init__(self, status_code: int, body: bytes = b"") -> None:
        self.status_code = status_code
        self._body = body
        self.request = httpx.Request("POST", "https://example/fake")

    async def aread(self) -> bytes:
        return self._body


# ---------------------------------------------------------------- #
# CodexOAuthChatModel.__init__                                     #
# ---------------------------------------------------------------- #


class TestCodexOAuthChatModelInit:
    def test_installs_create_wrapper(self) -> None:
        # Agentscope's ChatCompletion parser calls ``client.chat.
        # completions.create`` — if our wrapper isn't attached, a
        # real network call would leak to api.openai.com.
        auth = _FakeAuth()
        model = CodexOAuthChatModel(auth=auth, model_name="gpt-5.4")
        assert callable(model.client.chat.completions.create)

    def test_seeds_api_key_sentinel_when_none(self) -> None:
        # OpenAI SDK refuses construction without ``api_key`` — seed
        # with a sentinel so no caller can accidentally leak it in a
        # header (our wrapper redirects away from the default URL
        # anyway).
        auth = _FakeAuth()
        model = CodexOAuthChatModel(
            auth=auth, model_name="gpt-5.4", api_key=None,
        )
        # The sentinel is internal; what matters is construction
        # succeeded and the wrapper is installed.
        assert callable(model.client.chat.completions.create)

    def test_seeds_api_key_sentinel_when_empty_string(self) -> None:
        auth = _FakeAuth()
        model = CodexOAuthChatModel(
            auth=auth, model_name="gpt-5.4", api_key="",
        )
        assert callable(model.client.chat.completions.create)

    def test_keeps_explicit_api_key_if_given(self) -> None:
        # Nothing in the wrapper should overwrite a caller-supplied
        # key.  We can't easily inspect the SDK client's key field
        # across versions, so we just confirm construction doesn't
        # error.
        auth = _FakeAuth()
        model = CodexOAuthChatModel(
            auth=auth, model_name="gpt-5.4", api_key="sk-explicit",
        )
        assert callable(model.client.chat.completions.create)

    def test_stores_auth_reference(self) -> None:
        # The wrapper must hold the auth object so each call can
        # refresh its token.
        auth = _FakeAuth()
        model = CodexOAuthChatModel(auth=auth, model_name="gpt-5.4")
        assert model._auth is auth


# ---------------------------------------------------------------- #
# Wrapped create — invariants per call                             #
# ---------------------------------------------------------------- #


class TestWrappedCreate:
    """Exercises the monkey-patched
    ``client.chat.completions.create`` through both the streaming and
    non-streaming paths."""

    def test_streaming_returns_async_stream_and_refreshes_token(
        self,
    ) -> None:
        auth = _FakeAuth()
        model = CodexOAuthChatModel(auth=auth, model_name="gpt-5.4")

        async def run() -> Any:
            return await model.client.chat.completions.create(
                messages=[{"role": "user", "content": "hi"}],
                model="gpt-5.4",
                stream=True,
            )

        result = asyncio.run(run())
        assert isinstance(result, _CodexOAuthAsyncStream)
        # Wrapper hit the auth on the way in — streaming doesn't
        # touch the network until the first iteration, but the
        # header / URL prep already ran.
        assert auth.ensure_fresh_calls == 1
        assert auth.auth_headers_calls == 1

    def test_streaming_pops_stream_options_before_upstream(self) -> None:
        # agentscope sends ``stream_options={"include_usage": True}``.
        # The ChatGPT Responses backend rejects unknown fields — the
        # wrapper must strip it from the body.
        auth = _FakeAuth()
        model = CodexOAuthChatModel(auth=auth, model_name="gpt-5.4")

        async def run() -> Any:
            return await model.client.chat.completions.create(
                messages=[{"role": "user", "content": "hi"}],
                model="gpt-5.4",
                stream=True,
                stream_options={"include_usage": True},
            )

        result = asyncio.run(run())
        # We can't see the outbound body until the stream opens, but
        # we can see the captured body on the adapter.
        assert "stream_options" not in result._upstream_body

    def test_non_streaming_drains_upstream(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Pin httpx.AsyncClient → fake that yields a single
        # ``response.completed`` event with final content.
        auth = _FakeAuth()
        model = CodexOAuthChatModel(auth=auth, model_name="gpt-5.4")

        sse_body = (
            b"event: response.output_text.delta\n"
            b'data: {"delta":"Hello"}\n\n'
            b"event: response.completed\n"
            b'data: {"response":{"id":"resp_1","model":"gpt-5.4","'
            b'usage":{"input_tokens":5,"output_tokens":1,"total_tokens":6}}}\n\n'
        )
        _install_fake_httpx(monkeypatch, sse_body=sse_body)

        async def run() -> Any:
            return await model.client.chat.completions.create(
                messages=[{"role": "user", "content": "hi"}],
                model="gpt-5.4",
                stream=False,
            )

        result = asyncio.run(run())
        assert isinstance(result, ChatCompletion)


# ---------------------------------------------------------------- #
# _CodexOAuthAsyncStream adapter                                   #
# ---------------------------------------------------------------- #


class TestCodexOAuthAsyncStream:
    def test_yields_chat_completion_chunks(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        sse_body = (
            b"event: response.output_text.delta\n"
            b'data: {"delta":"Hel"}\n\n'
            b"event: response.output_text.delta\n"
            b'data: {"delta":"lo"}\n\n'
            b"event: response.completed\n"
            b'data: {"response":{"id":"resp_1","model":"gpt-5.4",'
            b'"usage":{"input_tokens":5,"output_tokens":2,"total_tokens":7}}}\n\n'
        )
        _install_fake_httpx(monkeypatch, sse_body=sse_body)

        adapter = _CodexOAuthAsyncStream(
            upstream_url="https://example/fake",
            upstream_body={"model": "gpt-5.4"},
            headers={"Authorization": "Bearer x"},
            state=StreamState(model="gpt-5.4"),
        )

        async def drain() -> list[ChatCompletionChunk]:
            out: list[ChatCompletionChunk] = []
            async for ch in adapter:
                out.append(ch)
            return out

        chunks = asyncio.run(drain())
        assert len(chunks) >= 1
        for ch in chunks:
            assert isinstance(ch, ChatCompletionChunk)
        # Cleanup ran on exhaustion.
        assert adapter._client is None
        assert adapter._iter is None

    def test_http_error_on_first_iteration_surfaces(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Upstream returns 401 → adapter must raise a clear
        # HTTPStatusError (callers up the stack rely on this to
        # distinguish auth problems from transport problems).
        _install_fake_httpx(
            monkeypatch, status_code=401, body=b'{"error":"expired"}',
        )

        adapter = _CodexOAuthAsyncStream(
            upstream_url="https://example/fake",
            upstream_body={"model": "gpt-5.4"},
            headers={},
            state=StreamState(model="gpt-5.4"),
        )

        async def drain() -> None:
            async for _ in adapter:
                pass

        with pytest.raises(httpx.HTTPStatusError) as exc_info:
            asyncio.run(drain())
        assert "401" in str(exc_info.value)
        # Cleanup ran even on the error path.
        assert adapter._client is None

    def test_aenter_aexit_contract(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Some callers wrap the adapter in ``async with`` — make sure
        # we don't blow up even if they never iterate.
        _install_fake_httpx(monkeypatch, sse_body=b"")

        async def run() -> None:
            adapter = _CodexOAuthAsyncStream(
                upstream_url="https://example/fake",
                upstream_body={"model": "gpt-5.4"},
                headers={},
                state=StreamState(model="gpt-5.4"),
            )
            async with adapter:
                pass

        asyncio.run(run())


# ---------------------------------------------------------------- #
# _raise_for_upstream_status                                       #
# ---------------------------------------------------------------- #


class TestRaiseForUpstreamStatus:
    def test_200_noop(self) -> None:
        resp = _FakeResponse(200)
        # Must not raise.
        _raise_for_upstream_status(resp)  # type: ignore[arg-type]

    def test_non_200_raises_http_status_error(self) -> None:
        resp = _FakeResponse(500)
        with pytest.raises(httpx.HTTPStatusError):
            _raise_for_upstream_status(resp)  # type: ignore[arg-type]

    @pytest.mark.parametrize("status", [400, 401, 403, 404, 429, 500, 502])
    def test_exception_includes_status_code(self, status: int) -> None:
        resp = _FakeResponse(status)
        with pytest.raises(httpx.HTTPStatusError) as exc_info:
            _raise_for_upstream_status(resp)  # type: ignore[arg-type]
        assert str(status) in str(exc_info.value)


# ---------------------------------------------------------------- #
# Strip-and-retry helpers (image URL fetch failures)               #
# ---------------------------------------------------------------- #


class TestExtractUnfetchableUrl:
    """The error-message parser ChatGPT returns when it can't fetch
    an image we sent.  Cover the standard sentence shape, the
    JSON-wrapped form (most common in production), and the
    no-match cases that must not yield a URL."""

    def test_extracts_url_from_plain_sentence(self) -> None:
        body = (
            "Error while downloading "
            "https://media.example/m?t=abc&exp=1. Upstream "
            "status code: 403."
        )
        assert (
            _extract_unfetchable_url(body)
            == "https://media.example/m?t=abc&exp=1"
        )

    def test_extracts_url_from_json_wrapped_error(self) -> None:
        body = (
            '{"error":{"message":"Error while downloading '
            'https://media.example/m?sig=xyz. Upstream status '
            'code: 403.","type":"invalid_request_error"}}'
        )
        url = _extract_unfetchable_url(body)
        assert url == "https://media.example/m?sig=xyz"

    def test_returns_none_for_non_download_error(self) -> None:
        assert _extract_unfetchable_url("rate limit exceeded") is None
        assert _extract_unfetchable_url("invalid api key") is None
        assert _extract_unfetchable_url("") is None

    def test_returns_none_when_marker_present_but_no_url(self) -> None:
        # Pathological: marker without a URL — fall back to None
        # rather than match something nonsensical.
        body = "Error while downloading something. Upstream"
        assert _extract_unfetchable_url(body) is None


class TestStripUnfetchableImageFromBody:
    def test_strips_matching_image_url_and_returns_true(self) -> None:
        body = {
            "input": [{
                "type": "message",
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "describe"},
                    {"type": "input_image",
                     "image_url": "https://x.example/a"},
                    {"type": "input_image",
                     "image_url": "https://x.example/b"},
                ],
            }],
        }
        stripped = _strip_unfetchable_image_from_body(
            body, "https://x.example/a",
        )
        assert stripped is True
        content = body["input"][0]["content"]
        # The matching image was replaced by a text placeholder;
        # the other image and the user text survive untouched.
        types = [c.get("type") for c in content]
        assert types == ["input_text", "input_text", "input_image"]
        assert content[1]["text"].startswith("[image previously sent")
        assert content[2]["image_url"] == "https://x.example/b"

    def test_no_match_returns_false_and_leaves_body_alone(self) -> None:
        body = {
            "input": [{
                "type": "message",
                "role": "user",
                "content": [
                    {"type": "input_image",
                     "image_url": "https://x.example/a"},
                ],
            }],
        }
        snapshot = json.loads(json.dumps(body))
        stripped = _strip_unfetchable_image_from_body(
            body, "https://different.example/z",
        )
        assert stripped is False
        assert body == snapshot

    def test_strips_across_multiple_input_entries(self) -> None:
        body = {
            "input": [
                {"type": "message", "role": "user", "content": [
                    {"type": "input_image", "image_url": "https://bad"},
                ]},
                {"type": "message", "role": "user", "content": [
                    {"type": "input_image", "image_url": "https://bad"},
                ]},
            ],
        }
        assert _strip_unfetchable_image_from_body(body, "https://bad")
        for entry in body["input"]:
            assert entry["content"][0]["type"] == "input_text"


# ---------------------------------------------------------------- #
# Reasoning-effort end-to-end                                      #
# ---------------------------------------------------------------- #


class TestReasoningEffortEndToEnd:
    """Proves the UI's ``generate_kwargs.reasoning_effort`` setting
    actually lands in the outbound Responses API body as
    ``reasoning.effort`` — the whole point of the Settings > Codex >
    Effort picker.  Failure modes this catches:

    * Effort silently dropped between agentscope kwargs and
      ``build_responses_body``.
    * Effort value mapped to a different shape (e.g. top-level
      ``effort`` instead of nested ``reasoning``) that the ChatGPT
      backend ignores.
    * A default overriding the user's explicit choice.

    Uses ``_CodexOAuthAsyncStream`` directly — that adapter is what
    carries the body to ``httpx.stream`` on first iteration, so
    inspecting the fake's captured payload proves what the backend
    would see.
    """

    @pytest.mark.parametrize(
        "effort", ["none", "low", "medium", "high", "xhigh"],
    )
    def test_effort_reaches_upstream_body(
        self,
        monkeypatch: pytest.MonkeyPatch,
        effort: str,
    ) -> None:
        captured: dict[str, Any] = {}

        class _Stream:
            def __init__(self) -> None:
                self.status_code = 200
                self.request = httpx.Request("POST", "https://example/fake")
            async def __aenter__(self) -> "_Stream":
                return self
            async def __aexit__(self, *_exc: Any) -> None:
                return None
            async def aiter_lines(self):
                if False:
                    yield ""
            async def aread(self) -> bytes:
                return b""

        class _FakeClient:
            def __init__(self, *_a: Any, **_kw: Any) -> None:
                pass
            async def __aenter__(self) -> "_FakeClient":
                return self
            async def __aexit__(self, *_exc: Any) -> None:
                return None
            def stream(
                self, method: str, url: str, *, json: dict, headers: dict,
            ) -> _Stream:
                captured["method"] = method
                captured["url"] = url
                captured["json"] = json
                captured["headers"] = headers
                return _Stream()
            async def aclose(self) -> None:
                return None

        monkeypatch.setattr(
            "qwenpaw.providers.codex_oauth_model.httpx.AsyncClient",
            _FakeClient,
        )

        auth = _FakeAuth()
        model = CodexOAuthChatModel(auth=auth, model_name="gpt-5.4")

        async def run() -> None:
            stream = await model.client.chat.completions.create(
                messages=[{"role": "user", "content": "hi"}],
                model="gpt-5.4",
                stream=True,
                reasoning_effort=effort,
            )
            # First __anext__ opens the HTTP connection — which is
            # when _CodexOAuthAsyncStream hands the body to httpx.
            try:
                async for _ in stream:
                    break
            except StopAsyncIteration:
                pass

        asyncio.run(run())

        # The UI-set effort must land here, exactly, as part of the
        # nested ``reasoning`` object.  Any other shape means the
        # ChatGPT backend will ignore the setting silently.
        body = captured.get("json")
        assert body is not None, "upstream never received a body"
        assert body.get("reasoning") == {"effort": effort}
        # Sanity: caller's stream_options never leaks (proves the
        # wrapper's strip still runs alongside the passthrough).
        assert "stream_options" not in body

    def test_effort_omitted_defaults_to_low(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # When the user leaves the picker empty (no
        # ``reasoning_effort`` in generate_kwargs), the wrapper must
        # default to ``"low"`` — matches the UI hint text.
        captured: dict[str, Any] = {}

        class _Stream:
            def __init__(self) -> None:
                self.status_code = 200
                self.request = httpx.Request("POST", "https://example/fake")
            async def __aenter__(self) -> "_Stream":
                return self
            async def __aexit__(self, *_exc: Any) -> None:
                return None
            async def aiter_lines(self):
                if False:
                    yield ""
            async def aread(self) -> bytes:
                return b""

        class _FakeClient:
            def __init__(self, *_a: Any, **_kw: Any) -> None:
                pass
            async def __aenter__(self) -> "_FakeClient":
                return self
            async def __aexit__(self, *_exc: Any) -> None:
                return None
            def stream(
                self, method: str, url: str, *, json: dict, headers: dict,
            ) -> _Stream:
                captured["json"] = json
                return _Stream()
            async def aclose(self) -> None:
                return None

        monkeypatch.setattr(
            "qwenpaw.providers.codex_oauth_model.httpx.AsyncClient",
            _FakeClient,
        )

        model = CodexOAuthChatModel(auth=_FakeAuth(), model_name="gpt-5.4")

        async def run() -> None:
            stream = await model.client.chat.completions.create(
                messages=[{"role": "user", "content": "hi"}],
                model="gpt-5.4",
                stream=True,
            )
            try:
                async for _ in stream:
                    break
            except StopAsyncIteration:
                pass

        asyncio.run(run())

        assert captured["json"].get("reasoning") == {"effort": "low"}

    def test_subsequent_call_picks_up_new_effort(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Regression guard for a cache-the-kwargs bug — changing the
        # effort in the UI must affect the NEXT call, not just
        # brand-new model instances.
        captured_bodies: list[dict] = []

        class _Stream:
            def __init__(self) -> None:
                self.status_code = 200
                self.request = httpx.Request("POST", "https://example/fake")
            async def __aenter__(self) -> "_Stream":
                return self
            async def __aexit__(self, *_exc: Any) -> None:
                return None
            async def aiter_lines(self):
                if False:
                    yield ""
            async def aread(self) -> bytes:
                return b""

        class _FakeClient:
            def __init__(self, *_a: Any, **_kw: Any) -> None:
                pass
            async def __aenter__(self) -> "_FakeClient":
                return self
            async def __aexit__(self, *_exc: Any) -> None:
                return None
            def stream(
                self, method: str, url: str, *, json: dict, headers: dict,
            ) -> _Stream:
                captured_bodies.append(json)
                return _Stream()
            async def aclose(self) -> None:
                return None

        monkeypatch.setattr(
            "qwenpaw.providers.codex_oauth_model.httpx.AsyncClient",
            _FakeClient,
        )

        model = CodexOAuthChatModel(auth=_FakeAuth(), model_name="gpt-5.4")

        async def one_call(effort: str) -> None:
            stream = await model.client.chat.completions.create(
                messages=[{"role": "user", "content": "hi"}],
                model="gpt-5.4",
                stream=True,
                reasoning_effort=effort,
            )
            try:
                async for _ in stream:
                    break
            except StopAsyncIteration:
                pass

        async def scenario() -> None:
            await one_call("low")
            await one_call("high")
            await one_call("xhigh")

        asyncio.run(scenario())

        efforts = [b["reasoning"]["effort"] for b in captured_bodies]
        assert efforts == ["low", "high", "xhigh"]


# ---------------------------------------------------------------- #
# httpx stubs                                                      #
# ---------------------------------------------------------------- #


# ---------------------------------------------------------------- #
# Dynamic model discovery via CodexAuth.list_models                #
# ---------------------------------------------------------------- #


class TestClientVersion:
    """Regression guard: the ``version`` / ``client_version`` the
    Codex backend sees decides which models it unlocks
    (e.g. gpt-5.5 requires ≥ 0.200.0).  We hardcode a floor because
    bumping it is a deliberate decision — silently falling back to
    an older string would re-lock newer models."""

    def test_default_version_is_at_least_0_200_0(self) -> None:
        from qwenpaw.providers.codex_auth import CODEX_CLIENT_VERSION

        parts = CODEX_CLIENT_VERSION.split(".")
        major, minor = int(parts[0]), int(parts[1])
        # 0.200.0 is the version that unlocks gpt-5.5 on
        # ChatGPT-account OAuth (probed 2026-04-24).  Anything older
        # silently filters gpt-5.5 out of /codex/models *and* makes
        # /codex/responses return model_not_found for it.
        assert (major, minor) >= (0, 200), (
            f"CODEX_CLIENT_VERSION={CODEX_CLIENT_VERSION} is too old — "
            "gpt-5.5 requires the 'version' header >= 0.200.0."
        )

    def test_env_override_takes_precedence(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Ops knob: flipping QWENPAW_CODEX_CLIENT_VERSION must
        # propagate without a code change.
        monkeypatch.setenv("QWENPAW_CODEX_CLIENT_VERSION", "9.9.9")
        # Re-import with the new env visible.
        import importlib
        import qwenpaw.providers.codex_auth as ca
        importlib.reload(ca)
        try:
            assert ca.CODEX_CLIENT_VERSION == "9.9.9"
        finally:
            monkeypatch.delenv("QWENPAW_CODEX_CLIENT_VERSION")
            importlib.reload(ca)  # restore default for other tests


class TestListModels:
    """Ensures ``CodexAuth.list_models`` hits the catalog endpoint,
    sends the right params, and returns the ``models`` array."""

    def test_hits_codex_models_endpoint_with_client_version(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from qwenpaw.providers.codex_auth import CodexAuth

        captured: dict[str, Any] = {}

        class _Resp:
            status_code = 200
            def json(self) -> dict:
                return {
                    "models": [
                        {"slug": "gpt-5.2", "visibility": "list"},
                        {"slug": "gpt-oss-120b", "visibility": "hide"},
                    ],
                }
            def raise_for_status(self) -> None:
                return None

        class _FakeClient:
            def __init__(self, *_a: Any, **_kw: Any) -> None:
                pass
            async def __aenter__(self) -> "_FakeClient":
                return self
            async def __aexit__(self, *_exc: Any) -> None:
                return None
            async def get(
                self, url: str, headers: dict, params: dict,
            ) -> _Resp:
                captured["url"] = url
                captured["params"] = params
                captured["headers"] = headers
                return _Resp()

        monkeypatch.setattr(
            "qwenpaw.providers.codex_auth.httpx.AsyncClient", _FakeClient,
        )

        auth = CodexAuth.__new__(CodexAuth)  # skip __init__
        auth._creds = None
        async def _fake_headers() -> dict:
            return {"Authorization": "Bearer x"}
        auth.auth_headers = _fake_headers  # type: ignore[attr-defined]
        type(auth).base_url = property(lambda _self: "https://fake/b")

        async def run() -> list[dict]:
            return await auth.list_models(client_version="0.99.0")

        models = asyncio.run(run())
        assert isinstance(models, list)
        assert len(models) == 2
        assert captured["url"].endswith("/codex/models")
        assert captured["params"] == {"client_version": "0.99.0"}
        assert captured["headers"]["Authorization"] == "Bearer x"


class TestFetchModelsCodexOAuth:
    """Tests the codex-oauth branch of ``OpenAIProvider.fetch_models``
    — filters out ``visibility=hide`` entries, preserves slugs, and
    falls back to an empty list on errors."""

    def test_filters_hidden_models_and_keeps_listed(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from qwenpaw.providers.openai_provider import OpenAIProvider

        provider = OpenAIProvider(
            id="codex-oauth",
            name="ChatGPT Codex (OAuth)",
            base_url="https://chatgpt.com/backend-api",
            api_key="oauth",
            api_key_prefix="",
            require_api_key=False,
            chat_model="OpenAIChatModel",
            models=[],
        )

        class _FakeAuthForDiscovery:
            async def list_models(
                self, *, client_version: str = "x", timeout: float = 10,
            ) -> list[dict]:
                return [
                    {
                        "slug": "gpt-5.2",
                        "display_name": "GPT-5.2",
                        "visibility": "list",
                    },
                    {
                        "slug": "gpt-oss-120b",
                        "display_name": "OSS 120B",
                        "visibility": "hide",
                    },
                    {
                        "slug": "",  # malformed — drop
                        "visibility": "list",
                    },
                    # unrelated junk — must not break the parser
                    "a bare string",
                ]

        monkeypatch.setattr(
            provider, "_get_codex_oauth",
            lambda: _FakeAuthForDiscovery(),
        )

        models = asyncio.run(provider.fetch_models())
        slugs = [m.id for m in models]
        assert slugs == ["gpt-5.2"]
        assert models[0].name == "GPT-5.2"
        assert models[0].probe_source == "codex-oauth-catalog"

    def test_discovery_failure_returns_empty_not_raise(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Discovery is best-effort — the UI must keep the user's
        # previously-saved models when the network is down, not blank
        # the list on a transient 502.
        from qwenpaw.providers.openai_provider import OpenAIProvider

        provider = OpenAIProvider(
            id="codex-oauth",
            name="ChatGPT Codex (OAuth)",
            base_url="https://chatgpt.com/backend-api",
            api_key="oauth",
            api_key_prefix="",
            require_api_key=False,
            chat_model="OpenAIChatModel",
            models=[],
        )

        class _BrokenAuth:
            async def list_models(
                self, *, client_version: str = "x", timeout: float = 10,
            ) -> list[dict]:
                raise httpx.ConnectError("backend down")

        monkeypatch.setattr(
            provider, "_get_codex_oauth", lambda: _BrokenAuth(),
        )

        models = asyncio.run(provider.fetch_models())
        assert models == []


def _install_fake_httpx(
    monkeypatch: pytest.MonkeyPatch,
    *,
    status_code: int = 200,
    sse_body: bytes = b"",
    body: bytes | None = None,
) -> None:
    """Swap :class:`httpx.AsyncClient` for a fake that yields the
    caller-pinned SSE stream on ``stream("POST", ...)``.

    The adapter only needs three things from httpx:
    - open a stream context
    - iterate ``aiter_lines`` / ``aread``
    - close the client

    We implement just those plus a compliant ``request`` attribute
    on the response so the HTTPStatusError surface stays truthful.
    """
    response_body = body if body is not None else sse_body

    class _Stream:
        def __init__(self, status: int, payload: bytes) -> None:
            self._status = status
            self._payload = payload
            self.status_code = status
            self.request = httpx.Request("POST", "https://example/fake")

        async def __aenter__(self) -> "_Stream":
            return self

        async def __aexit__(self, *_exc: Any) -> None:
            return None

        async def aiter_lines(self):
            for line in self._payload.decode(
                "utf-8", errors="replace",
            ).splitlines():
                yield line

        async def aread(self) -> bytes:
            return self._payload

    class _FakeClient:
        def __init__(self, *_a: Any, **_kw: Any) -> None:
            self.closed = False

        async def __aenter__(self) -> "_FakeClient":
            return self

        async def __aexit__(self, *_exc: Any) -> None:
            self.closed = True

        def stream(self, *_a: Any, **_kw: Any) -> _Stream:
            return _Stream(status_code, response_body)

        async def aclose(self) -> None:
            self.closed = True

    monkeypatch.setattr(
        "qwenpaw.providers.codex_oauth_model.httpx.AsyncClient",
        _FakeClient,
    )
