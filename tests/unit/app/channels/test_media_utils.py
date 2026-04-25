# -*- coding: utf-8 -*-
"""Unit tests for ``resolve_media_url`` / ``sign_media_path``.

The formerly-stub resolver is now the single place signing local
file paths into public URLs via the QwenPaw media server.  Channels
and ``view_video``'s fallback-model path both go through here, so
regressions here break a lot downstream — hence the dedicated suite.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from qwenpaw.app.channels import media_utils


# ---------------------------------------------------------------- #
# Passthrough cases — nothing to sign                              #
# ---------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_http_url_passthrough() -> None:
    # An HTTP URL needs no signing — hand it back unchanged without
    # bothering the media server at all.
    with patch.object(
        media_utils,
        "sign_media_path",
        AsyncMock(return_value="NO"),
    ) as signer:
        result = await media_utils.resolve_media_url(
            "https://example.com/a.mp4",
        )
    assert result == "https://example.com/a.mp4"
    signer.assert_not_awaited()


@pytest.mark.asyncio
async def test_data_url_passthrough() -> None:
    # Inline data URLs are fine as-is.
    inp = "data:video/mp4;base64,AAAA"
    with patch.object(
        media_utils,
        "sign_media_path",
        AsyncMock(return_value="NO"),
    ) as signer:
        result = await media_utils.resolve_media_url(inp)
    assert result == inp
    signer.assert_not_awaited()


@pytest.mark.asyncio
async def test_empty_input_passthrough() -> None:
    assert await media_utils.resolve_media_url("") == ""


# ---------------------------------------------------------------- #
# Local path — sign when possible, fall back to raw path otherwise  #
# ---------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_local_path_returns_signed_url_on_success(
    tmp_path: Path,
) -> None:
    src = tmp_path / "clip.mp4"
    src.write_bytes(b"x" * 16)
    signed = "https://media.example.com/media?sig=abc"
    with patch.object(
        media_utils,
        "sign_media_path",
        AsyncMock(return_value=signed),
    ) as signer:
        result = await media_utils.resolve_media_url(str(src))
    assert result == signed
    signer.assert_awaited_once()


@pytest.mark.asyncio
async def test_local_path_returns_raw_path_when_sign_fails(
    tmp_path: Path,
) -> None:
    # Media server unreachable ⇒ sign returns None ⇒ resolver falls
    # back to the original local path (no regression vs the former
    # stub behaviour).
    src = tmp_path / "clip.mp4"
    src.write_bytes(b"x" * 16)
    with patch.object(
        media_utils,
        "sign_media_path",
        AsyncMock(return_value=None),
    ):
        result = await media_utils.resolve_media_url(str(src))
    assert result == str(src)


@pytest.mark.asyncio
async def test_missing_path_still_returns_input() -> None:
    # Historical behaviour: pre-upload-by-other-means callers pass a
    # path that doesn't exist yet.  Resolver must not block or error.
    result = await media_utils.resolve_media_url("/tmp/does-not-exist.mp4")
    assert result == "/tmp/does-not-exist.mp4"


# ---------------------------------------------------------------- #
# sign_media_path HTTP-level behaviour                             #
# ---------------------------------------------------------------- #


class _FakeResponse:
    def __init__(self, status_code: int, body: dict | None = None) -> None:
        self.status_code = status_code
        self._body = body or {}
        self.text = str(body or "")

    def json(self) -> dict:
        return self._body


class _FakeAsyncClient:
    """Context-manager stand-in that returns whatever ``response`` the
    test pins to it.  Replaces ``httpx.AsyncClient`` for the duration
    of a single call."""

    def __init__(self, response: _FakeResponse) -> None:
        self._response = response
        self.last_call: dict | None = None

    async def __aenter__(self) -> "_FakeAsyncClient":
        return self

    async def __aexit__(self, *_exc) -> None:
        return None

    async def get(self, url: str, params: dict | None = None) -> _FakeResponse:
        self.last_call = {"url": url, "params": params}
        return self._response


@pytest.mark.asyncio
async def test_sign_media_path_happy_path() -> None:
    signed = "https://media.example.com/media?sig=xyz&exp=1"
    client = _FakeAsyncClient(
        _FakeResponse(200, {"url": signed, "expires": 1}),
    )
    with patch.object(
        httpx,
        "AsyncClient",
        lambda *a, **kw: client,
    ):
        result = await media_utils.sign_media_path("/tmp/x.mp4")
    assert result == signed
    assert client.last_call["params"]["path"] == "/tmp/x.mp4"


@pytest.mark.asyncio
async def test_sign_media_path_non_200_returns_none() -> None:
    client = _FakeAsyncClient(_FakeResponse(403, {"error": "denied"}))
    with patch.object(
        httpx,
        "AsyncClient",
        lambda *a, **kw: client,
    ):
        result = await media_utils.sign_media_path("/tmp/x.mp4")
    assert result is None


@pytest.mark.asyncio
async def test_sign_media_path_network_error_returns_none() -> None:
    class _Exploding:
        async def __aenter__(self):
            raise httpx.ConnectError("server down")

        async def __aexit__(self, *_exc):
            return None

    with patch.object(
        httpx,
        "AsyncClient",
        lambda *a, **kw: _Exploding(),
    ):
        result = await media_utils.sign_media_path("/tmp/x.mp4")
    assert result is None


@pytest.mark.asyncio
async def test_sign_media_path_forwards_auth_when_given() -> None:
    client = _FakeAsyncClient(
        _FakeResponse(200, {"url": "https://x/y", "expires": 1}),
    )
    with patch.object(
        httpx,
        "AsyncClient",
        lambda *a, **kw: client,
    ):
        await media_utils.sign_media_path("/tmp/x.mp4", auth="sekret")
    assert client.last_call["params"]["auth"] == "sekret"


@pytest.mark.asyncio
async def test_sign_media_path_empty_path_returns_none() -> None:
    assert await media_utils.sign_media_path("") is None
