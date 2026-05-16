# -*- coding: utf-8 -*-
"""Tests for the grok-image plugin's URL → local-path persistence.

The xAI image-generation surface returns either ``b64_json`` (already
saved by ``_save_b64``) or a short-lived ``https://imgen.x.ai/...``
URL.  Chat channels that attach the image — Signal especially — need a
real local file, not a URL.  ``_download_url`` is the fix; this file
locks in its behaviour.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import AsyncIterator, List
from unittest.mock import patch

import pytest


_PLUGIN_PATH = (
    Path(__file__).resolve().parents[3].parent
    / "plugins"
    / "tool"
    / "grok-image"
    / "grok_image_tool.py"
)


def _load_grok_image_tool():
    """Load grok_image_tool.py directly — it lives under
    ``plugins/tool/grok-image/`` which isn't a normal Python package
    on sys.path."""
    spec = importlib.util.spec_from_file_location(
        "grok_image_tool",
        str(_PLUGIN_PATH),
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["grok_image_tool"] = mod
    spec.loader.exec_module(mod)
    return mod


# ───────────────────────────── fake httpx stream ─────────────────────


class _FakeStreamResponse:
    def __init__(self, chunks: List[bytes]) -> None:
        self._chunks = chunks

    def raise_for_status(self) -> None:  # noqa: D401
        return None

    async def aiter_bytes(self) -> AsyncIterator[bytes]:
        for chunk in self._chunks:
            yield chunk


class _FakeStreamCM:
    def __init__(self, chunks: List[bytes]) -> None:
        self._chunks = chunks

    async def __aenter__(self) -> _FakeStreamResponse:
        return _FakeStreamResponse(self._chunks)

    async def __aexit__(self, *_exc) -> None:
        return None


class _FakeAsyncClient:
    """Stand-in for ``httpx.AsyncClient(timeout=...)`` covering the
    one method ``_download_url`` calls: ``stream("GET", url)`` as an
    async context manager."""

    def __init__(self, payload: bytes) -> None:
        self._payload = payload
        self.urls_seen: List[str] = []

    async def __aenter__(self) -> "_FakeAsyncClient":
        return self

    async def __aexit__(self, *_exc) -> None:
        return None

    def stream(self, method: str, url: str) -> _FakeStreamCM:
        assert method == "GET"
        self.urls_seen.append(url)
        return _FakeStreamCM([self._payload])


# ───────────────────────────── tests ─────────────────────────────────


@pytest.mark.asyncio
async def test_download_url_writes_file_with_sniffed_jpeg_extension(
    tmp_path,
) -> None:
    """xAI URLs usually end in .jpeg — the helper must respect that
    so chat channels (which sniff by extension) classify the file
    as an image rather than a generic ``.jpg`` / opaque blob."""
    mod = _load_grok_image_tool()
    fake_client = _FakeAsyncClient(b"\xff\xd8\xff" + b"x" * 64)

    with patch.object(mod, "DEFAULT_MEDIA_DIR", tmp_path), patch.object(
        mod.httpx,
        "AsyncClient",
        lambda *a, **kw: fake_client,
    ):
        path_str = await mod._download_url(
            "https://imgen.x.ai/xai-imgen/xai-tmp-imgen-abc.jpeg",
        )

    path = Path(path_str)
    assert path.exists()
    assert path.suffix == ".jpeg"
    assert path.parent == tmp_path / "grok_image"
    # File contains the streamed payload, not the URL string.
    assert path.read_bytes().startswith(b"\xff\xd8\xff")


@pytest.mark.asyncio
async def test_download_url_defaults_to_jpg_when_url_has_no_known_ext(
    tmp_path,
) -> None:
    """xAI sometimes returns presigned URLs without an obvious image
    extension.  We default to ``.jpg`` rather than blocking the
    download — chat channels sniff by content-type / file header,
    not the bare extension."""
    mod = _load_grok_image_tool()
    fake_client = _FakeAsyncClient(b"\x89PNG\r\n\x1a\n" + b"x" * 16)

    with patch.object(mod, "DEFAULT_MEDIA_DIR", tmp_path), patch.object(
        mod.httpx,
        "AsyncClient",
        lambda *a, **kw: fake_client,
    ):
        path_str = await mod._download_url(
            "https://imgen.x.ai/asset/no-extension-here",
        )

    path = Path(path_str)
    assert path.exists()
    assert path.suffix == ".jpg"
    assert path.parent == tmp_path / "grok_image"


@pytest.mark.asyncio
async def test_download_url_handles_png_extension(tmp_path) -> None:
    mod = _load_grok_image_tool()
    fake_client = _FakeAsyncClient(b"\x89PNG\r\n\x1a\n" + b"x" * 32)

    with patch.object(mod, "DEFAULT_MEDIA_DIR", tmp_path), patch.object(
        mod.httpx,
        "AsyncClient",
        lambda *a, **kw: fake_client,
    ):
        path_str = await mod._download_url(
            "https://imgen.x.ai/asset/preview.png",
        )

    assert Path(path_str).suffix == ".png"
