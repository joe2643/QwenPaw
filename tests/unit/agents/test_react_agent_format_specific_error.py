# -*- coding: utf-8 -*-
"""Tests for ``_is_format_specific_media_error`` guard.

Background:
    The agent's ``_reasoning_with_media_fallback`` / ``_summarizing``
    paths used to learn ``rejects_media=True`` in the capability cache
    every time a strip-and-retry succeeded.  That's wrong when the
    original error was *file-format* specific (e.g. animated WebP
    rejected with z.ai code ``1210 图片输入格式/解析错误``): the model
    can handle media — it just couldn't decode that *particular* file.

    Learning anyway poisons the process-scoped cache and turns every
    subsequent view_image call into a silent strip, even for PNG/JPG.
    Confirmed on 2026-05-12 from production copaw.log:
    a WebP sticker rejection at 13:01:10 made every later view_image
    skip the vision model.

    This test locks the new guard's pattern table so future edits
    don't silently drop a marker the cache regression depends on.
"""

from __future__ import annotations

import pytest

from qwenpaw.agents.react_agent import QwenPawAgent


# ---------------------------------------------------------------------------
# Format-specific markers — guard should fire (skip learning)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "error_text",
    [
        # z.ai / Zhipu — exact message from production
        "Error code: 400 - {'error': {'code': '1210', 'message': '图片输入格式/解析错误'}}",
        "{'code': '1210'}",
        "图片输入格式",
        "image format error",
        "image input format invalid",
        "decode error: cannot parse image",
        "parse error in webp container",
        # Mimo video transcode-recoverable rejections
        "Multimodal data is corrupted or cannot be processed.",
        "only mp4/wmv/mov/avi are supported",
        "invalid video format",
        "unsupported video format: av1",
        "unsupported image format: animated webp",
        # Generic codec rejection
        "codec libwebp animation not supported",
        "container webm not allowed",
    ],
)
def test_marker_match_skips_learning(error_text: str) -> None:
    """Every entry in the table must trigger the guard so the cache
    doesn't learn ``rejects_media=True`` from a transcode-able error.
    """
    exc = RuntimeError(error_text)
    assert QwenPawAgent._is_format_specific_media_error(exc) is True, (
        f"Expected guard to fire on {error_text!r} but it didn't — "
        "format-specific markers must stay in the guard's table."
    )


# ---------------------------------------------------------------------------
# True capability errors — guard should NOT fire (learn rejects_media)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "error_text",
    [
        # Generic capability rejection — no format markers
        "model does not support multimodal input",
        "vision input not enabled for this account",
        "Error code: 400 - {'error': {'message': 'multimodal not supported'}}",
        # Plain 400 with no media-format hint
        "Bad request",
        # Quota / auth
        "rate limited",
        "401 unauthorized",
    ],
)
def test_non_format_errors_allow_learning(error_text: str) -> None:
    """Errors without file-format markers should NOT be treated as
    format-specific — the cache SHOULD still learn ``rejects_media``
    in those cases (preserving the existing capability inference).
    """
    exc = RuntimeError(error_text)
    assert QwenPawAgent._is_format_specific_media_error(exc) is False, (
        f"Guard fired on {error_text!r} — would prevent learning a "
        "real capability gap."
    )


# ---------------------------------------------------------------------------
# Robustness
# ---------------------------------------------------------------------------


def test_case_insensitive_matching() -> None:
    """Marker matching must casefold — production errors arrive in
    mixed case from various providers."""
    exc = RuntimeError("IMAGE FORMAT IS INVALID")
    assert QwenPawAgent._is_format_specific_media_error(exc) is True


def test_chinese_marker_in_full_error_string() -> None:
    """The Chinese marker '图片输入格式' must match even when wrapped
    in surrounding English/JSON noise (the typical z.ai shape)."""
    exc = RuntimeError(
        "Error code: 400 - {'error': {'code': '1210', "
        "'message': '图片输入格式/解析错误'}}",
    )
    assert QwenPawAgent._is_format_specific_media_error(exc) is True


def test_empty_exception_does_not_match() -> None:
    exc = RuntimeError("")
    assert QwenPawAgent._is_format_specific_media_error(exc) is False
