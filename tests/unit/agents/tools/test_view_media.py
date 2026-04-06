# -*- coding: utf-8 -*-
"""Tests for view_media — view_image, view_video, signed URLs, keyframe extraction."""

import asyncio
import base64
import json
import os
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from copaw.agents.tools.view_media import (
    _extract_keyframes,
    _local_to_base64_source,
    _resolve_media_source,
    _validate_media_path,
    _IMAGE_EXTENSIONS,
    _VIDEO_EXTENSIONS,
    view_image,
    view_video,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_image(tmp_path):
    """Create a tiny valid PNG file."""
    # Minimal 1x1 PNG
    png_header = (
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
        b"\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00"
        b"\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00"
        b"\x05\x18\xd8N\x00\x00\x00\x00IEND\xaeB`\x82"
    )
    p = tmp_path / "test.png"
    p.write_bytes(png_header)
    return p


@pytest.fixture
def tmp_video(tmp_path):
    """Create a tiny fake MP4 file."""
    p = tmp_path / "test.mp4"
    p.write_bytes(b"\x00\x00\x00\x1cftypisom" + b"\x00" * 100)
    return p


@pytest.fixture
def tmp_text(tmp_path):
    """Create a text file (not media)."""
    p = tmp_path / "readme.txt"
    p.write_text("hello")
    return p


# ---------------------------------------------------------------------------
# _validate_media_path
# ---------------------------------------------------------------------------

class TestValidateMediaPath:

    def test_valid_image(self, tmp_image):
        resolved, err = _validate_media_path(str(tmp_image), _IMAGE_EXTENSIONS, "image")
        assert err is None
        assert resolved == tmp_image.resolve()

    def test_valid_video(self, tmp_video):
        resolved, err = _validate_media_path(str(tmp_video), _VIDEO_EXTENSIONS, "video")
        assert err is None

    def test_nonexistent_file(self):
        _, err = _validate_media_path("/tmp/does_not_exist_12345.png", _IMAGE_EXTENSIONS, "image")
        assert err is not None
        assert "does not exist" in err.content[0]["text"]

    def test_wrong_extension(self, tmp_text):
        _, err = _validate_media_path(str(tmp_text), _IMAGE_EXTENSIONS, "image")
        assert err is not None
        assert "not a supported" in err.content[0]["text"]


# ---------------------------------------------------------------------------
# _local_to_base64_source
# ---------------------------------------------------------------------------

class TestLocalToBase64:

    def test_encodes_correctly(self, tmp_image):
        source = _local_to_base64_source(tmp_image, "image")
        assert source["type"] == "base64"
        assert source["media_type"].startswith("image/")
        # Verify base64 is valid
        decoded = base64.b64decode(source["data"])
        assert decoded == tmp_image.read_bytes()

    def test_video_mime(self, tmp_video):
        source = _local_to_base64_source(tmp_video, "video")
        assert "video" in source["media_type"]


# ---------------------------------------------------------------------------
# _resolve_media_source
# ---------------------------------------------------------------------------

class TestResolveMediaSource:

    @patch("copaw.agents.tools.view_media._get_signed_url", return_value="https://media.example.com/signed")
    def test_prefers_signed_url(self, mock_sign, tmp_image):
        source = _resolve_media_source(tmp_image, "image")
        assert source["type"] == "url"
        assert source["url"] == "https://media.example.com/signed"

    @patch("copaw.agents.tools.view_media._get_signed_url", return_value=None)
    def test_image_fallback_to_base64(self, mock_sign, tmp_image):
        source = _resolve_media_source(tmp_image, "image")
        assert source["type"] == "base64"

    @patch("copaw.agents.tools.view_media._get_signed_url", return_value=None)
    def test_video_no_base64_fallback(self, mock_sign, tmp_video):
        """Videos must NEVER fallback to base64 (blows up context)."""
        source = _resolve_media_source(tmp_video, "video")
        assert source is None


# ---------------------------------------------------------------------------
# _extract_keyframes
# ---------------------------------------------------------------------------

class TestExtractKeyframes:

    def test_extracts_frames_from_real_video(self, tmp_path):
        """Integration test — requires ffmpeg. Generate a tiny test video first."""
        import subprocess
        video = tmp_path / "test_real.mp4"
        # Generate 3-second test video with ffmpeg
        try:
            subprocess.run([
                "ffmpeg", "-f", "lavfi", "-i", "color=c=blue:s=64x64:d=3",
                "-c:v", "libx264", "-pix_fmt", "yuv420p",
                "-y", str(video),
            ], capture_output=True, timeout=10)
        except Exception:
            pytest.skip("ffmpeg not available or failed to generate test video")

        if not video.exists():
            pytest.skip("ffmpeg failed to create test video")

        frames = _extract_keyframes(video, max_frames=3)
        assert len(frames) >= 1
        for frame_path, timestamp in frames:
            assert frame_path.exists()
            assert frame_path.suffix == ".jpg"
            assert "s" in timestamp  # e.g. "0.8s"

    def test_nonexistent_video_returns_empty(self, tmp_path):
        fake = tmp_path / "nonexistent.mp4"
        frames = _extract_keyframes(fake, max_frames=3)
        assert frames == []


# ---------------------------------------------------------------------------
# view_image (async)
# ---------------------------------------------------------------------------

class TestViewImage:

    @pytest.mark.asyncio
    async def test_valid_image_returns_image_block(self, tmp_image):
        with patch("copaw.agents.tools.view_media._get_signed_url", return_value=None):
            result = await view_image(str(tmp_image))
            types = [b.get("type") if isinstance(b, dict) else b["type"] for b in result.content]
            assert "image" in types
            assert "text" in types

    @pytest.mark.asyncio
    async def test_nonexistent_image_returns_error(self):
        result = await view_image("/tmp/no_such_image_99999.png")
        text = result.content[0]["text"]
        assert "does not exist" in text

    @pytest.mark.asyncio
    async def test_signed_url_used_when_available(self, tmp_image):
        with patch("copaw.agents.tools.view_media._get_signed_url", return_value="https://media.example.com/img"):
            result = await view_image(str(tmp_image))
            img_block = [b for b in result.content if (b.get("type") if isinstance(b, dict) else None) == "image"][0]
            assert img_block["source"]["type"] == "url"
            assert img_block["source"]["url"] == "https://media.example.com/img"


# ---------------------------------------------------------------------------
# view_video (async)
# ---------------------------------------------------------------------------

class TestViewVideo:

    @pytest.mark.asyncio
    async def test_signed_url_returns_video_block(self, tmp_video):
        with patch("copaw.agents.tools.view_media._get_signed_url", return_value="https://media.example.com/vid"):
            result = await view_video(str(tmp_video))
            types = [b.get("type") if isinstance(b, dict) else b["type"] for b in result.content]
            assert "video" in types

    @pytest.mark.asyncio
    async def test_no_server_extracts_keyframes(self, tmp_path):
        """Without media server, should extract keyframes instead of base64."""
        import subprocess
        video = tmp_path / "test.mp4"
        try:
            subprocess.run([
                "ffmpeg", "-f", "lavfi", "-i", "color=c=red:s=64x64:d=2",
                "-c:v", "libx264", "-pix_fmt", "yuv420p",
                "-y", str(video),
            ], capture_output=True, timeout=10)
        except Exception:
            pytest.skip("ffmpeg not available")

        if not video.exists():
            pytest.skip("ffmpeg failed")

        with patch("copaw.agents.tools.view_media._get_signed_url", return_value=None):
            result = await view_video(str(video))
            types = [b.get("type") if isinstance(b, dict) else b["type"] for b in result.content]
            # Should have image frames, NOT video block
            assert "video" not in types
            assert "image" in types
            assert "text" in types

    @pytest.mark.asyncio
    async def test_no_server_no_ffmpeg_returns_error(self, tmp_video):
        """If no server AND keyframe extraction fails, return error (not base64)."""
        with patch("copaw.agents.tools.view_media._get_signed_url", return_value=None):
            with patch("copaw.agents.tools.view_media._extract_keyframes", return_value=[]):
                result = await view_video(str(tmp_video))
                text = " ".join(b.get("text", "") for b in result.content if isinstance(b, dict))
                assert "media server" in text.lower() or "error" in text.lower()
                # Must NOT contain video or image blocks (no base64!)
                types = [b.get("type") for b in result.content if isinstance(b, dict)]
                assert "video" not in types
                assert "image" not in types

    @pytest.mark.asyncio
    async def test_nonexistent_video_returns_error(self):
        result = await view_video("/tmp/no_such_video_99999.mp4")
        text = result.content[0]["text"]
        assert "does not exist" in text

    @pytest.mark.asyncio
    async def test_video_never_uses_base64(self, tmp_video):
        """CRITICAL: Videos must never be base64 encoded (context explosion)."""
        with patch("copaw.agents.tools.view_media._get_signed_url", return_value=None):
            with patch("copaw.agents.tools.view_media._extract_keyframes", return_value=[]):
                result = await view_video(str(tmp_video))
                for block in result.content:
                    if isinstance(block, dict) and block.get("type") == "video":
                        source = block.get("source", {})
                        assert source.get("type") != "base64", \
                            "FATAL: Video served as base64 — this will blow up the context window!"


# ---------------------------------------------------------------------------
# Model capability check tests
# ---------------------------------------------------------------------------

class TestModelCapabilityCheck:

    @pytest.mark.asyncio
    async def test_video_unsupported_model_uses_keyframes(self, tmp_path):
        """When model doesn't support video, always extract keyframes even if media server is ON."""
        import subprocess
        video = tmp_path / "test.mp4"
        try:
            subprocess.run([
                "ffmpeg", "-f", "lavfi", "-i", "color=c=green:s=64x64:d=2",
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-y", str(video),
            ], capture_output=True, timeout=10)
        except Exception:
            pytest.skip("ffmpeg not available")
        if not video.exists():
            pytest.skip("ffmpeg failed")

        with patch("copaw.agents.tools.view_media._model_supports_video", return_value=False):
            result = await view_video(str(video))
            types = [b.get("type") for b in result.content if isinstance(b, dict)]
            # Should be keyframes (images), NOT video block
            assert "video" not in types
            assert "image" in types
            text = " ".join(b.get("text", "") for b in result.content if isinstance(b, dict))
            assert "does not support video" in text

    @pytest.mark.asyncio
    async def test_video_supported_model_sends_video(self, tmp_video):
        """When model supports video AND media server available, send video block."""
        with patch("copaw.agents.tools.view_media._model_supports_video", return_value=True):
            with patch("copaw.agents.tools.view_media._get_signed_url", return_value="https://media.example.com/v"):
                result = await view_video(str(tmp_video))
                types = [b.get("type") for b in result.content if isinstance(b, dict)]
                assert "video" in types

    @pytest.mark.asyncio
    async def test_video_unsupported_no_ffmpeg_returns_error(self, tmp_video):
        """Model doesn't support video AND keyframe extraction fails → error."""
        with patch("copaw.agents.tools.view_media._model_supports_video", return_value=False):
            with patch("copaw.agents.tools.view_media._extract_keyframes", return_value=[]):
                result = await view_video(str(tmp_video))
                text = " ".join(b.get("text", "") for b in result.content if isinstance(b, dict))
                assert "does not support video" in text
                types = [b.get("type") for b in result.content if isinstance(b, dict)]
                assert "video" not in types
                assert "image" not in types
