# -*- coding: utf-8 -*-
"""Load image or video files into the LLM context for analysis."""

import base64
import mimetypes
import os
import unicodedata
from pathlib import Path
from typing import Optional

from agentscope.message import ImageBlock, TextBlock, VideoBlock
from agentscope.tool import ToolResponse

_IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".webp",
    ".bmp",
    ".tiff",
    ".tif",
}

_VIDEO_EXTENSIONS = {
    ".mp4",
    ".webm",
    ".mpeg",
    ".mov",
    ".avi",
    ".mkv",
}


def _validate_media_path(
    file_path: str,
    allowed_extensions: set[str],
    mime_prefix: str,
) -> tuple[Path, Optional[ToolResponse]]:
    """Validate a media file path.

    Returns ``(resolved_path, None)`` on success or
    ``(_, error_response)`` on failure.
    """
    file_path = unicodedata.normalize(
        "NFC",
        os.path.expanduser(file_path),
    )
    resolved = Path(file_path).resolve()

    if not resolved.exists() or not resolved.is_file():
        return resolved, ToolResponse(
            content=[
                TextBlock(
                    type="text",
                    text=f"Error: {file_path} does not exist "
                    "or is not a file.",
                ),
            ],
        )

    ext = resolved.suffix.lower()
    mime, _ = mimetypes.guess_type(str(resolved))
    if ext not in allowed_extensions and (
        not mime or not mime.startswith(f"{mime_prefix}/")
    ):
        return resolved, ToolResponse(
            content=[
                TextBlock(
                    type="text",
                    text=f"Error: {resolved.name} is not a "
                    f"supported {mime_prefix} format.",
                ),
            ],
        )

    return resolved, None



# ---------------------------------------------------------------------------
# Media-server configuration
# Priority: agent running config > env vars > defaults
# ---------------------------------------------------------------------------

_DEFAULT_MEDIA_SERVER_URL = "http://localhost:8089"
_DEFAULT_MEDIA_TUNNEL_DOMAIN = ""
_DEFAULT_MEDIA_SECRET = "copaw-media-2026"
_DEFAULT_MAX_SIZE_MB = 100
_DEFAULT_MEDIA_ENABLED = False


def _get_media_config() -> dict:
    """Load media server config from agent running config, env vars, or defaults."""
    cfg = {
        "enabled": _DEFAULT_MEDIA_ENABLED,
        "server_url": _DEFAULT_MEDIA_SERVER_URL,
        "tunnel_domain": _DEFAULT_MEDIA_TUNNEL_DOMAIN,
        "media_secret": _DEFAULT_MEDIA_SECRET,
        "max_size_mb": _DEFAULT_MAX_SIZE_MB,
    }

    # Try loading from agent running config first
    try:
        from ...config.config import load_agent_config
        from ...config.utils import load_config

        root_config = load_config()
        agent_id = root_config.agents.active_agent
        agent_config = load_agent_config(agent_id)
        if agent_config.running and agent_config.running.media_server:
            ms = agent_config.running.media_server
            cfg["enabled"] = ms.enabled
            cfg["server_url"] = ms.server_url
            cfg["tunnel_domain"] = ms.tunnel_domain
            cfg["media_secret"] = ms.media_secret
            cfg["max_size_mb"] = ms.max_size_mb
            return cfg
    except Exception:
        pass

    # Fall back to env vars
    env_server = os.environ.get("COPAW_MEDIA_SERVER")
    env_domain = os.environ.get("COPAW_MEDIA_DOMAIN")
    env_secret = os.environ.get("COPAW_MEDIA_SECRET")
    env_max_mb = os.environ.get("COPAW_MEDIA_MAX_SIZE_MB")
    env_enabled = os.environ.get("COPAW_MEDIA_ENABLED")

    if env_server:
        cfg["server_url"] = env_server
    if env_domain:
        cfg["tunnel_domain"] = env_domain
    if env_secret:
        cfg["media_secret"] = env_secret
    if env_max_mb:
        try:
            cfg["max_size_mb"] = int(env_max_mb)
        except ValueError:
            pass
    if env_enabled is not None:
        cfg["enabled"] = env_enabled.lower() in ("1", "true", "yes")

    return cfg


def _get_signed_url(resolved: Path) -> str:
    """Get a signed public URL for a local media file via the media server.

    Falls back to base64 if media server is unavailable.
    """
    import urllib.request
    import json as _json

    media_cfg = _get_media_config()
    max_size = media_cfg["max_size_mb"] * 1024 * 1024

    size = resolved.stat().st_size
    if size > max_size:
        raise ValueError(
            f"{resolved.name} is {size / 1e6:.1f}MB, exceeds "
            f"{media_cfg['max_size_mb']}MB limit. "
            f"Compress first: ffmpeg -i {resolved} -vf scale=-2:480 -b:v 1M small.mp4"
        )

    if not media_cfg["enabled"]:
        return None

    try:
        url = f"{media_cfg['server_url']}/sign?path={resolved}&ttl=3600"
        resp = urllib.request.urlopen(url, timeout=5)
        data = _json.loads(resp.read())
        signed = data["url"]
        # If tunnel domain is set, rewrite URL to use public domain
        if media_cfg["tunnel_domain"]:
            from urllib.parse import urlparse, urlunparse
            parsed = urlparse(signed)
            tunnel = urlparse(media_cfg["tunnel_domain"])
            signed = urlunparse(parsed._replace(
                scheme=tunnel.scheme or "https",
                netloc=tunnel.netloc,
            ))
        return signed
    except Exception:
        # Fallback to base64 if media server is down
        return None


def _local_to_base64_source(resolved: Path, mime_prefix: str) -> dict:
    """Fallback: convert local file to base64 source dict."""
    data = resolved.read_bytes()
    b64 = base64.b64encode(data).decode("ascii")
    mime, _ = mimetypes.guess_type(str(resolved))
    if not mime:
        ext = resolved.suffix.lower()
        mime = f"{mime_prefix}/{ext.lstrip('.')}"
    return {"type": "base64", "media_type": mime, "data": b64}


def _resolve_media_source(resolved: Path, mime_prefix: str) -> dict:
    """Get media source — prefer signed URL, fallback to base64."""
    signed_url = _get_signed_url(resolved)
    if signed_url:
        return {"type": "url", "url": signed_url}
    return _local_to_base64_source(resolved, mime_prefix)


async def view_image(image_path: str) -> ToolResponse:
    """Load an image file into the LLM context so the model can see it.

    Use this after desktop_screenshot, browser_use, or any tool that
    produces an image file path.

    Args:
        image_path (`str`):
            Path to the image file to view.

    Returns:
        `ToolResponse`:
            An ImageBlock the model can inspect, or an error message.
    """
    resolved, err = _validate_media_path(
        image_path,
        _IMAGE_EXTENSIONS,
        "image",
    )
    if err is not None:
        return err

    return ToolResponse(
        content=[
            ImageBlock(
                type="image",
                source=_resolve_media_source(resolved, "image"),
            ),
            TextBlock(
                type="text",
                text=f"Image loaded: {resolved.name} (path: {resolved})",
            ),
        ],
    )


async def view_video(video_path: str) -> ToolResponse:
    """Load a video file into the LLM context so the model can see it.

    Use this when the user asks about a video file or when another
    tool produces a video file path.

    Args:
        video_path (`str`):
            Path to the video file to view.

    Returns:
        `ToolResponse`:
            A VideoBlock the model can inspect, or an error message.
    """
    resolved, err = _validate_media_path(
        video_path,
        _VIDEO_EXTENSIONS,
        "video",
    )
    if err is not None:
        return err

    return ToolResponse(
        content=[
            VideoBlock(
                type="video",
                source=_resolve_media_source(resolved, "video"),
            ),
            TextBlock(
                type="text",
                text=f"Video loaded: {resolved.name} (path: {resolved})",
            ),
        ],
    )
