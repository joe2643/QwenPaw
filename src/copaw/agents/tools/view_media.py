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



# Media server config — override via agent config or env vars
_MEDIA_SERVER_URL = os.environ.get("COPAW_MEDIA_SERVER", "http://localhost:8089")
_MEDIA_TUNNEL_DOMAIN = os.environ.get("COPAW_MEDIA_DOMAIN", "https://media.joe2643.work")
_MAX_MEDIA_SIZE = 100 * 1024 * 1024  # 100MB (media server limit)


def _get_signed_url(resolved: Path) -> str:
    """Get a signed public URL for a local media file via the media server.

    Falls back to base64 if media server is unavailable.
    """
    import urllib.request
    import json as _json

    size = resolved.stat().st_size
    if size > _MAX_MEDIA_SIZE:
        raise ValueError(
            f"{resolved.name} is {size / 1e6:.1f}MB, exceeds {_MAX_MEDIA_SIZE / 1e6:.0f}MB limit. "
            f"Compress first: ffmpeg -i {resolved} -vf scale=-2:480 -b:v 1M small.mp4"
        )
    try:
        url = f"{_MEDIA_SERVER_URL}/sign?path={resolved}&ttl=3600"
        resp = urllib.request.urlopen(url, timeout=5)
        data = _json.loads(resp.read())
        return data["url"]
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
