# -*- coding: utf-8 -*-
"""Load image or video files into the LLM context for analysis."""

import logging
import mimetypes
import os
import unicodedata
import urllib.parse
from pathlib import Path
from typing import Optional

from agentscope.message import ImageBlock, TextBlock, VideoBlock
from agentscope.tool import ToolResponse

logger = logging.getLogger(__name__)

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


def _is_url(path: str) -> bool:
    """Return True if *path* looks like an HTTP(S) URL."""
    return path.startswith(("http://", "https://"))


def _validate_url_extension(
    url: str,
    allowed_extensions: set[str],
    mime_prefix: str,
) -> Optional[ToolResponse]:
    """Optionally validate that the URL path has an allowed extension.

    Returns an error ``ToolResponse`` when the extension is clearly
    unsupported, or ``None`` to let it through (including when the URL
    has no recognisable extension, e.g. dynamic endpoints).
    """
    url_path = urllib.parse.urlparse(url).path
    ext = Path(url_path).suffix.lower()
    if not ext:
        return None
    mime, _ = mimetypes.guess_type(url_path)
    if ext not in allowed_extensions and (
        not mime or not mime.startswith(f"{mime_prefix}/")
    ):
        return ToolResponse(
            content=[
                TextBlock(
                    type="text",
                    text=f"Error: URL does not point to a "
                    f"supported {mime_prefix} format: {url}",
                ),
            ],
        )
    return None


def _validate_media_path(
    file_path: str,
    allowed_extensions: set[str],
    mime_prefix: str,
) -> tuple[Path, Optional[ToolResponse]]:
    """Validate a local media file path.

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


async def _probe_multimodal_if_needed(
    media_type: str = "image",
) -> bool | None:
    """Trigger a multimodal probe if capability is unknown (None).

    For ``image``: runs an image-only probe (~3s) and fires the full
    probe (image + video) as a background task so video support is
    persisted without blocking the caller.

    For ``video``: runs the full probe and waits for the video result,
    since video support cannot be inferred from the image probe alone.

    Uses the same agent-specific model resolution as
    ``_get_active_model_info`` so that per-agent model overrides are
    respected.

    Returns the probe result (True/False) for the requested media type,
    or None if no probe was needed or the probe failed.
    """
    try:
        from ..prompt import _get_active_model_info
        from ...providers.provider_manager import ProviderManager

        model_info, _ = _get_active_model_info()
        if model_info is None or model_info.supports_multimodal is not None:
            return None

        # Resolve agent-specific active model (mirrors _get_active_model_info)
        manager = ProviderManager.get_instance()
        active = None
        try:
            from ...app.agent_context import get_current_agent_id
            from ...config.config import load_agent_config

            agent_id = get_current_agent_id()
            agent_config = load_agent_config(agent_id)
            if agent_config.active_model:
                active = agent_config.active_model
        except Exception:
            pass
        if not active:
            active = manager.get_active_model()
        if not active:
            return None

        if media_type == "image":
            logger.info(
                "Multimodal capability unknown for %s/%s — "
                "running image-only probe...",
                active.provider_id,
                active.model,
            )
            result = await manager.probe_model_multimodal(
                active.provider_id,
                active.model,
                image_only=True,
            )
            supports = result.get("supports_image", False)
            logger.info(
                "Image probe completed for %s/%s: supports_image=%s",
                active.provider_id,
                active.model,
                supports,
            )
            # Fire full probe in background to persist video support too
            import asyncio

            asyncio.create_task(
                manager.probe_model_multimodal(
                    active.provider_id,
                    active.model,
                ),
            )
        else:
            # video: must run full probe to get video result
            logger.info(
                "Multimodal capability unknown for %s/%s — "
                "running full probe for video support...",
                active.provider_id,
                active.model,
            )
            result = await manager.probe_model_multimodal(
                active.provider_id,
                active.model,
            )
            supports = result.get("supports_video", False)
            logger.info(
                "Full probe completed for %s/%s: supports_video=%s",
                active.provider_id,
                active.model,
                supports,
            )
        return supports
    except Exception as e:
        logger.warning("Auto-probe in view_media failed: %s", e)
        return None


def _check_multimodal_support(media_type: str = "image") -> bool:
    """Check whether the active model supports the given media type (sync).

    For ``image``: returns True when supports_image or supports_multimodal
    is explicitly True.
    For ``video``: returns True only when supports_video is explicitly True.

    Returns False for unknown (None) or explicitly unsupported (False).
    The tool is still *registered*; the async probe path handles the
    probe-on-demand logic.
    """
    try:
        from ..prompt import _get_active_model_info

        model_info, _ = _get_active_model_info()
        if model_info is None:
            return True
        if media_type == "video":
            return model_info.supports_video is True
        # image: True if supports_image or the combined supports_multimodal
        return (
            model_info.supports_image is True
            or model_info.supports_multimodal is True
        )
    except Exception:
        return True


def _get_multimodal_fallback_hint(media_type: str, path: str) -> str:
    """Build a text hint for the model when multimodal is not available.

    The actual media block is still included in the response so the
    frontend/user can see it; the hint tells the agent it cannot perceive
    the media itself.
    """
    try:
        from ..prompt import get_active_model_multimodal_raw

        raw = get_active_model_multimodal_raw()
    except Exception:
        raw = None

    if raw is None:
        logger.warning(
            "view_%s was called but multimodal capability has not been "
            "confirmed for the active model. The %s at '%s' will be "
            "shown to the user but the model cannot see it. "
            "To fix, set supports_multimodal=true in provider settings.",
            media_type,
            media_type,
            path,
        )
        return (
            f"[Note: this model does not appear to support multimodal "
            f"input — no multimodal capability was detected. You cannot "
            f"see this {media_type}, but it has been shown to the user. "
            f"Inform the user that you cannot analyze the {media_type} "
            f"content. If they believe this model supports vision, they "
            f"can override this in provider settings by setting "
            f"`supports_multimodal: true`, then retry.]"
        )

    logger.warning(
        "view_%s was called but the active model explicitly does not "
        "support multimodal input. The %s at '%s' will be shown to "
        "the user but the model cannot see it.",
        media_type,
        media_type,
        path,
    )
    return (
        f"[Note: the current model does not support multimodal input — "
        f"you cannot see this {media_type}, but it has been shown to "
        f"the user. Inform the user that you cannot analyze the "
        f"{media_type} content. If they believe this model actually "
        f"supports vision, they can override `supports_multimodal: true` "
        f"in the provider settings, or switch to a vision-capable model.]"
    )


async def view_image(image_path: str) -> ToolResponse:
    """Load an image file into the LLM context so the model can see it.

    Use this after desktop_screenshot, browser_use, or any tool that
    produces an image file path.  Also accepts an HTTP(S) URL for
    online images — the URL is passed directly to the model without
    downloading.

    When the model does not support multimodal, the image is still
    returned (so the user/frontend can see it) along with a text hint
    telling the agent it cannot perceive the image. The downstream
    media-stripping pipeline will remove the ImageBlock before sending
    to the model.

    Args:
        image_path (`str`):
            Local path or HTTP(S) URL of the image to view.

    Returns:
        `ToolResponse`:
            An ImageBlock the model can inspect, or an error message.
    """
    # Determine whether we need a fallback hint
    fallback_hint: str | None = None
    if not _check_multimodal_support("image"):
        probe_result = await _probe_multimodal_if_needed("image")
        if probe_result is not True:
            fallback_hint = _get_multimodal_fallback_hint("image", image_path)

    if _is_url(image_path):
        err = _validate_url_extension(
            image_path,
            _IMAGE_EXTENSIONS,
            "image",
        )
        if err is not None:
            return err
        text_msg = (
            fallback_hint
            if fallback_hint
            else f"Image loaded from URL: {image_path}"
        )
        return ToolResponse(
            content=[
                ImageBlock(
                    type="image",
                    source={"type": "url", "url": image_path},
                ),
                TextBlock(type="text", text=text_msg),
            ],
        )

    resolved, err = _validate_media_path(
        image_path,
        _IMAGE_EXTENSIONS,
        "image",
    )
    if err is not None:
        return err

    text_msg = (
        fallback_hint if fallback_hint else f"Image loaded: {resolved.name}"
    )
    return ToolResponse(
        content=[
            ImageBlock(
                type="image",
                source={"type": "url", "url": str(resolved)},
            ),
            TextBlock(type="text", text=text_msg),
        ],
    )


_DEFAULT_VIDEO_FALLBACK_PROMPT = (
    "Describe this video in detail: what happens step by step, any "
    "on-screen text or captions, distinctive objects / people / "
    "locations, and the overall mood.  Be thorough so a model that "
    "cannot see the video can still reason about it from your "
    "description alone."
)

# Provider id prefixes that are OpenAI-chat-compat but expect Qwen's
# multimodal shape (``{"type":"video","video":[url]}`` — video is a
# list, not a ``source`` sub-object).  Seen across Aliyun Bailian,
# DashScope coding plan, Kimi, ModelScope, mimo, and the SkillClaw
# proxy that fronts Bailian locally.
_QWEN_FAMILY_PREFIXES = (
    "aliyun-",
    "bailian",
    "kimi-",
    "modelscope",
    "mimo",
)


def _is_qwen_family(provider_id: str) -> bool:
    p = (provider_id or "").lower()
    return any(p.startswith(x) for x in _QWEN_FAMILY_PREFIXES)


async def _build_fallback_video_messages(
    video_block: VideoBlock,
    prompt: str,
    provider_id: str,
) -> list[dict] | None:
    """Format ``messages`` for the fallback chat model in its native
    multimodal shape.  Returns ``None`` when we don't know how to
    shape the call for the target provider — the caller will fall
    through to the generic placeholder hint.

    Qwen-family providers need ``{"type":"video","video":[url]}``
    with a URL the upstream can actually fetch — tens of MB of
    base64 in the request body would 413 or time out.  We route
    through the shared :func:`resolve_media_url` so local files
    become signed media-server URLs (public via the Cloudflare
    tunnel when one is configured, loopback otherwise).  If the
    media server is unreachable / refuses the path, we skip the
    delegation and let the caller fall through to the placeholder
    hint — sending a local path to a cloud endpoint is guaranteed
    to fail.

    For providers we don't recognise (including Gemini, which has
    its own SDK video path via agentscope), we pass the VideoBlock
    as-is and rely on the upstream formatter to handle it.
    """
    from ...app.channels.media_utils import resolve_media_url

    source = video_block.get("source") or {}
    url = source.get("url") or ""

    if _is_qwen_family(provider_id):
        resolved = await resolve_media_url(url) if url else ""
        # Must be a URL the cloud endpoint can fetch — reject
        # anything that still looks like a raw local path.
        if not resolved.startswith(("http://", "https://", "data:")):
            return None
        # DashScope / Qwen-VL has two video modes on their
        # OpenAI-compat chat/completions endpoint:
        #   * ``type: "video_url"`` + ``video_url: {url}`` — single
        #     video file, Qwen samples frames server-side.
        #   * ``type: "video"``     + ``video: [frame_urls...]`` —
        #     pre-extracted frame list, must contain 4–8000 frames.
        # We have exactly one video URL, so use the single-file
        # mode.  Wrapping a single URL in the frame-list shape
        # trips the ``"sequence images should be (4, 8000)"``
        # validation server-side (seen in production).
        return [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video_url",
                        "video_url": {"url": resolved},
                    },
                    {"type": "text", "text": prompt},
                ],
            },
        ]

    # Unknown provider: pass agentscope-style VideoBlock and hope
    # the chat model's formatter translates it (agentscope's
    # Gemini path does; OpenAI's does not).
    return [
        {
            "role": "user",
            "content": [
                video_block,
                TextBlock(type="text", text=prompt),
            ],
        },
    ]


def _resolve_fallback_video_model() -> (
    "tuple[object, str, str] | None"
):
    """Return a ready-to-call chat model instance for the agent's
    configured ``fallback_video_model``, or ``None`` when none is set.

    Returns a ``(chat_model, provider_id, model_id)`` tuple so the
    caller can surface which fallback handled the request in logs
    / user-facing hints.
    """
    try:
        # Three dots: view_media lives at ``agents/tools/view_media.py``
        # so ``...app`` resolves to ``qwenpaw/app`` (not
        # ``qwenpaw/agents/app`` which doesn't exist).  Paired with
        # ``...config`` / ``...providers``.
        from ...app.agent_context import get_current_agent_id
        from ...config.config import load_agent_config
        from ...providers.provider_manager import ProviderManager

        try:
            agent_id = get_current_agent_id()
        except Exception:
            return None
        agent_config = load_agent_config(agent_id)
        fallback = getattr(agent_config, "fallback_video_model", None)
        if not fallback or not fallback.provider_id or not fallback.model:
            return None

        manager = ProviderManager.get_instance()
        provider = manager.get_provider(fallback.provider_id)
        if provider is None:
            logger.warning(
                "view_video: fallback provider '%s' not found",
                fallback.provider_id,
            )
            return None
        chat_model = provider.get_chat_model_instance(fallback.model)
        return chat_model, fallback.provider_id, fallback.model
    except Exception as e:
        logger.warning(
            "view_video: fallback model resolution failed: %s", e,
        )
        return None


async def _describe_video_via_fallback(
    video_block: VideoBlock,
    prompt: str,
    fallback: "tuple[object, str, str]",
) -> str | None:
    """One-shot call to the fallback video model.  Returns the text
    description, or ``None`` on failure (the caller substitutes the
    generic multimodal hint in that case).
    """
    chat_model, provider_id, model_id = fallback
    try:
        messages = await _build_fallback_video_messages(
            video_block, prompt, provider_id,
        )
        if messages is None:
            logger.warning(
                "view_video: cannot format video call for %s/%s "
                "(unknown shape or media-server signing failed); "
                "falling back to generic hint",
                provider_id, model_id,
            )
            return None
        logger.info(
            "view_video: delegating to fallback %s/%s (prompt len=%d)",
            provider_id, model_id, len(prompt),
        )
        response = await chat_model(messages)
        # Agentscope chat models can stream (AsyncGenerator) or return
        # a single ChatResponse depending on the ``stream`` init flag.
        # ``get_chat_model_instance`` defaults to ``stream=True``, so
        # we iterate and keep the final cumulative text.
        final_text = ""
        if hasattr(response, "__aiter__"):
            async for chunk in response:
                for block in getattr(chunk, "content", None) or []:
                    if (
                        isinstance(block, dict)
                        and block.get("type") == "text"
                    ):
                        # Streamed text blocks are cumulative in agentscope.
                        final_text = str(block.get("text") or final_text)
        else:
            for block in getattr(response, "content", None) or []:
                if isinstance(block, dict) and block.get("type") == "text":
                    final_text = str(block.get("text") or final_text)
        return final_text.strip() or None
    except Exception as e:
        logger.warning(
            "view_video: fallback %s/%s failed: %s",
            provider_id, model_id, e,
        )
        return None


async def view_video(
    video_path: str,
    prompt: str | None = None,
) -> ToolResponse:
    """Load a video file into the LLM context so the model can see it.

    Use this when the user asks about a video file or when another
    tool produces a video file path.  Also accepts an HTTP(S) URL —
    the URL is passed directly to the model without downloading.

    When the active model does not support video AND the agent has a
    ``fallback_video_model`` configured (Settings → Agent → Fallback
    Video Model), this tool delegates the video to that model with
    the supplied ``prompt`` (or a detailed default prompt when none
    is given) and returns the description as text.  The primary
    agent can then reason about the video's contents without
    multimodal support itself.

    When the active model does not support video and **no** fallback
    is configured, the video is still returned (so the user / frontend
    can see it) along with a text hint telling the agent it cannot
    perceive the video.

    Args:
        video_path (`str`):
            Local path or HTTP(S) URL of the video to view.
        prompt (`str | None`, optional):
            Question / instruction the fallback model should answer
            about the video.  Ignored when the active model itself
            supports video (in that case the agent reasons directly
            over the VideoBlock).  Defaults to a generic
            describe-everything prompt.

    Returns:
        `ToolResponse`:
            A VideoBlock the model can inspect, a fallback text
            description, or an error message.
    """
    # Step 1: resolve media path / URL into a VideoBlock first, because
    # both the native-model branch and the fallback branch need it.
    if _is_url(video_path):
        err = _validate_url_extension(
            video_path,
            _VIDEO_EXTENSIONS,
            "video",
        )
        if err is not None:
            return err
        video_block = VideoBlock(
            type="video",
            source={"type": "url", "url": video_path},
        )
        video_label = f"Video loaded from URL: {video_path}"
    else:
        resolved, err = _validate_media_path(
            video_path,
            _VIDEO_EXTENSIONS,
            "video",
        )
        if err is not None:
            return err
        video_block = VideoBlock(
            type="video",
            source={"type": "url", "url": str(resolved)},
        )
        video_label = f"Video loaded: {resolved.name}"

    # Step 2: check if the active model can see video natively.
    primary_supports_video = _check_multimodal_support("video")
    if not primary_supports_video:
        probe_result = await _probe_multimodal_if_needed("video")
        primary_supports_video = probe_result is True

    if primary_supports_video:
        # Active model handles video directly — return the block +
        # a short confirmation line.
        return ToolResponse(
            content=[video_block, TextBlock(type="text", text=video_label)],
        )

    # Step 3: active model can't see video — try the configured fallback.
    fallback = _resolve_fallback_video_model()
    if fallback is not None:
        effective_prompt = (
            prompt.strip() if isinstance(prompt, str) and prompt.strip()
            else _DEFAULT_VIDEO_FALLBACK_PROMPT
        )
        description = await _describe_video_via_fallback(
            video_block, effective_prompt, fallback,
        )
        if description:
            _, provider_id, model_id = fallback
            # Keep the VideoBlock in the response so the user /
            # frontend can still play the video; the primary model's
            # media-stripping pipeline will drop the block before it
            # reaches the model, leaving only the fallback's text.
            header = (
                f"[Video description from fallback model "
                f"{provider_id}/{model_id}]"
            )
            return ToolResponse(
                content=[
                    video_block,
                    TextBlock(type="text", text=header),
                    TextBlock(type="text", text=description),
                ],
            )
        # Fallback failed — fall through to the generic hint below.

    # Step 4: no fallback (or fallback itself failed) → generic hint.
    # The VideoBlock stays in the response so the user / frontend can
    # still see the video.  The normalizer (see
    # ``message_request_normalizer``) now strips video per-type when
    # the outgoing model can't process it and replaces the block with
    # a path-preserving text placeholder, so we don't need to drop
    # the block here to avoid the 413 ``Request Too Large`` that
    # used to fire on Claude OAuth.
    fallback_hint = _get_multimodal_fallback_hint("video", video_path)
    return ToolResponse(
        content=[
            video_block,
            TextBlock(type="text", text=fallback_hint),
        ],
    )
