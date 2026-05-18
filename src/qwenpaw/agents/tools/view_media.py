# -*- coding: utf-8 -*-
"""Load image or video files into the LLM context for analysis."""

import asyncio
import logging
import mimetypes
import os
import unicodedata
import urllib.parse
from pathlib import Path
from typing import Any, Optional

from agentscope.message import ImageBlock, TextBlock, VideoBlock
from agentscope.tool import ToolResponse

logger = logging.getLogger(__name__)


class _MimoUnsupportedFormatError(Exception):
    """Mimo (and likely other Qwen-family endpoints) returned a 400
    saying the multimodal payload is unprocessable.  Two flavours
    seen in production:

    - ``"Multimodal data is corrupted or cannot be processed."``
      → codec the decoder can't handle (e.g. AV1).
    - ``"only mp4/wmv/mov/avi are supported"``
      → container the endpoint refuses (e.g. .webm).

    Both are recoverable by transcoding to H.264-in-MP4 and
    retrying once.  Distinct from a generic non-200 so the caller
    can decide whether the retry is worth it; other 400s fall
    through to the existing ``return None`` placeholder path.
    """


_IMAGE_EXTENSIONS = {
    ".png",
    ".apng",
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


# Streaming platforms that serve HTML pages, not raw media bytes.
# Upstream vision endpoints (z.ai, mimo, qwen-vl) fetch ``video_url``
# server-side and expect direct mp4/webm; passing a YouTube /
# TikTok / X URL makes them get back HTML and 400 with ``1210
# 图片输入格式/解析错误``.  Reject these URLs early with an
# instructive error so the agent immediately knows to ``yt-dlp``
# them to a local file and retry — observed in WhatsApp group on
# 2026-05-12 (agent passed ``https://youtu.be/B9NGOONYnAo`` to
# view_video and ate the 1210).
_STREAMING_PLATFORM_HOST_HINTS = (
    "youtube.com",
    "youtu.be",
    "tiktok.com",
    "douyin.com",
    "bilibili.com",
    "x.com/i/videos/",
    "x.com/i/status/",
    "twitter.com",
    "instagram.com",
    "facebook.com/watch",
    "fb.watch",
    "vimeo.com",
)


def _is_streaming_platform_url(url: str) -> bool:
    """Return True for URLs hosted by platforms that serve HTML, not
    raw media bytes — those need an extractor (yt-dlp etc.) before
    they can be loaded into a vision model.
    """
    try:
        parsed = urllib.parse.urlparse(url)
    except Exception:
        return False
    host = (parsed.hostname or "").lower()
    full = (parsed.hostname or "") + (parsed.path or "")
    return any(
        hint in host or hint in full for hint in _STREAMING_PLATFORM_HOST_HINTS
    )


def _streaming_platform_error(url: str, media_type: str) -> ToolResponse:
    return ToolResponse(
        content=[
            TextBlock(
                type="text",
                text=(
                    f"Error: cannot load {media_type} directly from a "
                    f"streaming platform URL: {url}\n\n"
                    f"Vision models fetch ``{media_type}_url`` from the "
                    f"server side and need raw media bytes (mp4 / webm "
                    f"/ etc.), but streaming sites return HTML.  "
                    f"Download with yt-dlp first, then call view_"
                    f"{media_type} with the local path:\n\n"
                    f"```\n"
                    f"yt-dlp -f 'best[height<=720][ext=mp4]/best[height<=720]' "
                    f"-o '/tmp/yt_%(id)s.%(ext)s' '{url}'\n"
                    f"# then\n"
                    f"view_{media_type}(/tmp/yt_<id>.mp4, prompt=...)\n"
                    f"```"
                ),
            ),
        ],
    )


def _validate_url_extension(
    url: str,
    allowed_extensions: set[str],
    mime_prefix: str,
) -> Optional[ToolResponse]:
    """Optionally validate that the URL path has an allowed extension.

    Returns an error ``ToolResponse`` when the URL clearly cannot be
    loaded directly (streaming platform that returns HTML, or
    explicitly unsupported file extension), or ``None`` to let it
    through (including when the URL has no recognisable extension,
    e.g. dynamic endpoints).
    """
    if _is_streaming_platform_url(url):
        return _streaming_platform_error(url, mime_prefix)
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


async def _transcode_animated_webp_to_apng(src_path: str) -> str | None:
    """Convert an animated WebP (e.g. Signal sticker) to APNG.

    Z.AI's glm-5v-turbo (and similar OpenAI-compat endpoints) reject
    ``image/webp`` data URLs that carry animation chunks (VP8X/ANIM)
    with HTTP 400 ``"1210 图片输入格式/解析错误"``.  APNG preserves
    every frame + alpha channel and z.ai decodes it end-to-end —
    confirmed 2026-05-12 by curling the same sticker as PNG (works,
    single frame), GIF (rejected), APNG (works, model picked up
    animation-only details), original WebP (rejected).

    Static webp (no ANIM/ANMF chunks) usually works as-is; transcoding
    those would waste cycles and bloat the request body, so this
    helper is a no-op for them — caller decides which to invoke.

    Returns the new APNG path (sibling of ``src_path``) or ``None``
    on any failure.  Idempotent: reuses a non-empty sibling.
    """
    try:
        from PIL import Image, ImageSequence
    except ImportError:
        logger.warning(
            "view_image: Pillow not available — cannot transcode webp",
        )
        return None

    if not src_path or not os.path.exists(src_path):
        logger.warning(
            "view_image: webp transcode source missing: %s",
            src_path,
        )
        return None

    p = Path(src_path)
    out = p.with_name(p.stem + ".apng")
    if out.exists() and out.stat().st_size > 0:
        logger.debug(
            "view_image: reusing existing webp→apng transcode %s",
            out,
        )
        return str(out)

    def _do_transcode() -> str | None:
        try:
            with Image.open(src_path) as im:
                n_frames = getattr(im, "n_frames", 1)
                if n_frames < 2:
                    return None
                duration = im.info.get("duration") or 100
                frames = [f.copy() for f in ImageSequence.Iterator(im)]
            frames[0].save(
                out,
                format="PNG",
                save_all=True,
                append_images=frames[1:],
                duration=duration,
                loop=0,
            )
            return str(out)
        except Exception as e:
            logger.warning(
                "view_image: webp→apng transcode failed for %s: %s",
                src_path,
                e,
            )
            try:
                if out.exists():
                    out.unlink()
            except OSError:
                pass
            return None

    return await asyncio.to_thread(_do_transcode)


def _is_animated_webp(path: str) -> bool:
    """Cheap header sniff for animated WebP.  ``RIFF....WEBP`` container
    with a ``VP8X`` chunk that has the animation bit (``ANIM`` chunk
    follows).  Avoids loading Pillow for the common static-webp case.
    """
    try:
        with open(path, "rb") as f:
            head = f.read(200)
    except OSError:
        return False
    if not (head[:4] == b"RIFF" and head[8:12] == b"WEBP"):
        return False
    return b"ANIM" in head or b"ANMF" in head


async def _to_url_form_block(block: dict) -> dict:
    """Rewrite ``block.source.url`` to a media-server URL when it is a
    local path.  HTTP(S) / data URLs pass through.  Best-effort:
    returns the original block when signing fails so we don't regress
    in offline / tunnel-down scenarios.

    Keeps primary-model requests small even for big media — the active
    model receives a fetchable URL instead of inlining the file.
    """
    from ...app.channels.media_utils import resolve_media_url

    source = block.get("source") or {}
    url = source.get("url") or ""
    if not url:
        return block
    resolved = await resolve_media_url(url)
    if not resolved or resolved == url:
        return block
    return {**block, "source": {**source, "url": resolved}}


_DEFAULT_IMAGE_FALLBACK_PROMPT = (
    "Describe this image in detail: visible objects, people, on-screen "
    "text, colors, composition, and any notable context.  Be thorough "
    "so a model that cannot see the image can still reason about it "
    "from your description alone."
)


def _resolve_fallback_image_model() -> "tuple[Any, str, str] | None":
    """Return a ready-to-call chat model instance for the agent's
    configured ``fallback_image_model``, or ``None`` when none is set.

    Mirrors :func:`_resolve_fallback_video_model`.
    """
    try:
        from ...app.agent_context import get_current_agent_id
        from ...config.config import load_agent_config
        from ...providers.provider_manager import ProviderManager

        try:
            agent_id = get_current_agent_id()
        except Exception:
            return None
        agent_config = load_agent_config(agent_id)
        fallback = getattr(agent_config, "fallback_image_model", None)
        if not fallback or not fallback.provider_id or not fallback.model:
            return None

        manager = ProviderManager.get_instance()
        provider = manager.get_provider(fallback.provider_id)
        if provider is None:
            logger.warning(
                "view_image: fallback provider '%s' not found",
                fallback.provider_id,
            )
            return None
        chat_model = provider.get_chat_model_instance(fallback.model)
        return chat_model, fallback.provider_id, fallback.model
    except Exception as e:
        logger.warning(
            "view_image: fallback model resolution failed: %s",
            e,
        )
        return None


async def _build_fallback_image_messages(
    image_block: ImageBlock,
    prompt: str,
    provider_id: str,
) -> list[dict] | None:
    """Format ``messages`` for the fallback chat model.

    For Qwen-family providers we emit the native OpenAI-compat
    ``{"type":"image_url","image_url":{"url":...}}`` shape directly and
    dispatch through the httpx bypass — same rationale as the video
    path: keeps the wire shape we curl-tested and avoids any block-type
    surprises in agentscope's formatter.

    For everyone else, return the agentscope-native ``ImageBlock`` so
    each provider's formatter does its own translation; we still run
    the source URL through :func:`resolve_media_url` first so cloud
    endpoints get a fetchable HTTPS URL instead of a raw local path.
    Returns ``None`` when we can't produce a usable shape (e.g. local
    path that the media server refuses to sign for Qwen-family).
    """
    from ...app.channels.media_utils import resolve_media_url

    source = image_block.get("source") or {}
    url = source.get("url") or ""

    if _is_qwen_family(provider_id):
        resolved = await resolve_media_url(url) if url else ""
        if not resolved.startswith(("http://", "https://", "data:")):
            return None
        return [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": resolved},
                    },
                    {"type": "text", "text": prompt},
                ],
            },
        ]

    resolved_block = image_block
    if url:
        resolved = await resolve_media_url(url)
        if resolved and resolved != url:
            resolved_block = {
                **image_block,
                "source": {**source, "url": resolved},
            }
    return [
        {
            "role": "user",
            "content": [
                resolved_block,
                TextBlock(type="text", text=prompt),
            ],
        },
    ]


async def _describe_image_via_qwen_family_httpx(
    messages: list[dict],
    chat_model: "object",
    model_id: str,
) -> str | None:
    """POST the OpenAI-compat chat/completions request directly for
    Qwen-family providers.  Same rationale as the video bypass: avoids
    any formatter surprises around image content shape.
    """
    import httpx

    client = getattr(chat_model, "client", None)
    base_url = str(getattr(client, "base_url", "")).rstrip("/")
    api_key = getattr(client, "api_key", None) or ""
    if not base_url:
        logger.warning(
            "view_image: Qwen-family fallback %s has no base_url; "
            "cannot dispatch directly",
            model_id,
        )
        return None
    url = (
        f"{base_url}/chat/completions"
        if "/chat/completions" not in base_url
        else base_url
    )
    body = {"model": model_id, "messages": messages, "stream": False}
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    logger.info(
        "view_image: Qwen httpx POST → %s (model=%s, image_url=%s)",
        url,
        model_id,
        next(
            (
                (c.get("image_url") or {}).get("url", "?")
                for c in (messages[0].get("content") or [])
                if isinstance(c, dict) and c.get("type") == "image_url"
            ),
            "?",
        )[:120],
    )
    try:
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(180, connect=30),
        ) as hc:
            resp = await hc.post(url, json=body, headers=headers)
    except Exception as e:
        logger.warning(
            "view_image: Qwen fallback httpx call failed: %s",
            e,
        )
        return None

    if resp.status_code != 200:
        logger.warning(
            "view_image: Qwen fallback HTTP %d: %s",
            resp.status_code,
            resp.text[:400],
        )
        return None

    try:
        j = resp.json()
    except Exception as e:
        logger.warning(
            "view_image: Qwen fallback returned non-JSON body: %s",
            e,
        )
        return None

    choices = j.get("choices") or []
    if not choices:
        logger.warning(
            "view_image: Qwen fallback response missing choices: %s",
            str(j)[:400],
        )
        return None
    msg = (choices[0] or {}).get("message") or {}
    text = msg.get("content") or ""
    result = str(text).strip() or None
    logger.info(
        "view_image: Qwen fallback returned %d chars (usage=%s)",
        len(result or ""),
        j.get("usage"),
    )
    return result


async def _describe_image_via_fallback(
    image_block: ImageBlock,
    prompt: str,
    fallback: "tuple[Any, str, str]",
) -> str | None:
    """One-shot call to the fallback image model.  Returns the text
    description, or ``None`` on failure (the caller substitutes the
    generic multimodal hint in that case).
    """
    chat_model, provider_id, model_id = fallback
    try:
        messages = await _build_fallback_image_messages(
            image_block,
            prompt,
            provider_id,
        )
        if messages is None:
            logger.warning(
                "view_image: cannot format image call for %s/%s "
                "(unknown shape or media-server signing failed); "
                "falling back to generic hint",
                provider_id,
                model_id,
            )
            return None
        logger.info(
            "view_image: delegating to fallback %s/%s (prompt len=%d)",
            provider_id,
            model_id,
            len(prompt),
        )
        if _is_qwen_family(provider_id):
            return await _describe_image_via_qwen_family_httpx(
                messages,
                chat_model,
                model_id,
            )
        response = await chat_model(messages)
        final_text = ""
        chunk_count = 0
        seen_block_types: set[str] = set()
        if hasattr(response, "__aiter__"):
            async for chunk in response:
                chunk_count += 1
                for block in getattr(chunk, "content", None) or []:
                    if isinstance(block, dict):
                        seen_block_types.add(str(block.get("type", "?")))
                        if block.get("type") == "text":
                            final_text = str(
                                block.get("text") or final_text,
                            )
        else:
            for block in getattr(response, "content", None) or []:
                if isinstance(block, dict):
                    seen_block_types.add(str(block.get("type", "?")))
                    if block.get("type") == "text":
                        final_text = str(block.get("text") or final_text)

        result = final_text.strip() or None
        if not result:
            logger.warning(
                "view_image: fallback %s/%s returned empty "
                "(chunks=%d, block_types=%s) — model may not "
                "actually support image despite supports_image=True",
                provider_id,
                model_id,
                chunk_count,
                sorted(seen_block_types),
            )
        return result
    except Exception as e:
        logger.warning(
            "view_image: fallback %s/%s failed: %s",
            provider_id,
            model_id,
            e,
        )
        return None


async def view_image(
    image_path: str,
    prompt: str | None = None,
) -> ToolResponse:
    """Load an image file into the LLM context so the model can see it.

    Use this after desktop_screenshot, browser_use, or any tool that
    produces an image file path.  Also accepts an HTTP(S) URL for
    online images — the URL is passed directly to the model without
    downloading.

    When the active model does not support image AND the agent has a
    ``fallback_image_model`` configured (Settings → Agent → Fallback
    Image Model), this tool delegates the image to that model with the
    supplied ``prompt`` (or a detailed default prompt when none is
    given) and returns the description as text.  The primary agent
    can then reason about the image's contents without multimodal
    support itself.

    When the active model does not support image and **no** fallback
    is configured, the image is still returned (so the user/frontend
    can see it) along with a text hint telling the agent it cannot
    perceive the image.

    Args:
        image_path (`str`):
            Local path or HTTP(S) URL of the image to view.
        prompt (`str | None`, optional):
            Question / instruction the fallback model should answer
            about the image.  Ignored when the active model itself
            supports image (in that case the agent reasons directly
            over the ImageBlock).  Defaults to a generic
            describe-everything prompt.

    Returns:
        `ToolResponse`:
            An ImageBlock the model can inspect, a fallback text
            description, or an error message.
    """
    # Step 1: resolve media path / URL into an ImageBlock first.
    if _is_url(image_path):
        err = _validate_url_extension(
            image_path,
            _IMAGE_EXTENSIONS,
            "image",
        )
        if err is not None:
            return err
        image_block = ImageBlock(
            type="image",
            source={"type": "url", "url": image_path},
        )
        image_label = f"Image loaded from URL: {image_path}"
    else:
        resolved, err = _validate_media_path(
            image_path,
            _IMAGE_EXTENSIONS,
            "image",
        )
        if err is not None:
            return err
        # Animated WebP (Signal stickers etc.) trips z.ai and similar
        # OpenAI-compat endpoints with 1210 "image format/parse error";
        # APNG decodes end-to-end and preserves animation.  Static webp
        # is left alone — model support is widespread and transcoding
        # would inflate the request for no win.
        if resolved.suffix.lower() == ".webp" and _is_animated_webp(
            str(resolved),
        ):
            transcoded = await _transcode_animated_webp_to_apng(
                str(resolved),
            )
            if transcoded:
                logger.info(
                    "view_image: animated webp %s → apng %s",
                    resolved.name,
                    Path(transcoded).name,
                )
                resolved = Path(transcoded)
        image_block = ImageBlock(
            type="image",
            source={"type": "url", "url": str(resolved)},
        )
        image_label = f"Image loaded: {resolved.name}"

    # Step 2: check if the active model can see image natively.
    primary_supports_image = _check_multimodal_support("image")
    if not primary_supports_image:
        probe_result = await _probe_multimodal_if_needed("image")
        primary_supports_image = probe_result is True

    if primary_supports_image:
        effective_prompt = (
            prompt.strip()
            if isinstance(prompt, str) and prompt.strip()
            else _DEFAULT_IMAGE_FALLBACK_PROMPT
        )

        # Primary-path HTTP bypass for OpenAI-compat vision providers
        # (qwen-family + zhipu/z.ai).  Same rationale as the video
        # bypass — these endpoints refuse to decode inline media
        # blocks when conversation history contains prior view_image
        # / view_video tool_call / tool_result pairs.  Reuses the
        # fallback path so the agent receives a text description
        # instead of a hot ImageBlock that the next turn would have
        # to fight history-priors to look at.  Non-qwen-family
        # providers (Gemini, Claude vision) fall through to the
        # inline-block path below.
        primary = _resolve_primary_vision_model()
        if primary is not None and _is_qwen_family(primary[1]):
            logger.info(
                "view_image: primary bypass — calling %s/%s "
                "directly (clean no-history payload)",
                primary[1],
                primary[2],
            )
            description = await _describe_image_via_fallback(
                image_block,
                effective_prompt,
                primary,
            )
            if description:
                _, provider_id, model_id = primary
                header = (
                    f"[Above description produced by primary image "
                    f"model {provider_id}/{model_id}.]"
                )
                image_block_for_ui = await _to_url_form_block(image_block)
                return ToolResponse(
                    content=[
                        TextBlock(type="text", text=description),
                        TextBlock(type="text", text=header),
                        image_block_for_ui,
                    ],
                )
            logger.warning(
                "view_image: primary bypass returned empty for "
                "%s/%s — falling through to inline-block path",
                primary[1],
                primary[2],
            )

        # Inline-block path: keeps the request body small even for
        # big media and dodges providers that choke on inlined
        # payloads (zhipu coding-plan vs. a 39 MB video, observed
        # 2026-05-12).
        image_block = await _to_url_form_block(image_block)
        return ToolResponse(
            content=[
                image_block,
                TextBlock(type="text", text=image_label),
                TextBlock(
                    type="text",
                    text=f"User's question about this image:\n{effective_prompt}",
                ),
            ],
        )

    # Step 3: active model can't see image — try the configured fallback.
    fallback = _resolve_fallback_image_model()
    if fallback is not None:
        effective_prompt = (
            prompt.strip()
            if isinstance(prompt, str) and prompt.strip()
            else _DEFAULT_IMAGE_FALLBACK_PROMPT
        )
        description = await _describe_image_via_fallback(
            image_block,
            effective_prompt,
            fallback,
        )
        if description:
            _, provider_id, model_id = fallback
            # ORDER MATTERS — same rationale as view_video: put the
            # real description FIRST so the agent reads the answer
            # before any normalizer-replaced placeholder for the raw
            # ImageBlock.
            header = (
                f"[Above description produced by fallback image "
                f"model {provider_id}/{model_id}.]"
            )
            return ToolResponse(
                content=[
                    TextBlock(type="text", text=description),
                    TextBlock(type="text", text=header),
                    image_block,
                ],
            )
        # Fallback failed — fall through to the generic hint below.

    # Step 4: no fallback (or fallback itself failed) → generic hint.
    fallback_hint = _get_multimodal_fallback_hint("image", image_path)
    return ToolResponse(
        content=[
            image_block,
            TextBlock(type="text", text=fallback_hint),
        ],
    )


_DEFAULT_VIDEO_FALLBACK_PROMPT = (
    "Describe this video in detail: what happens step by step, any "
    "on-screen text or captions, distinctive objects / people / "
    "locations, and the overall mood.  Be thorough so a model that "
    "cannot see the video can still reason about it from your "
    "description alone."
)

# Provider id prefixes that speak OpenAI-compat /chat/completions but
# need the ``{"type":"video_url","video_url":{"url":...}}`` content
# block shape (matching DashScope / Qwen-VL docs).  agentscope's
# ``OpenAIChatFormatter`` doesn't understand ``video_url`` and silently
# drops it (``Unsupported block type ... skipped.``), so any model on
# one of these providers gets a text-only request unless we bypass
# the formatter with a direct httpx POST.
#
# ``zhipu`` covers ``zhipu-cn-codingplan`` / ``zhipu-intl-codingplan``
# (z.ai coding plan — glm-5v-turbo / GLM-4.6V).  Required as of
# 2026-05-12: A/B replay against a 48-msg production wire confirmed
# that glm-5v-turbo refuses to decode an inline video_url block when
# prior view_video turns sit in conversation history — only a no-
# history payload (same shape as the mimo fallback path) returns a
# real description.  Treating zhipu as Qwen-family lets the existing
# httpx bypass handle the primary path identically.
_QWEN_FAMILY_PREFIXES = (
    "aliyun-",
    "bailian",
    "kimi-",
    "modelscope",
    "mimo",
    "zhipu",
)


def _is_qwen_family(provider_id: str) -> bool:
    p = (provider_id or "").lower()
    return any(p.startswith(x) for x in _QWEN_FAMILY_PREFIXES)


# Markers seen in Mimo's HTTP 400 body when the upload is rejected
# for codec / container reasons.  Both flavours correspond to
# situations a transcode-to-H264-in-MP4 pass can resolve, so we
# treat them as the same "try transcoding once" signal.
_MIMO_FORMAT_REJECTION_MARKERS = (
    "Multimodal data is corrupted",  # AV1 etc. — late decode failure
    "only mp4/wmv/mov/avi",  # webm container — early reject
    "invalid video format",
)


def _is_format_rejection(body_text: str) -> bool:
    """Return True iff Mimo's 400 response body matches one of the
    known codec/container rejection markers.  Other 400s (auth,
    rate-limit, bad request shape) keep the existing behaviour
    (return None, no retry)."""
    text = body_text or ""
    return any(m in text for m in _MIMO_FORMAT_REJECTION_MARKERS)


async def _transcode_to_h264_mp4(src_path: str) -> str | None:
    """Transcode ``src_path`` to H.264 + AAC inside an MP4 container,
    640px wide (height auto), CRF 23 / veryfast.  Returns the new
    path (sibling of ``src_path``) or ``None`` on any failure.

    These knobs are the empirical sweet spot from the 2026-04-25
    AV1-rejection benchmark: ~10s wall on the local machine for an
    8-min 65 MB AV1 source, output ~63 MB H.264-in-MP4 that mimo
    accepts and analyses end-to-end (vs ~2m49s for VP9-in-MP4 which
    is also ~16% bigger and offers no decoder-coverage advantage).
    """
    if not src_path or not os.path.exists(src_path):
        logger.warning(
            "view_video: transcode source missing or unreadable: %s",
            src_path,
        )
        return None
    p = Path(src_path)
    # Distinct suffix so we never overwrite the original; idempotent
    # if the transcoded sibling already exists from a prior run.
    out = p.with_name(p.stem + ".h264.mp4")
    if out.exists() and out.stat().st_size > 0:
        logger.debug(
            "view_video: reusing existing transcode %s",
            out,
        )
        return str(out)
    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(p),
        "-vf",
        "scale=640:-2",
        "-c:v",
        "libx264",
        "-crf",
        "23",
        "-preset",
        "veryfast",
        "-c:a",
        "aac",
        "-b:a",
        "64k",
        str(out),
    ]
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
        )
        _, err = await proc.communicate()
    except FileNotFoundError:
        logger.warning(
            "view_video: ffmpeg binary not on PATH; cannot transcode",
        )
        return None
    if proc.returncode != 0:
        logger.warning(
            "view_video: ffmpeg transcode failed (rc=%s): %s",
            proc.returncode,
            (err or b"").decode("utf-8", errors="replace")[:300],
        )
        # Clean the half-written output so a future call can retry.
        try:
            if out.exists():
                out.unlink()
        except OSError:
            pass
        return None
    return str(out)


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

    # Unknown provider: keep agentscope's VideoBlock shape so the
    # downstream formatter can do its provider-specific translation,
    # but first run the source URL through the media server so
    # local paths become signed HTTPS URLs.  Almost every cloud
    # endpoint we route fallback traffic to (DeepSeek, ZAI, custom
    # OpenAI-compat providers) expects a fetchable URL — handing
    # them a bare local path leaks the path into the request body
    # and the request fails server-side.  ``resolve_media_url``
    # is a no-op for already-HTTP(S) sources so this is safe to
    # call unconditionally.
    resolved_block = video_block
    if url:
        resolved = await resolve_media_url(url)
        if resolved and resolved != url:
            resolved_block = {
                **video_block,
                "source": {**source, "url": resolved},
            }
    return [
        {
            "role": "user",
            "content": [
                resolved_block,
                TextBlock(type="text", text=prompt),
            ],
        },
    ]


def _resolve_primary_vision_model() -> "tuple[Any, str, str] | None":
    """Resolve the agent's *active* model into a ready-to-call
    ``(chat_model, provider_id, model_id)`` triple — same shape as
    :func:`_resolve_fallback_video_model` /
    :func:`_resolve_fallback_image_model`.  Lets the primary-path
    bypass reuse :func:`_describe_video_via_fallback` /
    :func:`_describe_image_via_fallback` (and the transcode-on-
    format-rejection retry for video) unchanged.

    Mirrors the agent-specific resolution in
    :func:`_probe_multimodal_if_needed` — agent ``active_model``
    override wins, then global ``ProviderManager.get_active_model()``.
    """
    try:
        from ...app.agent_context import get_current_agent_id
        from ...config.config import load_agent_config
        from ...providers.provider_manager import ProviderManager

        manager = ProviderManager.get_instance()
        active = None
        try:
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

        provider = manager.get_provider(active.provider_id)
        if provider is None:
            logger.warning(
                "view_media: primary provider '%s' not found",
                active.provider_id,
            )
            return None
        chat_model = provider.get_chat_model_instance(active.model)
        return chat_model, active.provider_id, active.model
    except Exception as e:
        logger.warning(
            "view_media: primary model resolution failed: %s",
            e,
        )
        return None


def _resolve_fallback_video_model() -> "tuple[Any, str, str] | None":
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
            "view_video: fallback model resolution failed: %s",
            e,
        )
        return None


async def _describe_video_via_qwen_family_httpx(
    messages: list[dict],
    chat_model: "object",
    model_id: str,
) -> str | None:
    """Bypass agentscope's ``OpenAIChatFormatter`` and POST the
    OpenAI-compat chat/completions request directly.

    agentscope's formatter only understands a short list of content
    block types (text / image / input_audio / tool_use / tool_result).
    Our Qwen-family video path uses ``{"type": "video_url", ...}`` —
    not on that list — so the formatter silently *drops* the block
    (``Unsupported block type video_url ... skipped.``) and Qwen
    receives only the text prompt, returning an empty / generic reply
    with no video_tokens used.  We call the upstream HTTP endpoint
    directly here to preserve the shape curl-tested against
    ``qwen3.6-plus``: 190k+ video_tokens, full description returned.
    """
    import httpx

    client = getattr(chat_model, "client", None)
    base_url = str(getattr(client, "base_url", "")).rstrip("/")
    api_key = getattr(client, "api_key", None) or ""
    if not base_url:
        logger.warning(
            "view_video: Qwen-family fallback %s has no base_url; "
            "cannot dispatch directly",
            model_id,
        )
        return None
    # The base_url typically already includes ``/v1``; don't double it.
    url = (
        f"{base_url}/chat/completions"
        if "/chat/completions" not in base_url
        else base_url
    )
    body = {"model": model_id, "messages": messages, "stream": False}
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    logger.info(
        "view_video: Qwen httpx POST → %s (model=%s, video_url=%s)",
        url,
        model_id,
        # Extract the URL the server will actually fetch, for debug.
        next(
            (
                (c.get("video_url") or {}).get("url", "?")
                for c in (messages[0].get("content") or [])
                if isinstance(c, dict) and c.get("type") == "video_url"
            ),
            "?",
        )[:120],
    )
    try:
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(300, connect=30),
        ) as hc:
            resp = await hc.post(url, json=body, headers=headers)
    except Exception as e:
        # Network-level failures aren't recoverable here.
        logger.warning(
            "view_video: Qwen fallback httpx call failed: %s",
            e,
        )
        return None

    if resp.status_code != 200:
        logger.warning(
            "view_video: Qwen fallback HTTP %d: %s",
            resp.status_code,
            resp.text[:400],
        )
        # Codec / container rejections are recoverable by
        # transcoding; raise a typed exception so the orchestrator
        # can detect this case and retry once with H.264.  Other
        # non-200s keep the existing ``return None`` semantics.
        if resp.status_code == 400 and _is_format_rejection(resp.text):
            raise _MimoUnsupportedFormatError(resp.text[:400])
        return None

    try:
        j = resp.json()
    except Exception as e:
        logger.warning(
            "view_video: Qwen fallback returned non-JSON body: %s",
            e,
        )
        return None

    choices = j.get("choices") or []
    if not choices:
        logger.warning(
            "view_video: Qwen fallback response missing choices: %s",
            str(j)[:400],
        )
        return None
    msg = (choices[0] or {}).get("message") or {}
    text = msg.get("content") or ""
    result = str(text).strip() or None
    logger.info(
        "view_video: Qwen fallback returned %d chars (usage=%s)",
        len(result or ""),
        j.get("usage"),
    )
    return result


async def _describe_video_via_fallback(
    video_block: VideoBlock,
    prompt: str,
    fallback: "tuple[Any, str, str]",
) -> str | None:
    """One-shot call to the fallback video model.  Returns the text
    description, or ``None`` on failure (the caller substitutes the
    generic multimodal hint in that case).
    """
    chat_model, provider_id, model_id = fallback
    try:
        messages = await _build_fallback_video_messages(
            video_block,
            prompt,
            provider_id,
        )
        if messages is None:
            logger.warning(
                "view_video: cannot format video call for %s/%s "
                "(unknown shape or media-server signing failed); "
                "falling back to generic hint",
                provider_id,
                model_id,
            )
            return None
        logger.info(
            "view_video: delegating to fallback %s/%s (prompt len=%d)",
            provider_id,
            model_id,
            len(prompt),
        )
        # Qwen-family providers need ``video_url`` content blocks
        # that agentscope's OpenAIChatFormatter doesn't understand.
        # Route around the formatter for those — every other
        # provider still goes through agentscope so its native
        # formatter (Gemini etc.) handles translation.
        if _is_qwen_family(provider_id):
            try:
                return await _describe_video_via_qwen_family_httpx(
                    messages,
                    chat_model,
                    model_id,
                )
            except _MimoUnsupportedFormatError as fmt_err:
                # Mimo rejected the codec/container.  Transcode the
                # local file to H.264 + AAC inside MP4 (the safe
                # superset across mimo/qwen-vl) and try once more.
                # We deliberately retry only once — a second
                # rejection means something else is wrong and the
                # user is better served by the placeholder hint
                # than another minute of transcode time.
                local_src = (video_block.get("source") or {}).get("url") or ""
                if not local_src or local_src.startswith(
                    ("http://", "https://", "data:"),
                ):
                    logger.warning(
                        "view_video: %s rejected format (%s) and source "
                        "is remote (%s) — cannot transcode; giving up",
                        model_id,
                        str(fmt_err)[:120],
                        local_src[:80],
                    )
                    return None
                logger.info(
                    "view_video: %s rejected format (%s); transcoding "
                    "%s → H.264-in-MP4 and retrying once",
                    model_id,
                    str(fmt_err)[:120],
                    local_src,
                )
                transcoded = await _transcode_to_h264_mp4(local_src)
                if not transcoded:
                    return None
                retry_block = {
                    **video_block,
                    "source": {
                        **(video_block.get("source") or {}),
                        "url": transcoded,
                    },
                }
                retry_messages = await _build_fallback_video_messages(
                    retry_block,
                    prompt,
                    provider_id,
                )
                if retry_messages is None:
                    return None
                try:
                    return await _describe_video_via_qwen_family_httpx(
                        retry_messages,
                        chat_model,
                        model_id,
                    )
                except _MimoUnsupportedFormatError as second_err:
                    logger.warning(
                        "view_video: transcode retry also rejected by "
                        "%s (%s); giving up",
                        model_id,
                        str(second_err)[:120],
                    )
                    return None
        response = await chat_model(messages)
        # Agentscope chat models can stream (AsyncGenerator) or return
        # a single ChatResponse depending on the ``stream`` init flag.
        # ``get_chat_model_instance`` defaults to ``stream=True``, so
        # we iterate and keep the final cumulative text.
        final_text = ""
        chunk_count = 0
        seen_block_types: set[str] = set()
        if hasattr(response, "__aiter__"):
            async for chunk in response:
                chunk_count += 1
                for block in getattr(chunk, "content", None) or []:
                    if isinstance(block, dict):
                        seen_block_types.add(str(block.get("type", "?")))
                        if block.get("type") == "text":
                            # Streamed text blocks are cumulative in agentscope.
                            final_text = str(
                                block.get("text") or final_text,
                            )
        else:
            for block in getattr(response, "content", None) or []:
                if isinstance(block, dict):
                    seen_block_types.add(str(block.get("type", "?")))
                    if block.get("type") == "text":
                        final_text = str(block.get("text") or final_text)

        result = final_text.strip() or None
        if not result:
            # Silent empty response is the trickiest failure mode —
            # no exception, no text, just a blank from the upstream.
            # Log enough to differentiate it from a real answer being
            # dropped later in the pipeline.
            logger.warning(
                "view_video: fallback %s/%s returned empty "
                "(chunks=%d, block_types=%s) — model may not "
                "actually support video despite supports_video=True",
                provider_id,
                model_id,
                chunk_count,
                sorted(seen_block_types),
            )
        return result
    except Exception as e:
        logger.warning(
            "view_video: fallback %s/%s failed: %s",
            provider_id,
            model_id,
            e,
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
        effective_prompt = (
            prompt.strip()
            if isinstance(prompt, str) and prompt.strip()
            else _DEFAULT_VIDEO_FALLBACK_PROMPT
        )

        # Primary-path HTTP bypass for OpenAI-compat vision providers
        # (qwen-family + zhipu/z.ai).  Call /chat/completions directly
        # with a clean no-history payload — same shape as the mimo
        # fallback path.  Required because these endpoints refuse to
        # decode an inline video_url block when conversation history
        # contains prior view_video tool_call / tool_result pairs:
        # 2026-05-12 A/B replay against a 48-msg production wire
        # confirmed six in-context variants all fail (`tool_call`
        # loop or empty content), only no-history returns a real
        # description.  Reuses :func:`_describe_video_via_fallback`
        # so the format-rejection → transcode → retry path comes
        # for free.  Other providers (Gemini, Claude vision, etc.)
        # fall through to the inline-block path below — their
        # formatters handle multimodal correctly in-context.
        primary = _resolve_primary_vision_model()
        if primary is not None and _is_qwen_family(primary[1]):
            logger.info(
                "view_video: primary bypass — calling %s/%s "
                "directly (clean no-history payload)",
                primary[1],
                primary[2],
            )
            description = await _describe_video_via_fallback(
                video_block,
                effective_prompt,
                primary,
            )
            if description:
                _, provider_id, model_id = primary
                header = (
                    f"[Above description produced by primary video "
                    f"model {provider_id}/{model_id}.]"
                )
                # Resolve to URL form for the user/frontend's copy
                # of the block so big local files don't bloat the
                # tool_result downstream.
                video_block_for_ui = await _to_url_form_block(video_block)
                return ToolResponse(
                    content=[
                        TextBlock(type="text", text=description),
                        TextBlock(type="text", text=header),
                        video_block_for_ui,
                    ],
                )
            logger.warning(
                "view_video: primary bypass returned empty for "
                "%s/%s — falling through to inline-block path",
                primary[1],
                primary[2],
            )

        # Inline-block path: for providers without a known bypass
        # (Gemini, Claude vision, etc.) and as the safety net when
        # the bypass above returned empty.  Normalize to URL form
        # so big local files don't get inlined (see ``view_image``
        # for rationale).
        video_block = await _to_url_form_block(video_block)
        return ToolResponse(
            content=[
                video_block,
                TextBlock(type="text", text=video_label),
                TextBlock(
                    type="text",
                    text=f"User's question about this video:\n{effective_prompt}",
                ),
            ],
        )

    # Step 3: active model can't see video — try the configured fallback.
    fallback = _resolve_fallback_video_model()
    if fallback is not None:
        effective_prompt = (
            prompt.strip()
            if isinstance(prompt, str) and prompt.strip()
            else _DEFAULT_VIDEO_FALLBACK_PROMPT
        )
        description = await _describe_video_via_fallback(
            video_block,
            effective_prompt,
            fallback,
        )
        if description:
            _, provider_id, model_id = fallback
            # ORDER MATTERS.  Agents skim tool_result.output
            # top-to-bottom; if the VideoBlock (which the normalizer
            # downstream replaces with a "[video at X removed —
            # this model cannot process video]" placeholder) sits at
            # index 0, Claude reads the scary placeholder first and
            # concludes the tool failed, even though the real answer
            # follows two slots down.  Observed in production:
            # agent quoted the placeholder verbatim as "return 仲係同樣"
            # despite a full 109-char Qwen description sitting just
            # below it.
            #
            # So put the description FIRST — the agent reads a real
            # answer up front, then the attribution, then (after
            # normalization) a benign note about the raw video block.
            header = (
                f"[Above description produced by fallback video "
                f"model {provider_id}/{model_id}.]"
            )
            return ToolResponse(
                content=[
                    TextBlock(type="text", text=description),
                    TextBlock(type="text", text=header),
                    video_block,
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
