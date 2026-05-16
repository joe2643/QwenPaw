# -*- coding: utf-8 -*-
# pylint: disable=too-many-return-statements,too-many-branches
"""Grok Imagine (Aurora) image generation + editing tools.

Two tools, mirroring upstream's gpt-image2 plugin convention:

- :func:`generate_image_grok` — text-to-image via ``POST /v1/images/generations``
- :func:`edit_image_grok` — image-to-image / multi-image edit via
  ``POST /v1/images/edits``

Credentials are resolved in this priority order for both tools:

  1. The plugin's per-agent ``api_key`` config (if the user pasted one).
  2. The xAI OAuth bearer from ``~/.xai/auth.json`` via :class:`XaiAuth`.
  3. The ``XAI_API_KEY`` environment variable.

Source images for ``edit_image_grok`` accept three forms:
  - http(s) URL (passed straight to xAI)
  - ``data:image/...;base64,...`` URI
  - local filesystem path (read off disk + inline base64; works even when
    qwenpaw isn't publicly reachable, capped at 20 MB pre-encoding).
"""

import base64
import logging
import mimetypes
import os
import time
from pathlib import Path
from typing import List, Optional

import httpx
from agentscope.message import ImageBlock, TextBlock
from agentscope.tool import ToolResponse

from qwenpaw.constant import DEFAULT_MEDIA_DIR

logger = logging.getLogger(__name__)

# Cap on raw bytes for an inline-base64 source image.  Big enough for
# typical phone-camera shots (~5-10 MB) with headroom, capped before
# base64 inflation pushes the request body past xAI's accepted range
# (undocumented, but ~30 MB requests have been observed to 413 in
# the wild).  A user with a genuinely huge PSD should resize first.
_MAX_LOCAL_IMAGE_BYTES = 20 * 1024 * 1024

# Quality is xAI's recommended default since 2026-05-15 (grok-imagine-image-pro
# was deprecated on that date).  Plain ``grok-imagine-image`` still works but
# is the fast/cheap tier.
DEFAULT_MODEL = "grok-imagine-image-quality"
# xAI splits the surface across two paths: ``/generations`` for pure
# text-to-image, ``/edits`` for image-to-image and multi-image edit.
# Sending ``image`` to ``/generations`` does NOT 4xx — the server
# silently ignores the field and runs a fresh text-to-image, which is
# why earlier "i2i" outputs had no visual relation to the input.
DEFAULT_BASE_URL = "https://api.x.ai/v1"
DEFAULT_ENDPOINT_GENERATE = f"{DEFAULT_BASE_URL}/images/generations"
DEFAULT_ENDPOINT_EDIT = f"{DEFAULT_BASE_URL}/images/edits"
DEFAULT_TIMEOUT_S = 120.0

# xAI's public docs (as of 2026-05) cap multi-image edit at 3 source
# images, but the real limit may differ per account tier and xAI has
# shipped limit increases without doc updates before.  We log a warning
# above this threshold rather than hard-clipping, so future limit bumps
# work without a client release — if the request 400s, the server's
# message is the source of truth.
DOCUMENTED_MAX_REFERENCE_IMAGES = 3

# xAI accepts the X.com canonical ratios; we expose the friendlier
# {landscape, square, portrait} triad to the LLM and map here.  The
# agent has trouble emitting exact ratios reliably (e.g. "16:9" vs
# "16x9") so giving it an enum keeps the call valid.
_ASPECT_MAP = {
    "landscape": "16:9",
    "square": "1:1",
    "portrait": "9:16",
}
_VALID_RESOLUTIONS = {"1k", "2k"}


# ============================================================================
# Public tools
# ============================================================================


async def generate_image_grok(
    prompt: str,
    aspect_ratio: str = "landscape",
    resolution: Optional[str] = None,
) -> ToolResponse:
    """Generate a new image with xAI Grok Imagine (Aurora).

    Text-to-image only.  For editing an existing image, use
    :func:`edit_image_grok` instead — it hits a different xAI endpoint
    (``/v1/images/edits``) that actually references the source.

    Args:
        prompt (str):
            Text description of the image.  Be specific — Aurora rewards
            detailed prompts with composition, lighting, and style cues.
        aspect_ratio (str, optional):
            One of "landscape" (16:9), "square" (1:1), "portrait" (9:16).
            Defaults to "landscape".
        resolution (str, optional):
            "1k" or "2k".  Defaults to the plugin's configured resolution
            (which itself defaults to "1k").  "2k" requires a Premium+
            xAI subscription and costs more.

    Returns:
        ToolResponse: Contains the generated image (URL or local path)
        and a text summary.
    """
    try:
        cfg = _get_tool_config("generate_image_grok") or {}

        if aspect_ratio not in _ASPECT_MAP:
            return _text_error(
                f"Invalid aspect_ratio '{aspect_ratio}'. "
                f"Must be one of: {', '.join(_ASPECT_MAP)}",
            )
        effective_resolution = (
            (resolution or cfg.get("resolution") or "1k").strip().lower()
        )
        if effective_resolution not in _VALID_RESOLUTIONS:
            return _text_error(
                f"Invalid resolution '{effective_resolution}'. "
                f"Must be one of: {', '.join(sorted(_VALID_RESOLUTIONS))}",
            )

        model = (cfg.get("model") or DEFAULT_MODEL).strip()
        endpoint = (
            (cfg.get("endpoint") or "").strip() or DEFAULT_ENDPOINT_GENERATE
        )
        timeout = _coerce_timeout(cfg.get("timeout"))

        bearer, source = await _resolve_bearer(cfg)
        if not bearer:
            return _missing_creds_error()

        payload = {
            "model": model,
            "prompt": prompt,
            "n": 1,
            "aspect_ratio": _ASPECT_MAP[aspect_ratio],
            "resolution": effective_resolution,
        }

        logger.info(
            "[grok-image] generate model=%s ratio=%s res=%s creds=%s",
            model,
            _ASPECT_MAP[aspect_ratio],
            effective_resolution,
            source,
        )

        return await _post_and_render(
            endpoint=endpoint,
            payload=payload,
            bearer=bearer,
            timeout=timeout,
            mode_label="text-to-image",
            prompt_echo=prompt,
            extras=(
                f"ratio={aspect_ratio}, res={effective_resolution}, "
                f"creds={source}"
            ),
        )

    except httpx.TimeoutException:
        return _text_error("Image generation timed out. Please try again.")
    except Exception as e:
        logger.error("[grok-image] generate failed: %s", e, exc_info=True)
        return _text_error(f"Image generation failed: {e}")


async def edit_image_grok(
    prompt: str,
    image_url: Optional[str] = None,
    reference_image_urls: Optional[List[str]] = None,
) -> ToolResponse:
    """Edit or compose images with xAI Grok Imagine (Aurora).

    Hits ``POST /v1/images/edits`` with one or more source images.

    Two sub-modes, picked automatically:

    - **Single-source edit**: just ``prompt`` + ``image_url``.  Grok
      transforms the supplied image while preserving relevant context
      (subject identity, layout, etc.).
    - **Multi-image edit**: ``prompt`` + ``reference_image_urls``.  Use
      for combining subjects, transferring styles, or composing scenes
      from references.

    Source-image inputs (both ``image_url`` and ``reference_image_urls``
    entries) accept three forms:

    - http(s) URL (passed straight to xAI)
    - ``data:image/...;base64,...`` URI
    - local filesystem path (read off disk + inline base64; works even
      when qwenpaw isn't publicly reachable, capped at 20 MB pre-encoding)

    Args:
        prompt (str):
            Edit instruction.  Describe the change you want Grok to apply
            while preserving relevant context from the source image(s).
        image_url (str, optional):
            Single source image for i2i editing.  Ignored if
            ``reference_image_urls`` is also supplied (use one or the
            other, not both).
        reference_image_urls (list[str], optional):
            Multiple source images for multi-image editing.  xAI's
            documented limit is 3 per request; we pass more through with
            a warning rather than clipping, so any future server-side
            bump works without a client release.

    Returns:
        ToolResponse: Contains the generated image (URL or local path)
        and a text summary.  Note: xAI charges per input image as well
        as per output image in edit modes.
    """
    try:
        if not image_url and not reference_image_urls:
            return _text_error(
                "edit_image_grok requires at least one source image "
                "(image_url or reference_image_urls).  For pure text-to-image, "
                "use generate_image_grok instead.",
            )

        cfg = _get_tool_config("edit_image_grok") or {}
        model = (cfg.get("model") or DEFAULT_MODEL).strip()
        endpoint = (
            (cfg.get("endpoint") or "").strip() or DEFAULT_ENDPOINT_EDIT
        )
        timeout = _coerce_timeout(cfg.get("timeout"))

        bearer, source = await _resolve_bearer(cfg)
        if not bearer:
            return _missing_creds_error()

        payload: dict = {
            "model": model,
            "prompt": prompt,
            "n": 1,
        }
        try:
            refs = [u for u in (reference_image_urls or []) if u]
            if refs:
                if len(refs) > DOCUMENTED_MAX_REFERENCE_IMAGES:
                    logger.warning(
                        "[grok-image] %d reference images > documented xAI "
                        "limit of %d; passing through anyway — server will "
                        "400 if your account/model rejects it.",
                        len(refs),
                        DOCUMENTED_MAX_REFERENCE_IMAGES,
                    )
                payload["image"] = [_to_image_payload(u) for u in refs]
                mode_label = f"multi-image-edit({len(refs)})"
            else:
                payload["image"] = _to_image_payload(image_url)  # type: ignore[arg-type]
                mode_label = "image-to-image"
        except _SourceImageError as e:
            return _text_error(str(e))

        logger.info(
            "[grok-image] edit model=%s mode=%s creds=%s",
            model,
            mode_label,
            source,
        )

        return await _post_and_render(
            endpoint=endpoint,
            payload=payload,
            bearer=bearer,
            timeout=timeout,
            mode_label=mode_label,
            prompt_echo=prompt,
            extras=f"creds={source}",
        )

    except httpx.TimeoutException:
        return _text_error("Image edit timed out. Please try again.")
    except Exception as e:
        logger.error("[grok-image] edit failed: %s", e, exc_info=True)
        return _text_error(f"Image edit failed: {e}")


# ============================================================================
# Shared helpers
# ============================================================================


class _SourceImageError(ValueError):
    """Raised when an i2i source image can't be turned into a payload.

    Caught by ``edit_image_grok`` and surfaced as a tool-level error so
    the agent gets a readable message instead of a 500.
    """


def _to_image_payload(src: str) -> dict:
    """Normalize an i2i source — URL, data URI, or local path — into
    the ``{"url": ..., "type": "image_url"}`` shape xAI expects.

    Local paths are read off disk, base64-encoded, and wrapped in a
    ``data:`` URI; this works regardless of network reachability (the
    caller's qwenpaw doesn't need to be publicly addressable for xAI
    to fetch the source).  http(s) URLs and pre-built data URIs pass
    through unchanged.

    Raises ``_SourceImageError`` for missing files, files past the
    inline-size cap, or unrecognised inputs.
    """
    if not src or not isinstance(src, str):
        raise _SourceImageError(f"empty / non-string image source: {src!r}")

    if src.startswith(("http://", "https://", "data:")):
        return {"url": src, "type": "image_url"}

    path = Path(os.path.expanduser(src))
    if not path.exists():
        raise _SourceImageError(
            f"image source not found on disk: {path} "
            f"(provide an http(s) URL, a data: URI, or an existing local path)",
        )
    if not path.is_file():
        raise _SourceImageError(f"image source is not a file: {path}")

    size = path.stat().st_size
    if size > _MAX_LOCAL_IMAGE_BYTES:
        raise _SourceImageError(
            f"image {path} is {size/1024/1024:.1f} MB; cap is "
            f"{_MAX_LOCAL_IMAGE_BYTES/1024/1024:.0f} MB for inline upload — "
            f"resize before retrying.",
        )

    mime, _ = mimetypes.guess_type(str(path))
    if not mime or not mime.startswith("image/"):
        # Fall back to png — xAI tolerates a wrong subtype as long as the
        # bytes are a real image, and guess_type misses HEIC/AVIF on some
        # systems where mimetypes lacks those mappings.
        mime = "image/png"

    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return {
        "url": f"data:{mime};base64,{encoded}",
        "type": "image_url",
    }


async def _post_and_render(
    *,
    endpoint: str,
    payload: dict,
    bearer: str,
    timeout: float,
    mode_label: str,
    prompt_echo: str,
    extras: str,
) -> ToolResponse:
    """Issue the HTTP POST + render the ToolResponse.  Shared by both tools."""
    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(
            endpoint,
            headers={
                "Authorization": f"Bearer {bearer}",
                "Content-Type": "application/json",
            },
            json=payload,
        )

    if resp.status_code != 200:
        return _text_error(_format_http_error(resp))

    data = resp.json()
    try:
        row = data["data"][0]
    except (KeyError, IndexError, TypeError):
        return _text_error(
            f"xAI returned unexpected response shape: {data!r}",
        )

    image_path: Optional[str] = None
    b64 = row.get("b64_json")
    url = row.get("url")
    if b64:
        image_path = _save_b64(b64)
    elif url:
        # xAI's CDN URLs (imgen.x.ai/...) are signed/short-lived and
        # chat channels (Signal, etc.) need a real local file to
        # attach — return path, not URL.
        image_path = await _download_url(url)
    if not image_path:
        return _text_error(
            "xAI response contained neither b64_json nor url",
        )

    model = payload.get("model", "grok-imagine-image-quality")
    return ToolResponse(
        content=[
            ImageBlock(
                type="image",
                source={"type": "url", "url": str(image_path)},
            ),
            TextBlock(
                type="text",
                text=(
                    f"Generated image with {model} "
                    f"(mode={mode_label}, {extras})\n"
                    f"Prompt: {prompt_echo}\n"
                    f"Saved to: {image_path}"
                ),
            ),
        ],
    )


def _text_error(msg: str) -> ToolResponse:
    return ToolResponse(
        content=[TextBlock(type="text", text=f"Error: {msg}")],
    )


def _missing_creds_error() -> ToolResponse:
    return _text_error(
        "No xAI credentials available. Run `qwenpaw xai login` "
        "or set XAI_API_KEY in your environment, or paste a key "
        "into the plugin config.",
    )


def _format_http_error(resp: httpx.Response) -> str:
    msg = f"xAI API error: HTTP {resp.status_code}"
    try:
        body = resp.json()
        if isinstance(body, dict):
            inner = body.get("error")
            if isinstance(inner, dict):
                msg += f" — {inner.get('message')}"
            elif isinstance(inner, str):
                msg += f" — {inner}"
            elif body.get("message"):
                msg += f" — {body['message']}"
    except Exception:
        snippet = resp.text[:200].strip()
        if snippet:
            msg += f" — {snippet}"
    if resp.status_code == 403:
        msg += (
            "  (403 typically means your xAI account lacks an active "
            "Premium+ or SuperGrok subscription required for image "
            "generation.)"
        )
    return msg


def _coerce_timeout(raw) -> float:
    """Per-tool timeout override; caps to the default when missing."""
    if isinstance(raw, (int, float)) and raw > 0:
        return float(raw)
    return DEFAULT_TIMEOUT_S


async def _resolve_bearer(cfg: dict) -> tuple[Optional[str], str]:
    """Return ``(bearer_token, source_label)``.

    Search order: plugin config api_key → XaiAuth OAuth file →
    XAI_API_KEY env.  First hit wins so users can deliberately override
    OAuth with a session key for ad-hoc testing.
    """
    user_key = (cfg.get("api_key") or "").strip()
    if user_key:
        return user_key, "plugin-config"

    try:
        from qwenpaw.providers.xai_auth import XaiAuth

        auth = XaiAuth()
        creds = await auth.ensure_fresh()
        return creds.access_token, "oauth"
    except FileNotFoundError:
        pass
    except Exception as e:
        logger.debug("[grok-image] XaiAuth unavailable: %s", e)

    env_key = (os.environ.get("XAI_API_KEY") or "").strip()
    if env_key:
        return env_key, "env"
    return None, "none"


def _save_b64(b64: str) -> str:
    media_dir = DEFAULT_MEDIA_DIR / "grok_image"
    media_dir.mkdir(parents=True, exist_ok=True)
    timestamp = int(time.time() * 1000)
    path = media_dir / f"grok_image_{timestamp}.png"
    path.write_bytes(base64.b64decode(b64))
    return str(path)


async def _download_url(url: str) -> str:
    media_dir = DEFAULT_MEDIA_DIR / "grok_image"
    media_dir.mkdir(parents=True, exist_ok=True)
    timestamp = int(time.time() * 1000)
    ext = ".jpg"
    for candidate in (".jpeg", ".jpg", ".png", ".webp"):
        if candidate in url.lower():
            ext = candidate
            break
    path = media_dir / f"grok_image_{timestamp}{ext}"
    async with httpx.AsyncClient(timeout=120.0) as client:
        async with client.stream("GET", url) as resp:
            resp.raise_for_status()
            with path.open("wb") as f:
                async for chunk in resp.aiter_bytes():
                    f.write(chunk)
    return str(path)


def _get_tool_config(tool_name: str) -> Optional[dict]:
    try:
        from qwenpaw.app.agent_context import get_current_agent_id
        from qwenpaw.plugins.registry import PluginRegistry

        registry = PluginRegistry()
        if not registry:
            return None
        agent_id = get_current_agent_id()
        if not agent_id:
            return None
        return registry.get_tool_config(tool_name, agent_id)
    except Exception as e:
        logger.debug("[grok-image] _get_tool_config(%s): %s", tool_name, e)
        return None
