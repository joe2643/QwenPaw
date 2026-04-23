# -*- coding: utf-8 -*-
"""Shared media URL utilities for channels.

Channels and tools call :func:`resolve_media_url` to turn a local media
path into whatever string the formatter / agent should see.  When the
QwenPaw media server is running (with or without a Cloudflare tunnel)
the function asks it to sign the path and returns a public (or loopback)
HTTPS URL that remote LLM endpoints can fetch without base64-encoding
the whole file.  When the server is unreachable or refuses the path
(outside its allowed directories), we fall back to the raw local path —
same as the historic behaviour, so existing callers never regress.

Single indirection point: view_video's fallback-model path uses this
too, keeping signing logic in one place.
"""

import logging
import os
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

# Env override for atypical deploys / tests.  Defaults target the
# local media server at its conventional port.
_MEDIA_SERVER_URL_ENV = "QWENPAW_MEDIA_SERVER_URL"
_MEDIA_SERVER_DEFAULT = "http://127.0.0.1:8089"
_DEFAULT_SIGN_TTL_S = 3600


def _media_server_base() -> str:
    return (os.environ.get(_MEDIA_SERVER_URL_ENV)
            or _MEDIA_SERVER_DEFAULT).rstrip("/")


async def sign_media_path(
    local_path: str,
    ttl_s: int = _DEFAULT_SIGN_TTL_S,
    auth: Optional[str] = None,
    timeout: float = 5.0,
) -> Optional[str]:
    """Ask the media server to sign an absolute file path and return a
    URL that fits within the server's tunnel policy (public HTTPS
    when a Cloudflare tunnel is active; ``http://127.0.0.1:8089/...``
    loopback otherwise).

    Returns ``None`` when the server isn't reachable, refuses the
    path (outside allowed directories, too large, wrong extension),
    or any other HTTP error — callers can then decide whether to
    fall through to the raw local path or abort.
    """
    if not local_path:
        return None
    try:
        params: dict = {"path": str(local_path), "ttl": ttl_s}
        if auth:
            params["auth"] = auth
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.get(
                f"{_media_server_base()}/sign", params=params,
            )
        if resp.status_code != 200:
            logger.debug(
                "sign_media_path: media_server /sign returned %d for %s: %s",
                resp.status_code, local_path, resp.text[:200],
            )
            return None
        body = resp.json()
        url = body.get("url")
        return url if isinstance(url, str) and url else None
    except Exception as e:
        logger.debug(
            "sign_media_path: media_server unreachable for %s: %s",
            local_path, e,
        )
        return None


async def resolve_media_url(local_path: str) -> str:
    """Return the media URL the agent should see for ``local_path``.

    Strategy:

    1. If the input already looks like an HTTP(S) or data URL, return
       it unchanged — nothing to sign.
    2. Otherwise try :func:`sign_media_path`.  If the media server
       returns a URL, use it (the server's tunnel config decides
       whether it's public).
    3. On any failure, return the raw local path — maintains the
       pre-signing behaviour so existing callers don't regress.

    The caller can still tell a remote URL from a local path by
    checking the ``http://`` / ``https://`` prefix on the return
    value, which is what :func:`view_video`'s Qwen-family path does
    to decide whether to route through the fallback model.
    """
    if not local_path:
        return str(local_path)
    text = str(local_path)

    # Already a URL or data URL — passthrough.
    if text.startswith(("http://", "https://", "data:")):
        return text

    # Signing is best-effort.  Back-compat: missing file still
    # returns the path (channels may upload it through other means
    # before the request goes out).
    if not os.path.exists(text):
        logger.debug("resolve_media_url: path does not exist: %s", text)
        return text

    signed = await sign_media_path(text)
    return signed or text
