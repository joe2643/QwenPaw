# -*- coding: utf-8 -*-
"""Embedded media file server for QwenPaw.

Runs as a process-level service — starts/stops with the qwenpaw daemon.
Single shared secret, no per-agent complexity.
"""

import asyncio
import hashlib
import hmac
import json
import logging
import mimetypes
import os
import secrets
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

try:
    from fastapi import FastAPI, HTTPException, Query
    from fastapi.responses import FileResponse
    import uvicorn

    _DEPS_AVAILABLE = True
except ImportError:
    _DEPS_AVAILABLE = False

# Single runtime secret (set when server starts, read by _get_media_config)
_runtime_secret: str = ""


class MediaServer:
    """Embedded media file server with signed URL access.

    Process-level singleton — created once at app startup, not per-agent.
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8089,
        secret: str = "",
        allowed_dirs: Optional[list] = None,
        max_size_mb: int = 100,
        tunnel_domain: str = "",
        tunnel_mode: str = "manual",
        named_tunnel_name: str = "",
        named_tunnel_hostname: str = "",
        named_tunnel_config_file: str = "",
        token_store_path: Optional[Path] = None,
    ):
        self.host = host
        self.port = port
        self.secret = secret or os.environ.get("QWENPAW_MEDIA_SECRET", "")
        self.allowed_dirs = allowed_dirs or ["/tmp"]
        self.max_size = max_size_mb * 1024 * 1024
        # user_tunnel_domain is what the operator configured; tunnel_domain is
        # the effective value used when signing URLs (driver URL wins).
        self.user_tunnel_domain = tunnel_domain
        self.tunnel_domain = tunnel_domain
        self.tunnel_mode = tunnel_mode  # "manual" | "quick" | "named"
        self.named_tunnel_name = named_tunnel_name
        self.named_tunnel_hostname = named_tunnel_hostname
        self.named_tunnel_config_file = named_tunnel_config_file
        self._tunnel_driver = None  # CloudflareTunnelDriver, lazily created
        self._server_task: Optional[asyncio.Task] = None
        self._app: Optional[FastAPI] = None
        # Token store is disk-backed so it survives copaw restarts.
        # Without persistence, a restart wipes every token — any
        # signed URL still inside an active conversation history
        # then 403s with "Invalid token" the next time ChatGPT (or
        # any consumer) tries to fetch it, even though the URL's
        # ``exp`` query param hasn't elapsed yet.  Default lives
        # next to the rest of the per-user copaw state under
        # ``WORKING_DIR``; tests pass a tmp path to keep production
        # untouched.
        if token_store_path is None:
            from ..constant import WORKING_DIR
            self._token_store_path: Path = (
                Path(WORKING_DIR).expanduser() / "media_token_store.json"
            )
        else:
            self._token_store_path = Path(token_store_path)
        self._token_store: dict[
            str,
            tuple[str, int],
        ] = self._load_token_store()

    def _create_app(self) -> "FastAPI":
        app = FastAPI(title="QwenPaw Media", docs_url=None, redoc_url=None)
        server = self

        @app.get("/health")
        async def health():
            return {"status": "ok", "service": "qwenpaw-media"}

        @app.get("/sign")
        async def sign_url(
            path: str = Query(...),
            ttl: int = Query(3600),
            auth: str = Query(""),
        ):
            if not hmac.compare_digest(auth, server.secret):
                raise HTTPException(403, "Unauthorized")
            resolved = Path(path).resolve()
            if not resolved.is_file():
                raise HTTPException(404, "File not found")
            if not any(
                resolved.is_relative_to(Path(d).resolve())
                for d in server.allowed_dirs
            ):
                raise HTTPException(403, "Path not in allowed directories")
            media_exts = {
                ".mp4",
                ".webm",
                ".mov",
                ".avi",
                ".mkv",
                ".mpeg",
                ".png",
                ".jpg",
                ".jpeg",
                ".gif",
                ".webp",
                ".bmp",
                ".mp3",
                ".wav",
                ".ogg",
                ".flac",
                ".m4a",
            }
            if resolved.suffix.lower() not in media_exts:
                raise HTTPException(
                    400,
                    f"Extension {resolved.suffix} not allowed",
                )
            if resolved.stat().st_size > server.max_size:
                raise HTTPException(413, "File too large")
            # Cap TTL at 24h
            ttl = min(ttl, 86400)
            expires = int(time.time()) + ttl
            raw_path = str(resolved)
            sig = server._sign(raw_path, expires)
            domain = (
                server.tunnel_domain.rstrip("/")
                if server.tunnel_domain
                else f"http://{server.host}:{server.port}"
            )
            token = secrets.token_urlsafe(24)
            server._token_store[token] = (raw_path, expires)
            server._cleanup_expired_tokens()
            # Persist the new token so a copaw restart doesn't make
            # already-issued URLs fail with "Invalid token".
            server._persist_token_store()
            return {
                "url": f"{domain}/media?t={token}&exp={expires}&sig={sig}",
                "expires": expires,
            }

        @app.get("/media")
        async def serve_media(
            t: str = Query(..., description="Opaque token for file path"),
            exp: int = Query(...),
            sig: str = Query(...),
        ):
            entry = server._token_store.get(t)
            if not entry:
                raise HTTPException(403, "Invalid token")
            raw_path, stored_exp = entry
            if not server._verify(raw_path, exp, sig):
                raise HTTPException(403, "Invalid or expired signature")
            resolved = Path(raw_path).resolve()
            if not any(
                resolved.is_relative_to(Path(d).resolve())
                for d in server.allowed_dirs
            ):
                raise HTTPException(403, "Path not in allowed directories")
            if not resolved.is_file():
                raise HTTPException(404, "File not found")
            media_exts = {
                ".mp4",
                ".webm",
                ".mov",
                ".avi",
                ".mkv",
                ".mpeg",
                ".png",
                ".jpg",
                ".jpeg",
                ".gif",
                ".webp",
                ".bmp",
                ".mp3",
                ".wav",
                ".ogg",
                ".flac",
                ".m4a",
            }
            if resolved.suffix.lower() not in media_exts:
                raise HTTPException(
                    400,
                    f"Extension {resolved.suffix} not allowed",
                )
            if resolved.stat().st_size > server.max_size:
                raise HTTPException(413, "File too large")
            return FileResponse(str(resolved), filename=resolved.name)

        return app

    def _cleanup_expired_tokens(self) -> None:
        """Remove expired entries from the token store to prevent memory leak."""
        now = int(time.time())
        expired = [k for k, v in self._token_store.items() if now > v[1]]
        for k in expired:
            del self._token_store[k]
        if expired:
            # Re-persist after a cleanup so the on-disk file doesn't
            # grow unbounded across restarts.
            self._persist_token_store()

    def _load_token_store(self) -> dict[str, tuple[str, int]]:
        """Read the token store from disk on startup, dropping any
        entries whose ``expires`` already elapsed.  Best-effort:
        a missing or corrupt file just yields an empty store —
        clients with already-issued tokens will 403 once until
        the URL is re-signed, same as the pre-persistence baseline.
        """
        path = self._token_store_path
        if not path.exists():
            return {}
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning(
                "media-server: token store at %s unreadable (%s); "
                "starting with empty store",
                path, e,
            )
            return {}
        if not isinstance(raw, dict):
            return {}
        now = int(time.time())
        loaded: dict[str, tuple[str, int]] = {}
        for token, entry in raw.items():
            try:
                raw_path, expires = entry[0], int(entry[1])
            except Exception:
                continue
            if expires > now:
                loaded[str(token)] = (str(raw_path), expires)
        if loaded:
            logger.info(
                "media-server: restored %d signed-URL token(s) "
                "from %s",
                len(loaded), path,
            )
        return loaded

    def _persist_token_store(self) -> None:
        """Atomically write the in-memory store back to disk via
        ``tmp + os.replace`` so a crash mid-write never leaves an
        empty file."""
        path = self._token_store_path
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(path.suffix + ".tmp")
            payload = {
                token: [raw_path, expires]
                for token, (raw_path, expires) in self._token_store.items()
            }
            tmp.write_text(
                json.dumps(payload, ensure_ascii=False),
                encoding="utf-8",
            )
            os.replace(tmp, path)
        except Exception as e:
            # Persistence is best-effort — falling back to in-memory
            # only is the same behaviour we had before this change,
            # so log and continue rather than failing the sign call.
            logger.warning(
                "media-server: failed to persist token store at %s: %s",
                path, e,
            )

    def _sign(self, file_path: str, expires: int) -> str:
        msg = f"{file_path}:{expires}"
        return hmac.new(
            self.secret.encode(),
            msg.encode(),
            hashlib.sha256,
        ).hexdigest()[:32]

    def _verify(self, file_path: str, expires: int, sig: str) -> bool:
        if time.time() > expires:
            return False
        return hmac.compare_digest(sig, self._sign(file_path, expires))

    async def start(self):
        global _runtime_secret
        if not _DEPS_AVAILABLE:
            logger.warning(
                "media-server: fastapi/uvicorn not installed, skipping",
            )
            return
        if not self.secret:
            self.secret = secrets.token_hex(32)
            logger.warning(
                "media-server: no secret configured, generated random secret",
            )
        _runtime_secret = self.secret
        # Check if port is already bound
        import socket as _socket

        _probe = _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM)
        try:
            _probe.bind((self.host, self.port))
            _probe.close()
        except OSError:
            _probe.close()
            logger.info(
                "media-server: port %d already in use, skipping",
                self.port,
            )
            return
        self._app = self._create_app()
        config = uvicorn.Config(
            self._app,
            host=self.host,
            port=self.port,
            log_level="warning",
        )
        server = uvicorn.Server(config)
        self._server_task = asyncio.create_task(
            server.serve(),
            name="media-server",
        )
        logger.info("media-server: started on %s:%s", self.host, self.port)

        if self.tunnel_mode in ("quick", "named"):
            await self._start_tunnel()

    async def stop(self):
        global _runtime_secret
        await self._stop_tunnel()
        if self._server_task and not self._server_task.done():
            self._server_task.cancel()
            try:
                await self._server_task
            except (asyncio.CancelledError, Exception):
                pass
        _runtime_secret = ""
        logger.info("media-server: stopped")

    async def _start_tunnel(self) -> None:
        """Spawn a Cloudflare Tunnel and update tunnel_domain."""
        if self._tunnel_driver is not None:
            return
        if self.tunnel_mode not in ("quick", "named"):
            return
        try:
            # Import lazily so environments without the tunnel extras still
            # run the plain media server.
            from ..tunnel import CloudflareTunnelDriver
        except Exception as exc:
            logger.warning(
                "media-server: cloudflare tunnel driver unavailable: %s",
                exc,
            )
            return
        try:
            if self.tunnel_mode == "quick":
                driver = CloudflareTunnelDriver(mode="quick")
            else:
                driver = CloudflareTunnelDriver(
                    mode="named",
                    tunnel_name=self.named_tunnel_name,
                    hostname=self.named_tunnel_hostname,
                    config_file=self.named_tunnel_config_file,
                )
        except ValueError as exc:
            # Missing required named-tunnel fields — log and skip so the
            # media server itself keeps serving on localhost.
            logger.error(
                "media-server: cannot start %s tunnel: %s",
                self.tunnel_mode,
                exc,
            )
            return
        try:
            info = await driver.start(self.port)
        except Exception as exc:
            logger.error(
                "media-server: failed to start %s tunnel: %s",
                self.tunnel_mode,
                exc,
            )
            return
        self._tunnel_driver = driver
        self.tunnel_domain = info.public_url
        logger.info(
            "media-server: %s tunnel ready at %s",
            self.tunnel_mode,
            info.public_url,
        )

    async def _stop_tunnel(self) -> None:
        """Stop the managed tunnel (if any) and restore the user's domain."""
        driver = self._tunnel_driver
        if driver is None:
            return
        self._tunnel_driver = None
        try:
            await driver.stop()
        except Exception as exc:
            logger.warning("media-server: error stopping tunnel: %s", exc)
        self.tunnel_domain = self.user_tunnel_domain

    async def reconcile_tunnel(
        self,
        tunnel_mode: str,
        named_tunnel_name: str = "",
        named_tunnel_hostname: str = "",
        named_tunnel_config_file: str = "",
    ) -> None:
        """Hot-switch tunnel mode without recreating the MediaServer.

        The PUT /config/media-server handler calls this after saving new
        config so toggling the Console UI produces immediate effect.
        """
        same_mode = tunnel_mode == self.tunnel_mode
        same_named = tunnel_mode != "named" or (
            named_tunnel_name == self.named_tunnel_name
            and named_tunnel_hostname == self.named_tunnel_hostname
            and named_tunnel_config_file == self.named_tunnel_config_file
        )
        if same_mode and same_named:
            # No state change — leave the (possibly running) driver alone.
            return

        # Stop any existing tunnel before applying new settings, otherwise we
        # could have two cloudflared subprocesses fighting over the same port.
        await self._stop_tunnel()
        self.tunnel_mode = tunnel_mode
        self.named_tunnel_name = named_tunnel_name
        self.named_tunnel_hostname = named_tunnel_hostname
        self.named_tunnel_config_file = named_tunnel_config_file

        if tunnel_mode in ("quick", "named"):
            await self._start_tunnel()

    def get_tunnel_url(self) -> str:
        """Return the current managed tunnel URL, or '' if none is running."""
        driver = self._tunnel_driver
        if driver is None:
            return ""
        return driver.get_public_url() or ""
