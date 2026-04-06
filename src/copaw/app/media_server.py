"""Embedded media file server for CoPaw.

Runs as a workspace service — starts/stops with the copaw daemon.
No separate systemd service needed.
"""

import asyncio
import hashlib
import hmac
import logging
import mimetypes
import os
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


class MediaServer:
    """Embedded media file server with signed URL access."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8089,
        secret: str = "",
        allowed_dirs: Optional[list] = None,
        max_size_mb: int = 100,
        tunnel_domain: str = "",
    ):
        self.host = host
        self.port = port
        self.secret = secret or os.environ.get("COPAW_MEDIA_SECRET", "")
        self.allowed_dirs = allowed_dirs or ["/tmp"]
        self.max_size = max_size_mb * 1024 * 1024
        self.tunnel_domain = tunnel_domain
        self._server_task: Optional[asyncio.Task] = None
        self._app: Optional[FastAPI] = None

    def _create_app(self) -> "FastAPI":
        app = FastAPI(title="CoPaw Media", docs_url=None, redoc_url=None)
        server = self

        @app.get("/health")
        async def health():
            return {"status": "ok", "service": "copaw-media"}

        @app.get("/sign")
        async def sign_url(
            path: str = Query(...),
            ttl: int = Query(3600),
        ):
            resolved = Path(path).resolve()
            if not resolved.is_file():
                raise HTTPException(404, "File not found")
            expires = int(time.time()) + ttl
            sig = server._sign(str(resolved), expires)
            domain = server.tunnel_domain.rstrip("/") if server.tunnel_domain else f"http://{server.host}:{server.port}"
            return {
                "url": f"{domain}/media?path={resolved}&exp={expires}&sig={sig}",
                "expires": expires,
            }

        @app.get("/media")
        async def serve_media(
            path: str = Query(...),
            exp: int = Query(...),
            sig: str = Query(...),
        ):
            if not server._verify(path, exp, sig):
                raise HTTPException(403, "Invalid or expired signature")
            resolved = Path(path).resolve()
            if not any(str(resolved).startswith(d) for d in server.allowed_dirs):
                raise HTTPException(403, "Path not in allowed directories")
            if not resolved.is_file():
                raise HTTPException(404, "File not found")
            media_exts = {
                ".mp4", ".webm", ".mov", ".avi", ".mkv", ".mpeg",
                ".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp",
                ".mp3", ".wav", ".ogg", ".flac", ".m4a",
            }
            if resolved.suffix.lower() not in media_exts:
                raise HTTPException(400, f"Extension {resolved.suffix} not allowed")
            if resolved.stat().st_size > server.max_size:
                raise HTTPException(413, "File too large")
            return FileResponse(str(resolved), filename=resolved.name)

        return app

    def _sign(self, file_path: str, expires: int) -> str:
        msg = f"{file_path}:{expires}"
        return hmac.new(self.secret.encode(), msg.encode(), hashlib.sha256).hexdigest()[:16]

    def _verify(self, file_path: str, expires: int, sig: str) -> bool:
        if time.time() > expires:
            return False
        return hmac.compare_digest(sig, self._sign(file_path, expires))

    async def start(self):
        if not _DEPS_AVAILABLE:
            logger.warning("media-server: fastapi/uvicorn not installed, skipping")
            return
        if not self.secret:
            logger.warning("media-server: no secret configured, skipping")
            return
        self._app = self._create_app()
        config = uvicorn.Config(
            self._app, host=self.host, port=self.port,
            log_level="warning",
        )
        server = uvicorn.Server(config)
        self._server_task = asyncio.create_task(server.serve(), name="media-server")
        logger.info("media-server: started on %s:%s", self.host, self.port)

    async def stop(self):
        if self._server_task and not self._server_task.done():
            self._server_task.cancel()
            try:
                await self._server_task
            except (asyncio.CancelledError, Exception):
                pass
            logger.info("media-server: stopped")
