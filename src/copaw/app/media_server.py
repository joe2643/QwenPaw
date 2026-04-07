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

# Runtime-accessible secrets keyed by agent_id so each agent can
# look up its own secret, and multiple agents share one server process.
_runtime_secrets: dict[str, str] = {}



class MediaServer:
    """Embedded media file server with signed URL access."""

    _instance: Optional["MediaServer"] = None

    @classmethod
    def get_or_create(cls, **kwargs) -> "MediaServer":
        """Return existing singleton or create a new instance.

        Registers per-agent allowed_dirs and secrets. Uses reference
        counting so the server stays alive until the last workspace stops.
        """
        agent_id = kwargs.get("agent_id", "default")
        if cls._instance is not None:
            # Register this agent's dirs (scoped, not merged)
            cls._instance._agent_dirs[agent_id] = kwargs.get("allowed_dirs", ["/tmp"])
            _runtime_secrets[agent_id] = kwargs.get("secret", "") or cls._instance.secret
            cls._instance._ref_count += 1
            return cls._instance
        instance = cls(**kwargs)
        instance._agent_dirs[agent_id] = kwargs.get("allowed_dirs", ["/tmp"])
        instance._ref_count = 1
        cls._instance = instance
        return instance

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8089,
        secret: str = "",
        allowed_dirs: Optional[list] = None,
        max_size_mb: int = 100,
        tunnel_domain: str = "",
        agent_id: str = "default",
    ):
        self.host = host
        self.port = port
        self.agent_id = agent_id
        self.secret = secret or os.environ.get("COPAW_MEDIA_SECRET", "")
        self.allowed_dirs = allowed_dirs or ["/tmp"]
        self.max_size = max_size_mb * 1024 * 1024
        self.tunnel_domain = tunnel_domain
        self._server_task: Optional[asyncio.Task] = None
        self._app: Optional[FastAPI] = None
        self._agent_dirs: dict[str, list[str]] = {}  # agent_id -> allowed_dirs
        self._ref_count = 0  # number of workspaces using this server
        self._token_store: dict[str, tuple[str, int, str]] = {}  # token -> (raw_path, expires, agent_id)

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
            auth: str = Query(""),
        ):
            # Resolve which agent this secret belongs to
            caller_agent = None
            for aid, sec in _runtime_secrets.items():
                if sec and hmac.compare_digest(auth, sec):
                    caller_agent = aid
                    break
            if caller_agent is None and hmac.compare_digest(auth, server.secret):
                caller_agent = server.agent_id
            if caller_agent is None:
                raise HTTPException(403, "Unauthorized")
            resolved = Path(path).resolve()
            if not resolved.is_file():
                raise HTTPException(404, "File not found")
            # Check THIS agent's allowed_dirs only
            agent_dirs = server._agent_dirs.get(caller_agent, [])
            if not any(resolved.is_relative_to(Path(d).resolve()) for d in agent_dirs):
                raise HTTPException(403, "Path not in allowed directories for this agent")
            media_exts = {
                ".mp4", ".webm", ".mov", ".avi", ".mkv", ".mpeg",
                ".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp",
                ".mp3", ".wav", ".ogg", ".flac", ".m4a",
            }
            if resolved.suffix.lower() not in media_exts:
                raise HTTPException(400, f"Extension {resolved.suffix} not allowed")
            if resolved.stat().st_size > server.max_size:
                raise HTTPException(413, "File too large")
            # Cap TTL at 24h
            ttl = min(ttl, 86400)
            expires = int(time.time()) + ttl
            raw_path = str(resolved)
            sig = server._sign(raw_path, expires)
            domain = server.tunnel_domain.rstrip("/") if server.tunnel_domain else f"http://{server.host}:{server.port}"
            # Opaque token instead of base64-encoded path (Finding 3)
            token = secrets.token_urlsafe(24)
            server._token_store[token] = (raw_path, expires, caller_agent)
            # Periodic cleanup of expired tokens
            server._cleanup_expired_tokens()
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
            raw_path, stored_exp, agent_id = entry
            if not server._verify(raw_path, exp, sig):
                raise HTTPException(403, "Invalid or expired signature")
            resolved = Path(raw_path).resolve()
            # Check agent-scoped dirs from the token
            agent_dirs = server._agent_dirs.get(agent_id, server.allowed_dirs)
            if not any(resolved.is_relative_to(Path(d).resolve()) for d in agent_dirs):
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

    def _cleanup_expired_tokens(self) -> None:
        """Remove expired entries from the token store to prevent memory leak."""
        now = int(time.time())
        expired = [k for k, v in self._token_store.items() if now > v[1]]
        for k in expired:
            del self._token_store[k]

    def _sign(self, file_path: str, expires: int) -> str:
        msg = f"{file_path}:{expires}"
        return hmac.new(self.secret.encode(), msg.encode(), hashlib.sha256).hexdigest()[:32]

    def _verify(self, file_path: str, expires: int, sig: str) -> bool:
        if time.time() > expires:
            return False
        return hmac.compare_digest(sig, self._sign(file_path, expires))

    async def start(self):
        if not _DEPS_AVAILABLE:
            logger.warning("media-server: fastapi/uvicorn not installed, skipping")
            return
        if not self.secret:
            import secrets as _secrets
            self.secret = _secrets.token_hex(32)
            logger.warning("media-server: no secret configured, generated random secret")
        # Register secret for this agent so _get_media_config() can find it
        _runtime_secrets[self.agent_id] = self.secret
        # Check if port is already bound (another agent started the server)
        import socket as _socket
        _probe = _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM)
        try:
            _probe.bind((self.host, self.port))
            _probe.close()
        except OSError:
            _probe.close()
            logger.info(
                "media-server: port %d already in use, skipping (shared with another agent)",
                self.port,
            )
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
        self._ref_count -= 1
        if self._ref_count > 0:
            logger.info("media-server: ref_count=%d, keeping alive", self._ref_count)
            return  # other workspaces still using it
        # Actually stop -- last reference released
        if self._server_task and not self._server_task.done():
            self._server_task.cancel()
            try:
                await self._server_task
            except (asyncio.CancelledError, Exception):
                pass
        MediaServer._instance = None
        logger.info("media-server: stopped (last reference released)")
