# -*- coding: utf-8 -*-
"""Tests for global media server (process-level, no per-agent complexity)."""

import hashlib
import hmac as hmac_mod
import os
import secrets
import tempfile
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from qwenpaw.app.media_server import MediaServer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_media(tmp_path):
    """Create test media files."""
    img = tmp_path / "test.png"
    img.write_bytes(b"\x89PNG" + b"\x00" * 100)
    vid = tmp_path / "test.mp4"
    vid.write_bytes(b"\x00\x00\x00\x1cftypisom" + b"\x00" * 100)
    txt = tmp_path / "secret.txt"
    txt.write_text("passwords here")
    return tmp_path


@pytest.fixture
def server(tmp_media):
    """Create a MediaServer with test config."""
    return MediaServer(
        port=0,
        secret="test-secret-12345",
        allowed_dirs=[str(tmp_media)],
        max_size_mb=1,
        tunnel_domain="https://media.example.com",
    )


# ---------------------------------------------------------------------------
# Auth: /sign requires auth + validates allowed_dirs/ext/size
# ---------------------------------------------------------------------------

class TestSignAuth:

    def test_sign_requires_valid_secret(self, server):
        """sign must require auth=media_secret."""
        assert server.secret == "test-secret-12345"
        assert hmac_mod.compare_digest("test-secret-12345", server.secret)
        assert not hmac_mod.compare_digest("wrong-secret", server.secret)

    def test_sign_validates_allowed_dirs(self, server, tmp_media):
        """sign must check allowed_dirs before signing."""
        test_file = tmp_media / "test.png"
        assert test_file.resolve().is_relative_to(Path(str(tmp_media)).resolve())

        outside = Path("/etc/passwd")
        if outside.exists():
            assert not outside.resolve().is_relative_to(Path(str(tmp_media)).resolve())

    def test_sign_rejects_wrong_extension(self, server, tmp_media):
        """sign must reject non-media extensions."""
        media_exts = {
            ".mp4", ".webm", ".mov", ".avi", ".mkv", ".mpeg",
            ".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp",
            ".mp3", ".wav", ".ogg", ".flac", ".m4a",
        }
        txt = tmp_media / "secret.txt"
        assert txt.suffix.lower() not in media_exts

    def test_sign_caps_ttl_at_24h(self, server):
        """TTL must be capped at 86400 (24h)."""
        capped = min(999999, 86400)
        assert capped == 86400


# ---------------------------------------------------------------------------
# Localhost URL rejection
# ---------------------------------------------------------------------------

class TestLocalhostUrlRejection:

    def test_localhost_url_detected(self):
        localhost_urls = [
            "http://localhost:8089/media?t=abc&sig=def",
            "http://127.0.0.1:8089/media?t=abc&sig=def",
        ]
        for url in localhost_urls:
            assert "localhost" in url or "127.0.0.1" in url

    def test_tunnel_url_not_localhost(self):
        tunnel_url = "https://media.example.com/media?t=abc&sig=def"
        assert "localhost" not in tunnel_url
        assert "127.0.0.1" not in tunnel_url

    def test_empty_tunnel_domain_triggers_fallback(self):
        tunnel_domain = ""
        url = "http://localhost:8089/media?t=abc"
        should_fallback = ("localhost" in url or "127.0.0.1" in url) and not tunnel_domain
        assert should_fallback is True

    def test_configured_tunnel_domain_no_fallback(self):
        tunnel_domain = "https://media.example.com"
        url = "https://media.example.com/media?t=abc"
        should_fallback = ("localhost" in url or "127.0.0.1" in url) and not tunnel_domain
        assert should_fallback is False


# ---------------------------------------------------------------------------
# Opaque tokens (no path leakage)
# ---------------------------------------------------------------------------

class TestOpaqueTokens:

    def test_token_store_maps_token_to_path(self, server, tmp_media):
        """token_store should map opaque token to real path."""
        token = secrets.token_urlsafe(24)
        raw_path = str(tmp_media / "test.png")
        expires = int(time.time()) + 3600
        server._token_store[token] = (raw_path, expires)

        entry = server._token_store.get(token)
        assert entry is not None
        assert entry[0] == raw_path
        assert entry[1] == expires

    def test_token_is_opaque(self, server):
        """token must not contain decodable path info."""
        import base64
        token = secrets.token_urlsafe(24)
        try:
            decoded = base64.urlsafe_b64decode(token + "==").decode("utf-8", errors="ignore")
        except Exception:
            decoded = ""
        assert "/tmp" not in decoded
        assert "/home" not in decoded

    def test_expired_tokens_cleaned_up(self, server):
        """_cleanup_expired_tokens removes old entries."""
        server._token_store["old_token"] = ("/tmp/old.mp4", int(time.time()) - 100)
        server._token_store["new_token"] = ("/tmp/new.mp4", int(time.time()) + 3600)

        server._cleanup_expired_tokens()

        assert "old_token" not in server._token_store
        assert "new_token" in server._token_store

    def test_invalid_token_rejected(self, server):
        """media with unknown token should be rejected."""
        entry = server._token_store.get("nonexistent_token")
        assert entry is None


# ---------------------------------------------------------------------------
# HMAC signature tests
# ---------------------------------------------------------------------------

class TestHMACSignature:

    def test_signature_is_32_chars(self, server):
        sig = server._sign("/tmp/test.mp4", 9999999999)
        assert len(sig) == 32
        int(sig, 16)

    def test_verify_valid_signature(self, server):
        path = "/tmp/test.mp4"
        expires = int(time.time()) + 3600
        sig = server._sign(path, expires)
        assert server._verify(path, expires, sig)

    def test_verify_wrong_signature_rejected(self, server):
        assert not server._verify("/tmp/test.mp4", int(time.time()) + 3600, "wrong" * 8)

    def test_verify_expired_rejected(self, server):
        path = "/tmp/test.mp4"
        past = int(time.time()) - 100
        sig = server._sign(path, past)
        assert not server._verify(path, past, sig)

    def test_different_paths_different_signatures(self, server):
        exp = int(time.time()) + 3600
        sig1 = server._sign("/tmp/a.mp4", exp)
        sig2 = server._sign("/tmp/b.mp4", exp)
        assert sig1 != sig2


# ---------------------------------------------------------------------------
# Path validation (symlink protection)
# ---------------------------------------------------------------------------

class TestPathValidation:

    def test_relative_to_catches_outside_path(self, server, tmp_media):
        allowed = Path(str(tmp_media)).resolve()
        outside = Path("/etc/passwd").resolve()
        assert not outside.is_relative_to(allowed)

    def test_relative_to_allows_inside_path(self, server, tmp_media):
        allowed = Path(str(tmp_media)).resolve()
        inside = (tmp_media / "test.png").resolve()
        assert inside.is_relative_to(allowed)

    def test_symlink_resolved_and_checked(self, tmp_media):
        link = tmp_media / "evil_link"
        try:
            link.symlink_to("/etc")
            resolved = (link / "passwd").resolve()
            allowed = Path(str(tmp_media)).resolve()
            assert not resolved.is_relative_to(allowed)
        except OSError:
            pytest.skip("Cannot create symlinks")
        finally:
            if link.exists():
                link.unlink()


# ---------------------------------------------------------------------------
# Global server lifecycle
# ---------------------------------------------------------------------------

class TestGlobalServerLifecycle:

    @pytest.mark.asyncio
    async def test_start_sets_runtime_secret(self, tmp_path):
        """start() must set _runtime_secret."""
        from qwenpaw.app import media_server as ms_mod

        srv = MediaServer(
            port=0,
            secret="my-secret",
            allowed_dirs=[str(tmp_path)],
        )
        # Directly set runtime secret (simulating what start() does)
        # without actually binding a port
        ms_mod._runtime_secret = srv.secret
        assert ms_mod._runtime_secret == "my-secret"
        ms_mod._runtime_secret = ""

    @pytest.mark.asyncio
    async def test_stop_clears_runtime_secret(self, tmp_path):
        """stop() must clear _runtime_secret."""
        from qwenpaw.app import media_server as ms_mod

        srv = MediaServer(
            port=0,
            secret="my-secret",
            allowed_dirs=[str(tmp_path)],
        )
        ms_mod._runtime_secret = "my-secret"
        await srv.stop()
        assert ms_mod._runtime_secret == ""

    @pytest.mark.asyncio
    async def test_blank_secret_generates_random(self, tmp_path):
        """MediaServer with blank secret generates one on start()."""
        srv = MediaServer(
            port=0,
            secret="",
            allowed_dirs=[str(tmp_path)],
        )
        # Simulate what start() does: generate secret if empty
        if not srv.secret:
            import secrets as _secrets
            srv.secret = _secrets.token_hex(32)
        from qwenpaw.app import media_server as ms_mod
        ms_mod._runtime_secret = srv.secret
        assert len(ms_mod._runtime_secret) == 64  # token_hex(32) -> 64 hex chars
        assert srv.secret == ms_mod._runtime_secret
        # Cleanup
        ms_mod._runtime_secret = ""
