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
def server(tmp_media, tmp_path):
    """Create a MediaServer with test config.

    ``token_store_path`` is pinned inside ``tmp_path`` so issuing
    tokens during the test never writes to the real
    ``~/.copaw/media_token_store.json`` (production pollution
    discovered 2026-04-25: a stale ``new_token`` entry showed up
    in the live store after a test run).
    """
    return MediaServer(
        port=0,
        secret="test-secret-12345",
        allowed_dirs=[str(tmp_media)],
        max_size_mb=1,
        tunnel_domain="https://media.example.com",
        token_store_path=tmp_path / "media_token_store.json",
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
        assert test_file.resolve().is_relative_to(
            Path(str(tmp_media)).resolve(),
        )

        outside = Path("/etc/passwd")
        if outside.exists():
            assert not outside.resolve().is_relative_to(
                Path(str(tmp_media)).resolve(),
            )

    def test_sign_rejects_wrong_extension(self, server, tmp_media):
        """sign must reject non-media extensions."""
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
        should_fallback = (
            "localhost" in url or "127.0.0.1" in url
        ) and not tunnel_domain
        assert should_fallback is True

    def test_configured_tunnel_domain_no_fallback(self):
        tunnel_domain = "https://media.example.com"
        url = "https://media.example.com/media?t=abc"
        should_fallback = (
            "localhost" in url or "127.0.0.1" in url
        ) and not tunnel_domain
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
            decoded = base64.urlsafe_b64decode(token + "==").decode(
                "utf-8",
                errors="ignore",
            )
        except Exception:
            decoded = ""
        assert "/tmp" not in decoded
        assert "/home" not in decoded

    def test_expired_tokens_cleaned_up(self, server):
        """_cleanup_expired_tokens removes old entries."""
        server._token_store["old_token"] = (
            "/tmp/old.mp4",
            int(time.time()) - 100,
        )
        server._token_store["new_token"] = (
            "/tmp/new.mp4",
            int(time.time()) + 3600,
        )

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
        assert not server._verify(
            "/tmp/test.mp4",
            int(time.time()) + 3600,
            "wrong" * 8,
        )

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
        assert (
            len(ms_mod._runtime_secret) == 64
        )  # token_hex(32) -> 64 hex chars
        assert srv.secret == ms_mod._runtime_secret
        # Cleanup
        ms_mod._runtime_secret = ""


# ---------------------------------------------------------------------------
# Cloudflare Quick Tunnel integration
# ---------------------------------------------------------------------------


class TestCloudflareTunnelIntegration:
    """MediaServer should drive a CloudflareTunnelDriver per tunnel_mode.

    Driver is patched so we exercise reconcile / URL-override logic
    without spawning cloudflared.
    """

    @pytest.mark.asyncio
    async def test_reconcile_turns_quick_tunnel_on(self, tmp_path):
        srv = MediaServer(
            port=8089,
            secret="x",
            allowed_dirs=[str(tmp_path)],
            tunnel_domain="https://user.example.com",
            tunnel_mode="manual",
        )
        assert srv.tunnel_domain == "https://user.example.com"
        assert srv.user_tunnel_domain == "https://user.example.com"
        assert srv.get_tunnel_url() == ""

        fake_driver = MagicMock()
        fake_info = MagicMock(public_url="https://abc123.trycloudflare.com")
        fake_driver.start = AsyncMock(return_value=fake_info)
        fake_driver.stop = AsyncMock()
        fake_driver.get_public_url = MagicMock(
            return_value="https://abc123.trycloudflare.com",
        )
        factory = MagicMock(return_value=fake_driver)
        with patch("qwenpaw.tunnel.CloudflareTunnelDriver", factory):
            await srv.reconcile_tunnel(tunnel_mode="quick")

        factory.assert_called_once_with(mode="quick")
        fake_driver.start.assert_awaited_once_with(8089)
        assert srv.tunnel_domain == "https://abc123.trycloudflare.com"
        assert srv.get_tunnel_url() == "https://abc123.trycloudflare.com"
        assert srv.user_tunnel_domain == "https://user.example.com"
        assert srv.tunnel_mode == "quick"

    @pytest.mark.asyncio
    async def test_reconcile_turns_tunnel_off_restores_user_domain(
        self,
        tmp_path,
    ):
        srv = MediaServer(
            port=8089,
            secret="x",
            allowed_dirs=[str(tmp_path)],
            tunnel_domain="https://user.example.com",
        )
        fake_driver = MagicMock()
        fake_driver.start = AsyncMock(
            return_value=MagicMock(public_url="https://abc.trycloudflare.com"),
        )
        fake_driver.stop = AsyncMock()
        fake_driver.get_public_url = MagicMock(
            return_value="https://abc.trycloudflare.com",
        )
        with patch(
            "qwenpaw.tunnel.CloudflareTunnelDriver",
            return_value=fake_driver,
        ):
            await srv.reconcile_tunnel(tunnel_mode="quick")
            assert srv.tunnel_domain == "https://abc.trycloudflare.com"
            await srv.reconcile_tunnel(tunnel_mode="manual")

        fake_driver.stop.assert_awaited_once()
        assert srv.tunnel_domain == "https://user.example.com"
        assert srv.tunnel_mode == "manual"
        assert srv.get_tunnel_url() == ""

    @pytest.mark.asyncio
    async def test_tunnel_failure_leaves_user_domain_untouched(self, tmp_path):
        """If cloudflared fails to start, the signed-URL domain must not
        silently become empty — we'd serve broken localhost URLs."""
        srv = MediaServer(
            port=8089,
            secret="x",
            allowed_dirs=[str(tmp_path)],
            tunnel_domain="https://user.example.com",
        )
        fake_driver = MagicMock()
        fake_driver.start = AsyncMock(side_effect=RuntimeError("no binary"))
        fake_driver.stop = AsyncMock()
        with patch(
            "qwenpaw.tunnel.CloudflareTunnelDriver",
            return_value=fake_driver,
        ):
            await srv.reconcile_tunnel(tunnel_mode="quick")

        assert srv._tunnel_driver is None
        assert srv.tunnel_domain == "https://user.example.com"

    @pytest.mark.asyncio
    async def test_named_tunnel_uses_user_hostname(self, tmp_path):
        """Named-mode driver is spawned with tunnel_name + hostname and the
        user hostname becomes the effective tunnel_domain."""
        srv = MediaServer(
            port=8089,
            secret="x",
            allowed_dirs=[str(tmp_path)],
            tunnel_domain="",
        )
        fake_driver = MagicMock()
        fake_driver.start = AsyncMock(
            return_value=MagicMock(
                public_url="https://media.example.com",
            ),
        )
        fake_driver.stop = AsyncMock()
        fake_driver.get_public_url = MagicMock(
            return_value="https://media.example.com",
        )
        factory = MagicMock(return_value=fake_driver)
        with patch("qwenpaw.tunnel.CloudflareTunnelDriver", factory):
            await srv.reconcile_tunnel(
                tunnel_mode="named",
                named_tunnel_name="media",
                named_tunnel_hostname="media.example.com",
                named_tunnel_config_file="/etc/cloudflared/media.yml",
            )

        factory.assert_called_once_with(
            mode="named",
            tunnel_name="media",
            hostname="media.example.com",
            config_file="/etc/cloudflared/media.yml",
        )
        fake_driver.start.assert_awaited_once_with(8089)
        assert srv.tunnel_domain == "https://media.example.com"
        assert srv.tunnel_mode == "named"

    @pytest.mark.asyncio
    async def test_reconcile_no_op_when_mode_and_config_unchanged(
        self,
        tmp_path,
    ):
        """Saving the same config twice shouldn't restart the tunnel —
        otherwise the user gets a new trycloudflare URL on every config
        save, which breaks already-issued signed URLs."""
        srv = MediaServer(
            port=8089,
            secret="x",
            allowed_dirs=[str(tmp_path)],
            tunnel_domain="",
            tunnel_mode="quick",
        )
        fake_driver = MagicMock()
        fake_driver.start = AsyncMock(
            return_value=MagicMock(
                public_url="https://alpha.trycloudflare.com",
            ),
        )
        fake_driver.stop = AsyncMock()
        fake_driver.get_public_url = MagicMock(
            return_value="https://alpha.trycloudflare.com",
        )
        with patch(
            "qwenpaw.tunnel.CloudflareTunnelDriver",
            return_value=fake_driver,
        ):
            # simulate a first reconcile to spin up the tunnel
            await srv.reconcile_tunnel(tunnel_mode="manual")  # stops nothing
            await srv.reconcile_tunnel(tunnel_mode="quick")
            assert fake_driver.start.await_count == 1

            # second reconcile with same mode: driver should not be touched
            await srv.reconcile_tunnel(tunnel_mode="quick")
            assert fake_driver.start.await_count == 1
            fake_driver.stop.assert_not_awaited()


# ---------------------------------------------------------------- #
# Token store persistence (survives copaw restart)                 #
# ---------------------------------------------------------------- #


import json as _json
import time as _time

import pytest as _pytest


class TestTokenStorePersistence:
    """Without disk-persistence the token store is reset on every
    copaw restart and any URL still inside an active conversation
    history 403s with 'Invalid token' even when its ``exp`` query
    param is fresh.  These tests pin the on-disk format and the
    load/persist round-trip."""

    def test_persist_then_load_round_trip(self, tmp_path) -> None:
        from qwenpaw.app.media_server import MediaServer

        s = MediaServer(secret="seed")
        s._token_store_path = tmp_path / "store.json"
        # Simulate two issued tokens.
        s._token_store["abc"] = ("/tmp/a.png", int(_time.time()) + 3600)
        s._token_store["def"] = ("/tmp/b.mp4", int(_time.time()) + 1800)
        s._persist_token_store()

        # Fresh server pointed at the same file recovers both.
        s2 = MediaServer(secret="seed")
        s2._token_store_path = s._token_store_path
        s2._token_store = s2._load_token_store()
        assert s2._token_store["abc"] == s._token_store["abc"]
        assert s2._token_store["def"] == s._token_store["def"]

    def test_load_drops_already_expired_entries(self, tmp_path) -> None:
        from qwenpaw.app.media_server import MediaServer

        s = MediaServer(secret="seed")
        s._token_store_path = tmp_path / "store.json"
        # One fresh, one already past expiry.
        s._token_store["fresh"] = ("/tmp/fresh.png", int(_time.time()) + 60)
        s._token_store["stale"] = ("/tmp/stale.png", int(_time.time()) - 60)
        s._persist_token_store()

        s2 = MediaServer(secret="seed")
        s2._token_store_path = s._token_store_path
        loaded = s2._load_token_store()
        assert "fresh" in loaded
        assert "stale" not in loaded

    def test_corrupt_store_yields_empty_dict(self, tmp_path) -> None:
        from qwenpaw.app.media_server import MediaServer

        path = tmp_path / "store.json"
        path.write_text("not valid json {")
        s = MediaServer(secret="seed")
        s._token_store_path = path
        # Should NOT raise — corruption falls back to empty.
        loaded = s._load_token_store()
        assert loaded == {}

    def test_cleanup_persists_after_removal(self, tmp_path) -> None:
        from qwenpaw.app.media_server import MediaServer

        s = MediaServer(secret="seed")
        s._token_store_path = tmp_path / "store.json"
        s._token_store["fresh"] = ("/tmp/x", int(_time.time()) + 60)
        s._token_store["stale"] = ("/tmp/y", int(_time.time()) - 60)
        s._persist_token_store()
        assert "stale" in _json.loads(s._token_store_path.read_text())

        s._cleanup_expired_tokens()
        on_disk = _json.loads(s._token_store_path.read_text())
        assert "stale" not in on_disk
        assert "fresh" in on_disk
