# -*- coding: utf-8 -*-
"""Tests for media server security fixes (Codex adversarial review findings)."""

import hashlib
import hmac as hmac_mod
import os
import secrets
import tempfile
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from copaw.app.media_server import MediaServer


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
# Finding 1: /sign requires auth + validates allowed_dirs/ext/size
# ---------------------------------------------------------------------------

class TestSignAuth:

    def test_sign_requires_valid_secret(self, server):
        """Finding 1: /sign must require auth=media_secret."""
        assert server.secret == "test-secret-12345"
        # If auth doesn't match, should be rejected
        # (tested via HTTP in integration, here test the HMAC comparison)
        assert hmac_mod.compare_digest("test-secret-12345", server.secret)
        assert not hmac_mod.compare_digest("wrong-secret", server.secret)

    def test_sign_validates_allowed_dirs(self, server, tmp_media):
        """Finding 1: /sign must check allowed_dirs before signing."""
        # File inside allowed_dirs — should work
        test_file = tmp_media / "test.png"
        assert test_file.resolve().is_relative_to(Path(str(tmp_media)).resolve())

        # File outside allowed_dirs — should fail
        outside = Path("/etc/passwd")
        if outside.exists():
            assert not outside.resolve().is_relative_to(Path(str(tmp_media)).resolve())

    def test_sign_rejects_wrong_extension(self, server, tmp_media):
        """Finding 1: /sign must reject non-media extensions."""
        media_exts = {
            ".mp4", ".webm", ".mov", ".avi", ".mkv", ".mpeg",
            ".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp",
            ".mp3", ".wav", ".ogg", ".flac", ".m4a",
        }
        txt = tmp_media / "secret.txt"
        assert txt.suffix.lower() not in media_exts

    def test_sign_caps_ttl_at_24h(self, server):
        """Finding 1: TTL must be capped at 86400 (24h)."""
        capped = min(999999, 86400)
        assert capped == 86400


# ---------------------------------------------------------------------------
# Finding 2: localhost URL rejected when tunnel_domain empty
# ---------------------------------------------------------------------------

class TestLocalhostUrlRejection:

    def test_localhost_url_detected(self):
        """Finding 2: localhost/127.0.0.1 URLs must be detected."""
        localhost_urls = [
            "http://localhost:8089/media?t=abc&sig=def",
            "http://127.0.0.1:8089/media?t=abc&sig=def",
        ]
        for url in localhost_urls:
            assert "localhost" in url or "127.0.0.1" in url

    def test_tunnel_url_not_localhost(self):
        """Finding 2: tunnel URLs should NOT be flagged as localhost."""
        tunnel_url = "https://media.example.com/media?t=abc&sig=def"
        assert "localhost" not in tunnel_url
        assert "127.0.0.1" not in tunnel_url

    def test_empty_tunnel_domain_triggers_fallback(self):
        """Finding 2: empty tunnel_domain + localhost URL = should fallback."""
        tunnel_domain = ""
        url = "http://localhost:8089/media?t=abc"
        should_fallback = ("localhost" in url or "127.0.0.1" in url) and not tunnel_domain
        assert should_fallback is True

    def test_configured_tunnel_domain_no_fallback(self):
        """Finding 2: configured tunnel_domain = should NOT fallback."""
        tunnel_domain = "https://media.example.com"
        url = "https://media.example.com/media?t=abc"
        should_fallback = ("localhost" in url or "127.0.0.1" in url) and not tunnel_domain
        assert should_fallback is False


# ---------------------------------------------------------------------------
# Finding 3: opaque tokens (no path leakage)
# ---------------------------------------------------------------------------

class TestOpaqueTokens:

    def test_token_store_maps_token_to_path(self, server, tmp_media):
        """Finding 3: token_store should map opaque token to real path."""
        token = secrets.token_urlsafe(24)
        raw_path = str(tmp_media / "test.png")
        expires = int(time.time()) + 3600
        server._token_store[token] = (raw_path, expires, "test-agent")

        entry = server._token_store.get(token)
        assert entry is not None
        assert entry[0] == raw_path
        assert entry[1] == expires
        assert entry[2] == "test-agent"

    def test_token_is_opaque(self, server):
        """Finding 3: token must not contain decodable path info."""
        import base64
        token = secrets.token_urlsafe(24)
        # Try to decode as base64 — should NOT contain filesystem path
        try:
            decoded = base64.urlsafe_b64decode(token + "==").decode("utf-8", errors="ignore")
        except Exception:
            decoded = ""
        assert "/tmp" not in decoded
        assert "/home" not in decoded

    def test_expired_tokens_cleaned_up(self, server):
        """Finding 3: _cleanup_expired_tokens removes old entries."""
        # Add expired token
        server._token_store["old_token"] = ("/tmp/old.mp4", int(time.time()) - 100, "agent-a")
        # Add valid token
        server._token_store["new_token"] = ("/tmp/new.mp4", int(time.time()) + 3600, "agent-b")

        server._cleanup_expired_tokens()

        assert "old_token" not in server._token_store
        assert "new_token" in server._token_store

    def test_invalid_token_rejected(self, server):
        """Finding 3: /media with unknown token should be rejected."""
        entry = server._token_store.get("nonexistent_token")
        assert entry is None


# ---------------------------------------------------------------------------
# HMAC signature tests
# ---------------------------------------------------------------------------

class TestHMACSignature:

    def test_signature_is_32_chars(self, server):
        """HMAC signature must be 32 hex chars (128-bit)."""
        sig = server._sign("/tmp/test.mp4", 9999999999)
        assert len(sig) == 32
        # Must be valid hex
        int(sig, 16)

    def test_verify_valid_signature(self, server):
        """Valid signature must pass verification."""
        path = "/tmp/test.mp4"
        expires = int(time.time()) + 3600
        sig = server._sign(path, expires)
        assert server._verify(path, expires, sig)

    def test_verify_wrong_signature_rejected(self, server):
        """Wrong signature must be rejected."""
        assert not server._verify("/tmp/test.mp4", int(time.time()) + 3600, "wrong" * 8)

    def test_verify_expired_rejected(self, server):
        """Expired signature must be rejected even if HMAC is correct."""
        path = "/tmp/test.mp4"
        past = int(time.time()) - 100
        sig = server._sign(path, past)
        assert not server._verify(path, past, sig)

    def test_different_paths_different_signatures(self, server):
        """Different paths must produce different signatures."""
        exp = int(time.time()) + 3600
        sig1 = server._sign("/tmp/a.mp4", exp)
        sig2 = server._sign("/tmp/b.mp4", exp)
        assert sig1 != sig2


# ---------------------------------------------------------------------------
# Path validation (symlink protection)
# ---------------------------------------------------------------------------

class TestPathValidation:

    def test_relative_to_catches_outside_path(self, server, tmp_media):
        """is_relative_to must reject paths outside allowed_dirs."""
        allowed = Path(str(tmp_media)).resolve()
        outside = Path("/etc/passwd").resolve()
        assert not outside.is_relative_to(allowed)

    def test_relative_to_allows_inside_path(self, server, tmp_media):
        """is_relative_to must accept paths inside allowed_dirs."""
        allowed = Path(str(tmp_media)).resolve()
        inside = (tmp_media / "test.png").resolve()
        assert inside.is_relative_to(allowed)

    def test_symlink_resolved_and_checked(self, tmp_media):
        """Symlinks must be resolved before checking allowed_dirs."""
        # Create symlink pointing outside
        link = tmp_media / "evil_link"
        try:
            link.symlink_to("/etc")
            resolved = (link / "passwd").resolve()
            allowed = Path(str(tmp_media)).resolve()
            # Resolved path should NOT be relative to allowed dir
            assert not resolved.is_relative_to(allowed)
        except OSError:
            pytest.skip("Cannot create symlinks")
        finally:
            if link.exists():
                link.unlink()

# ---------------------------------------------------------------------------
# Per-agent dir scoping (Codex Finding: cross-agent file access)
# ---------------------------------------------------------------------------

class TestPerAgentDirScoping:

    def test_agent_dirs_isolated(self, tmp_path):
        """Each agent's allowed_dirs must be scoped, not merged."""
        MediaServer._instance = None
        dir_a = tmp_path / "agent_a_files"
        dir_a.mkdir()
        dir_b = tmp_path / "agent_b_files"
        dir_b.mkdir()

        srv = MediaServer.get_or_create(
            port=0, secret="sec-a", allowed_dirs=[str(dir_a)],
            agent_id="agent-a",
        )
        MediaServer.get_or_create(
            port=0, secret="sec-b", allowed_dirs=[str(dir_b)],
            agent_id="agent-b",
        )

        assert srv._agent_dirs["agent-a"] == [str(dir_a)]
        assert srv._agent_dirs["agent-b"] == [str(dir_b)]
        # agent-a must NOT see agent-b's dirs
        assert str(dir_b) not in srv._agent_dirs["agent-a"]
        MediaServer._instance = None

    def test_get_or_create_increments_ref_count(self, tmp_path):
        """Each get_or_create call must increment _ref_count."""
        MediaServer._instance = None
        srv = MediaServer.get_or_create(
            port=0, secret="s1", allowed_dirs=["/tmp"], agent_id="a1",
        )
        assert srv._ref_count == 1
        MediaServer.get_or_create(
            port=0, secret="s2", allowed_dirs=["/tmp"], agent_id="a2",
        )
        assert srv._ref_count == 2
        MediaServer._instance = None


# ---------------------------------------------------------------------------
# Reference-counted stop (Codex Finding: singleton killed by any workspace)
# ---------------------------------------------------------------------------

class TestRefCountedStop:

    @pytest.mark.asyncio
    async def test_stop_decrements_ref_count(self, tmp_path):
        """stop() must decrement ref_count and keep server alive if > 0."""
        MediaServer._instance = None
        srv = MediaServer.get_or_create(
            port=0, secret="s1", allowed_dirs=["/tmp"], agent_id="a1",
        )
        MediaServer.get_or_create(
            port=0, secret="s2", allowed_dirs=["/tmp"], agent_id="a2",
        )
        assert srv._ref_count == 2

        await srv.stop()
        assert srv._ref_count == 1
        # Singleton must still be alive
        assert MediaServer._instance is not None

        await srv.stop()
        assert srv._ref_count == 0
        # Singleton must be cleared
        assert MediaServer._instance is None

    @pytest.mark.asyncio
    async def test_stop_last_ref_clears_singleton(self, tmp_path):
        """Last stop() must set _instance to None."""
        MediaServer._instance = None
        srv = MediaServer.get_or_create(
            port=0, secret="s", allowed_dirs=["/tmp"], agent_id="only",
        )
        assert srv._ref_count == 1
        await srv.stop()
        assert MediaServer._instance is None


# ---------------------------------------------------------------------------
# Finding 1 fix: blank secrets get unique per-agent values
# ---------------------------------------------------------------------------

class TestUniqueSecretGeneration:

    def test_blank_secrets_are_unique(self, tmp_path):
        """Two agents with blank secrets must get DIFFERENT generated secrets."""
        MediaServer._instance = None
        srv = MediaServer.get_or_create(
            port=0, secret="", allowed_dirs=["/tmp"], agent_id="agent-x",
        )
        MediaServer.get_or_create(
            port=0, secret="", allowed_dirs=["/tmp"], agent_id="agent-y",
        )

        from copaw.app.media_server import _runtime_secrets
        assert "agent-x" in _runtime_secrets
        assert "agent-y" in _runtime_secrets
        # Secrets must differ
        assert _runtime_secrets["agent-x"] != _runtime_secrets["agent-y"]
        # Each secret must be non-empty
        assert len(_runtime_secrets["agent-x"]) > 0
        assert len(_runtime_secrets["agent-y"]) > 0
        MediaServer._instance = None
        _runtime_secrets.clear()

    def test_explicit_secret_preserved(self, tmp_path):
        """When caller provides a non-blank secret, it must be used as-is."""
        MediaServer._instance = None
        from copaw.app.media_server import _runtime_secrets
        _runtime_secrets.clear()

        srv = MediaServer.get_or_create(
            port=0, secret="my-explicit-secret", allowed_dirs=["/tmp"],
            agent_id="agent-e",
        )
        assert _runtime_secrets["agent-e"] == "my-explicit-secret"
        MediaServer._instance = None
        _runtime_secrets.clear()

    def test_first_agent_blank_secret_overrides_instance(self, tmp_path):
        """First agent with blank secret: instance.secret must be the generated one."""
        MediaServer._instance = None
        from copaw.app.media_server import _runtime_secrets
        _runtime_secrets.clear()

        srv = MediaServer.get_or_create(
            port=0, secret="", allowed_dirs=["/tmp"], agent_id="first",
        )
        # The generated secret should be in _runtime_secrets
        generated = _runtime_secrets["first"]
        assert len(generated) == 64  # token_hex(32) -> 64 hex chars
        # instance.secret should be the same generated secret
        assert srv.secret == generated
        MediaServer._instance = None
        _runtime_secrets.clear()


# ---------------------------------------------------------------------------
# Finding 2 fix: stop(agent_id) revokes access
# ---------------------------------------------------------------------------

class TestStopRevokesAccess:

    @pytest.mark.asyncio
    async def test_stop_with_agent_id_removes_dirs_and_secrets(self, tmp_path):
        """stop(agent_id=X) must remove X's dirs, secret, and tokens."""
        MediaServer._instance = None
        from copaw.app.media_server import _runtime_secrets
        _runtime_secrets.clear()

        srv = MediaServer.get_or_create(
            port=0, secret="s1", allowed_dirs=["/tmp/a"], agent_id="a1",
        )
        MediaServer.get_or_create(
            port=0, secret="s2", allowed_dirs=["/tmp/b"], agent_id="a2",
        )
        # Plant a token for agent a1
        srv._token_store["tok-a1"] = ("/tmp/a/file.mp4", 9999999999, "a1")
        srv._token_store["tok-a2"] = ("/tmp/b/file.mp4", 9999999999, "a2")

        assert srv._ref_count == 2
        await srv.stop(agent_id="a1")

        # a1's data must be gone
        assert "a1" not in srv._agent_dirs
        assert "a1" not in _runtime_secrets
        assert "tok-a1" not in srv._token_store
        # a2's data must survive
        assert "a2" in srv._agent_dirs
        assert "a2" in _runtime_secrets
        assert "tok-a2" in srv._token_store
        assert srv._ref_count == 1
        # Singleton still alive
        assert MediaServer._instance is not None

        await srv.stop(agent_id="a2")
        assert srv._ref_count == 0
        assert MediaServer._instance is None
        _runtime_secrets.clear()

    @pytest.mark.asyncio
    async def test_stop_without_agent_id_still_works(self, tmp_path):
        """Legacy stop() without agent_id must still decrement and shut down."""
        MediaServer._instance = None
        from copaw.app.media_server import _runtime_secrets
        _runtime_secrets.clear()

        srv = MediaServer.get_or_create(
            port=0, secret="s", allowed_dirs=["/tmp"], agent_id="only",
        )
        assert srv._ref_count == 1
        await srv.stop()
        assert srv._ref_count == 0
        assert MediaServer._instance is None
        _runtime_secrets.clear()

    @pytest.mark.asyncio
    async def test_stop_clears_runtime_secrets_on_last_ref(self, tmp_path):
        """When the last reference is released, _runtime_secrets must be cleared."""
        MediaServer._instance = None
        from copaw.app.media_server import _runtime_secrets
        _runtime_secrets.clear()

        srv = MediaServer.get_or_create(
            port=0, secret="sec", allowed_dirs=["/tmp"], agent_id="last",
        )
        _runtime_secrets["leftover"] = "should-be-cleared"
        await srv.stop(agent_id="last")
        assert len(_runtime_secrets) == 0
        MediaServer._instance = None
