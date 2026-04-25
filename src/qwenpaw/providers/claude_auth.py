# -*- coding: utf-8 -*-
"""Claude Code OAuth credential loader + refresher.

Reads the shared ``~/.claude/.credentials.json`` file written by the
official ``claude`` CLI (Claude Code) and keeps the ``accessToken``
fresh by calling Anthropic's OAuth token endpoint when the token is
near expiry.  Mirrors ``codex_auth.py`` so the two OAuth paths stay
symmetric.

The file layout we expect (written by Claude Code login) ::

    {
      "claudeAiOauth": {
        "accessToken":      "sk-ant-oat01-...",
        "refreshToken":     "sk-ant-ort01-...",
        "expiresAt":        1756162077244,       # epoch ms
        "scopes":           ["user:inference", "user:profile"],
        "subscriptionType": "max"
      }
    }

Claude Code's OAuth requires the request to include ``Authorization:
Bearer ...`` (NOT ``x-api-key``) plus a specific ``anthropic-beta``
flag set, and the first ``system`` content block MUST be the literal
Claude Code identity string — without it, the Anthropic API rejects
the call as not coming from a trusted CLI.  See
``AnthropicProvider``'s OAuth mode for the system-prompt injection.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

logger = logging.getLogger(__name__)

# Public OAuth client id used by the official Claude Code CLI.  Not a
# secret — it's present in every public Claude-OAuth adapter (openclaw,
# opencode-claude-auth, ben-vargas gist).
CLAUDE_OAUTH_CLIENT_ID = "9d1c250a-e61b-44d9-88ed-5944d1962f5e"

# Primary refresh endpoint.  ``console.anthropic.com/v1/oauth/token``
# and ``platform.claude.com/v1/oauth/token`` are known fallbacks but
# Cloudflare WAF blocks them from some headless egress IPs; claude.ai
# is the one the current ``opencode-claude-auth`` targets, so lead with
# that and let an env var override if a deployment needs otherwise.
DEFAULT_TOKEN_ENDPOINT = "https://claude.ai/v1/oauth/token"

# String that MUST appear as the first ``system`` content block when
# calling ``/v1/messages`` with an OAuth token.  Anthropic validates
# byte-equality, so do not edit this line.
CLAUDE_CODE_IDENTITY = (
    "You are Claude Code, Anthropic's official CLI for Claude."
)

# A recent Claude Code CLI version string; what matters is that it
# parses as ``claude-cli/<semver>`` — upstream uses this for rough
# client-version bucketing, not for gating.
CLAUDE_CLI_VERSION = "2.1.90"

# Minimum required beta flags.  ``oauth-2025-04-20`` marks the request
# as OAuth-authed, ``claude-code-20250219`` marks it as coming from the
# CLI.  Missing either one → 401.  Other betas (interleaved-thinking,
# prompt-caching-scope, context-management) are feature-gates we stay
# out of unless a caller opts in.
CLAUDE_BASE_BETAS = "claude-code-20250219," "oauth-2025-04-20"

# Refresh the access_token this many seconds BEFORE ``expiresAt`` so an
# in-flight request never races the expiration edge.
REFRESH_SAFETY_MARGIN_S = 5 * 60


def _resolve_credentials_path() -> Path:
    """Resolve the credentials file, honouring ``CLAUDE_CONFIG_DIR``."""
    home = os.environ.get("CLAUDE_CONFIG_DIR") or "~/.claude"
    return Path(os.path.expanduser(home)) / ".credentials.json"


@dataclass
class ClaudeCredential:
    access_token: str
    refresh_token: str
    expires_at_ms: int
    credentials_path: Path
    scopes: list[str]
    subscription_type: str | None = None

    @property
    def seconds_until_expiry(self) -> int:
        return max(0, int((self.expires_at_ms - time.time() * 1000) / 1000))

    @property
    def needs_refresh(self) -> bool:
        return self.seconds_until_expiry <= REFRESH_SAFETY_MARGIN_S


class ClaudeAuth:
    """Stateful manager for ``~/.claude/.credentials.json``.

    Usage::

        auth = ClaudeAuth()                   # raises if not logged in
        headers = await auth.auth_headers()   # refreshes on demand
        kwargs  = await auth.client_kwargs()  # for anthropic.AsyncAnthropic
    """

    def __init__(self, credentials_path: Path | None = None) -> None:
        self._path = credentials_path or _resolve_credentials_path()
        # ``asyncio.Lock`` so concurrent ``ensure_fresh()`` callers in
        # the same event loop coalesce into one refresh round-trip —
        # otherwise they both see the token as ``needs_refresh``,
        # both POST to the OAuth endpoint, and the loser's rotated
        # refresh_token lands on disk, invalidating the winner's.
        self._lock = asyncio.Lock()
        self._creds: ClaudeCredential | None = None
        self._load()

    # ------------------------------------------------------------- #
    # Disk I/O                                                       #
    # ------------------------------------------------------------- #

    def _load(self) -> None:
        if not self._path.exists():
            raise FileNotFoundError(
                f"Claude Code credentials file not found: {self._path}. "
                "Run `claude login` once to populate it.",
            )
        raw = json.loads(self._path.read_text())
        oauth = raw.get("claudeAiOauth") or {}
        access = oauth.get("accessToken")
        refresh = oauth.get("refreshToken")
        expires_at = oauth.get("expiresAt")
        if (
            not access
            or not refresh
            or not isinstance(expires_at, (int, float))
        ):
            raise ValueError(
                f"Claude Code credentials file {self._path} is missing "
                "accessToken / refreshToken / expiresAt under "
                "'claudeAiOauth' (re-run `claude login`?).",
            )

        self._creds = ClaudeCredential(
            access_token=str(access),
            refresh_token=str(refresh),
            expires_at_ms=int(expires_at),
            credentials_path=self._path,
            scopes=list(oauth.get("scopes") or []),
            subscription_type=oauth.get("subscriptionType"),
        )

    def _save(self, *, tokens: dict[str, Any]) -> None:
        """Write refreshed tokens back atomically, preserving other fields."""
        raw = json.loads(self._path.read_text())
        raw.setdefault("claudeAiOauth", {})
        raw["claudeAiOauth"].update(tokens)
        tmp = self._path.with_suffix(self._path.suffix + ".tmp")
        tmp.write_text(json.dumps(raw, indent=2))
        os.chmod(tmp, 0o600)
        os.replace(tmp, self._path)

    # ------------------------------------------------------------- #
    # Refresh                                                        #
    # ------------------------------------------------------------- #

    async def _refresh(self) -> None:
        assert self._creds is not None
        endpoint = os.environ.get(
            "QWENPAW_CLAUDE_OAUTH_TOKEN_URL",
            DEFAULT_TOKEN_ENDPOINT,
        )
        logger.info(
            "[ClaudeAuth] refreshing access_token via %s (expires in %ds)",
            endpoint,
            self._creds.seconds_until_expiry,
        )
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                endpoint,
                data={
                    "grant_type": "refresh_token",
                    "client_id": CLAUDE_OAUTH_CLIENT_ID,
                    "refresh_token": self._creds.refresh_token,
                },
                headers={
                    "Content-Type": "application/x-www-form-urlencoded",
                },
            )
            resp.raise_for_status()
            body = resp.json()

        new_access = body.get("access_token")
        new_refresh = body.get("refresh_token") or self._creds.refresh_token
        expires_in = body.get("expires_in")
        if not new_access:
            raise RuntimeError(
                "Claude OAuth refresh succeeded but no access_token in "
                f"response: {body!r}",
            )
        # ``expires_in`` is seconds; default to 10h (the value Claude
        # Code observes in practice) when missing.
        exp_ms = int(time.time() * 1000) + int(
            (expires_in if isinstance(expires_in, (int, float)) else 36_000)
            * 1000,
        )

        # Caller holds ``self._lock``; we update disk + in-memory
        # state together so the next ``needs_refresh`` check sees the
        # fresh token.
        self._save(
            tokens={
                "accessToken": new_access,
                "refreshToken": new_refresh,
                "expiresAt": exp_ms,
            },
        )
        self._creds = ClaudeCredential(
            access_token=new_access,
            refresh_token=new_refresh,
            expires_at_ms=exp_ms,
            credentials_path=self._path,
            scopes=self._creds.scopes,
            subscription_type=self._creds.subscription_type,
        )
        logger.info(
            "[ClaudeAuth] refreshed — new expiry in %ds",
            self._creds.seconds_until_expiry,
        )

    # ------------------------------------------------------------- #
    # Public API                                                     #
    # ------------------------------------------------------------- #

    async def ensure_fresh(self) -> ClaudeCredential:
        if self._creds is None:
            self._load()
        assert self._creds is not None
        # Lock the "check + refresh" pair so concurrent callers in the
        # same event loop coalesce into one refresh round-trip.
        async with self._lock:
            if self._creds.needs_refresh:
                await self._refresh()
            return self._creds

    def default_headers(self) -> dict[str, str]:
        """OAuth-specific headers that must be merged into every
        ``/v1/messages`` call.  Does NOT include ``Authorization``
        (the anthropic SDK builds that from ``auth_token``).
        """
        return {
            "anthropic-beta": CLAUDE_BASE_BETAS,
            "x-app": "cli",
            "user-agent": f"claude-cli/{CLAUDE_CLI_VERSION} (external, cli)",
        }

    async def auth_headers(self) -> dict[str, str]:
        """Full header bundle including ``Authorization``.  Use this
        when calling the API directly via httpx (e.g., smoke tests);
        when going through the anthropic SDK, prefer ``client_kwargs``
        + ``auth_token`` on the SDK client.
        """
        creds = await self.ensure_fresh()
        h = {
            "Authorization": f"Bearer {creds.access_token}",
            "anthropic-version": "2023-06-01",
        }
        h.update(self.default_headers())
        return h

    async def client_kwargs(self) -> dict[str, Any]:
        """kwargs to hand to ``anthropic.AsyncAnthropic(**kwargs)`` so
        every request is OAuth-authed with the correct headers."""
        creds = await self.ensure_fresh()
        return {
            "api_key": None,
            "auth_token": creds.access_token,
            "default_headers": self.default_headers(),
        }

    @property
    def access_token(self) -> str | None:
        return self._creds.access_token if self._creds else None


# -------------------------------------------------------------------------
# CLI smoke test — `python -m qwenpaw.providers.claude_auth`
# -------------------------------------------------------------------------


async def _smoke() -> None:
    auth = ClaudeAuth()
    creds = await auth.ensure_fresh()
    headers = await auth.auth_headers()
    masked = {
        k: (
            v[:12] + "..." + v[-6:]
            if k == "Authorization" and len(v) > 30
            else v
        )
        for k, v in headers.items()
    }
    print(f"credentials_path: {creds.credentials_path}")
    print(f"subscription:     {creds.subscription_type}")
    print(f"scopes:           {creds.scopes}")
    print(f"expires_in:       {creds.seconds_until_expiry}s")
    print(f"headers:          {masked}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(_smoke())
