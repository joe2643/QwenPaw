# -*- coding: utf-8 -*-
"""Codex OAuth credential loader + refresher.

Reads the shared ``~/.codex/auth.json`` file written by the official
`@openai/codex` CLI (same format OpenClaw uses via `pi-ai/oauth`) and
keeps the ``access_token`` fresh by calling OpenAI's OAuth token
endpoint when the JWT is near expiry.

The file layout we expect::

    {
      "auth_mode": "chatgpt" | "apikey",
      "OPENAI_API_KEY": "...",          # optional
      "tokens": {
        "id_token":      "eyJ...",
        "access_token":  "eyJ...",
        "refresh_token": "rt_...",
        "account_id":    "uuid",
      },
      "last_refresh":   "2026-04-21T12:00:00Z"
    }
"""

from __future__ import annotations

import base64
import json
import logging
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

logger = logging.getLogger(__name__)

# Public Codex-CLI OAuth client (same identifier openai/codex uses).
OPENAI_OAUTH_CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
OPENAI_TOKEN_ENDPOINT = "https://auth.openai.com/oauth/token"

# ChatGPT backend that actually serves Codex traffic.
DEFAULT_CHATGPT_BACKEND = "https://chatgpt.com/backend-api"

# Refresh the access_token this many seconds BEFORE its JWT `exp` so an
# in-flight request never races the expiration edge.
REFRESH_SAFETY_MARGIN_S = 5 * 60


def _decode_jwt_exp_ms(jwt: str) -> int | None:
    """Best-effort extract the ``exp`` claim (in ms) from a JWT.

    Codex's access_token is a JWT signed by OpenAI; the ``exp`` claim
    tells us when the server will stop accepting it.  Returns ``None``
    when the token isn't a parseable JWT — caller falls back to an
    mtime-based heuristic.
    """
    try:
        payload_b64 = jwt.split(".")[1]
        # base64url, padded to a multiple of 4.
        payload_b64 += "=" * (-len(payload_b64) % 4)
        payload = json.loads(
            base64.urlsafe_b64decode(payload_b64.encode("ascii")),
        )
        exp = payload.get("exp")
        if isinstance(exp, (int, float)):
            return int(exp * 1000)
    except Exception:
        pass
    return None


def _resolve_auth_path() -> Path:
    home = os.environ.get("CODEX_HOME") or "~/.codex"
    return Path(os.path.expanduser(home)) / "auth.json"


@dataclass
class CodexCredential:
    access_token: str
    refresh_token: str
    id_token: str | None
    account_id: str | None
    expires_at_ms: int  # wall-clock ms, from JWT exp or fallback
    auth_path: Path
    auth_mode: str = "chatgpt"

    @property
    def seconds_until_expiry(self) -> int:
        return max(0, int((self.expires_at_ms - time.time() * 1000) / 1000))

    @property
    def needs_refresh(self) -> bool:
        return self.seconds_until_expiry <= REFRESH_SAFETY_MARGIN_S


class CodexAuth:
    """Stateful manager for ``~/.codex/auth.json``.

    Usage::

        auth = CodexAuth()                # raises FileNotFoundError if not logged in
        headers = await auth.auth_headers()   # refreshes on demand
    """

    def __init__(self, auth_path: Path | None = None) -> None:
        self._auth_path = auth_path or _resolve_auth_path()
        self._lock = threading.Lock()
        self._creds: CodexCredential | None = None
        self._load()

    # ------------------------------------------------------------- #
    # Disk I/O                                                       #
    # ------------------------------------------------------------- #

    def _load(self) -> None:
        if not self._auth_path.exists():
            raise FileNotFoundError(
                f"Codex auth file not found: {self._auth_path}. "
                "Run `codex login` once to populate it.",
            )
        raw = json.loads(self._auth_path.read_text())
        tokens = raw.get("tokens") or {}
        access_token = tokens.get("access_token")
        refresh_token = tokens.get("refresh_token")
        if not access_token or not refresh_token:
            raise ValueError(
                f"Codex auth file {self._auth_path} missing access_token / refresh_token "
                "(did `codex login` complete? Or is auth_mode=apikey?)",
            )

        # JWT exp → ms; fall back to file mtime + 1h if unparseable.
        exp_ms = _decode_jwt_exp_ms(access_token)
        if exp_ms is None:
            exp_ms = int(self._auth_path.stat().st_mtime * 1000) + 60 * 60 * 1000

        self._creds = CodexCredential(
            access_token=access_token,
            refresh_token=refresh_token,
            id_token=tokens.get("id_token"),
            account_id=tokens.get("account_id"),
            expires_at_ms=exp_ms,
            auth_path=self._auth_path,
            auth_mode=str(raw.get("auth_mode") or "chatgpt"),
        )

    def _save(self, *, tokens: dict[str, Any]) -> None:
        """Write refreshed tokens back atomically, preserving other fields."""
        raw = json.loads(self._auth_path.read_text())
        raw.setdefault("tokens", {})
        raw["tokens"].update(tokens)
        raw["last_refresh"] = time.strftime(
            "%Y-%m-%dT%H:%M:%SZ", time.gmtime(),
        )
        tmp = self._auth_path.with_suffix(self._auth_path.suffix + ".tmp")
        tmp.write_text(json.dumps(raw, indent=2))
        os.chmod(tmp, 0o600)
        os.replace(tmp, self._auth_path)

    # ------------------------------------------------------------- #
    # Refresh                                                        #
    # ------------------------------------------------------------- #

    async def _refresh(self) -> None:
        """Exchange the refresh_token for a fresh access_token."""
        assert self._creds is not None
        logger.info(
            "[CodexAuth] refreshing access_token (expires in %ds)",
            self._creds.seconds_until_expiry,
        )
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                OPENAI_TOKEN_ENDPOINT,
                data={
                    "grant_type": "refresh_token",
                    "client_id": OPENAI_OAUTH_CLIENT_ID,
                    "refresh_token": self._creds.refresh_token,
                    "scope": "openid profile email offline_access",
                },
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )
            resp.raise_for_status()
            body = resp.json()

        new_access = body.get("access_token")
        new_refresh = body.get("refresh_token") or self._creds.refresh_token
        new_id = body.get("id_token") or self._creds.id_token
        if not new_access:
            raise RuntimeError(
                "Codex token refresh succeeded but no access_token in response",
            )

        exp_ms = _decode_jwt_exp_ms(new_access) or (int(time.time() * 1000) + 3600 * 1000)
        tokens_to_save: dict[str, Any] = {
            "access_token": new_access,
            "refresh_token": new_refresh,
        }
        if new_id:
            tokens_to_save["id_token"] = new_id

        with self._lock:
            self._save(tokens=tokens_to_save)
            self._creds = CodexCredential(
                access_token=new_access,
                refresh_token=new_refresh,
                id_token=new_id,
                account_id=self._creds.account_id,
                expires_at_ms=exp_ms,
                auth_path=self._auth_path,
                auth_mode=self._creds.auth_mode,
            )
        logger.info(
            "[CodexAuth] refreshed — new expiry in %ds",
            self._creds.seconds_until_expiry,
        )

    # ------------------------------------------------------------- #
    # Public API                                                     #
    # ------------------------------------------------------------- #

    async def ensure_fresh(self) -> CodexCredential:
        if self._creds is None:
            self._load()
        assert self._creds is not None
        if self._creds.needs_refresh:
            await self._refresh()
        return self._creds

    async def auth_headers(self) -> dict[str, str]:
        """Return HTTP headers ready to attach to a ChatGPT backend call."""
        creds = await self.ensure_fresh()
        h = {
            "Authorization": f"Bearer {creds.access_token}",
            # Codex CLI also sends these:
            "OpenAI-Beta": "responses=experimental",
            "originator": "codex_cli_rs",
            "version": "0.122.0",
        }
        if creds.account_id:
            h["chatgpt-account-id"] = creds.account_id
        return h

    @property
    def account_id(self) -> str | None:
        return self._creds.account_id if self._creds else None

    @property
    def base_url(self) -> str:
        return os.environ.get(
            "QWENPAW_CODEX_BACKEND_URL",
            DEFAULT_CHATGPT_BACKEND,
        )


# -------------------------------------------------------------------------
# CLI smoke test — `python -m qwenpaw.providers.codex_auth`
# -------------------------------------------------------------------------

async def _smoke() -> None:
    """Refresh if needed + print header sketch (no secrets revealed)."""
    auth = CodexAuth()
    creds = await auth.ensure_fresh()
    headers = await auth.auth_headers()
    masked = {
        k: (v[:12] + "..." + v[-6:] if k == "Authorization" and len(v) > 30 else v)
        for k, v in headers.items()
    }
    print(f"auth_path: {creds.auth_path}")
    print(f"auth_mode: {creds.auth_mode}")
    print(f"account_id: {creds.account_id}")
    print(f"expires_in: {creds.seconds_until_expiry}s")
    print(f"base_url:  {auth.base_url}")
    print(f"headers:   {masked}")


if __name__ == "__main__":
    import asyncio
    logging.basicConfig(level=logging.INFO)
    asyncio.run(_smoke())
