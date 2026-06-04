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
import contextlib
import errno
import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

try:
    import fcntl  # POSIX-only; absent on Windows.
except ImportError:  # pragma: no cover - non-POSIX fallback
    fcntl = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

# Public Codex-CLI OAuth client (same identifier openai/codex uses).
OPENAI_OAUTH_CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
OPENAI_TOKEN_ENDPOINT = "https://auth.openai.com/oauth/token"

# ChatGPT backend that actually serves Codex traffic.
DEFAULT_CHATGPT_BACKEND = "https://chatgpt.com/backend-api"

# Refresh the access_token this many seconds BEFORE its JWT `exp` so an
# in-flight request never races the expiration edge.
REFRESH_SAFETY_MARGIN_S = 5 * 60

# ``version`` header + ``client_version`` query param advertised to
# ChatGPT's Codex backend.  The backend gates which models this
# token can reach on this value:
#
#   0.122.0: sees gpt-5.4 / 5.4-mini / 5.3-codex / 5.2
#   0.200.0: above + gpt-5.5
#
# The Codex CLI source is the moving source of truth — bump when a
# newer model is gated behind a newer client string.  Env override
# lets ops flip this without a redeploy.  Too-fresh values never 400
# (the backend only cares that it's >= the gate), so erring high is
# safe.
CODEX_CLIENT_VERSION: str = os.environ.get(
    "QWENPAW_CODEX_CLIENT_VERSION",
    "0.200.0",
)


def _decode_jwt_claims(jwt: str) -> dict | None:
    """Best-effort base64url-decode the payload segment of a JWT and
    return the claims dict.  Returns ``None`` when the token isn't a
    parseable JWT.  Used for both the ``exp`` claim on the access_token
    and the ChatGPT account-info claims on the id_token.
    """
    try:
        payload_b64 = jwt.split(".")[1]
        payload_b64 += "=" * (-len(payload_b64) % 4)
        return json.loads(
            base64.urlsafe_b64decode(payload_b64.encode("ascii")),
        )
    except Exception:
        return None


def _decode_jwt_exp_ms(jwt: str) -> int | None:
    """Best-effort extract the ``exp`` claim (in ms) from a JWT."""
    claims = _decode_jwt_claims(jwt)
    if claims is None:
        return None
    exp = claims.get("exp")
    if isinstance(exp, (int, float)):
        return int(exp * 1000)
    return None


@dataclass
class CodexAccountInfo:
    """ChatGPT-account metadata extracted from the id_token claims.
    All fields optional — tokens issued to API-key mode carry no
    ChatGPT account info."""

    email: str | None = None
    plan_type: str | None = (
        None  # "pro" / "plus" / "team" / "business" / "enterprise"
    )
    org_title: str | None = None
    subscription_active_until: str | None = None  # ISO8601 from OpenAI


def _extract_account_info(id_token: str | None) -> CodexAccountInfo:
    """Pull ChatGPT account info out of the id_token claims.

    OpenAI's id_token nests all ChatGPT-specific claims under the
    ``https://api.openai.com/auth`` key, alongside top-level OIDC
    claims like ``email``.  We surface just the human-facing subset
    the UI shows (plan, email, org) and skip internal IDs.
    """
    if not id_token:
        return CodexAccountInfo()
    claims = _decode_jwt_claims(id_token)
    if not claims:
        return CodexAccountInfo()
    openai_auth = claims.get("https://api.openai.com/auth") or {}
    orgs = openai_auth.get("organizations") or []
    default_org = next(
        (o for o in orgs if isinstance(o, dict) and o.get("is_default")),
        orgs[0] if orgs else None,
    )
    return CodexAccountInfo(
        email=claims.get("email")
        if isinstance(claims.get("email"), str)
        else None,
        plan_type=openai_auth.get("chatgpt_plan_type"),
        org_title=(
            default_org.get("title") if isinstance(default_org, dict) else None
        ),
        subscription_active_until=openai_auth.get(
            "chatgpt_subscription_active_until",
        ),
    )


def _resolve_auth_path() -> Path:
    home = os.environ.get("CODEX_HOME") or "~/.codex"
    return Path(os.path.expanduser(home)) / "auth.json"


# How long a process will wait to acquire the cross-process refresh lock
# before giving up and proceeding without it (so a stale/orphaned lock can
# never hard-block token refresh forever).  A refresh round-trip is well
# under this; the wait only matters when a *sibling* process is mid-refresh.
_REFRESH_LOCK_TIMEOUT_S = 30.0
_REFRESH_LOCK_POLL_S = 0.2


@contextlib.contextmanager
def _cross_process_refresh_lock(auth_path: Path):
    """Serialize codex-token refreshes across *every* process that shares the
    same ``~/.codex/auth.json``.

    Root cause this guards against (see
    research/codex-oauth-revoke-diagnosis-20260602.md): OpenAI rotates the
    ``refresh_token`` on every refresh and runs reuse-detection — if two
    processes (the long-lived ``copaw app`` runtime *and* a transient
    ``copaw task`` / cron worker, which each spawn their own interpreter and
    therefore their own ``threading.Lock``) refresh concurrently, the slower
    one replays an already-rotated refresh_token and OpenAI revokes the whole
    token family → fleet-wide ``token_revoked``.

    A ``flock`` on a sibling lock-file makes refresh fleet-wide single-flight:
    at most one process performs a refresh at a time, and every other process
    blocks until the holder has written the new tokens to disk (after which
    they re-read disk and adopt the fresh token instead of refreshing again).

    Degrades gracefully:
      * On non-POSIX / no-``fcntl`` builds it is a no-op (single-process
        ``threading.Lock`` still applies).
      * If the lock cannot be acquired within ``_REFRESH_LOCK_TIMEOUT_S`` it
        proceeds anyway (logged) so a crashed lock-holder cannot deadlock
        token refresh permanently.
    """
    if fcntl is None:
        # No cross-process locking primitive available; rely on the
        # in-process threading.Lock only.
        yield False
        return

    lock_path = auth_path.with_suffix(auth_path.suffix + ".refresh.lock")
    fd = None
    acquired = False
    try:
        try:
            lock_path.parent.mkdir(parents=True, exist_ok=True)
        except OSError:
            pass
        fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR, 0o600)
        deadline = time.monotonic() + _REFRESH_LOCK_TIMEOUT_S
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                acquired = True
                break
            except OSError as exc:
                if exc.errno not in (errno.EACCES, errno.EAGAIN):
                    raise
                if time.monotonic() >= deadline:
                    logger.warning(
                        "[CodexAuth] refresh lock busy >%.0fs — proceeding "
                        "without cross-process lock (possible stale holder)",
                        _REFRESH_LOCK_TIMEOUT_S,
                    )
                    break
                time.sleep(_REFRESH_LOCK_POLL_S)
        yield acquired
    finally:
        if fd is not None:
            try:
                if acquired:
                    fcntl.flock(fd, fcntl.LOCK_UN)
            except OSError:
                pass
            with contextlib.suppress(OSError):
                os.close(fd)


@dataclass
class CodexCredential:
    access_token: str
    refresh_token: str
    id_token: str | None
    account_id: str | None
    expires_at_ms: int  # wall-clock ms, from JWT exp or fallback
    auth_path: Path
    auth_mode: str = "chatgpt"
    account_info: CodexAccountInfo = field(default_factory=CodexAccountInfo)

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
        # Track the file mtime we loaded from so ``ensure_fresh`` can
        # detect external rewrites (a fresh ``codex login`` completing
        # while CoPaw is running) and transparently pick up new tokens
        # without a process restart.
        self._loaded_mtime_ns: int = 0
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
        # Read mtime BEFORE reading content so a concurrent write
        # doesn't leave us with new content tagged as stale.  Using
        # ``st_mtime_ns`` avoids float rounding issues at sub-second
        # resolution.
        stat = self._auth_path.stat()
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
            exp_ms = int(stat.st_mtime * 1000) + 60 * 60 * 1000

        id_token = tokens.get("id_token")
        self._creds = CodexCredential(
            access_token=access_token,
            refresh_token=refresh_token,
            id_token=id_token,
            account_id=tokens.get("account_id"),
            expires_at_ms=exp_ms,
            auth_path=self._auth_path,
            auth_mode=str(raw.get("auth_mode") or "chatgpt"),
            account_info=_extract_account_info(id_token),
        )
        self._loaded_mtime_ns = stat.st_mtime_ns

    def _save(self, *, tokens: dict[str, Any]) -> None:
        """Write refreshed tokens back atomically, preserving other fields."""
        raw = json.loads(self._auth_path.read_text())
        raw.setdefault("tokens", {})
        raw["tokens"].update(tokens)
        raw["last_refresh"] = time.strftime(
            "%Y-%m-%dT%H:%M:%SZ",
            time.gmtime(),
        )
        tmp = self._auth_path.with_suffix(self._auth_path.suffix + ".tmp")
        tmp.write_text(json.dumps(raw, indent=2))
        os.chmod(tmp, 0o600)
        os.replace(tmp, self._auth_path)
        # Advance our mtime marker so the freshly-rewritten file — which
        # we just authored — is not treated as an external change by
        # ``ensure_fresh``.  Without this, the very next call would
        # redundantly ``_load`` the same bytes we just wrote.
        self._loaded_mtime_ns = self._auth_path.stat().st_mtime_ns

    # ------------------------------------------------------------- #
    # Refresh                                                        #
    # ------------------------------------------------------------- #

    async def _refresh(self) -> None:
        """Exchange the refresh_token for a fresh access_token.

        Fleet-safe: the actual network refresh (which OpenAI rotates the
        refresh_token on, and runs reuse-detection against) is serialized
        across every process sharing this ``auth.json`` by a cross-process
        ``flock``.  Once the lock is held we *re-read disk first* — if a
        sibling process already refreshed while we were waiting, we simply
        adopt its fresh token and skip the network call, so the
        already-rotated refresh_token is never replayed (which is what
        triggers OpenAI's family-wide ``token_revoked``).
        """
        assert self._creds is not None
        with _cross_process_refresh_lock(self._auth_path) as locked:
            # Post-lock disk recheck: a sibling may have refreshed while we
            # blocked on the lock.  Re-read the newest tokens from disk and,
            # if they are now fresh, adopt them and return without hitting
            # the network (avoids refresh_token reuse → family revoke).
            try:
                disk_mtime_ns = self._auth_path.stat().st_mtime_ns
            except OSError:
                disk_mtime_ns = self._loaded_mtime_ns
            if disk_mtime_ns > self._loaded_mtime_ns:
                with self._lock:
                    self._load()
                assert self._creds is not None
            if not self._creds.needs_refresh:
                logger.info(
                    "[CodexAuth] token already refreshed by another process "
                    "(adopted from disk, fresh for %ds) — skipping network "
                    "refresh",
                    self._creds.seconds_until_expiry,
                )
                return

            logger.info(
                "[CodexAuth] refreshing access_token (expires in %ds)%s",
                self._creds.seconds_until_expiry,
                "" if locked else " [WITHOUT cross-process lock]",
            )
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(
                    OPENAI_TOKEN_ENDPOINT,
                    data={
                        "grant_type": "refresh_token",
                        "client_id": OPENAI_OAUTH_CLIENT_ID,
                        # Always use the newest refresh_token on disk (set by
                        # the recheck above), never a stale in-memory copy.
                        "refresh_token": self._creds.refresh_token,
                        "scope": "openid profile email offline_access",
                    },
                    headers={
                        "Content-Type": "application/x-www-form-urlencoded",
                    },
                )
                resp.raise_for_status()
                body = resp.json()

            new_access = body.get("access_token")
            new_refresh = body.get("refresh_token") or self._creds.refresh_token
            new_id = body.get("id_token") or self._creds.id_token
            if not new_access:
                raise RuntimeError(
                    "Codex token refresh succeeded but no access_token in "
                    "response",
                )

            exp_ms = _decode_jwt_exp_ms(new_access) or (
                int(time.time() * 1000) + 3600 * 1000
            )
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
        else:
            # Pick up out-of-band rewrites (eg. a fresh ``codex login``
            # on the host while CoPaw keeps running).  stat() failures
            # fall through to the cached creds — the next refresh will
            # surface any real problem.
            try:
                disk_mtime_ns = self._auth_path.stat().st_mtime_ns
            except OSError:
                disk_mtime_ns = self._loaded_mtime_ns
            if disk_mtime_ns > self._loaded_mtime_ns:
                logger.info(
                    "[CodexAuth] auth.json changed on disk — reloading",
                )
                self._load()
        assert self._creds is not None
        if self._creds.needs_refresh:
            await self._refresh()
        return self._creds

    def reload(self) -> CodexCredential:
        """Force-reread ``auth.json`` from disk, regardless of mtime.
        Used by the console's explicit Reload action.  Synchronous —
        a subsequent request will refresh the token lazily if needed.
        """
        with self._lock:
            self._load()
        assert self._creds is not None
        return self._creds

    async def auth_headers(self) -> dict[str, str]:
        """Return HTTP headers ready to attach to a ChatGPT backend call."""
        creds = await self.ensure_fresh()
        h = {
            "Authorization": f"Bearer {creds.access_token}",
            # Codex CLI also sends these:
            "OpenAI-Beta": "responses=experimental",
            "originator": "codex_cli_rs",
            "version": CODEX_CLIENT_VERSION,
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

    async def list_models(
        self,
        *,
        client_version: str | None = None,
        timeout: float = 10.0,
    ) -> list[dict]:
        """Query the ChatGPT backend's model catalogue for this account.

        Endpoint: ``GET {base_url}/codex/models?client_version=...``.
        Requires a bearer token — refreshes before the call.

        Returns the raw ``models`` array from the backend (each entry is
        a dict with ``slug`` / ``display_name`` / ``visibility`` /
        ``supported_reasoning_levels`` etc).  Callers filter /
        transform into ``ModelInfo`` instances themselves so we don't
        couple this low-level loader to the provider schema.

        The ``client_version`` default matches the one sent on
        ``auth_headers`` so the backend always sees a consistent
        client identity.
        """
        headers = await self.auth_headers()
        effective_version = client_version or CODEX_CLIENT_VERSION
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.get(
                f"{self.base_url}/codex/models",
                headers=headers,
                params={"client_version": effective_version},
            )
            resp.raise_for_status()
            data = resp.json()
        models = data.get("models")
        return models if isinstance(models, list) else []


# -------------------------------------------------------------------------
# CLI smoke test — `python -m qwenpaw.providers.codex_auth`
# -------------------------------------------------------------------------


async def _smoke() -> None:
    """Refresh if needed + print header sketch (no secrets revealed)."""
    auth = CodexAuth()
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
