# -*- coding: utf-8 -*-
"""Signal subprocess client: signal-cli jsonRpc over stdin/stdout.

Spawns `signal-cli -a <account> --output=json jsonRpc` as a child process and
speaks newline-delimited JSON-RPC 2.0 over its pipes. Replaces the previous
HTTP+SSE `SignalDaemon` (which required signal-cli running as an external
HTTP service).

Design notes:
- One subprocess per SignalChannel instance; lifecycle owned here.
- Requests keyed by auto-incrementing id, correlated via asyncio.Futures.
- Inbound notifications (method="receive") dispatched to on_notify callback.
- On process exit, supervisor respawns with backoff 5s → 60s.
"""
from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import shutil
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

NotifyCallback = Callable[[Dict[str, Any]], Awaitable[None]]

_RPC_TIMEOUT = 15.0
_BACKOFF_START = 5.0
_BACKOFF_MAX = 60.0
_TERM_GRACE_SEC = 3.0

def _iter_signal_cli_pids(account: str) -> "list[int]":
    """Find signal-cli processes belonging to the current user that
    target ``account``.  Uses /proc (Linux only) so there's no new
    dependency on psutil.  Returns an empty list on non-Linux or
    when /proc is unreadable — safer to skip the probe than to
    raise and block spawn entirely.
    """
    import sys
    if not sys.platform.startswith("linux"):
        return []
    proc_dir = Path("/proc")
    if not proc_dir.is_dir():
        return []
    my_uid = os.getuid()
    my_pid = os.getpid()
    hits: list[int] = []
    for entry in proc_dir.iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        if pid == my_pid:
            continue
        try:
            # Match UID first so we never touch another user's
            # processes even when the cmdline coincidentally looks
            # right.
            status = (entry / "status").read_text(encoding="utf-8")
        except OSError:
            continue
        if f"Uid:\t{my_uid}" not in status:
            continue
        try:
            cmdline = (entry / "cmdline").read_bytes()
        except OSError:
            continue
        if not cmdline:
            continue
        args = cmdline.split(b"\x00")
        if not any(b"signal-cli" in a for a in args):
            continue
        # ``-a <account>`` must appear so we only target the
        # contested account — a jsonRpc on a different phone number
        # is unrelated and must not be killed.
        arg_list = [a.decode("utf-8", "replace") for a in args if a]
        try:
            idx = arg_list.index("-a")
            if idx + 1 >= len(arg_list) or arg_list[idx + 1] != account:
                continue
        except ValueError:
            continue
        hits.append(pid)
    return hits


def _sticker_pack_uri(pack_id: str, pack_key: str) -> str:
    """Canonical sticker-pack install URI signal-cli's ``addStickerPack``
    expects.  Same fragment layout Signal Desktop ships in ``signal.art``
    share links — the RPC parses ``pack_id`` / ``pack_key`` from the
    fragment rather than taking them as separate fields.
    """
    return f"https://signal.art/addstickers/#pack_id={pack_id}&pack_key={pack_key}"


class SignalSubprocessClient:
    """signal-cli subprocess client speaking JSON-RPC over stdin/stdout."""

    def __init__(
        self,
        account: str,
        signal_cli_path: str = "signal-cli",
        extra_args: Optional[List[str]] = None,
        data_dir: Optional[Path | str] = None,
    ):
        self._account = account
        self._signal_cli_path = signal_cli_path
        self._extra_args = list(extra_args or [])
        if data_dir is None or data_dir == "":
            self._data_dir: Optional[Path] = None
        else:
            self._data_dir = Path(data_dir).expanduser()
        self._proc: Optional[asyncio.subprocess.Process] = None
        self._reader_task: Optional[asyncio.Task] = None
        self._stderr_task: Optional[asyncio.Task] = None
        self._supervisor_task: Optional[asyncio.Task] = None
        self._pending: Dict[int, asyncio.Future] = {}
        self._next_id = 1
        self._on_notify: Optional[NotifyCallback] = None
        self._connected = asyncio.Event()
        self._stopping = False
        self._write_lock = asyncio.Lock()

    # ── Public API ────────────────────────────────────────────────────

    @property
    def connected(self) -> bool:
        return self._connected.is_set()

    @property
    def account(self) -> str:
        return self._account

    async def connect(self, on_notify: NotifyCallback) -> bool:
        """Start supervisor; returns True if initial spawn succeeded."""
        if self._supervisor_task and not self._supervisor_task.done():
            return self._connected.is_set()
        self._on_notify = on_notify
        self._stopping = False
        if not self._binary_available():
            logger.error(
                "signal-cli not found (looked for %r). Signal channel disabled.\n"
                "  Install — pick the distribution for your platform:\n"
                "    Linux x86_64: download signal-cli-X.Y.Z-Linux-native.tar.gz\n"
                "                  (~97 MB, self-contained, no Java runtime needed)\n"
                "                  https://github.com/AsamK/signal-cli/releases\n"
                "    Linux ARM64 / macOS / Windows: download signal-cli-X.Y.Z.tar.gz\n"
                "                  (~98 MB JAR bundle, requires Java 21+)\n"
                "                  + brew install openjdk@21 (mac) /\n"
                "                    apt install openjdk-21-jre (Debian/Ubuntu) /\n"
                "                    MSI installer (Windows)\n"
                "  Then either put signal-cli on your $PATH or set\n"
                "  channels.signal.signal_cli_path to an absolute path in config.",
                self._signal_cli_path,
            )
            return False
        self._supervisor_task = asyncio.create_task(
            self._supervise(), name="signal_supervisor",
        )
        # Wait for first successful spawn (or initial failure)
        try:
            await asyncio.wait_for(self._connected.wait(), timeout=10.0)
            return True
        except asyncio.TimeoutError:
            logger.error("signal: subprocess failed to become ready within 10s")
            return False

    async def disconnect(self) -> None:
        """Stop supervisor and terminate subprocess."""
        self._stopping = True
        if self._supervisor_task:
            self._supervisor_task.cancel()
            try:
                await self._supervisor_task
            except (asyncio.CancelledError, Exception):
                pass
            self._supervisor_task = None
        await self._terminate_proc()
        self._connected.clear()
        # Fail any remaining pending futures
        self._fail_pending(ConnectionResetError("signal subprocess disconnected"))
        logger.info("signal: subprocess client stopped")

    async def call(
        self,
        method: str,
        params: Optional[Dict[str, Any]] = None,
        timeout: float = _RPC_TIMEOUT,
    ) -> Any:
        """Issue a JSON-RPC request and await the response."""
        if not self._connected.is_set() or not self._proc or self._proc.returncode is not None:
            raise ConnectionError("signal subprocess not connected")
        req_id = self._next_id
        self._next_id += 1
        payload: Dict[str, Any] = {"jsonrpc": "2.0", "id": req_id, "method": method}
        if params:
            payload["params"] = params
        loop = asyncio.get_running_loop()
        fut: asyncio.Future = loop.create_future()
        self._pending[req_id] = fut
        line = json.dumps(payload, ensure_ascii=False) + "\n"
        try:
            async with self._write_lock:
                assert self._proc.stdin is not None
                self._proc.stdin.write(line.encode("utf-8"))
                await self._proc.stdin.drain()
        except Exception as e:
            self._pending.pop(req_id, None)
            raise ConnectionError(f"signal stdin write failed: {e}") from e
        try:
            return await asyncio.wait_for(fut, timeout=timeout)
        except asyncio.TimeoutError:
            self._pending.pop(req_id, None)
            raise
        finally:
            self._pending.pop(req_id, None)

    # ── Convenience RPCs ──────────────────────────────────────────────

    async def send_message(
        self,
        target: str,
        text: str,
        is_group: bool = False,
        quote_timestamp: int = 0,
        quote_author: str = "",
        attachments: Optional[List[str]] = None,
        text_style: Optional[List[str]] = None,
        mentions: Optional[List[str]] = None,
    ) -> Optional[int]:
        """Send a Signal message. Returns sent timestamp on success."""
        params: Dict[str, Any] = {"account": self._account}
        if text:
            params["message"] = text
        if text_style:
            params["text-style"] = text_style
        if mentions:
            params["mention"] = mentions
        if is_group:
            params["groupId"] = target
        else:
            params["recipients"] = [target]
        if quote_timestamp and quote_author:
            params["quoteTimestamp"] = quote_timestamp
            params["quoteAuthor"] = quote_author
        if attachments:
            params["attachments"] = attachments
        try:
            result = await self.call("send", params)
        except Exception as e:
            logger.error("signal: send failed to %s: %s", target, e)
            return None
        if isinstance(result, dict) and "timestamp" in result:
            return int(result["timestamp"])
        logger.error("signal: unexpected send result: %r", result)
        return None

    async def send_reaction(
        self,
        target: str,
        emoji: str,
        target_author: str,
        target_timestamp: int,
        is_group: bool = False,
        remove: bool = False,
    ) -> bool:
        params: Dict[str, Any] = {
            "account": self._account,
            "emoji": emoji,
            "targetAuthor": target_author,
            "targetTimestamp": target_timestamp,
        }
        if remove:
            params["remove"] = True
        if is_group:
            params["groupId"] = target
        else:
            params["recipients"] = [target]
        try:
            await self.call("sendReaction", params)
            return True
        except Exception as e:
            logger.error("signal: sendReaction failed: %s", e)
            return False

    async def send_typing(
        self, target: str, start: bool = True, is_group: bool = False,
    ) -> None:
        method = "sendTyping" if start else "stopTyping"
        params: Dict[str, Any] = {"account": self._account}
        if is_group:
            params["groupId"] = target
        else:
            params["recipients"] = [target]
        try:
            await self.call(method, params)
        except Exception:
            # Typing is best-effort
            pass

    async def download_attachment(
        self, attachment_id: str, dest_dir: Path,
    ) -> Optional[Path]:
        """Download an attachment. signal-cli >=0.13 autosaves to
        ~/.local/share/signal-cli/attachments/<id>; we try that first, then
        fall back to the getAttachment RPC (which returns base64 data).

        The returned path is **always** placed under ``dest_dir`` —
        the callers rely on the media server being able to sign it,
        and the media server's ``allowed_dirs`` allow-list covers
        CoPaw's channel dirs but not signal-cli's default autosave
        location.  When signal-cli has already decoded the blob to
        its own attachments dir we copy it across rather than
        pointing at the original; the signal-cli directory still
        acts as a secondary cache.
        """
        # Fast path: file already saved by signal-cli — copy into
        # dest_dir so media-server signing + resolve_media_url work.
        default_dir = Path.home() / ".local" / "share" / "signal-cli" / "attachments"
        candidate = default_dir / attachment_id
        if candidate.is_file():
            try:
                dest_dir.mkdir(parents=True, exist_ok=True)
                # Preserve the signal-cli filename (already includes
                # extension for modern signal-cli versions — e.g.
                # ``<id>.jpg``).  Fall back to the bare id when not.
                dest = dest_dir / candidate.name
                if not dest.exists() or dest.stat().st_size != candidate.stat().st_size:
                    import shutil
                    shutil.copy2(candidate, dest)
                return dest
            except Exception as e:
                logger.warning(
                    "signal: failed to copy attachment %s into %s: %s "
                    "— falling back to signal-cli path",
                    attachment_id, dest_dir, e,
                )
                return candidate
        # RPC fallback
        try:
            result = await self.call(
                "getAttachment", {"account": self._account, "id": attachment_id},
            )
        except Exception as e:
            logger.error("signal: getAttachment failed: %s", e)
            return None
        if isinstance(result, str):
            p = Path(result)
            return p if p.is_file() else None
        if isinstance(result, dict) and result.get("data"):
            dest_dir.mkdir(parents=True, exist_ok=True)
            ctype = result.get("contentType", "application/octet-stream")
            ext = ctype.split("/")[-1] or "bin"
            dest = dest_dir / f"signal_att_{attachment_id[:8]}.{ext}"
            try:
                dest.write_bytes(base64.b64decode(result["data"]))
            except Exception as e:
                logger.error("signal: attachment decode failed: %s", e)
                return None
            return dest
        return None

    async def list_sticker_packs(self) -> List[Dict[str, Any]]:
        """List known sticker packs (installed + uploaded by this account).

        signal-cli's ``listStickerPacks`` returns an array of
        ``{packId, packKey, title, author, installed, stickers:
        [{id, emoji, contentType, fileName}]}`` objects.  We
        forward the raw list verbatim — the tool wrapper shapes it
        into something the agent can reason about.  Empty list on
        error (the RPC is read-only, so callers treat it as "no
        packs" rather than surfacing the subprocess failure).
        """
        try:
            result = await self.call("listStickerPacks")
        except Exception as e:
            logger.warning("signal: listStickerPacks failed: %s", e)
            return []
        if isinstance(result, list):
            return result
        if isinstance(result, dict):
            packs = result.get("packs")
            if isinstance(packs, list):
                return packs
        logger.warning(
            "signal: listStickerPacks returned unexpected shape: %r",
            type(result).__name__,
        )
        return []

    async def add_sticker_pack(
        self,
        pack_id: str,
        pack_key: str,
    ) -> bool:
        """Install a sticker pack by id + key.  Equivalent to
        tapping a signal.art share link.  Idempotent — re-installing
        an already-installed pack returns success."""
        try:
            await self.call(
                "addStickerPack",
                {"uri": _sticker_pack_uri(pack_id, pack_key)},
            )
            return True
        except Exception as e:
            logger.error(
                "signal: addStickerPack %s failed: %s",
                pack_id[:12], e,
            )
            return False

    async def upload_sticker_pack(self, manifest_path: str) -> Optional[str]:
        """Upload a sticker pack from a directory (containing
        ``manifest.json`` + numbered ``<id>.webp`` files) or a zip.

        Returns the signal.art share URL on success — callers parse
        that URL for ``pack_id`` + ``pack_key``.  Returns ``None`` on
        any RPC failure; the caller is expected to surface that to
        the agent so it can retry or show the user a specific error.
        """
        try:
            result = await self.call(
                "uploadStickerPack",
                {"path": manifest_path},
                timeout=120.0,
            )
        except Exception as e:
            logger.error("signal: uploadStickerPack failed: %s", e)
            return None
        # signal-cli returns either a bare URL string or
        # ``{"url": "..."}`` depending on version.  Tolerate both.
        if isinstance(result, str):
            return result
        if isinstance(result, dict):
            url = result.get("url") or result.get("packUrl")
            if isinstance(url, str):
                return url
        logger.warning(
            "signal: uploadStickerPack returned unexpected shape: %r",
            type(result).__name__,
        )
        return None

    async def send_sticker_message(
        self,
        target: str,
        pack_id: str,
        sticker_id: int,
        is_group: bool = False,
    ) -> Optional[int]:
        """Send a sticker by pack reference.  ``pack_id`` + ``sticker_id``
        must correspond to a pack the *sender* has access to (either
        installed or uploaded by this account); the recipient auto-
        fetches the sticker from Signal's sticker CDN on receipt.
        """
        params: Dict[str, Any] = {
            "account": self._account,
            "sticker": f"{pack_id}:{sticker_id}",
        }
        if is_group:
            params["groupId"] = target
        else:
            params["recipients"] = [target]
        try:
            result = await self.call("send", params)
        except Exception as e:
            logger.error(
                "signal: send sticker %s:%s to %s failed: %s",
                pack_id[:12], sticker_id, target[:20], e,
            )
            return None
        if isinstance(result, dict) and "timestamp" in result:
            return int(result["timestamp"])
        return None

    async def get_sticker(
        self,
        pack_id: str,
        sticker_id: int,
        dest_dir: Path,
        *,
        pack_key: Optional[str] = None,
    ) -> Optional[Path]:
        """Fetch a single sticker by ``packId:stickerId``.

        Uses signal-cli's ``getSticker`` RPC, which returns the image
        base64-encoded.  When the pack isn't locally installed yet
        (``getSticker`` 500s with "StickerPackNotFoundException" or
        similar) and a ``pack_key`` is supplied, we try
        ``addStickerPack`` once and retry — this is the common case
        for an inbound sticker from a pack the bot has never seen.

        The sticker is persisted as
        ``signal_sticker_<pack_prefix>_<id>.webp`` under ``dest_dir``
        (lined up with :meth:`download_attachment` so the media
        server's ``allowed_dirs`` signing covers the path).  Returns
        ``None`` when the RPC can't produce bytes after both
        attempts.
        """
        params = {"packId": pack_id, "stickerId": sticker_id}

        async def _rpc() -> Any:
            return await self.call("getSticker", params)

        try:
            result = await _rpc()
        except Exception as e:
            # Retry path: install-then-fetch when we have the key.
            if pack_key:
                try:
                    await self.call(
                        "addStickerPack",
                        {"uri": _sticker_pack_uri(pack_id, pack_key)},
                    )
                    result = await _rpc()
                except Exception as retry_exc:
                    logger.warning(
                        "signal: getSticker %s:%s failed even after "
                        "addStickerPack: %s",
                        pack_id[:12], sticker_id, retry_exc,
                    )
                    return None
            else:
                logger.warning(
                    "signal: getSticker %s:%s failed: %s",
                    pack_id[:12], sticker_id, e,
                )
                return None

        # signal-cli may return the raw base64 string or a dict with
        # ``data`` key — mirror ``getAttachment`` handling.
        if isinstance(result, dict):
            payload = result.get("data") or result.get("sticker")
        else:
            payload = result
        if not isinstance(payload, str) or not payload:
            logger.warning(
                "signal: getSticker %s:%s returned no data: %r",
                pack_id[:12], sticker_id, type(result).__name__,
            )
            return None

        try:
            raw = base64.b64decode(payload)
        except Exception as e:
            logger.error("signal: sticker base64 decode failed: %s", e)
            return None

        try:
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest = (
                dest_dir / f"signal_sticker_{pack_id[:8]}_{sticker_id}.webp"
            )
            dest.write_bytes(raw)
            return dest
        except OSError as e:
            logger.error("signal: sticker write failed: %s", e)
            return None

    async def whoami(self) -> Optional[Dict[str, Any]]:
        try:
            return await self.call("version")
        except Exception:
            return None

    # ── Subprocess supervision ────────────────────────────────────────

    def _binary_available(self) -> bool:
        # Allow absolute paths; otherwise look on PATH
        if os.path.sep in self._signal_cli_path or self._signal_cli_path.startswith("./"):
            return Path(self._signal_cli_path).is_file()
        return shutil.which(self._signal_cli_path) is not None

    def _build_cmd(self) -> List[str]:
        cmd = [self._signal_cli_path]
        # signal-cli requires -c BEFORE account-scoped flags like -a.
        if self._data_dir is not None:
            cmd += ["-c", str(self._data_dir)]
        if self._account:
            cmd += ["-a", self._account]
        cmd += ["--output=json", "jsonRpc"]
        cmd += self._extra_args
        return cmd

    async def _supervise(self) -> None:
        backoff = _BACKOFF_START
        while not self._stopping:
            spawned = await self._spawn_once()
            if not spawned:
                if self._stopping:
                    return
                logger.warning(
                    "signal: subprocess spawn failed; retrying in %.0fs", backoff,
                )
                try:
                    await asyncio.sleep(backoff)
                except asyncio.CancelledError:
                    return
                backoff = min(backoff * 2, _BACKOFF_MAX)
                continue
            backoff = _BACKOFF_START
            # Wait for the process to exit
            assert self._proc is not None
            try:
                rc = await self._proc.wait()
            except asyncio.CancelledError:
                await self._terminate_proc()
                return
            logger.warning("signal: subprocess exited with code %s", rc)
            self._connected.clear()
            self._fail_pending(ConnectionResetError("signal subprocess exited"))
            if self._stopping:
                return
            logger.info("signal: respawning in %.0fs", backoff)
            try:
                await asyncio.sleep(backoff)
            except asyncio.CancelledError:
                return
            backoff = min(backoff * 2, _BACKOFF_MAX)

    async def _reap_account_orphans(self) -> None:
        """Look for foreign signal-cli processes bound to this account
        and request them to quit before we spawn our own.

        "Foreign" = this user's UID, targets our ``-a <account>``,
        but is not the current ``self._proc`` subprocess.  Covers
        the orphan-from-crash case: a prior copaw exited (port
        bind failure, OOM, SIGKILL) but its signal-cli child got
        reparented to init and is still holding the Java
        ``FileChannel`` lock on the account db.

        Two-stage kill: SIGTERM + ~3s grace, then SIGKILL so we
        still make progress even if the orphan is wedged.  The
        alternative — waiting for signal-cli's own lock contention
        prompt to time out — silently manifests as 2-min
        ``uploadStickerPack`` stalls.
        """
        import signal as _sig
        import sys

        if not sys.platform.startswith("linux"):
            return
        my_child = self._proc.pid if self._proc else -1
        candidates = [
            pid for pid in _iter_signal_cli_pids(self._account)
            if pid != my_child
        ]
        if not candidates:
            return
        logger.warning(
            "signal: reaping %d orphan signal-cli pid(s) on account %s: %s",
            len(candidates), self._account, candidates,
        )
        for pid in candidates:
            try:
                os.kill(pid, _sig.SIGTERM)
            except ProcessLookupError:
                pass
            except PermissionError:
                logger.warning(
                    "signal: cannot SIGTERM orphan pid %d (EPERM) — "
                    "skipping", pid,
                )
        # Give them a moment to unwind the lock cleanly.
        for _ in range(30):  # up to 3s in 100ms steps
            still = [
                pid for pid in _iter_signal_cli_pids(self._account)
                if pid != my_child
            ]
            if not still:
                return
            await asyncio.sleep(0.1)
        # Still alive — force.  This is the fallback path; most
        # signal-cli versions honour SIGTERM within a second.
        still = [
            pid for pid in _iter_signal_cli_pids(self._account)
            if pid != my_child
        ]
        for pid in still:
            logger.warning(
                "signal: orphan pid %d did not exit on SIGTERM, sending "
                "SIGKILL", pid,
            )
            try:
                os.kill(pid, _sig.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass

    async def _spawn_once(self) -> bool:
        # Ensure the data-dir exists so signal-cli can create its SQLite
        # store on first link. mkdir is a no-op on existing dirs.
        if self._data_dir is not None:
            try:
                self._data_dir.mkdir(parents=True, exist_ok=True)
            except OSError as e:
                logger.error(
                    "signal: failed to create data_dir %s: %s",
                    self._data_dir, e,
                )
                return False
        # Pre-spawn: reap any orphan signal-cli that still owns the
        # account-file lock.  Without this, the fresh jsonRpc daemon
        # just stalls with ``Config file is in use by another instance,
        # waiting…`` until an RPC timeout fires (2 min for
        # ``uploadStickerPack``), which surfaces to the agent as an
        # opaque upload failure.  Skips cleanly when there aren't any.
        await self._reap_account_orphans()
        cmd = self._build_cmd()
        logger.info("signal: spawning %s", " ".join(cmd))
        try:
            self._proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                # New session detaches from copaw's controlling
                # terminal so TTY signals (SIGINT, SIGHUP) from a
                # dev shell don't take signal-cli down with us.
                # We deliberately DON'T use ``PR_SET_PDEATHSIG``
                # here — that signal is bound to the parent
                # *thread*, not the parent process, and asyncio's
                # fork happens on a helper thread whose lifetime
                # is shorter than a single child.  Using it caused
                # a SIGTERM storm every few seconds, killing the
                # very signal-cli we just spawned.  The orphan
                # safety net in ``_reap_account_orphans`` handles
                # the "copaw crashed and left a lock-holding
                # signal-cli behind" case at next-spawn time.
                start_new_session=True,
            )
        except FileNotFoundError:
            logger.error("signal: binary not found: %s", self._signal_cli_path)
            return False
        except Exception as e:
            logger.exception("signal: failed to spawn: %s", e)
            return False
        # Start reader / stderr tasks
        self._reader_task = asyncio.create_task(
            self._read_stdout(), name="signal_reader",
        )
        self._stderr_task = asyncio.create_task(
            self._read_stderr(), name="signal_stderr",
        )
        self._connected.set()
        return True

    async def _terminate_proc(self) -> None:
        proc = self._proc
        if not proc:
            return
        if proc.returncode is None:
            try:
                proc.terminate()
            except ProcessLookupError:
                pass
            try:
                await asyncio.wait_for(proc.wait(), timeout=_TERM_GRACE_SEC)
            except asyncio.TimeoutError:
                logger.warning("signal: SIGTERM grace expired; sending SIGKILL")
                try:
                    proc.kill()
                except ProcessLookupError:
                    pass
                try:
                    await proc.wait()
                except Exception:
                    pass
        for task in (self._reader_task, self._stderr_task):
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    pass
        self._reader_task = None
        self._stderr_task = None
        self._proc = None

    async def _read_stdout(self) -> None:
        proc = self._proc
        if not proc or not proc.stdout:
            return
        try:
            while True:
                line = await proc.stdout.readline()
                if not line:
                    return  # EOF
                text = line.decode("utf-8", errors="replace").strip()
                if not text:
                    continue
                try:
                    msg = json.loads(text)
                except json.JSONDecodeError:
                    logger.debug("signal: non-JSON stdout line ignored: %s", text[:200])
                    continue
                await self._dispatch(msg)
        except asyncio.CancelledError:
            return
        except Exception:
            logger.exception("signal: reader task crashed")

    async def _read_stderr(self) -> None:
        proc = self._proc
        if not proc or not proc.stderr:
            return
        try:
            while True:
                line = await proc.stderr.readline()
                if not line:
                    return
                text = line.decode("utf-8", errors="replace").rstrip()
                if text:
                    logger.warning("signal-cli: %s", text)
        except asyncio.CancelledError:
            return
        except Exception:
            logger.exception("signal: stderr task crashed")

    async def _dispatch(self, msg: Dict[str, Any]) -> None:
        # Response to a pending call
        if "id" in msg and (msg.get("jsonrpc") == "2.0" or "result" in msg or "error" in msg):
            req_id = msg["id"]
            try:
                req_id_int = int(req_id)
            except (TypeError, ValueError):
                req_id_int = None
            fut = self._pending.pop(req_id_int, None) if req_id_int is not None else None
            if fut and not fut.done():
                err = msg.get("error")
                if err:
                    fut.set_exception(
                        RuntimeError(f"signal RPC error: {err}"),
                    )
                else:
                    fut.set_result(msg.get("result"))
            return
        # Inbound notification
        method = msg.get("method")
        if method == "receive":
            params = msg.get("params") or {}
            if self._on_notify:
                try:
                    await self._on_notify(params)
                except Exception:
                    logger.exception("signal: on_notify handler failed")
            return
        # Anything else — just log
        logger.debug("signal: ignored stdout frame: %s", json.dumps(msg)[:200])

    def _fail_pending(self, exc: BaseException) -> None:
        pending = self._pending
        self._pending = {}
        for fut in pending.values():
            if not fut.done():
                fut.set_exception(exc)
