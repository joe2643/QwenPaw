# -*- coding: utf-8 -*-
"""Cloudflare Tunnel driver — supports Quick Tunnels and Named Tunnels.

Quick Tunnel (``mode="quick"``): runs ``cloudflared tunnel --url
http://localhost:<port>`` and extracts the generated
``*.trycloudflare.com`` URL from stderr. No account required. URL rotates
on every restart.

Named Tunnel (``mode="named"``): runs ``cloudflared tunnel [--config
<file>] run [--url http://localhost:<port>] <tunnel_name>``. Requires the
tunnel to be pre-created via ``cloudflared tunnel login`` + ``tunnel
create``, with DNS CNAME already pointing at it. The driver does not
know the hostname from the subprocess — the caller supplies it.
"""
from __future__ import annotations

import asyncio
import logging
import os
import re
import signal
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Literal, Optional

from .binary_manager import BinaryManager

logger = logging.getLogger(__name__)

# Pattern to extract the public URL from Quick Tunnel stderr output.
_URL_RE = re.compile(r"https://[a-zA-Z0-9\-]+\.trycloudflare\.com")

# Signatures in Named Tunnel stderr that mean "connection registered OK".
_NAMED_READY_RE = re.compile(
    r"Registered tunnel connection|Connection .* registered|"
    r"Connection registered",
)


@dataclass
class TunnelInfo:
    """Information about a running Cloudflare Tunnel."""

    public_url: str  # "https://abc123.trycloudflare.com" or user hostname
    public_wss_url: str  # wss://... equivalent
    started_at: datetime
    pid: Optional[int] = None


class CloudflareTunnelDriver:
    """Manage a Cloudflare Tunnel subprocess (Quick or Named).

    Quick Tunnel usage::

        driver = CloudflareTunnelDriver()  # mode="quick" default
        info = await driver.start(8089)
        print(info.public_url)  # https://<random>.trycloudflare.com

    Named Tunnel usage::

        driver = CloudflareTunnelDriver(
            mode="named",
            tunnel_name="media",
            hostname="media.example.com",
        )
        info = await driver.start(8089)
        print(info.public_url)  # https://media.example.com
    """

    def __init__(
        self,
        binary_manager: BinaryManager | None = None,
        mode: Literal["quick", "named"] = "quick",
        tunnel_name: str = "",
        hostname: str = "",
        config_file: str = "",
    ) -> None:
        self._binary_mgr = binary_manager or BinaryManager()
        self._mode = mode
        self._tunnel_name = tunnel_name
        self._hostname = hostname
        self._config_file = config_file
        self._process: Optional[asyncio.subprocess.Process] = None
        self._info: Optional[TunnelInfo] = None
        self._monitor_task: Optional[asyncio.Task] = None

        if mode == "named" and not tunnel_name:
            raise ValueError("named tunnel requires tunnel_name")
        if mode == "named" and not hostname:
            raise ValueError(
                "named tunnel requires hostname (used as public URL)",
            )

    async def start(self, local_port: int) -> TunnelInfo:
        """Start the tunnel and return connection info.

        For Quick Tunnels: blocks until the public URL is detected in
        cloudflared stderr (typically 2-5 seconds).
        For Named Tunnels: blocks until cloudflared reports that the
        tunnel connection has been registered with the edge.
        """
        if self._process and self._process.returncode is None:
            await self.stop()

        # Clean up any orphan cloudflared instance that survived a previous
        # crash.  Our own ``stop()`` only touches ``self._process``, so if
        # the parent CoPaw was killed ungracefully (SIGKILL, uncaught
        # exception), the child cloudflared is re-parented to init (PPID=1)
        # and keeps serving the tunnel.  When CoPaw restarts and spawns a
        # new cloudflared for the same named tunnel, Cloudflare ends up
        # with two concurrent edge connections for one tunnel — wastes
        # quota and causes subtle routing flaps.
        await self._kill_orphans(local_port)

        binary = await self._binary_mgr.get_binary_path()

        cmd = self._build_command(binary, local_port)
        logger.info(
            "Starting cloudflared %s tunnel -> http://localhost:%d (cmd=%s)",
            self._mode,
            local_port,
            " ".join(cmd),
        )

        self._process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
        )

        if self._mode == "quick":
            url = await self._wait_for_url(timeout=30)
            if not url:
                await self.stop()
                raise RuntimeError(
                    "cloudflared did not emit a trycloudflare URL "
                    "within 30 seconds",
                )
        else:
            ready = await self._wait_for_named_ready(timeout=30)
            if not ready:
                await self.stop()
                raise RuntimeError(
                    f"cloudflared named tunnel {self._tunnel_name!r} did "
                    "not register a connection within 30 seconds",
                )
            url = f"https://{self._hostname.lstrip('/').rstrip('/')}"
            # Strip scheme prefix if user already supplied full URL
            if self._hostname.startswith(("http://", "https://")):
                url = self._hostname.rstrip("/")

        self._info = TunnelInfo(
            public_url=url,
            public_wss_url=url.replace("https://", "wss://").replace(
                "http://",
                "ws://",
            ),
            started_at=datetime.now(timezone.utc),
            pid=self._process.pid,
        )
        logger.info("Tunnel ready: %s (pid=%s)", url, self._process.pid)

        self._monitor_task = asyncio.create_task(
            self._monitor(),
            name="tunnel_monitor",
        )

        return self._info

    def _build_command(self, binary: str, local_port: int) -> list[str]:
        """Build the cloudflared argv for the current mode."""
        if self._mode == "quick":
            return [
                binary,
                "tunnel",
                "--url",
                f"http://localhost:{local_port}",
            ]
        # Named tunnel: pass --url so all ingress goes to MediaServer
        # regardless of what the user's config.yml / dashboard ingress says.
        # --url is a valid override for `tunnel run`.
        cmd = [binary, "tunnel"]
        if self._config_file:
            cmd += ["--config", self._config_file]
        cmd += [
            "run",
            "--url",
            f"http://localhost:{local_port}",
            self._tunnel_name,
        ]
        return cmd

    async def stop(self) -> None:
        """Terminate the tunnel subprocess."""
        if self._monitor_task and not self._monitor_task.done():
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
            self._monitor_task = None

        if self._process and self._process.returncode is None:
            logger.info("Stopping cloudflared (pid=%s)", self._process.pid)
            self._process.terminate()
            try:
                await asyncio.wait_for(self._process.wait(), timeout=5)
            except asyncio.TimeoutError:
                self._process.kill()
                await self._process.wait()
        self._process = None
        self._info = None

    async def _kill_orphans(self, local_port: int) -> None:
        """Terminate any stale cloudflared process that is already serving
        this tunnel identity.

        Matches by the argv we build ourselves (``--url http://localhost:<port>``
        plus, for named tunnels, the tunnel name or config file).  That is
        precise enough to avoid touching unrelated cloudflared instances the
        user may be running for other tunnels.
        """
        # Unique-to-us argv tokens.  Every orphan launched by this driver
        # will have them all; any hit on a subset is still safe because we
        # require *all* the tokens to match.
        port_token = f"http://localhost:{local_port}"
        match_tokens: list[str] = [port_token]
        if self._mode == "named" and self._tunnel_name:
            match_tokens.append(self._tunnel_name)
        elif self._mode == "named" and self._config_file:
            match_tokens.append(self._config_file)

        try:
            import psutil  # type: ignore
        except ImportError:
            psutil = None  # type: ignore

        my_pid = self._process.pid if self._process else os.getpid()
        killed: list[int] = []

        if psutil is not None:
            for proc in psutil.process_iter(["pid", "name", "cmdline"]):
                try:
                    if proc.info["pid"] == my_pid:
                        continue
                    name = (proc.info["name"] or "").lower()
                    if "cloudflared" not in name:
                        continue
                    cmdline = " ".join(proc.info["cmdline"] or [])
                    if all(tok in cmdline for tok in match_tokens):
                        logger.warning(
                            "Cleaning up orphan cloudflared pid=%d (%s)",
                            proc.info["pid"],
                            cmdline[:120],
                        )
                        proc.terminate()
                        killed.append(proc.info["pid"])
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            # Give terminate() a moment, then force anything still alive.
            if killed:
                _, still = psutil.wait_procs(
                    [psutil.Process(p) for p in killed if psutil.pid_exists(p)],
                    timeout=3,
                )
                for proc in still:
                    try:
                        proc.kill()
                    except psutil.NoSuchProcess:
                        pass
            return

        # Fallback: pgrep + kill if psutil isn't available.
        try:
            pattern = "cloudflared.*" + ".*".join(
                tok.replace("/", r"\/") for tok in match_tokens
            )
            proc = await asyncio.create_subprocess_exec(
                "pgrep", "-f", pattern,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
            )
            out, _ = await proc.communicate()
            pids = [
                int(p) for p in out.decode().split()
                if p.isdigit() and int(p) != my_pid
            ]
            for pid in pids:
                logger.warning("Cleaning up orphan cloudflared pid=%d", pid)
                try:
                    os.kill(pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
            # Short grace period before force-kill
            if pids:
                await asyncio.sleep(2)
                for pid in pids:
                    try:
                        os.kill(pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
        except Exception as exc:
            logger.warning("orphan cloudflared cleanup failed: %s", exc)

    async def health_check(self) -> bool:
        """Return True if the tunnel process is running."""
        return self._process is not None and self._process.returncode is None

    def get_public_url(self) -> str | None:
        """Return the current public URL, or None if not running."""
        return self._info.public_url if self._info else None

    def get_info(self) -> TunnelInfo | None:
        """Return the current TunnelInfo, or None if not running."""
        return self._info

    async def _wait_for_url(self, timeout: float = 30) -> str | None:
        """Read cloudflared stderr until a Quick Tunnel public URL appears."""
        return await self._scan_stderr(_URL_RE, timeout, return_match=True)

    async def _wait_for_named_ready(self, timeout: float = 30) -> bool:
        """Read stderr until a Named Tunnel connection registration line."""
        hit = await self._scan_stderr(_NAMED_READY_RE, timeout)
        return hit is not None

    async def _scan_stderr(
        self,
        pattern: re.Pattern[str],
        timeout: float,
        return_match: bool = False,
    ) -> str | None:
        """Scan stderr for `pattern`. Returns match group(0) if
        return_match, else a truthy sentinel when matched, else None."""
        if not self._process or not self._process.stderr:
            return None
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        while loop.time() < deadline:
            try:
                line = await asyncio.wait_for(
                    self._process.stderr.readline(),
                    timeout=max(0.1, deadline - loop.time()),
                )
            except asyncio.TimeoutError:
                if loop.time() >= deadline:
                    break
                continue
            if not line:
                if self._process.returncode is not None:
                    break
                continue
            text = line.decode("utf-8", errors="replace").strip()
            logger.debug("cloudflared: %s", text)
            match = pattern.search(text)
            if match:
                return match.group(0) if return_match else "ok"
        return None

    async def _drain_stderr(self) -> None:
        """Read and discard stderr to prevent pipe buffer from filling."""
        if not self._process or not self._process.stderr:
            return
        while True:
            line = await self._process.stderr.readline()
            if not line:
                break
            logger.debug(
                "cloudflared: %s",
                line.decode("utf-8", errors="replace").strip(),
            )

    async def _monitor(self) -> None:
        """Drain stderr and log unexpected exit without auto-restart."""
        # Keep reading stderr so the pipe buffer doesn't fill and
        # block cloudflared.  _drain_stderr returns when the process
        # closes its stderr (i.e. exits).
        await self._drain_stderr()

        if not self._process:
            return

        try:
            await self._process.wait()
        except asyncio.CancelledError:
            return

        rc = self._process.returncode
        logger.warning(
            "cloudflared exited with code %s; not restarting Quick Tunnel "
            "automatically because a new public URL would be issued.",
            rc,
        )

        # Clear tunnel info so callers know the tunnel is no longer available.
        self._info = None
        return
