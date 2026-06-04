# -*- coding: utf-8 -*-
"""Unit tests for the Claude Code (acpx) connection-test endpoint.

The endpoint composes two best-effort checks (npx subprocess + on-disk
credentials) into a single ``ClaudeAcpxStatus`` payload.  Each branch
of the matrix — both pass / cli only / creds only / both fail / cli
times out — has a distinct user-visible failure mode in the UI tile,
so we pin them all explicitly.

We stub ``asyncio.create_subprocess_exec`` and the ``ClaudeAuth``
constructor rather than running real subprocesses so CI never installs
anything from npm.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from qwenpaw.app.routers.providers import router


app = FastAPI()
app.include_router(router, prefix="/api")


@pytest.fixture
def api_client():
    transport = ASGITransport(app=app)
    return AsyncClient(transport=transport, base_url="http://test")


# ---------------------------------------------------------------- #
# Fakes                                                            #
# ---------------------------------------------------------------- #


class _FakeProc:
    """Minimal ``asyncio.subprocess.Process`` stand-in — only the
    surface ``claude_acpx_test_connection`` reads."""

    def __init__(
        self,
        *,
        returncode: int = 0,
        stdout: bytes = b"",
        stderr: bytes = b"",
        hang: bool = False,
    ) -> None:
        self.returncode = returncode
        self._stdout = stdout
        self._stderr = stderr
        self._hang = hang
        self.killed = False
        # Fake PID: os.getpgid(9999) raises ProcessLookupError, exercising
        # the timeout handler's proc.kill() fallback path.
        self.pid = 9999

    async def communicate(self) -> tuple[bytes, bytes]:
        if self._hang:
            # Block forever to exercise the wait_for timeout branch.
            await asyncio.Event().wait()
        return self._stdout, self._stderr

    def kill(self) -> None:
        self.killed = True

    async def wait(self) -> int:
        """Reaped by the timeout handler after kill()."""
        return self.returncode


class _FakeCreds:
    """Stand-in for ``ClaudeCredential`` — only attributes the endpoint
    reads."""

    def __init__(
        self,
        *,
        credentials_path: str = "/fake/.claude/.credentials.json",
        seconds_until_expiry: int = 3600,
    ) -> None:
        self.credentials_path = credentials_path
        self.seconds_until_expiry = seconds_until_expiry


class _FakeAuth:
    def __init__(self, creds: _FakeCreds | None) -> None:
        self._creds = creds


# ---------------------------------------------------------------- #
# Tests                                                            #
# ---------------------------------------------------------------- #


def _patch_subprocess(proc: _FakeProc):
    """Returns a context manager that swaps
    ``asyncio.create_subprocess_exec`` for one yielding the given fake.
    The endpoint imports ``asyncio`` inline, so we patch on the asyncio
    module itself rather than a re-bound name.
    """

    async def _fake_create_subprocess_exec(
        *_args: Any,
        **_kwargs: Any,
    ) -> _FakeProc:
        return proc

    return patch.object(
        asyncio,
        "create_subprocess_exec",
        side_effect=_fake_create_subprocess_exec,
    )


def _patch_claude_auth(auth: _FakeAuth | type[Exception]):
    """Replace the ``ClaudeAuth`` constructor used by the endpoint.
    Pass an instance to succeed, or pass ``FileNotFoundError`` (or any
    Exception subclass) to fail."""

    if isinstance(auth, type) and issubclass(auth, BaseException):

        def _ctor(*_a: Any, **_kw: Any) -> _FakeAuth:
            raise auth("boom")

    else:

        def _ctor(*_a: Any, **_kw: Any) -> _FakeAuth:
            return auth

    return patch(
        "qwenpaw.providers.claude_auth.ClaudeAuth",
        side_effect=_ctor,
    )


def _patch_resolve_path(path_str: str = "/fake/.claude/.credentials.json"):
    from pathlib import Path

    return patch(
        "qwenpaw.providers.claude_auth._resolve_credentials_path",
        return_value=Path(path_str),
    )


async def test_both_checks_pass(api_client) -> None:
    proc = _FakeProc(returncode=0, stdout=b"acpx 1.2.3\n")
    auth = _FakeAuth(_FakeCreds())
    with _patch_subprocess(proc), _patch_claude_auth(auth), _patch_resolve_path():
        async with api_client:
            resp = await api_client.get(
                "/api/models/claude-acpx/test-connection",
            )
    assert resp.status_code == 200
    body = resp.json()
    assert body["acpx_cli_available"] is True
    assert body["acpx_version"] == "acpx 1.2.3"
    assert body["claude_credentials_present"] is True
    # Path comes from the (fake) credentials object, not the resolver.
    assert body["credentials_path"] == "/fake/.claude/.credentials.json"
    assert body["expires_in_s"] == 3600
    assert body["error"] is None


async def test_cli_ok_credentials_missing(api_client) -> None:
    # User installed acpx but never ran ``claude login``.
    proc = _FakeProc(returncode=0, stdout=b"acpx 1.2.3\n")
    with _patch_subprocess(proc), _patch_claude_auth(
        FileNotFoundError,
    ), _patch_resolve_path():
        async with api_client:
            resp = await api_client.get(
                "/api/models/claude-acpx/test-connection",
            )
    assert resp.status_code == 200
    body = resp.json()
    assert body["acpx_cli_available"] is True
    assert body["claude_credentials_present"] is False
    # error string must surface the credentials-side failure.
    assert body["error"] is not None
    assert "boom" in body["error"]


async def test_cli_missing_credentials_present(api_client) -> None:
    # ``npx`` ran but acpx package failed (e.g. registry blocked).
    proc = _FakeProc(returncode=1, stderr=b"E404 acpx\n")
    auth = _FakeAuth(_FakeCreds())
    with _patch_subprocess(proc), _patch_claude_auth(auth), _patch_resolve_path():
        async with api_client:
            resp = await api_client.get(
                "/api/models/claude-acpx/test-connection",
            )
    assert resp.status_code == 200
    body = resp.json()
    assert body["acpx_cli_available"] is False
    assert body["acpx_version"] is None
    assert body["claude_credentials_present"] is True
    # Error string includes the npm exit code and a tail of stderr.
    assert "exit 1" in body["error"]
    assert "E404" in body["error"]


async def test_npx_not_on_path(api_client) -> None:
    # User has neither Node nor npx — most aggressive failure.
    auth = _FakeAuth(_FakeCreds())

    async def _raise_fnfe(*_a: Any, **_kw: Any) -> _FakeProc:
        raise FileNotFoundError(2, "No such file or directory: 'npx'")

    with patch.object(
        asyncio,
        "create_subprocess_exec",
        side_effect=_raise_fnfe,
    ), _patch_claude_auth(auth), _patch_resolve_path():
        async with api_client:
            resp = await api_client.get(
                "/api/models/claude-acpx/test-connection",
            )
    assert resp.status_code == 200
    body = resp.json()
    assert body["acpx_cli_available"] is False
    assert body["error"] is not None
    assert "npx not found" in body["error"]


async def test_cli_timeout_kills_process(api_client) -> None:
    # ``npm install`` got stuck on a slow link — endpoint must
    # timeout, kill the subprocess, and report cleanly instead of
    # hanging the UI tile probe.
    proc = _FakeProc(hang=True)
    auth = _FakeAuth(_FakeCreds())

    # Patch wait_for to skip the real 10s wait — we still want to
    # exercise the timeout branch logic, just not actually wait.
    real_wait_for = asyncio.wait_for

    async def _fast_wait_for(coro: Any, timeout: float) -> Any:
        # Cancel the inner coroutine immediately so communicate()
        # doesn't keep its task alive past the test.
        return await real_wait_for(coro, timeout=0.05)

    with _patch_subprocess(proc), _patch_claude_auth(auth), _patch_resolve_path(), patch.object(
        asyncio,
        "wait_for",
        side_effect=_fast_wait_for,
    ):
        async with api_client:
            resp = await api_client.get(
                "/api/models/claude-acpx/test-connection",
            )
    assert resp.status_code == 200
    body = resp.json()
    assert body["acpx_cli_available"] is False
    assert body["error"] is not None
    assert "timed out" in body["error"]
    assert proc.killed, (
        "Endpoint must kill the hung subprocess on timeout — leaking "
        "zombies will eventually exhaust the host's process table."
    )


async def test_both_checks_fail_combines_errors(api_client) -> None:
    # Worst case: nothing works.  The single ``error`` field should
    # surface BOTH messages so the UI can show the full picture
    # without making a second round-trip.
    proc = _FakeProc(returncode=127, stderr=b"command not found\n")
    with _patch_subprocess(proc), _patch_claude_auth(
        FileNotFoundError,
    ), _patch_resolve_path():
        async with api_client:
            resp = await api_client.get(
                "/api/models/claude-acpx/test-connection",
            )
    assert resp.status_code == 200
    body = resp.json()
    assert body["acpx_cli_available"] is False
    assert body["claude_credentials_present"] is False
    assert "exit 127" in body["error"]
    assert "boom" in body["error"]
    # Composition is via ``"; "`` so both messages are visible.
    assert "; " in body["error"]
