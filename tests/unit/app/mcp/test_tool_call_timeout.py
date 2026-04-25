# -*- coding: utf-8 -*-
"""Unit tests for MCP tool-call timeout plumbing.

Regression guard for the WhatsApp-group hang we saw in production:
a stdio MCP subprocess (z.ai vision) whose backend was unreachable
retried internally forever, leaving ``session.call_tool`` awaiting
a reply that never came.  The agent reply task froze, the typing
indicator loop kept firing, new user messages queued up, and
``/stop`` couldn't cancel the hung reply because ``task.cancel()``
doesn't propagate through an uninterruptible await.

After the fix, ``tool_call_timeout`` bounds every call at the MCP
layer (pushed through ``read_timeout_seconds`` on the session) AND
the wrapper is a plain ``await`` so external ``task.cancel()``
unwinds promptly.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from qwenpaw.app.mcp.stateful_client import (
    HttpStatefulClient,
    StdIOStatefulClient,
)
from qwenpaw.config.config import MCPClientConfig


# ---------------------------------------------------------------- #
# Config schema                                                    #
# ---------------------------------------------------------------- #


class TestMCPClientConfigTimeoutField:
    def test_default_is_none(self):
        # ``None`` preserves pre-fix behaviour — no timeout.
        c = MCPClientConfig(name="x", command="true")
        assert c.tool_call_timeout is None

    def test_round_trip(self):
        src = {"name": "x", "command": "true", "tool_call_timeout": 45.0}
        c = MCPClientConfig.model_validate(src)
        assert c.tool_call_timeout == 45.0
        assert c.model_dump()["tool_call_timeout"] == 45.0


# ---------------------------------------------------------------- #
# Client stores the configured timeout                             #
# ---------------------------------------------------------------- #


class TestClientStoresTimeout:
    def test_stdio_stores_timeout(self):
        c = StdIOStatefulClient(
            name="s",
            command="true",
            tool_call_timeout=30.0,
        )
        assert c._tool_call_timeout == 30.0

    def test_stdio_default_is_none(self):
        c = StdIOStatefulClient(name="s", command="true")
        assert c._tool_call_timeout is None

    def test_http_stores_timeout(self):
        c = HttpStatefulClient(
            name="h",
            transport="streamable_http",
            url="http://example.com",
            tool_call_timeout=15.0,
        )
        assert c._tool_call_timeout == 15.0

    def test_http_default_is_none(self):
        c = HttpStatefulClient(
            name="h",
            transport="streamable_http",
            url="http://example.com",
        )
        assert c._tool_call_timeout is None


# ---------------------------------------------------------------- #
# Runtime behaviour of call_tool                                   #
# ---------------------------------------------------------------- #


class _HangingSession:
    """Stand-in for ``mcp.ClientSession`` that hangs forever on
    ``call_tool`` unless ``read_timeout_seconds`` is supplied, in
    which case it honours the bound by raising TimeoutError after
    the deadline (faithful to the mcp library contract)."""

    def __init__(self) -> None:
        self.last_call: dict[str, Any] | None = None

    async def call_tool(
        self,
        name: str,
        arguments: dict,
        read_timeout_seconds: Any = None,
    ) -> Any:
        self.last_call = {
            "name": name,
            "arguments": arguments,
            "read_timeout_seconds": read_timeout_seconds,
        }
        if read_timeout_seconds is None:
            # Forever — caller must cancel externally.
            await asyncio.Event().wait()
            return None  # pragma: no cover
        # Convert timedelta → seconds for the simulated wait.
        if hasattr(read_timeout_seconds, "total_seconds"):
            deadline_s = read_timeout_seconds.total_seconds()
        else:
            deadline_s = float(read_timeout_seconds)
        try:
            await asyncio.wait_for(
                asyncio.Event().wait(),
                timeout=deadline_s,
            )
        except asyncio.TimeoutError:
            # mcp library raises its own error; for our purposes any
            # timeout-shaped exception is sufficient to prove the
            # deadline was honoured.
            raise


def _connected_client(
    session: _HangingSession,
    timeout: float | None,
) -> StdIOStatefulClient:
    c = StdIOStatefulClient(
        name="fake",
        command="true",
        tool_call_timeout=timeout,
    )
    c.session = session  # type: ignore[assignment]
    c.is_connected = True
    return c


@pytest.mark.asyncio
async def test_timeout_is_forwarded_to_session() -> None:
    sess = _HangingSession()
    client = _connected_client(sess, timeout=5.0)
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(
            client.call_tool("ping", {}),
            timeout=0.2,
        )
    # No matter whether the inner call timed out or the outer
    # wait_for did, the read_timeout_seconds we set MUST have been
    # forwarded to the session — that's the load-bearing change.
    assert sess.last_call is not None
    assert sess.last_call["read_timeout_seconds"] is not None
    td = sess.last_call["read_timeout_seconds"]
    assert hasattr(td, "total_seconds")
    assert td.total_seconds() == pytest.approx(5.0)


@pytest.mark.asyncio
async def test_no_timeout_passes_none_to_session() -> None:
    # Back-compat: a client without tool_call_timeout must continue
    # to call the session without read_timeout_seconds so the
    # library's default behaviour is preserved.
    sess = _HangingSession()
    client = _connected_client(sess, timeout=None)
    task = asyncio.create_task(client.call_tool("ping", {}))
    # Give the task a tick to enter the await.
    await asyncio.sleep(0.05)
    assert sess.last_call is not None
    assert sess.last_call["read_timeout_seconds"] is None
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.asyncio
async def test_external_cancel_propagates_promptly() -> None:
    # Part C proof: task.cancel() from the outside (/stop) reaches
    # an uninterruptible-looking MCP call and unwinds it, because
    # our wrapper is a plain ``await`` — no suppressed cancellation
    # handlers in the way.
    sess = _HangingSession()
    client = _connected_client(sess, timeout=None)
    task = asyncio.create_task(client.call_tool("ping", {}))
    await asyncio.sleep(0.05)
    cancelled_at = asyncio.get_event_loop().time()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    elapsed = asyncio.get_event_loop().time() - cancelled_at
    # Must unwind in well under a second — in practice this is
    # effectively immediate; we're only ruling out "cancel is
    # silently swallowed" regressions.
    assert elapsed < 0.5, f"cancel took {elapsed:.3f}s — propagation broken"


@pytest.mark.asyncio
async def test_timeout_fires_cleanly_when_set() -> None:
    # Set a tight timeout and confirm that without external cancel
    # we still get out of the call — no forever-hang.
    sess = _HangingSession()
    client = _connected_client(sess, timeout=0.1)
    with pytest.raises(asyncio.TimeoutError):
        await client.call_tool("ping", {})


@pytest.mark.asyncio
async def test_arguments_preserved_through_wrapper() -> None:
    # Regression guard: the timeout-wrapping code path must not
    # eat or mutate the caller's arguments.
    sess = _HangingSession()
    client = _connected_client(sess, timeout=0.05)
    with pytest.raises(asyncio.TimeoutError):
        await client.call_tool("echo", {"foo": 1, "bar": "baz"})
    assert sess.last_call["name"] == "echo"
    assert sess.last_call["arguments"] == {"foo": 1, "bar": "baz"}


# ---------------------------------------------------------------- #
# get_callable_function plumbs the timeout into MCPToolFunction    #
# ---------------------------------------------------------------- #


class _MinimalTool:
    """Duck-type stand-in for ``mcp.types.Tool`` — MCPToolFunction
    only needs ``name``, ``description``, ``inputSchema``."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.description = ""
        self.inputSchema = {"type": "object", "properties": {}}


@pytest.mark.asyncio
async def test_get_callable_function_propagates_configured_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Toolkit.register_mcp_client calls ``get_callable_function``
    with ``execution_timeout=None``; our override must substitute
    the configured value so the returned ``MCPToolFunction``'s
    ``self.timeout`` is set.  Without this, agentscope-registered
    tools bypass the timeout entirely.
    """
    sess = _HangingSession()
    client = _connected_client(sess, timeout=12.0)
    # Pretend we've already listed tools.
    client._cached_tools = [_MinimalTool("ping")]  # type: ignore[attr-defined]

    # Helper: agentscope stores ``timeout`` as either a float or a
    # ``timedelta`` depending on version; normalise for the assert.
    def _secs(t: Any) -> float:
        if hasattr(t, "total_seconds"):
            return t.total_seconds()
        return float(t)

    # Call the override with no explicit timeout (Toolkit's code path).
    fn = await client.get_callable_function("ping", wrap_tool_result=True)
    assert _secs(fn.timeout) == 12.0

    # Explicit override at call-site still wins.
    fn2 = await client.get_callable_function(
        "ping",
        wrap_tool_result=True,
        execution_timeout=3.0,
    )
    assert _secs(fn2.timeout) == 3.0
