# -*- coding: utf-8 -*-
"""Tests for service_factories — specifically _MediaServerHandle."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from qwenpaw.app.workspace.service_factories import _MediaServerHandle


class TestMediaServerHandle:

    @pytest.mark.asyncio
    async def test_stop_passes_agent_id(self):
        """Handle.stop() must call server.stop(agent_id=<agent>)."""
        mock_server = AsyncMock()
        handle = _MediaServerHandle(mock_server, "agent-42")

        await handle.stop()
        mock_server.stop.assert_awaited_once_with(agent_id="agent-42")

    @pytest.mark.asyncio
    async def test_start_delegates(self):
        """Handle.start() must delegate to server.start()."""
        mock_server = AsyncMock()
        handle = _MediaServerHandle(mock_server, "agent-42")

        await handle.start()
        mock_server.start.assert_awaited_once()

    def test_getattr_delegates(self):
        """Attribute access must delegate to the underlying server."""
        mock_server = MagicMock()
        mock_server.port = 8089
        mock_server.host = "127.0.0.1"
        handle = _MediaServerHandle(mock_server, "agent-42")

        assert handle.port == 8089
        assert handle.host == "127.0.0.1"
