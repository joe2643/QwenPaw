# -*- coding: utf-8 -*-
"""ACP client and server exports.

.core holds pure-Python config/exception types that the rest of QwenPaw
imports (via config.py). Those must never fail.

.server and .service both depend on the external ``acp`` package, which
isn't listed in pyproject deps — on hosts without it, the daemon would
crash at startup. Guard those behind a try/except so merely ``from
qwenpaw.agents.acp import ACPConfig`` keeps working, and only users who
actually run the ACP agent see the ImportError.
"""

from .core import (
    ACPAgentConfig,
    ACPConfig,
    ACPConfigurationError,
    ACPProtocolError,
    ACPSessionError,
    ACPTransportError,
    ACPErrors,
    PermissionResolution,
    SuspendedPermission,
)

try:
    from .server import QwenPawACPAgent, run_qwenpaw_agent
    from .service import ACPService, get_acp_service, init_acp_service

    _ACP_AVAILABLE = True
except ImportError as _exc:  # pragma: no cover - optional dep
    _ACP_IMPORT_ERROR = _exc

    _ACP_AVAILABLE = False

    class _ACPMissing:
        """Placeholder that raises only when touched."""

        def __init__(self, *_a, **_kw):
            raise ImportError(
                "ACP server support requires the 'acp' package. "
                "Install with: pip install acp",
            ) from _ACP_IMPORT_ERROR

    QwenPawACPAgent = _ACPMissing  # type: ignore[assignment,misc]
    run_qwenpaw_agent = _ACPMissing  # type: ignore[assignment]
    ACPService = _ACPMissing  # type: ignore[assignment,misc]
    get_acp_service = _ACPMissing  # type: ignore[assignment]
    init_acp_service = _ACPMissing  # type: ignore[assignment]


__all__ = [
    "ACPAgentConfig",
    "ACPConfig",
    "ACPErrors",
    "ACPConfigurationError",
    "ACPProtocolError",
    "ACPSessionError",
    "ACPTransportError",
    "ACPService",
    "QwenPawACPAgent",
    "get_acp_service",
    "init_acp_service",
    "PermissionResolution",
    "run_qwenpaw_agent",
    "SuspendedPermission",
]
