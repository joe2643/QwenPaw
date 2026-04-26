# -*- coding: utf-8 -*-
"""Claude Code (acpx) chat-completions wrapper — Lane C SKELETON.

CoPaw provider that routes ``chat.completions.create`` calls through
``acpx`` — a CLI tool speaking the Agent Client Protocol (ACP) to the
Claude Code IDE-grade agent.  Stateful per-conversation sessions reuse
Claude Code's prompt cache so cache-prefix prefix is stable across
turns (vs the stateless Anthropic-direct path).

Hybrid tool-execution mode (decided 2026-04-26 in the design doc):

* Claude Code proposes tools via ACP ``tool_call`` notifications;
* CoPaw EXECUTES tools via ACP client-side ``fs/*`` + ``terminal/*``
  method handlers, routing through CoPaw's existing MCP / security
  guardian / tool dispatch stack;
* Permission flow integrates with CoPaw's existing security guardians
  (file_guardian, shell_evasion_guardian, …); ACP
  ``session/request_permission`` maps to CoPaw's permission engine;
* Tool results flow back to Claude via ACP ``tool_call_update`` with
  status=completed and content arrays — and ALSO surface in CoPaw's
  normal logging/UI path.

Lane status (2026-04-26):

* Lane A (merged): ``acpx_translate`` translator + Stateful session
  registry skeleton.
* Lane B (in flight): daemon process management +
  ``fs/*`` / ``terminal/*`` ACP handlers.
* Lane C (this lane): provider tile registration + SDK-shape stub
  + dispatch wiring so the UI surfaces show up and
  ``AnthropicProvider.get_chat_model_instance`` reaches a real class.
* Lane D (next): replace the ``NotImplementedError`` stub in
  :meth:`ClaudeAcpxChatModel._install_wrapper` with daemon dispatch
  + session-registry plumbing (currently only imported in a comment
  on the stub so a future ``grep claude_acpx_session_registry`` finds
  this entry-point).

Design mirrors
:class:`qwenpaw.providers.codex_oauth_model.CodexOAuthChatModel`:

* Subclass rather than composition — agentscope's response parsing
  logic runs on the same SDK types the real OpenAI client returns,
  so we synthesise those types instead of forking the parser.
* ``_install_wrapper`` mutates ``self.client.chat.completions.create``
  at init time; once Lane B/D land, each call will hit the daemon's
  ``acpx claude -s <session_name>`` path and translate ACP
  ``session/update`` notifications back into ``ChatCompletionChunk``
  objects via :mod:`acpx_translate`.
"""

from __future__ import annotations

import logging
from typing import Any

from agentscope.model import OpenAIChatModel

logger = logging.getLogger(__name__)


class ClaudeAcpxChatModel(OpenAIChatModel):
    """Claude Code (acpx) variant of :class:`OpenAIChatModel`.

    v1 Lane C STUB — the ``_install_wrapper`` override below replaces
    the SDK's ``client.chat.completions.create`` with a coroutine that
    raises :class:`NotImplementedError`.  This lets the UI surface, the
    provider registry, and the dispatch path light up end-to-end so
    Lane B (daemon + handlers) and Lane D (chat wiring) can land
    independently without churning the integration layer.

    Once Lane B/D fill in the body, this class will:

    1. read ``agent_id`` / ``session_id`` ContextVars set by
       :class:`react_agent.ReActAgent`;
    2. call
       :func:`qwenpaw.providers.claude_acpx_session_registry.get_or_mint`
       to map ``(agent_id, session_id, model)`` to a stable acpx
       session name (with drift detection over the message history);
    3. apply effort/generate_kwargs deltas via ``acpx ... set ...``
       one-shots before the prompt;
    4. enqueue the prompt blocks against the long-lived daemon and
       stream ACP ``session/update`` notifications back as
       ``ChatCompletionChunk`` objects (via
       :func:`qwenpaw.providers.acpx_translate
       .translate_acp_updates_to_chat_chunks`).
    """

    def __init__(
        self,
        *,
        model_name: str,
        stream: bool = True,
        generate_kwargs: dict[str, Any] | None = None,
        client_kwargs: dict[str, Any] | None = None,
        api_key: str | None = None,
        stream_tool_parsing: bool = False,
    ) -> None:
        # OpenAI SDK refuses to construct without an ``api_key`` (even
        # though our ``_install_wrapper`` redirects every request away
        # from its default base URL).  Seed with a harmless sentinel —
        # it never reaches the wire.  Mirrors
        # :class:`CodexOAuthChatModel` (codex_oauth_model.py:212-214).
        if not api_key:
            api_key = "claude-acpx-unused"
        super().__init__(
            model_name=model_name,
            stream=stream,
            api_key=api_key,
            stream_tool_parsing=stream_tool_parsing,
            client_kwargs=client_kwargs or {},
            generate_kwargs=generate_kwargs or {},
        )
        self._install_wrapper()

    def _install_wrapper(self) -> None:
        """Override the SDK's ``chat.completions.create`` with our acpx
        path.  v1 Lane C stub — Lane B/D will fill in daemon dispatch
        + session-registry plumbing
        (:mod:`qwenpaw.providers.claude_acpx_session_registry`).
        Until then, real calls raise :class:`NotImplementedError`."""

        async def _wrapped_create(**call_kwargs: Any) -> Any:
            # Intentional: raise loudly instead of silently returning
            # a stub completion so an accidental dispatch surfaces as
            # a clear traceback at agent-call time rather than masking
            # as missing assistant output.
            raise NotImplementedError(
                "ClaudeAcpxChatModel: daemon backend pending Lane B/D. "
                "Provider tile + dispatch land in Lane C; full chat "
                "wiring lands in Lane D.",
            )

        self.client.chat.completions.create = _wrapped_create  # type: ignore[method-assign]
