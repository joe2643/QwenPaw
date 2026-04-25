# -*- coding: utf-8 -*-
# pylint: disable=too-many-branches,too-many-statements,unused-argument
# pylint: disable=too-many-public-methods,unnecessary-pass
"""
Base Channel: bound to AgentRequest/AgentResponse, unified by process.
"""
from __future__ import annotations

import asyncio
import json
import logging
from abc import ABC
from typing import (
    Optional,
    Dict,
    Any,
    List,
    Union,
    AsyncIterator,
    AsyncGenerator,
    Callable,
    TYPE_CHECKING,
)

from agentscope_runtime.engine.schemas.agent_schemas import (
    RunStatus,
    ContentType,
    TextContent,
    ImageContent,
    VideoContent,
    AudioContent,
    FileContent,
    RefusalContent,
    MessageType,
)

from .renderer import MessageRenderer, RenderStyle
from .schema import ChannelType
from ...config.utils import load_config

# Optional callback to enqueue payload (set by manager)
EnqueueCallback = Optional[Callable[[Any], None]]

# Called when a user-originated reply was sent (channel, user_id, session_id)
OnReplySent = Optional[Callable[[str, str, str], None]]

logger = logging.getLogger(__name__)


_TOOL_OUTPUT_MESSAGE_TYPES = {
    MessageType.FUNCTION_CALL_OUTPUT,
    MessageType.PLUGIN_CALL_OUTPUT,
    MessageType.MCP_TOOL_CALL_OUTPUT,
}

if TYPE_CHECKING:
    from agentscope_runtime.engine.schemas.agent_schemas import (
        AgentRequest,
        AgentResponse,
        Event,
    )

# process: accepts AgentRequest, streams Event
# (including message events with status completed)
ProcessHandler = Callable[[Any], AsyncIterator["Event"]]

# Outgoing part = runtime content types (no Dict[str, Any])
OutgoingContentPart = Union[
    TextContent,
    ImageContent,
    VideoContent,
    AudioContent,
    FileContent,
    RefusalContent,
]


class BaseChannel(ABC):
    """Base for all channels. Queue lives in ChannelManager; channel defines
    how to consume via consume_one().
    """

    channel: ChannelType

    # If True, manager creates a queue and consumer loop for this channel.
    uses_manager_queue: bool = True

    # If True, replace_channel() stops the old channel BEFORE starting the
    # new one to avoid resource conflicts (e.g. exclusive SQLite locks).
    requires_sequential_restart: bool = False

    @classmethod
    def doctor_connectivity_notes(
        cls,
        agent_id: str,
        config: Any,
        *,
        timeout: float,
    ) -> list[str]:
        """Optional ``copaw doctor --deep`` reachability checks.

        Override in custom channels. Default: no extra checks
        (built-in channels use shared probes in ``doctor_connectivity``
        unless this returns notes).

        Args:
            agent_id: Profile id from ``agents.profiles``.
            config: Channel subsection (Pydantic model or dict for extras).
            timeout: Seconds for TCP/HTTP probes.

        Returns:
            Informational lines (empty if OK / skipped).
        """
        return []

    def __init__(
        self,
        process: ProcessHandler,
        on_reply_sent: OnReplySent = None,
        show_tool_details: bool = True,
        filter_tool_messages: bool = False,
        filter_thinking: bool = False,
        dm_policy: str = "open",
        group_policy: str = "open",
        allow_from: Optional[list] = None,
        deny_message: str = "",
        require_mention: bool = False,
    ):
        self._process = process
        self._on_reply_sent = on_reply_sent
        self._show_tool_details = show_tool_details
        self._filter_tool_messages = filter_tool_messages
        self._filter_thinking = filter_thinking
        self.dm_policy = dm_policy or "open"
        self.group_policy = group_policy or "open"
        self.allow_from = set(allow_from or [])
        self.deny_message = deny_message or ""
        self.require_mention = require_mention
        self._enqueue: EnqueueCallback = None
        self._workspace = None
        cfg = load_config()
        internal_tools = frozenset(
            name
            for name, tc in cfg.tools.builtin_tools.items()
            if not tc.display_to_user
        )
        self._render_style = RenderStyle(
            show_tool_details=show_tool_details,
            filter_tool_messages=filter_tool_messages,
            filter_thinking=filter_thinking,
            internal_tools=internal_tools,
        )
        self._renderer = MessageRenderer(self._render_style)
        self._http: Optional[Any] = None
        # Debounce: content from messages that had no text; merged when text
        # arrives. Key = session_id.
        self._pending_content_by_session: Dict[str, List[Any]] = {}
        # Time debounce: merge native payloads within _debounce_seconds.
        # Set > 0 in subclass (e.g. 0.3). Key = get_debounce_key(payload).
        self._debounce_seconds: float = 0.0
        self._debounce_pending: Dict[str, List[Any]] = {}
        self._debounce_timers: Dict[str, asyncio.Task[None]] = {}

    def _is_native_payload(self, payload: Any) -> bool:
        """True if payload is a native dict that can be time-debounced."""
        return isinstance(payload, dict) and "content_parts" in payload

    def get_debounce_key(self, payload: Any) -> str:
        """
        Key for time debounce (same key = same conversation).
        Delegates to ``resolve_session_id`` so every channel gets
        session-scoped isolation automatically.
        """
        if isinstance(payload, dict):
            sender_id = payload.get("sender_id") or ""
            meta = payload.get("meta") or {}
            return payload.get("session_id") or self.resolve_session_id(
                sender_id,
                meta,
            )
        return getattr(payload, "session_id", "") or ""

    def merge_native_items(self, items: List[Any]) -> Any:
        """
        Merge multiple native payloads into one. Override for
        channel-specific merge (e.g. meta keys). Default: concat
        content_parts, merge meta (reply_future, reply_loop, etc.).
        """
        if not items:
            return None
        first = items[0] if isinstance(items[0], dict) else {}
        merged_parts: List[Any] = []
        merged_meta: Dict[str, Any] = dict(first.get("meta") or {})
        for it in items:
            p = it if isinstance(it, dict) else {}
            merged_parts.extend(p.get("content_parts") or [])
            m = p.get("meta") or {}
            for k in (
                "reply_future",
                "reply_loop",
                "incoming_message",
                "conversation_id",
                "message_id",
            ):
                if k in m:
                    merged_meta[k] = m[k]
        return {
            "channel_id": first.get("channel_id") or self.channel,
            "sender_id": first.get("sender_id") or "",
            "content_parts": merged_parts,
            "meta": merged_meta,
        }

    def merge_requests(self, requests: List[Any]) -> Any:
        """
        Merge multiple AgentRequest payloads (same session) into one.
        Used when manager drains same-session queue: concatenate
        input[0].content from all, keep first request's meta/session.
        Returns one request; None if requests empty.
        """
        if not requests:
            return None
        first = requests[0]
        if len(requests) == 1:
            return first
        all_contents: List[Any] = []
        for req in requests:
            inp = getattr(req, "input", None) or []
            if inp and hasattr(inp[0], "content"):
                all_contents.extend(getattr(inp[0], "content") or [])
        if not all_contents:
            return first
        msg = first.input[0]
        if hasattr(msg, "model_copy"):
            new_msg = msg.model_copy(update={"content": all_contents})
        else:
            new_msg = msg
            setattr(new_msg, "content", all_contents)
        if hasattr(first, "model_copy"):
            return first.model_copy(
                update={"input": [new_msg]},
            )
        first.input[0] = new_msg
        return first

    def _on_debounce_buffer_append(
        self,
        key: str,
        payload: Any,
        existing_items: List[Any],
    ) -> None:
        """
        Hook when appending to time-debounce buffer (existing_items
        non-empty). Override e.g. to unblock previous reply_future.
        """
        del key
        del payload
        del existing_items

    def _content_has_text(self, contents: List[Any]) -> bool:
        """True if contents has at least one TEXT or REFUSAL with non-empty."""
        if not contents:
            return False
        for c in contents:
            t = getattr(c, "type", None)
            if (
                t == ContentType.TEXT
                and (getattr(c, "text", None) or "").strip()
            ):
                return True
            if (
                t == ContentType.REFUSAL
                and (getattr(c, "refusal", None) or "").strip()
            ):
                return True
        return False

    def _content_has_audio(self, contents: List[Any]) -> bool:
        """True if contents has at least one AUDIO block."""
        return any(
            getattr(c, "type", None) == ContentType.AUDIO
            for c in (contents or [])
        )

    def _apply_no_text_debounce(
        self,
        session_id: str,
        content_parts: List[Any],
    ) -> tuple[bool, List[Any]]:
        """
        Debounce: if content has no text, buffer and return (False, []).
        If has text, return (True, merged) with any buffered content prepended.
        Audio-only messages bypass debounce and are processed immediately
        (voice messages are standalone user input, not partial uploads).
        """
        if not self._content_has_text(content_parts):
            if self._content_has_audio(content_parts):
                # Audio-only messages (e.g. voice messages) should be
                # processed immediately — they are complete user input.
                pending = self._pending_content_by_session.pop(
                    session_id,
                    [],
                )
                merged = pending + list(content_parts)
                return (True, merged)
            self._pending_content_by_session.setdefault(
                session_id,
                [],
            ).extend(content_parts)
            logger.debug(
                "channel debounce: no text, buffered session_id=%s",
                session_id[:24] if session_id else "",
            )
            return (False, [])
        pending = self._pending_content_by_session.pop(session_id, [])
        merged = pending + list(content_parts)
        return (True, merged)

    def _check_allowlist(
        self,
        sender_id: str,
        is_group: bool,
    ) -> tuple[bool, Optional[str]]:
        """Check sender against allowlist policy."""
        policy = self.group_policy if is_group else self.dm_policy
        if policy == "open":
            return True, None
        if sender_id in self.allow_from:
            return True, None
        if self.deny_message:
            return False, self.deny_message
        if is_group:
            return (
                False,
                "Sorry, this bot is only available to authorized users.",
            )
        return False, (
            "Sorry, you are not authorized to use this bot. "
            "Please contact the administrator to add your ID "
            f"to the allowlist. Your ID: {sender_id}"
        )

    def _check_group_mention(
        self,
        is_group: bool,
        meta: dict,
    ) -> bool:
        """Return True if message should be processed under mention policy."""
        if not is_group or not self.require_mention:
            return True
        return bool(
            meta.get("bot_mentioned") or meta.get("has_bot_command"),
        )

    def set_enqueue(self, cb: EnqueueCallback) -> None:
        """Set enqueue callback (called by ChannelManager)."""
        self._enqueue = cb

    def set_workspace(
        self,
        workspace,
        command_registry=None,
    ) -> None:
        """Set workspace reference for TaskTracker access.

        Args:
            workspace: Workspace instance with task_tracker and chat_manager
            command_registry: CommandRegistry for control command detection
        """
        self._workspace = workspace
        self._command_registry = command_registry

    def _extract_chat_name(self, payload: Any) -> str:
        """Extract chat name from payload for chat creation.

        Args:
            payload: Message payload (dict or AgentRequest)

        Returns:
            Chat name (truncated to 50 chars)
        """
        try:
            if isinstance(payload, dict):
                parts = payload.get("content_parts", [])
                if parts:
                    first = parts[0]
                    if isinstance(first, dict):
                        text = first.get("text", "")
                    elif hasattr(first, "text"):
                        text = first.text
                    else:
                        text = str(first)
                    if text:
                        return text[:50]
                return "New Chat"
            if hasattr(payload, "input") and payload.input:
                msg = payload.input[0]
                if hasattr(msg, "content") and msg.content:
                    content = msg.content[0]
                    if hasattr(content, "text"):
                        return content.text[:50]
            return "New Chat"
        except Exception as e:
            logger.warning(
                f"Failed to extract chat name from payload: {e}",
                exc_info=True,
            )
            return "New Chat"

    async def _consume_with_tracker(
        self,
        request: "AgentRequest",
        payload: Any,
    ) -> None:
        """Consume message with TaskTracker registration for cancellation.

        TaskTracker is used to track the running task so /stop can cancel it.
        Message serialization is ensured by UnifiedQueueManager which queues
        messages per (channel, session, priority).

        Args:
            request: AgentRequest
            payload: Original payload
        """
        session_id = getattr(request, "session_id", "") or ""
        user_id = getattr(request, "user_id", "") or ""
        channel_id = getattr(request, "channel", self.channel)

        chat = await self._workspace.chat_manager.get_or_create_chat(
            session_id,
            user_id,
            channel_id,
            name=self._extract_chat_name(payload),
        )

        logger.info(
            f"_consume_with_tracker: chat_id={chat.id} "
            f"session={session_id[:30]}",
        )

        queue, is_new = await self._workspace.task_tracker.attach_or_start(
            chat.id,
            payload,
            self._stream_with_tracker,
        )

        if is_new:
            try:
                async for _ in self._workspace.task_tracker.stream_from_queue(
                    queue,
                    chat.id,
                ):
                    pass
            except asyncio.CancelledError:
                logger.info(
                    f"Task cancelled: chat_id={chat.id} "
                    f"session={session_id[:30]}",
                )
                raise
        else:
            logger.warning(
                f"Message ignored (task already running): "
                f"chat_id={chat.id} session={session_id[:30]}. "
                f"This should not happen with UnifiedQueueManager.",
            )

    async def _stream_with_tracker(
        self,
        payload: Any,
    ) -> AsyncGenerator[str, None]:
        """Stream events through TaskTracker for task tracking.

        This method wraps _process and yields SSE-formatted events.
        Called by TaskTracker.attach_or_start to enable task cancellation.

        Args:
            payload: Message payload (dict or AgentRequest)

        Yields:
            SSE-formatted event strings
        """
        request = self._payload_to_request(payload)

        if isinstance(payload, dict):
            send_meta = dict(payload.get("meta") or {})
            if payload.get("session_webhook"):
                send_meta["session_webhook"] = payload["session_webhook"]
        else:
            send_meta = getattr(request, "channel_meta", None) or {}

        bot_prefix = getattr(self, "bot_prefix", None) or getattr(
            self,
            "_bot_prefix",
            "",
        )
        if bot_prefix and "bot_prefix" not in send_meta:
            send_meta = {**send_meta, "bot_prefix": bot_prefix}

        to_handle = self.get_to_handle_from_request(request)

        await self._before_consume_process(request)

        last_response = None
        process_iterator = None
        try:
            process_iterator = self._process(request)
            # Codex OAuth (gpt-5.x) intersperses scratch-style
            # preamble text between tool calls.  The agentscope
            # runtime adapter emits each as a ``MESSAGE`` event
            # immediately followed by a ``PLUGIN_CALL`` event in
            # the same agent turn.  We can't tell preamble from
            # final reply by content alone — but we can tell by
            # what comes next: if a ``PLUGIN_CALL`` lands before
            # the next ``MESSAGE`` / ``response``, the buffered
            # text is preamble and gets dropped.  Otherwise it
            # gets flushed as a normal channel send.
            #
            # Only enabled for codex-oauth agents: Claude / Qwen
            # preambles ("I'll fetch that") are intentional and
            # users want them.  We resolve the agent via the
            # channel's bound workspace because the
            # ``get_current_agent_id`` ContextVar isn't set until
            # ``AgentRunner.stream_query`` runs — which happens
            # AFTER ``_stream_with_tracker`` enters this loop, so
            # reading it here always returned ``None`` and the
            # buffering code was a silent no-op.
            from ...config.config import load_agent_config
            buffer_preamble = False
            try:
                workspace = getattr(self, "_workspace", None)
                agent_id = (
                    getattr(workspace, "agent_id", "") if workspace else ""
                )
                if agent_id:
                    agent_cfg = load_agent_config(agent_id)
                    active = getattr(agent_cfg, "active_model", None)
                    provider = (
                        getattr(active, "provider_id", "") or ""
                    ).lower()
                    buffer_preamble = (provider == "codex-oauth")
            except Exception:
                buffer_preamble = False
            logger.info(
                "channel _stream_with_tracker: buffer_preamble=%s "
                "(workspace_agent=%s)",
                buffer_preamble,
                getattr(getattr(self, "_workspace", None), "agent_id", None),
            )

            pending_message_send: Optional[tuple] = None  # (event, meta)

            def _serialize(ev: Any) -> str:
                if hasattr(ev, "model_dump_json"):
                    return ev.model_dump_json()
                if hasattr(ev, "json"):
                    return ev.json()
                return json.dumps({"text": str(ev)})

            def _yield_sse(ev: Any) -> str:
                return f"data: {_serialize(ev)}\n\n"

            def _yield_sse_as_reasoning(ev: Any) -> str:
                """Re-serialize ``ev`` as a ``REASONING`` message so
                Console UI renders it in the thinking pane instead
                of the bot-reply pane.  Mutates a copy — original
                event is left untouched in case downstream readers
                of the iterator (tracing, telemetry) still need its
                original type.
                """
                try:
                    if hasattr(ev, "model_copy"):
                        ev2 = ev.model_copy(
                            update={"type": MessageType.REASONING},
                        )
                    else:
                        ev2 = ev
                        try:
                            ev2.type = MessageType.REASONING
                        except Exception:
                            pass
                    return _yield_sse(ev2)
                except Exception:
                    return _yield_sse(ev)

            async def _flush_pending():
                nonlocal pending_message_send
                if pending_message_send is not None:
                    pend_event, pend_meta = pending_message_send
                    pending_message_send = None
                    await self.on_event_message_completed(
                        request, to_handle, pend_event, pend_meta,
                    )

            async for event in process_iterator:
                obj = getattr(event, "object", None)
                status = getattr(event, "status", None)
                msg_type = getattr(event, "type", "")

                # ── SSE yield decision ────────────────────────────
                # For codex-oauth MESSAGE.MESSAGE events we *defer*
                # the SSE yield: if a tool call follows we re-yield
                # the buffered event as REASONING (Console UI shows
                # it in the thinking pane); if a MESSAGE / response
                # follows we yield it as-is (it was the final reply).
                # All other events yield immediately so the Console
                # UI never sees a stall.
                deferred_sse = (
                    buffer_preamble
                    and obj == "message"
                    and status == RunStatus.Completed
                    and msg_type != MessageType.REASONING
                )
                if not deferred_sse:
                    yield _yield_sse(event)

                if obj == "content":
                    if await self.on_event_content(
                        request,
                        to_handle,
                        event,
                        send_meta,
                    ):
                        continue
                if obj == "message" and status == RunStatus.Completed:
                    # MESSAGE.REASONING is always suppressed at the
                    # channel.  Console UI already saw the SSE yield
                    # above (deferred_sse is False for REASONING).
                    if msg_type == MessageType.REASONING:
                        await _flush_pending()
                        continue
                    if buffer_preamble:
                        # Hold this MESSAGE until we see what
                        # comes next.  Pending event's SSE has not
                        # been yielded yet — we yield it below
                        # either as MESSAGE (final reply) or as
                        # REASONING (preamble dropped from channel
                        # but visible in Console UI thinking pane).
                        if pending_message_send is not None:
                            # Two MESSAGEs in a row → first one
                            # was the actual final answer, flush
                            # its SSE + dispatch.
                            prior_event, _ = pending_message_send
                            yield _yield_sse(prior_event)
                            await _flush_pending()
                        pending_message_send = (event, send_meta)
                    else:
                        await self.on_event_message_completed(
                            request,
                            to_handle,
                            event,
                            send_meta,
                        )
                elif obj == "message" and status != RunStatus.Completed:
                    # Non-completed message events don't reach the
                    # channel.  Leave any pending alone.
                    pass
                elif obj == "response":
                    # End of stream — pending was the final reply,
                    # flush its SSE + dispatch normally.
                    if pending_message_send is not None:
                        prior_event, _ = pending_message_send
                        yield _yield_sse(prior_event)
                        await _flush_pending()
                    last_response = event
                    await self.on_event_response(request, event)
                else:
                    # Only an OUTGOING tool *call* (not a tool
                    # *output* / result) confirms the previous
                    # MESSAGE was preamble.  Tool outputs may carry
                    # a media-bearing MESSAGE synthesised by
                    # ``runner.utils._build_media_message_from_block``
                    # — dropping that as preamble would silently
                    # eat the user's image / sticker / video.
                    is_tool_call = msg_type in (
                        "function_call",
                        "plugin_call",
                        "mcp_call",
                    )
                    if pending_message_send is not None and is_tool_call:
                        prior_event, _ = pending_message_send
                        yield _yield_sse_as_reasoning(prior_event)
                        logger.info(
                            "channel: dropped codex-oauth preamble text "
                            "before %s/%s (re-yielded as REASONING for UI)",
                            obj, msg_type,
                        )
                        pending_message_send = None

            # Defensive flush in case the iterator ended without a
            # response event.  Anything still pending here was the
            # final reply, so yield + dispatch.
            if pending_message_send is not None:
                prior_event, _ = pending_message_send
                yield _yield_sse(prior_event)
                await _flush_pending()

            err_msg = self._get_response_error_message(last_response)
            if err_msg:
                await self._on_consume_error(
                    request,
                    to_handle,
                    f"Error: {err_msg}",
                )
            else:
                await self._on_process_completed(
                    request,
                    to_handle,
                    send_meta,
                )

            if self._on_reply_sent:
                args = self.get_on_reply_sent_args(request, to_handle)
                self._on_reply_sent(self.channel, *args)

        except asyncio.CancelledError:
            logger.info(
                f"channel task cancelled: "
                f"session={getattr(request, 'session_id', '')[:30]}",
            )
            if process_iterator is not None:
                await process_iterator.aclose()
            raise

        except Exception as e:
            logger.exception(
                f"channel _stream_with_tracker failed: {e}, "
                f"session={getattr(request, 'session_id', 'N/A')[:30]}, "
                f"agent={to_handle}",
            )
            await self._on_consume_error(
                request,
                to_handle,
                "Internal error",
            )
            raise

    @classmethod
    def from_env(
        cls,
        process: ProcessHandler,
        on_reply_sent: OnReplySent = None,
    ) -> "BaseChannel":
        raise NotImplementedError

    @classmethod
    def from_config(
        cls,
        process: ProcessHandler,
        config: Any,
        on_reply_sent: OnReplySent = None,
        show_tool_details: bool = True,
        filter_tool_messages: bool = False,
        filter_thinking: bool = False,
    ) -> "BaseChannel":
        raise NotImplementedError

    def resolve_session_id(
        self,
        sender_id: str,
        channel_meta: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Map sender and optional channel meta to session_id.
        Override in subclasses for channel-specific session keys
        (e.g. short suffix of conversation_id for cron lookup).
        """
        return f"{self.channel}:{sender_id}"

    def build_agent_request_from_user_content(
        self,
        channel_id: str,
        sender_id: str,
        session_id: str,
        content_parts: List[Any],
        channel_meta: Optional[Dict[str, Any]] = None,
    ) -> "AgentRequest":
        """
        Build AgentRequest from runtime content parts (Message content list).
        Use agentscope_runtime Message/Content types; no intermediate envelope.
        Subclasses call this after parsing native payload to content_parts.
        """
        from agentscope_runtime.engine.schemas.agent_schemas import (
            AgentRequest,
            Message,
            Role,
        )

        if not content_parts:
            content_parts = [
                TextContent(type=ContentType.TEXT, text=" "),
            ]
        msg = Message(
            type=MessageType.MESSAGE,
            role=Role.USER,
            content=content_parts,
        )
        return AgentRequest(
            session_id=session_id,
            user_id=sender_id,
            input=[msg],
            channel=channel_id,
        )

    def build_agent_request_from_native(
        self,
        native_payload: Any,
    ) -> "AgentRequest":
        """
        Convert channel-native message payload to AgentRequest.
        Subclasses must implement: parse native -> content_parts (runtime
        Content types), session_id, then build_agent_request_from_user_content.
        Attach channel_meta to result for send path:
        request.channel_meta = meta.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement "
            "build_agent_request_from_native(native_payload)",
        )

    def _payload_to_request(self, payload: Any) -> "AgentRequest":
        """
        Convert queue payload to AgentRequest. Default: if payload looks like
        AgentRequest (has session_id, input), return it; else
        build_agent_request_from_native(payload). Override if needed.
        """
        if payload is None:
            raise ValueError("payload is None")
        if hasattr(payload, "session_id") and hasattr(payload, "input"):
            return payload
        return self.build_agent_request_from_native(payload)

    def get_to_handle_from_request(self, request: "AgentRequest") -> str:
        """
        Resolve send target (to_handle) from AgentRequest. Default: user_id.
        Override for channels that send by session_id (e.g. Feishu).
        """
        return getattr(request, "user_id", "") or ""

    def get_on_reply_sent_args(
        self,
        request: "AgentRequest",
        to_handle: str,
    ) -> tuple:
        """
        Args for _on_reply_sent(channel, *args). Default: (to_handle,
        session_id). Override e.g. to pass (user_id, session_id).
        """
        session_id = (
            getattr(request, "session_id", "") or f"{self.channel}:{to_handle}"
        )
        return (to_handle, session_id)

    async def refresh_webhook_or_token(self) -> None:
        """
        Optional: refresh webhook URL or API token. Override for channels
        that need periodic or on-401 refresh. Default no-op.
        """

    async def consume_one(self, payload: Any) -> None:
        """
        Process one payload from the manager-owned queue. If
        _debounce_seconds > 0 and payload is native (dict with
        content_parts), append to buffer and flush after delay;
        otherwise call _consume_one_request(payload). Messages
        with no text are buffered until text arrives (see
        _apply_no_text_debounce). Override only when you need
        a different flow (e.g. print).
        """
        if self._debounce_seconds > 0 and self._is_native_payload(payload):
            key = self.get_debounce_key(payload)
            if key in self._debounce_pending and self._debounce_pending[key]:
                self._on_debounce_buffer_append(
                    key,
                    payload,
                    self._debounce_pending[key],
                )
            self._debounce_pending.setdefault(key, []).append(payload)
            old = self._debounce_timers.pop(key, None)
            if old and not old.done():
                old.cancel()

            async def flush(k: str) -> None:
                await asyncio.sleep(self._debounce_seconds)
                items = self._debounce_pending.pop(k, [])
                self._debounce_timers.pop(k, None)
                if not items:
                    return
                merged = self.merge_native_items(items)
                if not merged:
                    return
                await self._consume_one_request(merged)

            self._debounce_timers[key] = asyncio.create_task(flush(key))
            return
        await self._consume_one_request(payload)

    def _extract_query_from_payload(self, payload: Any) -> str:
        """Extract query text from payload for command detection.

        Channels may prepend context parts (group history, reply-to
        blocks) to content_parts before the user's actual message. We
        skip those so that slash commands like ``/new`` are detected
        even when history context sits at index 0.

        Args:
            payload: Native dict or AgentRequest

        Returns:
            Query text string (empty if not found)
        """
        # Context-marker prefixes that channels use to inject bounded
        # untrusted metadata (group history, reply-to, etc.). These are
        # never the user's actual message.
        SKIP_PREFIXES = ("=== ", "[Replying")

        def _pick(parts) -> str:
            first_text = ""
            for part in parts:
                if isinstance(part, dict) and part.get("type") == "text":
                    text = part.get("text") or ""
                elif hasattr(part, "type") and part.type == "text":
                    text = getattr(part, "text", "") or ""
                else:
                    continue
                if not first_text:
                    first_text = text
                if not any(text.startswith(p) for p in SKIP_PREFIXES):
                    return text
            return first_text

        if isinstance(payload, dict):
            return _pick(payload.get("content_parts") or [])
        if hasattr(payload, "input"):
            inp = payload.input or []
            if inp and hasattr(inp[0], "content"):
                return _pick(inp[0].content or [])
        return ""

    def _debounce_payload(self, payload: Any) -> bool:
        """Apply no-text debounce on payload; return False if buffered."""
        if isinstance(payload, dict):
            content_parts = payload.get("content_parts") or []
        elif hasattr(payload, "input") and payload.input:
            content_parts = getattr(payload.input[0], "content", None) or []
        else:
            return True

        if not content_parts:
            return True

        session_id = self.get_debounce_key(payload)
        should_process, merged = self._apply_no_text_debounce(
            session_id,
            content_parts,
        )
        if not should_process:
            return False

        # Write merged parts back so downstream paths see full content.
        if isinstance(payload, dict):
            payload["content_parts"] = merged
        elif hasattr(payload, "input") and payload.input:
            first = payload.input[0]
            if hasattr(first, "model_copy"):
                payload.input[0] = first.model_copy(
                    update={"content": merged},
                )
            elif hasattr(first, "content"):
                first.content = merged
        return True

    async def _consume_one_request(self, payload: Any) -> None:
        """
        Convert payload to request, apply no-text debounce, run _process,
        send messages, handle errors and on_reply_sent. Used by
        consume_one (direct or after time-debounce flush).

        If workspace is available, routes through TaskTracker for tracking.
        Control commands bypass TaskTracker for immediate response.
        """
        logger.debug(
            "base _consume_one_request: "
            f"has_workspace={self._workspace is not None}",
        )

        if not self._debounce_payload(payload):
            return

        if self._workspace is not None and self._command_registry is not None:
            query_text = self._extract_query_from_payload(payload)
            logger.debug(
                f"base _consume_one_request: query={query_text[:50]}",
            )
            is_control = self._command_registry.is_control_command(
                query_text,
            )
            logger.debug(
                f"base _consume_one_request: is_control={is_control}",
            )
            if not is_control:
                request = self._payload_to_request(payload)
                await self._consume_with_tracker(request, payload)
                return

        request = self._payload_to_request(payload)
        # Build meta from payload so session_webhook is never lost when
        # request has no channel_meta (e.g. AgentRequest schema has no field).
        if isinstance(payload, dict):
            meta_from_payload = dict(payload.get("meta") or {})
            if payload.get("session_webhook"):
                meta_from_payload["session_webhook"] = payload[
                    "session_webhook"
                ]
            # Always attach so channel _before_consume_process can use it
            # (e.g. Feishu save receive_id for cron send).
            setattr(request, "channel_meta", meta_from_payload)
        to_handle = self.get_to_handle_from_request(request)
        await self._before_consume_process(request)
        # Prefer meta built from payload so session_webhook is present when
        # request.channel_meta is missing (AgentRequest may not have the attr).
        if isinstance(payload, dict):
            send_meta = dict(payload.get("meta") or {})
            if payload.get("session_webhook"):
                send_meta["session_webhook"] = payload["session_webhook"]
        else:
            send_meta = getattr(request, "channel_meta", None) or {}
        bot_prefix = getattr(self, "bot_prefix", None) or getattr(
            self,
            "_bot_prefix",
            "",
        )
        if bot_prefix and "bot_prefix" not in send_meta:
            send_meta = {**send_meta, "bot_prefix": bot_prefix}
        logger.info(
            "base _consume_one_request: send_meta has_session_webhook=%s",
            bool((send_meta or {}).get("session_webhook")),
        )
        await self._run_process_loop(request, to_handle, send_meta)

    async def _run_process_loop(
        self,
        request: "AgentRequest",
        to_handle: str,
        send_meta: Dict[str, Any],
    ) -> None:
        """
        Run _process and send events. Override to use channel-specific
        loop (e.g. DingTalk _process_one_request with webhook sends).
        """
        last_response = None
        try:
            async for event in self._process(request):
                obj = getattr(event, "object", None)
                status = getattr(event, "status", None)
                if obj == "content":
                    if await self.on_event_content(
                        request,
                        to_handle,
                        event,
                        send_meta,
                    ):
                        continue
                if obj == "message" and status == RunStatus.Completed:
                    # Suppress preamble text from tool-using turns —
                    # ``runner.utils`` promotes that text from
                    # ``MESSAGE`` to ``REASONING`` so we can drop it
                    # here without losing visibility on the Console
                    # UI (the SSE ``yield`` above already shipped the
                    # event upstream).  Final reply messages keep
                    # ``MESSAGE`` and reach the channel as before.
                    msg_type = getattr(event, "type", MessageType.MESSAGE)
                    if msg_type != MessageType.REASONING:
                        await self.on_event_message_completed(
                            request,
                            to_handle,
                            event,
                            send_meta,
                        )
                elif obj == "response":
                    last_response = event
                    await self.on_event_response(request, event)
            err_msg = self._get_response_error_message(last_response)
            if err_msg:
                await self._on_consume_error(
                    request,
                    to_handle,
                    f"Error: {err_msg}",
                )
            else:
                await self._on_process_completed(
                    request,
                    to_handle,
                    send_meta,
                )
            if self._on_reply_sent:
                args = self.get_on_reply_sent_args(request, to_handle)
                self._on_reply_sent(self.channel, *args)
        except Exception:
            logger.exception("channel consume_one failed")
            await self._on_consume_error(
                request,
                to_handle,
                "An error occurred while processing your request.",
            )

    def _get_response_error_message(self, last_response: Any) -> Optional[str]:
        """
        Extract error message from runtime response event.
        Handles AgentResponse.error or Event wrapper (e.g. .data / .response).
        """
        if not last_response:
            return None
        resp = last_response
        if getattr(last_response, "data", None) is not None:
            resp = last_response.data
        elif getattr(last_response, "response", None) is not None:
            resp = last_response.response
        err = getattr(resp, "error", None)
        if not err:
            return None
        if hasattr(err, "message"):
            return getattr(err, "message", None) or str(err)
        if isinstance(err, dict):
            return err.get("message") or str(err)
        return str(err)

    async def _before_consume_process(self, request: "AgentRequest") -> None:
        """
        Hook called once per consume_one before running _process. Override
        to e.g. save receive_id for send path (Feishu).
        """

    async def on_event_content(
        self,
        request: "AgentRequest",
        to_handle: str,
        event: Any,
        send_meta: Dict[str, Any],
    ) -> bool:
        """Hook: one content event. Return True if handled."""
        del request
        if getattr(event, "type", None) != ContentType.DATA:
            return False
        status = getattr(event, "status", None)
        if status != RunStatus.InProgress:
            return False
        if self._filter_tool_messages:
            return False
        data = getattr(event, "data", None) or {}
        if not isinstance(data, dict) or "output" not in data:
            return False
        body = self._format_stream_tool_output_body(event)
        if not body:
            return False
        await self.send_content_parts(
            to_handle,
            [TextContent(text=body)],
            send_meta,
        )
        return True

    async def on_event_message_completed(
        self,
        request: "AgentRequest",
        to_handle: str,
        event: Any,
        send_meta: Dict[str, Any],
    ) -> None:
        """
        Hook: one message event completed. Default: send_message_content.
        Override for batch/debounce (e.g. DingTalk merge then send).
        """
        await self.send_message_content(to_handle, event, send_meta)

    async def on_event_response(
        self,
        request: "AgentRequest",
        event: Any,
    ) -> None:
        """Hook: response event received. Default: no-op."""

    async def _on_process_completed(
        self,
        request: "AgentRequest",
        to_handle: str,
        send_meta: Dict[str, Any],
    ) -> None:
        """Hook called after all events processed without error.

        Override for post-processing (e.g. Feishu DONE reaction).
        """

    async def _on_consume_error(
        self,
        request: Any,
        to_handle: str,
        err_text: str,
    ) -> None:
        """
        Called when consume_one hits an error or response.error. Default:
        send err_text via send_content_parts. Override to send via channel
        API (e.g. imessage _send_sync).
        """
        await self.send_content_parts(
            to_handle,
            [TextContent(type=ContentType.TEXT, text=err_text)],
            getattr(request, "channel_meta", None) or {},
        )

    async def send_response(
        self,
        to_handle: str,
        response: "AgentResponse",
        meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Convert AgentResponse to this channel's reply and send.
        Default: take last message text from output and call
        send(to_handle, text, meta).
        Subclasses may override to support image, video attachments.
        """
        text = self._response_to_text(response)
        await self.send(to_handle, text or "", meta)

    def _message_to_content_parts(
        self,
        message: Any,
    ) -> List[OutgoingContentPart]:
        """
        Convert a Message (object=='message') into sendable parts.
        Delegates to self._renderer; override _renderer or _render_style
        for channel-specific formatting.
        """
        return self._renderer.message_to_parts(message)

    async def send_message_content(
        self,
        to_handle: str,
        message: Any,
        meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Send all content of a Message
        (text, image, video, audio, file, refusal).
        Subclasses may override send_content_parts for channel-specific
        multi-part sending.
        """
        parts = self._message_to_content_parts(message)
        if not parts:
            logger.debug(
                f"channel send_message_content: no parts for to_handle="
                f"{to_handle}, skip send",
            )
            return
        logger.debug(
            f"channel send_message_content: to_handle={to_handle} "
            f"parts_count={len(parts)} "
            f"part_types={[getattr(p, 'type', None) for p in parts]}",
        )
        await self.send_content_parts(to_handle, parts, meta)

    def _truncate_stream_tool_chunk(
        self,
        text: Any,
        limit: int = 72,
    ) -> str:
        preview = " ".join(str(text or "").split()).strip()
        if len(preview) > limit:
            return preview[:limit] + "..."
        return preview

    def _format_stream_tool_output_body(
        self,
        event: Any,
    ) -> Optional[str]:
        data = getattr(event, "data", None) or {}
        if not isinstance(data, dict):
            return None
        output = data.get("output")
        if isinstance(output, str):
            try:
                output = json.loads(output)
            except json.JSONDecodeError:
                return None
        if not isinstance(output, list):
            return None

        tool_name = data.get("name") or "tool"
        chunks: List[str] = []
        seen_chunks: set[str] = set()
        for block in output:
            if not isinstance(block, dict):
                continue
            block_type = block.get("type")
            raw_text = ""
            if block_type == "text":
                raw_text = str(block.get("text") or "")
            elif block_type == "thinking":
                raw_text = str(block.get("thinking") or "")
            if not raw_text.strip():
                continue
            preview = self._truncate_stream_tool_chunk(raw_text)
            if not preview or preview in seen_chunks:
                continue
            seen_chunks.add(preview)
            chunks.append(preview)
        if not chunks:
            return None
        return f"⌛️ **{tool_name}**:\n" + "\n".join(
            f"`{text}`" for text in chunks
        )

    async def send_content_parts(
        self,
        to_handle: str,
        parts: List[OutgoingContentPart],
        meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Send a list of content parts.
        Default: merge text/refusal into one text, append media URLs as
        fallback, send one message; optionally call send_media for each
        media part if overridden.
        """
        text_parts: List[str] = []
        media_parts: List[OutgoingContentPart] = []
        for p in parts:
            t = getattr(p, "type", None)
            if t == ContentType.TEXT and getattr(p, "text", None):
                text_parts.append(p.text or "")
            elif t == ContentType.REFUSAL and getattr(p, "refusal", None):
                text_parts.append(p.refusal or "")
            elif t in (
                ContentType.IMAGE,
                ContentType.VIDEO,
                ContentType.AUDIO,
                ContentType.FILE,
            ):
                media_parts.append(p)
        body = "\n".join(text_parts) if text_parts else ""
        prefix = (meta or {}).get("bot_prefix", "") or ""
        if prefix and body:
            body = prefix + "  " + body
        for m in media_parts:
            t = getattr(m, "type", None)
            if t == ContentType.IMAGE and getattr(m, "image_url", None):
                body += f"\n[Image: {m.image_url}]"
            elif t == ContentType.VIDEO and getattr(m, "video_url", None):
                body += f"\n[Video: {m.video_url}]"
            elif t == ContentType.FILE and (
                getattr(m, "file_url", None) or getattr(m, "file_id", None)
            ):
                body += f"\n[File: {m.file_url or m.file_id}]"
            elif t == ContentType.AUDIO and getattr(m, "data", None):
                body += "\n[Audio]"
        if body.strip():
            logger.debug(
                f"channel send_content_parts: to_handle={to_handle} "
                f"body_len={len(body)} preview="
                f"{body[:120] + '...' if len(body) > 120 else body}",
            )
            await self.send(to_handle, body.strip(), meta)
        for m in media_parts:
            await self.send_media(to_handle, m, meta)

    async def send_media(
        self,
        to_handle: str,
        part: OutgoingContentPart,
        meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Send a single media part (image, video, audio, file).
        Default: no-op (already appended to text in send_content_parts).
        Subclasses override to send real attachments.
        """
        pass

    def _response_to_text(self, response: "AgentResponse") -> str:
        """Extract reply text from the last ``message``-type output item.

        Searches backwards so trailing reasoning / tool-output items are
        skipped.
        """
        if not response.output:
            return ""

        last_msg = None
        for msg in reversed(response.output):
            if msg.type == MessageType.MESSAGE and msg.content:
                last_msg = msg
                break

        if not last_msg:
            return ""
        parts = []
        for c in last_msg.content:
            if getattr(c, "type", None) == ContentType.TEXT and getattr(
                c,
                "text",
                None,
            ):
                parts.append(c.text)
            elif getattr(c, "type", None) == ContentType.REFUSAL and getattr(
                c,
                "refusal",
                None,
            ):
                parts.append(c.refusal)
        return "".join(parts)

    def clone(self, config) -> "BaseChannel":
        """Clone a new channel instance with updated config, cloning
        process and on_reply_sent from self.

        Subclasses must implement from_config(process, config, on_reply_sent).

        show_tool_details is global config (not in channel config), so we
        preserve from self. filter_tool_messages and filter_thinking are
        per-channel config, so we read from new config.

        workspace_dir is the agent's per-agent credential/state root. It
        is NOT in `config` (it's agent-level, not channel-level), so we
        must carry it across from self — otherwise channels that rely on
        the workspace_dir fallback in their `data_dir`/`auth_dir`
        resolver (WhatsApp, Signal, …) silently fall back to the
        install-wide WORKING_DIR on hot reload, breaking per-agent
        credential isolation. Subclasses that accept `workspace_dir` in
        `from_config` store it as `self._workspace_dir`; we introspect
        that here and pass it back. Channels without that attribute are
        unaffected.
        """
        kwargs = {
            "process": self._process,
            "config": config,
            "on_reply_sent": self._on_reply_sent,
            "show_tool_details": getattr(self, "_show_tool_details", True),
            "filter_tool_messages": getattr(
                config,
                "filter_tool_messages",
                False,
            ),
            "filter_thinking": getattr(
                config,
                "filter_thinking",
                False,
            ),
        }
        ws = getattr(self, "_workspace_dir", None)
        if ws is not None:
            kwargs["workspace_dir"] = ws
        return self.__class__.from_config(**kwargs)

    async def update_config(self, config) -> bool:
        """Try to update config in-place without restart.

        Returns True if applied successfully (no restart needed).
        Returns False if a full clone+replace is required.
        Default: returns False (subclasses override to support hot patching).
        """
        return False

    async def health_check(self) -> Dict[str, Any]:
        """Return health status for this channel.

        Default implementation returns a basic status dict.
        Subclasses can override to add channel-specific checks
        (e.g. webhook reachability, token validity, polling status).

        Returns:
            Dict with at least: channel, status ("healthy" / "unhealthy"),
            and optional detail, error fields.
        """
        return {
            "channel": self.channel,
            "status": "healthy",
            "detail": "Channel is loaded and running.",
        }

    async def start(self) -> None:
        raise NotImplementedError

    async def stop(self) -> None:
        raise NotImplementedError

    async def send(
        self,
        to_handle: str,
        text: str,
        meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Subclass implements: send one text
        (and optional attachments) to to_handle.
        """
        raise NotImplementedError

    def to_handle_from_target(self, *, user_id: str, session_id: str) -> str:
        """Map cron dispatch target to channel-specific to_handle.

        Default: use user_id. For many channels, this is enough.
        Discord proactive send relies on meta['channel_id'] or
         meta['user_id'] anyway.
        """
        return user_id

    async def send_event(
        self,
        *,
        user_id: str,
        session_id: str,
        event: "Event",
        meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Send a runner Event to this channel (non-stream).

        We only send when event is a completed message, then reuse
        send_message_content().
        """
        # Delay import to avoid hard dependency at module import time

        obj = getattr(event, "object", None)
        status = getattr(event, "status", None)

        if obj != "message" or status != RunStatus.Completed:
            return

        to_handle = self.to_handle_from_target(
            user_id=user_id,
            session_id=session_id,
        )
        await self.send_message_content(to_handle, event, meta)
