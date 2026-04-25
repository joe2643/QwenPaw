# -*- coding: utf-8 -*-
"""WhatsApp channel: neonize (whatsmeow Go backend).

Features:
- Text messages (DM + group)
- Image/audio/document send and receive
- Mentions
- Reactions
- QR code / pair code authentication
- Group shared sessions
"""

from __future__ import annotations

import asyncio
import datetime
import logging
import os
import re
from pathlib import Path
from typing import Any, Optional, Dict, List, Union

from agentscope_runtime.engine.schemas.agent_schemas import (
    TextContent,
    ImageContent,
    AudioContent,
    FileContent,
    VideoContent,
    ContentType,
    RunStatus,
)

from ....config.config import WhatsAppConfig
from ..media_utils import resolve_media_url
from ..base import (
    BaseChannel,
    OnReplySent,
    ProcessHandler,
    OutgoingContentPart,
)

logger = logging.getLogger(__name__)


class _WhatsAppAlbumBuffer:
    """In-progress album collation state.

    WhatsApp's album protocol delivers an ``albumMessage`` header
    that announces ``expectedImageCount`` + ``expectedVideoCount``,
    then the children arrive as independent
    ``imageMessage`` / ``videoMessage`` payloads from the same
    sender within roughly one second.  No formal album_id ties
    children to the header — we collate on (chat, sender) plus a
    short timeout.

    The header's ``content_parts`` is empty (the album body itself
    carries no media) but it can carry the reply context (quote
    parts).  We stash everything on this buffer and flush when the
    expected count is reached or the timer fires, whichever is
    earlier — partial flushes are better than dropping a child if
    one delivery is lost.
    """

    __slots__ = (
        "header_msg_id",
        "header_msg",
        "header_timestamp",
        "quote_parts",
        "expected",
        "gathered_parts",
        "gathered_paths",
        "gathered_body_parts",
        "timeout_task",
        "flushed",
    )

    def __init__(
        self,
        header_msg_id: str,
        header_msg: Any,
        header_timestamp: Any,
        quote_parts: List[Any],
        expected: int,
    ) -> None:
        self.header_msg_id = header_msg_id
        self.header_msg = header_msg
        self.header_timestamp = header_timestamp
        self.quote_parts: List[Any] = quote_parts
        self.expected: int = expected
        self.gathered_parts: List[Any] = []
        self.gathered_paths: List[str] = []
        # Captions arrive on the children, not the album header —
        # join them in arrival order so the agent sees the user's
        # caption alongside the media.
        self.gathered_body_parts: List[str] = []
        self.timeout_task: Optional[asyncio.Task] = None
        self.flushed: bool = False

    def is_complete(self) -> bool:
        """``True`` when the expected number of children have arrived
        — ready for an immediate flush without waiting for timeout.
        Counted by media parts (image / video) only; text-only parts
        from a caption don't count toward the album quota."""
        if self.expected <= 0:
            return False
        media_count = sum(
            1
            for p in self.gathered_parts
            if isinstance(p, (ImageContent, VideoContent))
        )
        return media_count >= self.expected


WHATSAPP_MAX_TEXT_LENGTH = 4096

# Single source of truth for the timestamp string the agent sees on
# WhatsApp inbound — render in the host's local timezone so the model
# never has to reason across zones (whichever zone the operator runs
# in is the one that matches their own clock).  Returns ``""`` on
# any parse failure so the caller can substitute the raw value or
# omit the prefix.
def _format_local_timestamp(
    ts,
    style: str = "long",
) -> str:
    """Render ``ts`` (epoch seconds, epoch ms, str, or
    ``datetime.datetime``) in the host's local timezone.

    ``style="long"``  → ``"2026年4月25日 19:40:11 JST"`` (history block)
    ``style="short"`` → ``"2026-04-25 19:40 JST"`` (envelope prefix)

    The trailing label is whatever the system reports via
    ``time.tzname`` for this moment (handles DST transitions
    correctly because we resolve via ``astimezone()`` per call).
    """
    try:
        if isinstance(ts, datetime.datetime):
            # Naive datetime → assume local; aware datetime → convert.
            dt = (
                ts.astimezone()
                if ts.tzinfo is not None
                else ts.astimezone()
            )
        else:
            ts_val = float(ts)
            if ts_val > 1e12:
                ts_val /= 1000  # epoch milliseconds → seconds
            dt = datetime.datetime.fromtimestamp(ts_val).astimezone()
    except (TypeError, ValueError, OverflowError):
        return ""
    tz_label = dt.strftime("%Z") or ""
    if style == "short":
        return (
            dt.strftime("%Y-%m-%d %H:%M ") + tz_label
        ).rstrip()
    return (
        f"{dt.year}年{dt.month}月{dt.day}日 "
        + dt.strftime("%H:%M:%S ")
        + tz_label
    ).rstrip()
from ....constant import WORKING_DIR

_MEDIA_DIR = WORKING_DIR / "media" / "whatsapp"
# Default auth_dir: WORKING_DIR/credentials/whatsapp/default when no
# workspace_dir is passed. When a workspace_dir IS passed (agent-scoped
# install), the default becomes workspace_dir/credentials/whatsapp/default
# so each agent gets its own WhatsApp session DB. Explicit `auth_dir` in
# the channel config overrides both.
_DEFAULT_AUTH_DIR = WORKING_DIR / "credentials" / "whatsapp" / "default"


def _resolve_wa_auth_dir(
    explicit_auth_dir: str,
    workspace_dir: Optional["Path"] = None,
) -> "Path":
    """Compute WhatsApp auth_dir path consistently across channel + router.

    Priority: explicit > workspace > WORKING_DIR default.
    """
    if explicit_auth_dir:
        return Path(explicit_auth_dir).expanduser()
    if workspace_dir is not None:
        return (
            Path(workspace_dir).expanduser()
            / "credentials"
            / "whatsapp"
            / "default"
        )
    return _DEFAULT_AUTH_DIR


try:
    from neonize.aioze.client import NewAClient
    from neonize.events import (
        MessageEv,
        ConnectedEv,
        DisconnectedEv,
        ConnectFailureEv,
        KeepAliveTimeoutEv,
        QREv,
    )
    from neonize.utils import build_jid

    NEONIZE_AVAILABLE = True
except ImportError:
    NEONIZE_AVAILABLE = False
    NewAClient = None
    logger.warning(
        "neonize-qwenpaw not installed. WhatsApp channel unavailable. "
        "Install: pip install qwenpaw[whatsapp] "
        "(or explicitly: pip install neonize-qwenpaw)",
    )


def _jid_to_str(jid) -> str:
    """Convert JID protobuf to readable string."""
    if hasattr(jid, "User") and jid.User:
        return (
            f"{jid.User}@{jid.Server}" if hasattr(jid, "Server") else jid.User
        )
    return str(jid)


def _str_to_jid(s: str):
    """Convert string to JID. Handles phone numbers and group IDs."""
    if "@" in s:
        user, server = s.split("@", 1)
        return build_jid(user, server)
    # Phone number
    num = s.lstrip("+")
    return build_jid(num, "s.whatsapp.net")


def _is_group_jid(jid) -> bool:
    """Check if JID is a group."""
    server = getattr(jid, "Server", "")
    return server == "g.us"


class WhatsAppChannel(BaseChannel):
    """WhatsApp channel using neonize (whatsmeow)."""

    channel = "whatsapp"
    uses_manager_queue = True
    requires_sequential_restart = True

    def __init__(
        self,
        process: ProcessHandler,
        enabled: bool = False,
        auth_dir: str = "",
        workspace_dir: Optional[Path] = None,
        on_reply_sent: OnReplySent = None,
        show_tool_details: bool = True,
        filter_tool_messages: bool = False,
        filter_thinking: bool = False,
        dm_policy: str = "open",
        group_policy: str = "open",
        allow_from: Optional[list] = None,
        deny_message: str = "",
        require_mention: bool = False,
        send_read_receipts: bool = True,
        text_chunk_limit: int = WHATSAPP_MAX_TEXT_LENGTH,
        self_chat_mode: bool = False,
        ack_reaction_thinking: str = "🤔",
        ack_reaction_done: str = "👀",
        ack_reaction_error: str = "⚠️",
        **kwargs,
    ):
        super().__init__(
            process,
            on_reply_sent=on_reply_sent,
            show_tool_details=show_tool_details,
            filter_tool_messages=filter_tool_messages,
            filter_thinking=filter_thinking,
            dm_policy=dm_policy,
            group_policy=group_policy,
            allow_from=allow_from,
            deny_message=deny_message,
            require_mention=require_mention,
        )
        self.enabled = enabled
        self._workspace_dir = (
            Path(workspace_dir).expanduser() if workspace_dir else None
        )
        self._auth_dir = _resolve_wa_auth_dir(auth_dir, self._workspace_dir)
        self._send_read_receipts = send_read_receipts
        self._text_chunk_limit = text_chunk_limit
        self._self_chat_mode = self_chat_mode
        self._ack_reaction_thinking = ack_reaction_thinking or ""
        self._ack_reaction_done = ack_reaction_done or ""
        self._ack_reaction_error = ack_reaction_error or ""
        self._groups: List[str] = kwargs.get("groups") or []
        self._group_allow_from: List[str] = (
            kwargs.get("group_allow_from") or []
        )
        self._reply_to_trigger: bool = kwargs.get("reply_to_trigger", True)
        self._pending_quote_msgs: Dict[
            str,
            Any,
        ] = {}  # chat_jid -> raw neonize message
        self._media_dir = _MEDIA_DIR
        self._client: Optional[Any] = None
        self._lid_cache: Dict[
            str,
            Dict[str, str],
        ] = {}  # lid -> {"phone": "+852...", "name": "Joe"}
        self._connected = False
        self._stopping = False  # set by stop() so DisconnectedEv handlers don't auto-reconnect during shutdown
        self._reconnect_lock: Optional[
            asyncio.Lock
        ] = None  # lazy-init in _auto_reconnect (asyncio.Lock needs a loop)
        self._connect_task = None
        self._my_jid = None
        self._bot_phone = ""
        self._bot_lid = ""
        self._group_history: Dict[
            str,
            list,
        ] = {}  # chat_jid -> [{sender, body, ts}]
        self._group_history_limit = 50
        # Scratch buffer used by _extract_message_content to hand raw local
        # paths back to the dispatch loop for group-history storage. Reset
        # at the top of each _extract_message_content call.
        self._last_extracted_media_paths: List[str] = []

        # Inbound-media path index: ``(chat_jid_str, msg_id) ->
        # [local_path, ...]``.  Keyed on the WhatsApp stanzaID so a
        # later reply whose contextInfo.stanzaID points at a past
        # album can resolve the actual image paths (the album
        # header proto carries no media keys — children do, but
        # WhatsApp's reply ID points at the header).  Bounded by
        # ``_inbound_media_limit`` to cap memory; FIFO eviction.
        self._inbound_media: Dict[tuple[str, str], list[str]] = {}
        self._inbound_media_order: list[tuple[str, str]] = []
        self._inbound_media_limit: int = 200

        # Album buffers: keyed by (chat_jid_str, sender_jid_str).
        # WhatsApp encodes a multi-image / multi-video send as an
        # ``albumMessage`` header (with ``expectedImageCount`` /
        # ``expectedVideoCount``) followed by N independent
        # ``imageMessage`` / ``videoMessage`` payloads from the same
        # sender within ~1 s.  No formal album_id ties children to
        # the header, so we collate by (sender, chat) plus a short
        # timeout, dispatch one combined Msg to the agent instead of
        # N fragmented turns + one silently-dropped header.
        self._album_buffers: Dict[
            tuple[str, str],
            "_WhatsAppAlbumBuffer",
        ] = {}
        # Time window after the album header during which arriving
        # imageMessage/videoMessage from the same (sender, chat) are
        # treated as album children.  WhatsApp normally delivers all
        # children within < 1 s; 5 s gives plenty of slack for
        # network jitter without risking accidental collation of a
        # genuinely-fresh next message.
        self._album_timeout_s: float = 5.0

        if self.enabled and not NEONIZE_AVAILABLE:
            logger.error("whatsapp: neonize not installed, channel disabled")
            self.enabled = False

        if self.enabled:
            logger.info(
                "whatsapp: channel initialized (auth_dir=%s)",
                self._auth_dir,
            )

    @classmethod
    def from_config(
        cls,
        process: ProcessHandler,
        config: Union[WhatsAppConfig, dict],
        on_reply_sent: OnReplySent = None,
        show_tool_details: bool = True,
        filter_tool_messages: bool = False,
        filter_thinking: bool = False,
        workspace_dir: Path | None = None,
        **kwargs,
    ) -> "WhatsAppChannel":
        if isinstance(config, dict):
            c = config
        elif hasattr(config, "model_dump"):
            c = config.model_dump()
        else:
            c = vars(config) if hasattr(config, "__dict__") else dict(config)
        return cls(
            process=process,
            enabled=bool(c.get("enabled", False)),
            auth_dir=c.get("auth_dir") or "",
            workspace_dir=workspace_dir,
            on_reply_sent=on_reply_sent,
            show_tool_details=show_tool_details,
            filter_tool_messages=filter_tool_messages,
            filter_thinking=filter_thinking,
            dm_policy=c.get("dm_policy") or "open",
            group_policy=c.get("group_policy") or "open",
            allow_from=c.get("allow_from") or [],
            deny_message=c.get("deny_message") or "",
            require_mention=c.get("require_mention", False),
            send_read_receipts=c.get("send_read_receipts", True),
            text_chunk_limit=c.get(
                "text_chunk_limit",
                WHATSAPP_MAX_TEXT_LENGTH,
            ),
            self_chat_mode=c.get("self_chat_mode", False),
            ack_reaction_thinking=c.get("ack_reaction_thinking", "🤔"),
            ack_reaction_done=c.get("ack_reaction_done", "👀"),
            ack_reaction_error=c.get("ack_reaction_error", "⚠️"),
            groups=c.get("groups") or [],
            group_allow_from=c.get("group_allow_from") or [],
            reply_to_trigger=c.get("reply_to_trigger", True),
        )

    async def update_config(self, config) -> bool:
        """Patch config in-place without restarting neonize.

        Returns False if auth_dir or enabled changed (needs full restart).
        """
        if isinstance(config, dict):
            c = config
        elif hasattr(config, "model_dump"):
            c = config.model_dump()
        else:
            c = vars(config) if hasattr(config, "__dict__") else dict(config)

        # Fields that require full restart
        new_auth_dir = c.get("auth_dir") or ""
        new_auth_path = _resolve_wa_auth_dir(new_auth_dir, self._workspace_dir)
        if new_auth_path != self._auth_dir:
            logger.info(
                "whatsapp: update_config: auth_dir changed, needs restart",
            )
            return False

        new_enabled = bool(c.get("enabled", False))
        if new_enabled != self.enabled:
            logger.info(
                "whatsapp: update_config: enabled changed, needs restart",
            )
            return False

        # If the existing neonize client is dead (e.g. server-forced logout
        # or zombie after EOF with no DisconnectedEv), in-place reload
        # would preserve a useless client and every subsequent send
        # would fail with "device JID missing" / "websocket not
        # connected".  Force a full restart so the channel teardown +
        # fresh start picks up any newly-paired credentials in
        # ``neonize.db`` and fires ``ConnectedEv`` cleanly.
        if not self._connected:
            logger.info(
                "whatsapp: update_config: neonize client is dead "
                "(_connected=False) — triggering full restart so the "
                "fresh client re-reads device credentials.",
            )
            return False

        # Soft-patchable fields
        self._send_read_receipts = c.get("send_read_receipts", True)
        self._text_chunk_limit = c.get(
            "text_chunk_limit",
            WHATSAPP_MAX_TEXT_LENGTH,
        )
        self._self_chat_mode = c.get("self_chat_mode", False)
        self._ack_reaction_thinking = c.get("ack_reaction_thinking", "") or ""
        self._ack_reaction_done = c.get("ack_reaction_done", "") or ""
        self._ack_reaction_error = c.get("ack_reaction_error", "") or ""
        self._groups = c.get("groups") or []
        self._group_allow_from = c.get("group_allow_from") or []
        self._reply_to_trigger = c.get("reply_to_trigger", True)

        # BaseChannel fields
        self.dm_policy = c.get("dm_policy") or "open"
        self.group_policy = c.get("group_policy") or "open"
        self.allow_from = set(c.get("allow_from") or [])
        self.deny_message = c.get("deny_message") or ""
        self.require_mention = c.get("require_mention", False)
        self._filter_tool_messages = c.get("filter_tool_messages", False)
        self._filter_thinking = c.get("filter_thinking", False)

        logger.info("whatsapp: config updated in-place (neonize preserved)")
        return True

    # ── Lifecycle ─────────────────────────────────────────────────────

    async def _rewire_handlers(self) -> None:
        """Re-register every ``@self._client.event(...)`` handler.

        Used after ``_auto_reconnect`` recreates the neonize client
        to pick up a fresh device (see the docstring in
        ``_auto_reconnect``).  The handlers originally live inside
        ``start()``'s body to close over ``self``, and the decorator
        binds them to whatever ``self._client`` was at registration
        time — swapping the client without re-binding leaves the new
        client deaf to every event.  Keep this body in sync with the
        handler definitions in ``start()``.
        """
        # Call start() again but only the handler-binding portion
        # would require threading a flag into start(); simpler to let
        # this method be a deliberate mirror, and the test
        # ``test_rewire_handlers_matches_start`` enforces the pair
        # stay in sync.
        from neonize.events import (
            MessageEv,
            ConnectedEv,
            QREv,
            DisconnectedEv,
            ConnectFailureEv,
            KeepAliveTimeoutEv,
        )

        @self._client.event(MessageEv)
        async def on_message(client, message):
            logger.debug("whatsapp: MessageEv received (rewired)")
            await self._on_message(client, message)

        @self._client.event(ConnectedEv)
        async def on_connected(client, evt):
            self._connected = True
            self._my_jid = client.me
            try:
                import sqlite3

                def _read_device_jid():
                    _db = str(self._auth_dir / "neonize.db")
                    _conn = sqlite3.connect(_db)
                    try:
                        return _conn.execute(
                            "SELECT jid, lid FROM whatsmeow_device LIMIT 1",
                        ).fetchone()
                    finally:
                        _conn.close()

                row = await asyncio.to_thread(_read_device_jid)
                if row:
                    jid_str = row[0] or ""
                    lid_str = row[1] or ""
                    bot_phone = (
                        jid_str.split(":")[0]
                        if ":" in jid_str
                        else jid_str.split("@")[0]
                    )
                    bot_lid = (
                        lid_str.split(":")[0]
                        if ":" in lid_str
                        else lid_str.split("@")[0]
                    )
                    self._my_jid = _str_to_jid(bot_phone)
                    self._bot_phone = bot_phone
                    self._bot_lid = bot_lid
                    if bot_lid and bot_phone:
                        self._lid_cache[f"{bot_lid}@lid"] = {
                            "phone": bot_phone,
                            "name": "bot",
                            "lid": f"{bot_lid}@lid",
                        }
                    logger.info(
                        "whatsapp: connected as phone=%s lid=%s (rewired)",
                        bot_phone,
                        bot_lid,
                    )
            except Exception as e:
                logger.warning(
                    "whatsapp: failed to read JID from DB (rewired): %s",
                    e,
                )

        @self._client.event(QREv)
        async def on_qr(client, evt):
            logger.info(
                "whatsapp: QR code event received during reconnect "
                "(new pair required)",
            )

        def _schedule_reconnect(delay: int, reason: str) -> None:
            if self._stopping:
                return
            logger.warning(
                "whatsapp: %s — scheduling auto-reconnect in %ds (rewired)",
                reason,
                delay,
            )
            loop = asyncio.get_running_loop()
            loop.call_later(
                delay,
                lambda: loop.create_task(self._auto_reconnect()),
            )

        @self._client.event(DisconnectedEv)
        async def on_disconnected(client, evt):
            self._connected = False
            _schedule_reconnect(10, "DISCONNECTED")

        @self._client.event(ConnectFailureEv)
        async def on_connect_failure(client, evt):
            self._connected = False
            reason = getattr(evt, "reason", "unknown")
            _schedule_reconnect(30, f"connection failure (reason={reason})")

        @self._client.event(KeepAliveTimeoutEv)
        async def on_keepalive_timeout(client, evt):
            _schedule_reconnect(15, "keepalive timeout")

    async def start(self) -> None:
        if not self.enabled:
            return

        self._auth_dir.mkdir(parents=True, exist_ok=True)
        db_path = str(self._auth_dir / "neonize.db")

        logger.info("whatsapp: starting channel (db=%s)", db_path)
        self._client = NewAClient(name=db_path)

        # Register event handlers BEFORE connecting
        @self._client.event(MessageEv)
        async def on_message(client, message):
            logger.debug("whatsapp: MessageEv received")
            await self._on_message(client, message)

        @self._client.event(ConnectedEv)
        async def on_connected(client, evt):
            self._connected = True
            self._my_jid = client.me

            # Read bot JID from database (client.me may be empty at connect time)
            try:
                import sqlite3

                def _read_device_jid():
                    _db = str(self._auth_dir / "neonize.db")
                    _conn = sqlite3.connect(_db)
                    try:
                        return _conn.execute(
                            "SELECT jid, lid FROM whatsmeow_device LIMIT 1",
                        ).fetchone()
                    finally:
                        _conn.close()

                row = await asyncio.to_thread(_read_device_jid)
                if row:
                    jid_str = (
                        row[0] or ""
                    )  # e.g. "817089933036:1@s.whatsapp.net"
                    lid_str = row[1] or ""  # e.g. "229661330157571:1@lid"
                    # Extract phone number and LID
                    bot_phone = (
                        jid_str.split(":")[0]
                        if ":" in jid_str
                        else jid_str.split("@")[0]
                    )
                    bot_lid = (
                        lid_str.split(":")[0]
                        if ":" in lid_str
                        else lid_str.split("@")[0]
                    )
                    self._my_jid = _str_to_jid(bot_phone)
                    self._bot_phone = bot_phone
                    self._bot_lid = bot_lid
                    if bot_lid and bot_phone:
                        self._lid_cache[f"{bot_lid}@lid"] = {
                            "phone": bot_phone,
                            "name": "bot",
                            "lid": f"{bot_lid}@lid",
                        }
                    logger.info(
                        "whatsapp: connected as phone=%s lid=%s",
                        bot_phone,
                        bot_lid,
                    )
            except Exception as e:
                logger.warning("whatsapp: failed to read JID from DB: %s", e)
            # Enable delivery receipts (double check marks) - gated on config
            if self._send_read_receipts:
                try:
                    await client.set_force_activate_delivery_receipts(True)
                    logger.info("whatsapp: delivery receipts activated")
                except Exception as e:
                    logger.warning("whatsapp: delivery receipts failed: %s", e)

        @self._client.event(QREv)
        async def on_qr(client, evt):
            logger.info(
                "whatsapp: QR code event received (authentication needed)",
            )

        def _schedule_reconnect(delay: int, reason: str) -> None:
            # Don't auto-reconnect during a deliberate stop() — the disconnect
            # is expected and scheduling a reconnect would race the shutdown.
            if self._stopping:
                logger.debug(
                    "whatsapp: %s — skipping auto-reconnect (stop in progress)",
                    reason,
                )
                return
            logger.warning(
                "whatsapp: %s — scheduling auto-reconnect in %ds",
                reason,
                delay,
            )
            # Use get_running_loop() + create_task() — get_event_loop() and
            # ensure_future() are deprecated as entry points in Python 3.10+.
            # This is safe here because the event handlers that call
            # _schedule_reconnect are async and always run inside a loop.
            loop = asyncio.get_running_loop()
            loop.call_later(
                delay,
                lambda: loop.create_task(self._auto_reconnect()),
            )

        @self._client.event(DisconnectedEv)
        async def on_disconnected(client, evt):
            self._connected = False
            _schedule_reconnect(10, "DISCONNECTED")

        @self._client.event(ConnectFailureEv)
        async def on_connect_failure(client, evt):
            self._connected = False
            reason = getattr(evt, "reason", "unknown")
            _schedule_reconnect(30, f"connection failure (reason={reason})")

        @self._client.event(KeepAliveTimeoutEv)
        async def on_keepalive_timeout(client, evt):
            _schedule_reconnect(15, "keepalive timeout")

        # Start connection - connect_task must be kept running
        try:
            self._connect_task = await self._client.connect()
            logger.info(
                "whatsapp: channel started, waiting for authentication...",
            )
            # Give time for connection to establish
            await asyncio.sleep(2)
            logger.info(
                "whatsapp: channel status - connected=%s, client=%s, task=%s",
                self._connected,
                self._client is not None,
                self._connect_task is not None,
            )
        except Exception:
            logger.exception("whatsapp: failed to start")

    async def stop(self) -> None:
        if not self.enabled:
            return

        # Flag so the DisconnectedEv / ConnectFailureEv handlers installed in
        # start() don't schedule an auto-reconnect while we're deliberately
        # shutting down.
        self._stopping = True

        if self._connect_task:
            self._connect_task.cancel()
            try:
                await self._connect_task
            except (asyncio.CancelledError, Exception):
                pass

        if self._client:
            try:
                await self._client.disconnect()
            except Exception:
                pass
        self._connected = False
        logger.info("whatsapp: channel stopped")

    # ── Inbound message handler ───────────────────────────────────────

    async def _extract_message_content(self, client, msg, msg_id) -> tuple:
        """Extract body text and content parts from a WhatsApp message.

        Returns ``(body, content_parts)`` for backwards compatibility. The
        raw local-media paths that were downloaded here are also recorded
        on ``self._last_extracted_media_paths`` so the group-history store
        can reference them directly without having to reason about whether
        a content_part's ``image_url`` is a local path, a file:// URL, a
        signed HTTPS URL, or base64.
        """
        body = ""
        content_parts: List[Any] = []
        media_local_paths: List[str] = []
        # Clear the per-call scratch buffer on entry so stale paths from a
        # previous message don't leak into the next.
        self._last_extracted_media_paths = media_local_paths

        # Text message
        if msg.conversation:
            body = msg.conversation
        elif (
            msg.HasField("extendedTextMessage")
            and msg.extendedTextMessage.text
        ):
            body = msg.extendedTextMessage.text

        # Resolve LID mentions in body (e.g. @229661330157571 -> @+85251159218)
        if body:
            import re as _re

            lid_mentions = _re.findall(r"@(\d{12,20})", body)
            for lid_num in lid_mentions:
                lid_key = f"{lid_num}@lid"
                if lid_key not in self._lid_cache:
                    try:
                        from neonize.utils import build_jid

                        lid_jid = build_jid(lid_num, "lid")
                        await self._resolve_lid(client, lid_key, lid_jid)
                    except Exception:
                        pass
                cached = self._lid_cache.get(lid_key, {})
                phone = cached.get("phone", "")
                if phone:
                    body = body.replace(f"@{lid_num}", f"@+{phone}")

            content_parts.append(TextContent(type=ContentType.TEXT, text=body))

        # Image
        if msg.HasField("imageMessage"):
            img_msg = msg.imageMessage
            caption = img_msg.caption or ""
            if caption and not body:
                body = caption
                content_parts.append(
                    TextContent(type=ContentType.TEXT, text=caption),
                )
            try:
                self._media_dir.mkdir(parents=True, exist_ok=True)
                path = self._media_dir / f"wa_img_{msg_id}.jpg"
                await client.download_any(msg, path=str(path))
                media_local_paths.append(str(path))
                media_url = await resolve_media_url(str(path))
                content_parts.append(
                    ImageContent(type=ContentType.IMAGE, image_url=media_url),
                )
            except Exception as e:
                logger.warning("whatsapp: image download failed: %s", e)

        # Audio/voice
        if msg.HasField("audioMessage"):
            try:
                self._media_dir.mkdir(parents=True, exist_ok=True)
                ext = "ogg" if msg.audioMessage.ptt else "m4a"
                path = self._media_dir / f"wa_audio_{msg_id}.{ext}"
                await client.download_any(msg, path=str(path))
                media_local_paths.append(str(path))
                media_url = await resolve_media_url(str(path))
                content_parts.append(
                    AudioContent(type=ContentType.AUDIO, data=media_url),
                )
            except Exception as e:
                logger.warning("whatsapp: audio download failed: %s", e)

        # Document
        if msg.HasField("documentMessage"):
            try:
                self._media_dir.mkdir(parents=True, exist_ok=True)
                raw_fname = msg.documentMessage.fileName or f"wa_doc_{msg_id}"
                # Sanitize filename to prevent path traversal
                fname = Path(raw_fname).name or f"wa_doc_{msg_id}"
                path = self._media_dir / fname
                await client.download_any(msg, path=str(path))
                media_local_paths.append(str(path))
                media_url = await resolve_media_url(str(path))
                content_parts.append(
                    FileContent(type=ContentType.FILE, file_url=media_url),
                )
            except Exception as e:
                logger.warning("whatsapp: document download failed: %s", e)

        # Video (mp4 by default; mimeType can override but neonize's download_any
        # already chooses the right container).
        if msg.HasField("videoMessage"):
            vid_msg = msg.videoMessage
            caption = getattr(vid_msg, "caption", "") or ""
            if caption and not body:
                body = caption
                content_parts.append(
                    TextContent(type=ContentType.TEXT, text=caption),
                )
            try:
                self._media_dir.mkdir(parents=True, exist_ok=True)
                path = self._media_dir / f"wa_vid_{msg_id}.mp4"
                await client.download_any(msg, path=str(path))
                media_local_paths.append(str(path))
                media_url = await resolve_media_url(str(path))
                content_parts.append(
                    VideoContent(type=ContentType.VIDEO, video_url=media_url),
                )
            except Exception as e:
                logger.warning("whatsapp: video download failed: %s", e)

        # Sticker (usually webp — surface as ImageContent so vision models can read).
        if msg.HasField("stickerMessage"):
            try:
                self._media_dir.mkdir(parents=True, exist_ok=True)
                path = self._media_dir / f"wa_sticker_{msg_id}.webp"
                await client.download_any(msg, path=str(path))
                media_local_paths.append(str(path))
                media_url = await resolve_media_url(str(path))
                content_parts.append(
                    ImageContent(type=ContentType.IMAGE, image_url=media_url),
                )
            except Exception as e:
                logger.warning("whatsapp: sticker download failed: %s", e)

        return body, content_parts

    async def _extract_quote_content(
        self,
        client,
        msg,
        chat_str: str = "",
    ) -> List[Any]:
        """Extract the content of a quoted/replied-to WhatsApp message.

        Returns content parts (text + media) so the agent has full context
        of what the user is responding to — not just a stanza ID.

        Note: quotedMessage is a stripped-down proto — media download keys
        are usually absent, so we extract text/captions and describe media
        types rather than attempting (and failing) to download.

        ``chat_str`` lets the album-quote branch reverse-lookup
        local paths from ``self._inbound_media`` (the album header
        carries no media keys, only an ``expectedImageCount``, so
        without the cache we'd only know the count, not the files).
        """
        # contextInfo lives on extendedTextMessage, imageMessage, etc.
        # ``albumMessage`` is included so that replies whose quoted
        # target is itself a multi-image / multi-video album still
        # surface a reply block (the album's contextInfo points at
        # the quoted message; the album body itself is just an
        # announcement of how many media items will arrive next).
        ctx = None
        for field in (
            "extendedTextMessage",
            "imageMessage",
            "videoMessage",
            "audioMessage",
            "documentMessage",
            "stickerMessage",
            "albumMessage",
        ):
            if msg.HasField(field):
                sub = getattr(msg, field)
                if hasattr(sub, "contextInfo"):
                    ctx = sub.contextInfo
                    break
        if not ctx:
            return []
        if not (
            ctx.HasField("quotedMessage")
            if hasattr(ctx, "HasField")
            else False
        ):
            return []

        quoted_msg = ctx.quotedMessage
        participant = getattr(ctx, "participant", "") or ""
        if isinstance(participant, str):
            sender_label = (
                participant.split("@")[0]
                if "@" in participant
                else participant
            )
        elif hasattr(participant, "User"):
            sender_label = participant.User
        else:
            sender_label = "unknown"

        # Resolve LID to phone/name for display
        # Only treat as LID if the participant JID server is "lid", not just because it is numeric
        is_lid = False
        if isinstance(participant, str) and "@lid" in participant:
            is_lid = True
        elif (
            hasattr(participant, "Server")
            and getattr(participant, "Server", "") == "lid"
        ):
            is_lid = True
        lid_key = f"{sender_label}@lid" if is_lid and sender_label else ""
        if lid_key:
            if lid_key not in self._lid_cache:
                try:
                    from neonize.utils import build_jid

                    lid_jid = build_jid(sender_label, "lid")
                    await self._resolve_lid(client, lid_key, lid_jid)
                except Exception:
                    pass
            cached = self._lid_cache.get(lid_key, {})
            phone = cached.get("phone", "")
            name = cached.get("name", "")
            if phone and name:
                sender_label = f"+{phone} ({name})"
            elif phone:
                sender_label = f"+{phone}"

        # Extract what we can from the quoted message proto
        quote_body = ""
        # Each entry: ``"image: /path/to/file.jpg"`` when the download
        # succeeded, ``"image"`` alone when we couldn't pull the bytes.
        # Agents use the path to feed the quoted media into tools
        # (codex image i2i, view_video, transcribe) — without it they
        # can only describe the reference, not act on it.
        media_types: list[str] = []
        # ImageContent blocks to return alongside the text block so
        # vision-capable models also see the image inline.
        extra_parts: list[Any] = []

        # Text
        if getattr(quoted_msg, "conversation", ""):
            quote_body = quoted_msg.conversation
        elif (
            quoted_msg.HasField("extendedTextMessage")
            and quoted_msg.extendedTextMessage.text
        ):
            quote_body = quoted_msg.extendedTextMessage.text

        stanza_id = getattr(ctx, "stanzaId", "") or "quote"
        stanza_key = stanza_id[:12] if stanza_id else "quote"

        async def _try_download(ext: str) -> str | None:
            """Download the currently-iterating quoted media to
            ``wa_quote_<stanza>.<ext>`` under the channel media dir.
            Returns the absolute path on success; ``None`` if the
            media key is missing (the common case for forwarded
            quotes) or the download errored for any reason.
            """
            try:
                self._media_dir.mkdir(parents=True, exist_ok=True)
                path = self._media_dir / f"wa_quote_{stanza_key}.{ext}"
                await client.download_any(quoted_msg, path=str(path))
                if path.exists() and path.stat().st_size > 0:
                    return str(path)
            except Exception:
                pass
            return None

        # Detect media types present in quoted message.  For each
        # downloadable type we attempt ``client.download_any`` and,
        # on success, emit the path inline in the text block so the
        # agent can reference it without guessing.
        if quoted_msg.HasField("imageMessage"):
            caption = getattr(quoted_msg.imageMessage, "caption", "") or ""
            if caption and not quote_body:
                quote_body = caption
            img_path = await _try_download("jpg")
            if img_path:
                media_types.append(f"image: {img_path}")
                extra_parts.append(
                    ImageContent(
                        type=ContentType.IMAGE,
                        image_url=await resolve_media_url(img_path),
                    ),
                )
            else:
                media_types.append("image")

        if quoted_msg.HasField("videoMessage"):
            vid_path = await _try_download("mp4")
            media_types.append(f"video: {vid_path}" if vid_path else "video")

        if quoted_msg.HasField("audioMessage"):
            ptt = getattr(quoted_msg.audioMessage, "ptt", False)
            label = "voice note" if ptt else "audio"
            aud_path = await _try_download("ogg")
            media_types.append(f"{label}: {aud_path}" if aud_path else label)

        if quoted_msg.HasField("documentMessage"):
            fname = getattr(quoted_msg.documentMessage, "fileName", "") or ""
            base_label = f"file: {fname}" if fname else "document"
            # Preserve the original extension when we can — some
            # downstream tools sniff file type from the path.
            ext = fname.rsplit(".", 1)[-1] if "." in fname else "bin"
            doc_path = await _try_download(ext)
            media_types.append(
                f"{base_label} ({doc_path})" if doc_path else base_label,
            )

        if quoted_msg.HasField("stickerMessage"):
            st_path = await _try_download("webp")
            media_types.append(f"sticker: {st_path}" if st_path else "sticker")

        # AlbumMessage: WhatsApp's multi-image / multi-video container.
        # The album body itself doesn't carry the actual media keys —
        # those live on the children — but we cached the children's
        # local paths against the header's stanza ID when they came
        # in.  Reverse-lookup that cache so the agent sees real
        # ``image: /tmp/...`` paths it can feed to tools, falling
        # back to the count placeholder when the cache misses
        # (album was sent before bot was running, restart cleared
        # the index, etc.).
        if quoted_msg.HasField("albumMessage"):
            cached_paths = self._lookup_inbound_media(
                chat_str, stanza_id,
            )
            for p in cached_paths:
                ext = Path(p).suffix.lower()
                if ext in {".jpg", ".jpeg", ".png", ".gif",
                           ".webp", ".bmp"}:
                    media_types.append(f"image: {p}")
                    extra_parts.append(ImageContent(
                        type=ContentType.IMAGE,
                        image_url=await resolve_media_url(p),
                    ))
                elif ext in {".mp4", ".mov", ".avi", ".webm",
                             ".mkv", ".mpeg"}:
                    media_types.append(f"video: {p}")
                else:
                    media_types.append(f"file: {p}")
            if not cached_paths:
                # Cache miss — fall back to the count placeholder.
                ai = getattr(
                    quoted_msg.albumMessage, "expectedImageCount", 0,
                )
                av = getattr(
                    quoted_msg.albumMessage, "expectedVideoCount", 0,
                )
                counts = []
                if ai:
                    counts.append(f"{ai} image{'s' if ai != 1 else ''}")
                if av:
                    counts.append(f"{av} video{'s' if av != 1 else ''}")
                media_types.append(
                    f"album with {' + '.join(counts)}"
                    if counts else "album",
                )

        if not quote_body and not media_types:
            return []

        block = TextContent(
            type=ContentType.TEXT,
            text=self._format_reply_context(
                sender=sender_label,
                body=quote_body,
                media_types=media_types,
            ),
        )
        return [block, *extra_parts]

    def _record_inbound_media(
        self,
        chat_str: str,
        msg_id: str,
        paths: list[str],
    ) -> None:
        """Cache the local media paths for an inbound message so a
        later reply (where ``contextInfo.stanzaID`` points back at
        this msg_id) can resolve actual files instead of an opaque
        ``"album with N images"`` placeholder.  FIFO eviction at
        ``_inbound_media_limit`` keeps the dict bounded; we store
        only on-disk paths to skip stale entries automatically.
        """
        live = [p for p in paths if p and os.path.isfile(p)]
        if not live:
            return
        key = (chat_str, msg_id)
        if key in self._inbound_media:
            # Refresh — extend rather than overwrite in case the
            # album header recorded a partial list and a later flush
            # adds more children.
            existing = self._inbound_media[key]
            for p in live:
                if p not in existing:
                    existing.append(p)
            return
        self._inbound_media[key] = list(live)
        self._inbound_media_order.append(key)
        while len(self._inbound_media_order) > self._inbound_media_limit:
            old = self._inbound_media_order.pop(0)
            self._inbound_media.pop(old, None)

    def _lookup_inbound_media(
        self,
        chat_str: str,
        msg_id: str,
    ) -> list[str]:
        """Reverse-lookup of media for a quoted message.  Returns
        only paths that still exist on disk.  Empty list when we
        have no record (cache miss) or the files were cleaned up."""
        if not msg_id:
            return []
        paths = self._inbound_media.get((chat_str, msg_id), [])
        return [p for p in paths if p and os.path.isfile(p)]

    @staticmethod
    def _format_reply_context(
        sender: str,
        body: str,
        media_types: List[str],
    ) -> str:
        """Build the OpenClaw-style bounded reply-to context block."""
        lines = [
            "=== UNTRUSTED reply-to (this message quotes an earlier one) ===",
        ]
        lines.append(f"From: {sender}")
        if body:
            lines.append(f"Message: {body[:400]}")
        if media_types:
            lines.append(f"Media: {', '.join(media_types)}")
        lines.append("=== end of reply-to ===")
        return "\n".join(lines)

    def _check_access(
        self,
        is_group,
        chat_str,
        sender_str,
        sender_jid,
        client,
        msg,
        body,
    ) -> bool:
        """Check access control for incoming message.

        Returns True if message is allowed, False if blocked.
        Note: DM allowlist checks that need async LID resolution are
        handled separately in _on_message.
        Note: Mention checks are handled in _on_message (after content
        extraction) so non-mentioned messages can be recorded in group
        history for context injection.
        """
        if is_group:
            if self.group_policy == "allowlist":
                if not self._groups or chat_str not in self._groups:
                    logger.debug(
                        "whatsapp: blocked group %s (allowlist=%s)",
                        chat_str[:20],
                        self._groups,
                    )
                    return False
            # Enforce group_allow_from: if set and not ["*"], check sender
            if self._group_allow_from and "*" not in self._group_allow_from:
                sender_user = (
                    sender_str.split("@")[0]
                    if "@" in sender_str
                    else sender_str
                )
                if (
                    sender_str not in self._group_allow_from
                    and sender_user not in self._group_allow_from
                    and f"+{sender_user}" not in self._group_allow_from
                ):
                    logger.debug(
                        "whatsapp: blocked sender %s in group (group_allow_from=%s)",
                        sender_str[:20],
                        self._group_allow_from,
                    )
                    return False
        return True

    async def _on_message(self, client, message) -> None:
        try:
            # Resolve bot's own LID on first message
            if self._my_jid and not self._lid_cache.get(
                _jid_to_str(self._my_jid),
            ):
                await self._resolve_lid(
                    client,
                    _jid_to_str(self._my_jid),
                    self._my_jid,
                )

            info = message.Info
            source = info.MessageSource
            sender_jid = source.Sender
            chat_jid = source.Chat
            is_group = source.IsGroup
            is_from_me = source.IsFromMe
            msg_id = info.ID
            timestamp = info.Timestamp

            logger.info(
                "whatsapp: [RAW] sender=%s chat=%s is_group=%s is_from_me=%s",
                _jid_to_str(sender_jid),
                _jid_to_str(chat_jid),
                is_group,
                is_from_me,
            )

            # Skip own messages unless self_chat_mode
            if is_from_me and not self._self_chat_mode:
                logger.debug(
                    "whatsapp: skipping own message (self_chat_mode=%s)",
                    self._self_chat_mode,
                )
                return

            sender_str = _jid_to_str(sender_jid)
            chat_str = _jid_to_str(chat_jid)

            # Extract message content via helper
            msg = message.Message
            body, content_parts = await self._extract_message_content(
                client,
                msg,
                msg_id,
            )
            # Snapshot the raw local-media paths _extract_message_content
            # populated via its scratch buffer, so we can pass them to the
            # group-history store below without being vulnerable to a
            # later message overwriting the buffer.
            # Index by stanza ID so a later reply can reverse-lookup
            # paths even though WhatsApp's reply contextInfo only
            # carries the parent message's ID, not its media keys.
            if self._last_extracted_media_paths and msg_id:
                self._record_inbound_media(
                    chat_str, msg_id, self._last_extracted_media_paths,
                )
            media_local_paths = list(self._last_extracted_media_paths)

            # Extract quoted/replied-to message content
            quote_parts = await self._extract_quote_content(
                client, msg, chat_str=chat_str,
            )
            if quote_parts:
                content_parts = quote_parts + content_parts

            # Album collation hook ───────────────────────────────
            # WhatsApp's multi-image / multi-video send arrives as
            # (1) an ``albumMessage`` header with empty media body
            # and (2) N independent ``imageMessage`` /
            # ``videoMessage`` payloads from the same sender within
            # ~1 s.  Collate them into a single user turn so the
            # agent sees the album as one message rather than N
            # fragmented turns + a silently-dropped header (the
            # header has no media → ``content_parts`` empty → the
            # check below would ``return`` on it).  See
            # ``_handle_album_inbound`` for the buffer mechanics.
            if await self._handle_album_inbound(
                client=client,
                message=message,
                msg=msg,
                msg_id=msg_id,
                sender_jid=sender_jid,
                chat_jid=chat_jid,
                is_group=is_group,
                timestamp=timestamp,
                sender_str=sender_str,
                chat_str=chat_str,
                body=body,
                content_parts=content_parts,
                quote_parts=quote_parts,
                media_local_paths=media_local_paths,
            ):
                # Buffered (header or pending child) — dispatch
                # will fire from ``_flush_album`` when complete.
                return

            if not content_parts:
                return

            await self._dispatch_inbound_message(
                client=client,
                message=message,
                msg=msg,
                msg_id=msg_id,
                sender_jid=sender_jid,
                chat_jid=chat_jid,
                is_group=is_group,
                timestamp=timestamp,
                sender_str=sender_str,
                chat_str=chat_str,
                body=body,
                content_parts=content_parts,
                media_local_paths=media_local_paths,
                info=info,
            )

        except Exception:
            logger.exception("whatsapp: error processing message")

    async def _dispatch_inbound_message(
        self,
        *,
        client,
        message,
        msg,
        msg_id: str,
        sender_jid,
        chat_jid,
        is_group: bool,
        timestamp: Any,
        sender_str: str,
        chat_str: str,
        body: str,
        content_parts: List[Any],
        media_local_paths: List[str],
        info,
    ) -> None:
        """Dispatch an inbound message that's already been
        extracted, quote-merged, and (for albums) collated.

        Pulled out of ``_on_message`` so the album-flush path can
        reuse it after gathering N children — both code paths
        share the access-control, mention-gate, group-history,
        envelope, channel_meta, and queue-enqueue logic below.
        """
        try:
            # Access control (sync checks: group allowlist)
            if not self._check_access(
                is_group,
                chat_str,
                sender_str,
                sender_jid,
                client,
                msg,
                body,
            ):
                return

            # Group mention gate — record non-mentioned messages for context
            # Slash commands (/new, /stop, /clear, etc.) bypass mention gate
            is_slash_command = bool(body and body.lstrip().startswith("/"))
            # Compute actual mention status up-front so channel_meta reflects
            # reality rather than "we reached here, so assume mentioned".
            # DMs: implicitly addressed to the bot; mark as mentioned.
            # Groups: run the real check regardless of require_mention so
            # downstream metadata consumers see truth.
            bot_mentioned_actual = (
                self._is_bot_mentioned(msg, body) if is_group else True
            )
            if is_group and self.require_mention and not is_slash_command:
                if not bot_mentioned_actual:
                    # Buffer for later context injection when bot IS mentioned
                    if body or content_parts:
                        # Resolve LID to phone/name for readable history
                        if sender_str.endswith("@lid"):
                            await self._resolve_lid(
                                client,
                                sender_str,
                                sender_jid,
                            )
                        display = self._format_sender(sender_str)
                        # Store the raw local paths (tracked by
                        # _extract_message_content) rather than peeking at
                        # content_parts' URL attributes — those may be
                        # resolved HTTPS URLs once resolve_media_url starts
                        # doing signed-URL substitution, and os.path.isfile
                        # would silently drop them otherwise. Keep only the
                        # paths that still exist on disk right now.
                        media_paths = [
                            p
                            for p in media_local_paths
                            if p and os.path.isfile(p)
                        ]
                        history = self._group_history.setdefault(chat_str, [])
                        history.append(
                            {
                                "sender": display,
                                "body": body or "[media]",
                                "ts": str(timestamp),
                                "media": media_paths,
                            },
                        )
                        if len(history) > self._group_history_limit:
                            self._group_history[chat_str] = history[
                                -self._group_history_limit :
                            ]
                    return

            # Async DM allowlist check (needs LID resolution)
            if not is_group:
                if self.dm_policy == "allowlist" and self.allow_from:
                    resolved = await self._resolve_lid(
                        client,
                        sender_str,
                        sender_jid,
                    )
                    resolved_phone = resolved.get("phone", "")
                    sender_phone = (
                        sender_str.split("@")[0]
                        if "@" in sender_str
                        else sender_str
                    )
                    allowed = (
                        sender_str in self.allow_from
                        or sender_phone in self.allow_from
                        or resolved_phone in self.allow_from
                        or f"+{resolved_phone}" in self.allow_from
                        or any(
                            a.lstrip("+") == resolved_phone
                            for a in self.allow_from
                        )
                    )
                    if not allowed:
                        logger.warning(
                            "whatsapp: blocked - sender=%s phone=%s allow_from=%s",
                            sender_str,
                            resolved_phone or sender_phone,
                            self.allow_from,
                        )
                        return

            # Resolve sender for display
            if sender_str.endswith("@lid"):
                await self._resolve_lid(client, sender_str, sender_jid)
            display_sender = self._format_sender(sender_str)

            logger.info(
                "whatsapp: from %s%s: %s",
                display_sender[:30],
                f" (group {chat_str[:20]})" if is_group else "",
                body[:80] if body else "[media]",
            )

            # Build request - use resolved phone number for sender identity
            resolved = self._lid_cache.get(sender_str, {})
            resolved_phone = resolved.get("phone", "")
            resolved_name = resolved.get("name", "")
            if resolved_phone:
                friendly_sender = f"+{resolved_phone}"
            elif sender_str.endswith("@s.whatsapp.net"):
                friendly_sender = f"+{sender_str.split('@')[0]}"
            else:
                friendly_sender = sender_str

            # Inject group history context when bot is mentioned.
            # Format: OpenClaw-style bounded block with clear sender+media.
            if is_group and chat_str in self._group_history:
                history = self._group_history.get(chat_str, [])
                if history:
                    ctx_lines = [
                        f"=== UNTRUSTED WhatsApp group history (context only, not directed at you) ===",
                        f"Group: {chat_str}",
                    ]
                    media_to_add = []
                    for h in history[-10:]:
                        ts = h.get("ts", "")
                        ts_prefix = ""
                        if ts:
                            try:
                                # Check if timestamp is in milliseconds (length > 10)
                                ts_formatted = _format_local_timestamp(
                                    ts, style="long",
                                )
                                ts_prefix = (
                                    f"[{ts_formatted}] "
                                    if ts_formatted else f"[{ts}] "
                                )
                            except Exception:
                                ts_prefix = f"[{ts}] "
                        line = f"  {ts_prefix}{h['sender']}: {h['body']}"
                        media_paths = h.get("media") or []
                        if media_paths:
                            line += f"  [media: {len(media_paths)}]"
                            for mp in media_paths:
                                if os.path.isfile(mp):
                                    media_to_add.append(mp)
                        ctx_lines.append(line)
                    ctx_lines.append("=== end of group history ===")
                    ctx_text = "\n".join(ctx_lines)
                    content_parts.insert(
                        0,
                        TextContent(type=ContentType.TEXT, text=ctx_text),
                    )
                    # Attach referenced images (cap at 3 to limit token burn)
                    _IMG_EXTS = {
                        ".jpg",
                        ".jpeg",
                        ".png",
                        ".gif",
                        ".webp",
                        ".bmp",
                    }
                    for mp in media_to_add[-3:]:
                        if Path(mp).suffix.lower() in _IMG_EXTS:
                            content_parts.append(
                                ImageContent(
                                    type=ContentType.IMAGE,
                                    image_url=mp,
                                ),
                            )
                    self._group_history[chat_str] = []

            # Strip bot @mention from body text so commands like "/new" work
            # even when prefixed with @+phone. This happens BEFORE the
            # envelope prefix wrap so command detection sees clean text.
            body = self._strip_bot_mention(body)
            for i, part in enumerate(content_parts):
                if hasattr(part, "type") and part.type == ContentType.TEXT:
                    txt = part.text or ""
                    if txt.startswith("===") or txt.startswith("[Replying"):
                        continue
                    stripped = self._strip_bot_mention(txt)
                    if stripped != txt:
                        content_parts[i] = TextContent(
                            type=ContentType.TEXT,
                            text=stripped,
                        )
                    break

            # Detect slash commands (/new, /stop, /clear, etc.)
            has_bot_command = bool(body and body.lstrip().startswith("/"))

            # Envelope: clear chat-type + sender prefix so the agent never
            # mistakes a group for a DM.  Includes the WhatsApp send
            # timestamp in local-system tz so the model can reason
            # about "when was this sent" without guessing —
            # particularly useful when the user replies hours after
            # an earlier turn.
            # Group:  [2026-04-25 19:40 JST] [WhatsApp group {chat_jid}] Joe HO (+85251159218): text
            # DM:     [2026-04-25 19:40 JST] [WhatsApp DM] +85251159218: text
            sender_label = friendly_sender
            if resolved_name:
                sender_label = f"{resolved_name} ({friendly_sender})"
            ts_short = _format_local_timestamp(timestamp, style="short")
            ts_prefix = f"[{ts_short}] " if ts_short else ""
            if is_group:
                envelope = (
                    f"{ts_prefix}[WhatsApp group {chat_str}] {sender_label}"
                )
            else:
                envelope = f"{ts_prefix}[WhatsApp DM] {sender_label}"
            for i, part in enumerate(content_parts):
                if hasattr(part, "type") and part.type == ContentType.TEXT:
                    txt = part.text or ""
                    if txt.startswith("===") or txt.startswith("[Replying"):
                        continue
                    content_parts[i] = TextContent(
                        type=ContentType.TEXT,
                        text=f"{envelope}: {txt}",
                    )
                    break

            effective_sender = (
                f"group:{chat_str}" if is_group else friendly_sender
            )
            # Also resolve chat LID to phone for send target
            send_chat_jid = chat_str
            if chat_str.endswith("@lid"):
                chat_resolved = self._lid_cache.get(chat_str, {})
                chat_phone = chat_resolved.get("phone", "")
                if chat_phone:
                    send_chat_jid = f"{chat_phone}@s.whatsapp.net"

            # Resolve typing JID for typing loop during response
            typing_jid = chat_jid
            if chat_str.endswith("@lid"):
                c_info = self._lid_cache.get(chat_str, {})
                c_phone = c_info.get("phone", "")
                if c_phone:
                    typing_jid = _str_to_jid(c_phone)

            channel_meta = {
                "platform": "whatsapp",
                "chat_jid": send_chat_jid,
                "sender_jid": sender_str,
                "sender_phone": friendly_sender,
                "sender_name": resolved_name,
                "is_group": is_group,
                "msg_id": msg_id,
                "timestamp": timestamp,
                "bot_phone": f"+{self._bot_phone}" if self._bot_phone else "",
                "bot_lid": self._bot_lid,
                "has_bot_command": has_bot_command,
                # bot_mentioned reflects whether the bot was actually
                # @-mentioned — NOT whether we "passed the mention gate".
                # DMs are True (implicitly addressed to bot); groups
                # reflect the real _is_bot_mentioned() result.
                "bot_mentioned": bot_mentioned_actual,
            }
            session_id = self.resolve_session_id(
                effective_sender,
                channel_meta,
            )
            request = self.build_agent_request_from_user_content(
                channel_id=self.channel,
                sender_id=effective_sender,
                session_id=session_id,
                content_parts=content_parts,
                channel_meta=channel_meta,
            )
            request.channel_meta = channel_meta
            # Store typing info on request (NOT in channel_meta — JID/client are not JSON-serializable)
            request._wa_typing_jid = typing_jid
            request._wa_typing_client = client
            # Store ack-reaction target so _stream_with_tracker can clear it
            request._wa_ack_chat_jid = chat_jid
            request._wa_ack_sender_jid = sender_jid
            request._wa_ack_msg_id = msg_id
            # Store raw neonize message for reply-to quoting
            request._wa_raw_message = message

            # Mark as read
            if self._send_read_receipts:
                try:
                    await client.mark_read(
                        [info.ID],
                        chat_jid,
                        sender_jid,
                    )
                except Exception:
                    pass

            # For slash commands (/new, /stop, /clear, etc.), strip the
            # envelope prefix ([WhatsApp group xxx] Sender: ...) so the
            # command registry sees the raw command text.
            # Also remove group history context — it prepends text that
            # breaks command detection (query must start with /command).
            if has_bot_command:
                content_parts = [
                    p
                    for p in content_parts
                    if not (
                        hasattr(p, "text")
                        and isinstance(p.text, str)
                        and p.text.startswith("=== UNTRUSTED")
                    )
                ]
                for i, part in enumerate(content_parts):
                    if hasattr(part, "text") and part.text.startswith(
                        "[WhatsApp ",
                    ):
                        # Format is: [WhatsApp group xxx] Name (+phone): text
                        # Strip up to the first ": " after the closing bracket.
                        bracket_end = part.text.find("] ")
                        if bracket_end > 0:
                            after_bracket = part.text[bracket_end + 2 :]
                            idx = after_bracket.find(": ")
                            if idx > 0:
                                raw_text = after_bracket[idx + 2 :]
                                content_parts[i] = TextContent(
                                    type=ContentType.TEXT,
                                    text=raw_text,
                                )
                        break
                request = self.build_agent_request_from_user_content(
                    channel_id=self.channel,
                    sender_id=effective_sender,
                    session_id=session_id,
                    content_parts=content_parts,
                    channel_meta=channel_meta,
                )
                request.channel_meta = channel_meta
                request._wa_typing_jid = typing_jid
                request._wa_typing_client = client
                request._wa_ack_chat_jid = chat_jid
                request._wa_ack_sender_jid = sender_jid
                request._wa_ack_msg_id = msg_id
                request._wa_raw_message = message

            # Fire "thinking" reaction (fire-and-forget) so the user
            # knows the bot has picked up their message before the
            # agent's reply lands.
            if self._ack_reaction_thinking:
                asyncio.create_task(
                    self._send_reaction(
                        client,
                        chat_jid,
                        sender_jid,
                        msg_id,
                        self._ack_reaction_thinking,
                    ),
                )

            # Route through UnifiedQueueManager (via self._enqueue) so
            # each (whatsapp, session_id, priority) gets its own queue
            # and worker task. Messages from different chats process
            # in parallel; same-chat messages still serialize
            # (prevents races inside one conversation).
            #
            # NOTE: direct `await self.consume_one(request)` would
            # block neonize's message callback until the agent's full
            # response finishes, which serializes ALL inbound traffic
            # (DM + group messages stack up). Fall back to direct call
            # if no enqueue callback is attached (unit tests set up
            # channels without the manager).
            if self._enqueue is not None:
                self._enqueue(request)
            else:
                await self.consume_one(request)

        except Exception:
            logger.exception("whatsapp: error processing message")

    async def _handle_album_inbound(
        self,
        *,
        client,
        message,
        msg,
        msg_id: str,
        sender_jid,
        chat_jid,
        is_group: bool,
        timestamp: Any,
        sender_str: str,
        chat_str: str,
        body: str,
        content_parts: List[Any],
        quote_parts: List[Any],
        media_local_paths: List[str],
    ) -> bool:
        """Album-collation gate.  Returns ``True`` when the inbound
        was buffered (album header or pending child) and the caller
        should NOT continue with normal dispatch — the flush will
        fire from ``_flush_album`` once all children arrive (or the
        timeout fires).  Returns ``False`` for ordinary messages
        that should fall through to normal dispatch.
        """
        key = (chat_str, sender_str)

        # 1. Album header — start a buffer and arm the timeout.
        if msg.HasField("albumMessage"):
            album = msg.albumMessage
            expected = int(getattr(album, "expectedImageCount", 0)) + int(
                getattr(album, "expectedVideoCount", 0),
            )
            # Empty album (no expected media) — nothing to wait
            # for; let it fall through and get dropped by the
            # ``if not content_parts`` guard.
            if expected <= 0:
                return False

            # Cancel any prior buffer for the same key (a fresh
            # album header before the previous one finished is a
            # signal that the previous album was lost or aborted).
            existing = self._album_buffers.pop(key, None)
            if (
                existing
                and existing.timeout_task
                and not existing.timeout_task.done()
            ):
                existing.timeout_task.cancel()

            buf = _WhatsAppAlbumBuffer(
                header_msg_id=msg_id,
                header_msg=message,
                header_timestamp=timestamp,
                quote_parts=list(quote_parts or []),
                expected=expected,
            )
            self._album_buffers[key] = buf

            async def _on_timeout(
                b: _WhatsAppAlbumBuffer = buf,
                k=key,
            ) -> None:
                try:
                    await asyncio.sleep(self._album_timeout_s)
                    if not b.flushed and self._album_buffers.get(k) is b:
                        logger.info(
                            "whatsapp: album timeout (key=%s, gathered=%d/%d) "
                            "— flushing partial",
                            k,
                            len(b.gathered_parts),
                            b.expected,
                        )
                        await self._flush_album(
                            buffer=b,
                            client=client,
                            chat_jid=chat_jid,
                            sender_jid=sender_jid,
                            is_group=is_group,
                            sender_str=sender_str,
                            chat_str=chat_str,
                        )
                except asyncio.CancelledError:
                    pass

            buf.timeout_task = asyncio.create_task(
                _on_timeout(),
                name=f"wa-album-flush-{chat_str}",
            )
            logger.info(
                "whatsapp: album header buffered (key=%s, expected=%d, "
                "has_quote=%s)",
                key,
                expected,
                bool(quote_parts),
            )
            return True

        # 2. Possible child of an in-flight album.
        buf = self._album_buffers.get(key)
        if buf is None or buf.flushed:
            return False

        # Only image/video parts count as album children.  Pure-
        # text or audio messages from the same sender mid-album
        # are user-initiated and shouldn't be silently swallowed
        # — let them dispatch normally.
        media_parts = [
            p
            for p in content_parts
            if isinstance(p, (ImageContent, VideoContent))
        ]
        if not media_parts:
            return False

        # Gather ALL parts the child carried (image + caption text),
        # not just the media — otherwise the caption TextContent
        # ``_extract_message_content`` appended for an
        # ``imageMessage.caption`` is dropped and the merged
        # dispatch ends up with content_parts == [images...].
        # Downstream ``_apply_no_text_debounce`` then sees no text
        # block and buffers the request indefinitely waiting for
        # one — observed 2026-04-25 as "agent never responds to
        # albums" even though the album header buffered cleanly.
        buf.gathered_parts.extend(content_parts)
        buf.gathered_paths.extend(media_local_paths)
        if body:
            buf.gathered_body_parts.append(body)

        if buf.is_complete():
            if buf.timeout_task and not buf.timeout_task.done():
                buf.timeout_task.cancel()
            await self._flush_album(
                buffer=buf,
                client=client,
                chat_jid=chat_jid,
                sender_jid=sender_jid,
                is_group=is_group,
                sender_str=sender_str,
                chat_str=chat_str,
            )
        return True

    async def _flush_album(
        self,
        *,
        buffer: "_WhatsAppAlbumBuffer",
        client,
        chat_jid,
        sender_jid,
        is_group: bool,
        sender_str: str,
        chat_str: str,
    ) -> None:
        """Emit one combined inbound for everything we've collected
        so far on ``buffer`` and clear the buffer.  Idempotent: a
        second call (e.g. timeout firing right after a count-driven
        flush) is a no-op."""
        if buffer.flushed:
            return
        buffer.flushed = True
        # Drop the buffer from the dict so a follow-up media
        # message starts a fresh, separate dispatch instead of
        # being swallowed as a "child" of the just-flushed album.
        key = (chat_str, sender_str)
        if self._album_buffers.get(key) is buffer:
            del self._album_buffers[key]

        merged_body = " ".join(
            b for b in buffer.gathered_body_parts if b
        ).strip()
        merged_content = list(buffer.quote_parts) + list(buffer.gathered_parts)

        # Index the album header → child paths so a later reply
        # whose ``contextInfo.stanzaID`` points back at this album
        # can resolve real local paths instead of an opaque count
        # placeholder.  See ``_record_inbound_media`` for the
        # eviction policy.
        if buffer.gathered_paths:
            self._record_inbound_media(
                chat_str, buffer.header_msg_id, buffer.gathered_paths,
            )

        # Header info objects for routing — info comes from the
        # original albumMessage proto; sender/chat already passed
        # in as args so we don't need to re-resolve.
        header_message = buffer.header_msg
        info = header_message.Info
        try:
            await self._dispatch_inbound_message(
                client=client,
                message=header_message,
                msg=header_message.Message,
                msg_id=buffer.header_msg_id,
                sender_jid=sender_jid,
                chat_jid=chat_jid,
                is_group=is_group,
                timestamp=buffer.header_timestamp,
                sender_str=sender_str,
                chat_str=chat_str,
                body=merged_body,
                content_parts=merged_content,
                media_local_paths=buffer.gathered_paths,
                info=info,
            )
        except Exception:
            logger.exception(
                "whatsapp: album flush dispatch failed (key=%s)",
                key,
            )

    def _is_bot_mentioned(self, msg, body: str) -> bool:
        """Check if bot is mentioned in message or message is a reply to bot."""
        if not self._my_jid:
            return False

        my_lid = self._bot_lid or getattr(self._my_jid, "User", "") or ""
        my_phone = self._bot_phone or ""
        if not my_phone:
            bot_jid_str = _jid_to_str(self._my_jid) if self._my_jid else ""
            bot_info = self._lid_cache.get(bot_jid_str, {})
            my_phone = bot_info.get("phone", "")

        # 1. Check body text for @LID or @phone, with a word-boundary guard
        #    so "@85211111" doesn't false-match inside "@852111112345".
        if my_lid and re.search(rf"@{re.escape(my_lid)}(?!\w)", body):
            return True
        if my_phone:
            # Match @+phone or @phone (no + prefix), both with the boundary.
            if re.search(rf"@\+?{re.escape(my_phone)}(?!\d)", body):
                return True

        # WhatsApp JID/LID user portion can carry a ``:<device>`` suffix
        # (e.g. ``229661330157571:2@lid`` for the 2nd linked device on
        # that LID).  ``_bot_lid`` / ``_bot_phone`` were already stripped
        # of the device suffix in the ConnectedEv handler, so raw
        # splits that only remove ``@<server>`` leave ``229661330157571:2``
        # and the equality check silently fails.  This tripped up
        # reply-to-bot detection in groups (reply to any bot message →
        # bot thinks it isn't mentioned → ignores the reply).
        def _normalize_user(s: str) -> str:
            if not s:
                return ""
            s = s.split("@", 1)[0]
            s = s.split(":", 1)[0]
            return s

        # 2. Check WhatsApp native mention (contextInfo.mentionedJID)
        ctx = None
        if msg.HasField("extendedTextMessage"):
            ctx = msg.extendedTextMessage.contextInfo
        if ctx:
            # Check mentionedJID - must be EXACT match, not substring
            if ctx.mentionedJID:
                for jid in ctx.mentionedJID:
                    if hasattr(jid, "User"):
                        jid_user = _normalize_user(str(jid.User))
                    elif isinstance(jid, str):
                        jid_user = _normalize_user(jid)
                    else:
                        continue
                    if jid_user and (
                        jid_user == my_lid or jid_user == my_phone
                    ):
                        return True

            # 3. Reply-to bot message counts as mention
            if ctx.HasField("quotedMessage") or getattr(ctx, "stanzaId", ""):
                quoted_participant = getattr(ctx, "participant", "") or ""
                if isinstance(quoted_participant, str):
                    qp_user = _normalize_user(quoted_participant)
                elif hasattr(quoted_participant, "User"):
                    qp_user = _normalize_user(str(quoted_participant.User))
                else:
                    qp_user = ""
                if qp_user and (qp_user == my_lid or qp_user == my_phone):
                    logger.debug("whatsapp: reply-to-bot detected")
                    return True

        logger.debug(
            "whatsapp: mention check failed - my_lid=%s my_phone=%s body=%s",
            my_lid,
            my_phone,
            body[:60],
        )
        return False

    async def _resolve_lid(
        self,
        client,
        lid_str: str,
        lid_jid=None,
    ) -> Dict[str, str]:
        """Resolve LID to phone number + name. Caches results."""
        if lid_str in self._lid_cache:
            return self._lid_cache[lid_str]

        result = {"phone": "", "name": "", "lid": lid_str}

        if not lid_str.endswith("@lid"):
            # Already a phone number
            result["phone"] = lid_str.split("@")[0]
            return result

        lid_user = lid_str.split("@")[0]

        # Try get_pn_from_lid
        if client and lid_jid:
            try:
                pn_jid = await client.get_pn_from_lid(lid_jid)
                if pn_jid and hasattr(pn_jid, "User") and pn_jid.User:
                    result["phone"] = pn_jid.User
                    logger.info(
                        "whatsapp: LID %s -> phone %s",
                        lid_user,
                        result["phone"],
                    )
            except Exception as e:
                logger.debug(
                    "whatsapp: get_pn_from_lid failed for %s: %s",
                    lid_user,
                    e,
                )

        # Try contact store for name
        if client:
            try:
                contact = client.contact
                if contact:
                    info = contact.get(lid_jid or lid_str)
                    if info and hasattr(info, "FullName"):
                        result["name"] = info.FullName or ""
            except Exception:
                pass

        self._lid_cache[lid_str] = result
        return result

    def _format_sender(self, lid_str: str) -> str:
        """Format sender for display: +85251159218 (Joe) or fallback to LID."""
        cached = self._lid_cache.get(lid_str, {})
        phone = cached.get("phone", "")
        name = cached.get("name", "")
        if phone and name:
            return f"+{phone} ({name})"
        if phone:
            return f"+{phone}"
        return lid_str

    def _strip_bot_mention(self, text: str) -> str:
        """Remove bot @mention (e.g. '@+817089933036') from text.

        Handles both @phone and @LID forms so that slash commands
        preceded by a mention ("@+817089933036 /new") are recognized.
        """
        if not text:
            return text
        import re as _re

        patterns = []
        if self._bot_phone:
            patterns.append(rf"^@\+?{_re.escape(self._bot_phone)}\s*")
        if self._bot_lid:
            patterns.append(rf"^@{_re.escape(self._bot_lid)}\s*")
        out = text
        for pat in patterns:
            out = _re.sub(pat, "", out).strip()
        return out

    # ── Outbound send ─────────────────────────────────────────────────

    async def _auto_reconnect(self):
        """Attempt to reconnect the WhatsApp neonize client after disconnect.

        Uses a lock (instantiated lazily in __init__ / here so asyncio.Lock
        has a running loop) to prevent concurrent reconnect attempts. Retries
        with exponential backoff indefinitely (capped at 5 minutes).
        """
        # Bail if a deliberate stop() is in progress — no reconnect race.
        if self._stopping:
            logger.debug("whatsapp: auto-reconnect skipped (stop in progress)")
            return

        if self._reconnect_lock is None:
            self._reconnect_lock = asyncio.Lock()

        if self._reconnect_lock.locked():
            logger.debug("whatsapp: reconnect already in progress, skipping")
            return

        async with self._reconnect_lock:
            backoff = 10
            attempt = 0
            while True:  # retry until connected or stop() bails out
                # Re-check _stopping inside the loop so a stop() that lands
                # after we entered the lock can short-circuit the retry.
                if self._stopping:
                    logger.info(
                        "whatsapp: stop() detected — aborting reconnect loop at attempt %d",
                        attempt,
                    )
                    return
                attempt += 1
                logger.info("whatsapp: reconnect attempt %d...", attempt)
                try:
                    # Stop old connection if still lingering
                    if self._connect_task and not self._connect_task.done():
                        self._connect_task.cancel()
                        try:
                            await self._connect_task
                        except (asyncio.CancelledError, Exception):
                            pass

                    # After 2 failed reuses, recreate the neonize client
                    # from scratch.  Reason: a server-forced logout (EOF
                    # that doesn't surface as ``DisconnectedEv``) wipes
                    # the ``whatsmeow_device`` row from the backing db.
                    # The in-memory ``self._client`` caches the old
                    # device identity at ``__init__`` time, so
                    # ``_client.connect()`` alone re-dials the socket
                    # but can't re-pair.  Rebuilding the client forces
                    # neonize to re-read the db — if a fresh QR scan
                    # landed in between, the new device row picks up
                    # and pairing completes cleanly.
                    if attempt > 2:
                        try:
                            db_path = str(self._auth_dir / "neonize.db")
                            self._client = NewAClient(name=db_path)
                            # Re-register every event handler on the
                            # fresh client by re-running start().  We
                            # can't just call start() (it checks
                            # ``self.enabled`` and would re-bind the
                            # task), so inline the minimum: the handler
                            # wiring happens inside the new connect
                            # call's event loop anyway.
                            logger.info(
                                "whatsapp: recreated neonize client "
                                "(attempt %d) — re-reading device from db",
                                attempt,
                            )
                            # Fall through to start() so handlers are
                            # registered against the new client.
                            await self._rewire_handlers()
                        except Exception as e:
                            logger.error(
                                "whatsapp: client recreation failed: %s",
                                e,
                            )

                    # Re-connect
                    self._connect_task = await self._client.connect()
                    # Wait for ConnectedEv to set self._connected. Break the
                    # wait into short sleeps so stop() still has a chance to
                    # interrupt us before we claim success.
                    for _ in range(5):
                        if self._stopping:
                            logger.info(
                                "whatsapp: stop() during reconnect wait — aborting",
                            )
                            return
                        await asyncio.sleep(1)
                    if self._connected:
                        logger.info(
                            "whatsapp: reconnected on attempt %d",
                            attempt,
                        )
                        return
                except Exception as e:
                    logger.error(
                        "whatsapp: reconnect attempt %d failed: %s",
                        attempt,
                        e,
                    )

                # Sleep-with-wake so stop() can still interrupt the backoff.
                for _ in range(backoff):
                    if self._stopping:
                        return
                    await asyncio.sleep(1)
                backoff = min(backoff * 2, 300)  # cap at 5 minutes

                # Log periodic status
                if attempt % 10 == 0:
                    logger.warning(
                        "whatsapp: still trying to reconnect (attempt %d, backoff=%ds)",
                        attempt,
                        backoff,
                    )

    async def send(
        self,
        to_handle: str,
        text: str,
        meta: Optional[dict] = None,
    ) -> None:
        if not self.enabled or not self._client or not self._connected:
            return
        if not text:
            return

        # Replace LID mentions with phone numbers + strip + prefix for neonize mention detection
        import re as _re

        for lid_str, info in self._lid_cache.items():
            phone = info.get("phone", "")
            if phone and lid_str.endswith("@lid"):
                lid_num = lid_str.split("@")[0]
                text = text.replace(f"@{lid_num}", f"@{phone}")
                text = text.replace(f"@+{lid_num}", f"@{phone}")
        # Also convert @+phone to @phone (neonize needs digits only after @)
        text = _re.sub(r"@\+(\d{5,16})", lambda m: "@" + m.group(1), text)

        meta = meta or {}
        chat_jid_str = meta.get("chat_jid") or to_handle
        jid = _str_to_jid(chat_jid_str)

        # Extract [Image: /path] patterns — restricted to media dir to prevent
        # LLM-driven file exfiltration (e.g. /etc/passwd)
        img_re = re.compile(r"\[Image: (file:///[^\]]+|/[^\]]+)\]")
        img_matches = img_re.findall(text)
        safe_dir = str(self._media_dir.resolve())
        for m in img_matches:
            p = m.replace("file://", "") if m.startswith("file://") else m
            resolved = str(Path(p).resolve())
            if Path(resolved).is_relative_to(safe_dir) and os.path.isfile(
                resolved,
            ):
                try:
                    await self._client.send_image(jid, resolved)
                    logger.info("whatsapp: sent image %s", resolved)
                except Exception as e:
                    logger.warning("whatsapp: image send failed: %s", e)
            elif os.path.isfile(p):
                logger.warning(
                    "whatsapp: blocked send of %s — outside media dir %s",
                    p,
                    safe_dir,
                )
            text = text.replace(f"[Image: {m}]", "").strip()

        if not text:
            return

        chunks = self._chunk_text(text)
        # Reply-to: quote the original inbound message on the first chunk
        chat_jid_key = meta.get("chat_jid") or to_handle
        quote_msg = (
            self._pending_quote_msgs.pop(chat_jid_key, None)
            if self._reply_to_trigger
            else None
        )
        for i, chunk in enumerate(chunks):
            try:
                logger.info(
                    "whatsapp: SENDING to %s: %s",
                    _jid_to_str(jid),
                    chunk[:100],
                )
                if i == 0 and quote_msg is not None and self._client:
                    # Build reply message quoting the inbound trigger
                    reply_built = await self._client.build_reply_message(
                        message=chunk,
                        quoted=quote_msg,
                    )
                    await self._client.send_message(jid, reply_built)
                else:
                    await self._client.send_message(jid, chunk)
            except Exception as e:
                logger.error("whatsapp: send failed: %s", e)

    async def send_media(
        self,
        to_handle: str,
        part: OutgoingContentPart,
        meta: Optional[dict] = None,
    ) -> None:
        """Send a single outbound media attachment (image / video / audio / file).

        This is the primary outbound media path for WhatsApp. It is
        invoked by the base channel's ``send_content_parts`` for every
        non-text content block in the agent's reply — so when the
        agent returns an ``ImageBlock`` / ``AudioBlock`` / etc.
        (including via the ``send_file_to_user`` tool), the file
        ends up here.

        Extracts the local path from the content part (``image_url``,
        ``video_url``, ``file_url``/``file_id`` or ``data``), strips the
        ``file://`` scheme if present, then dispatches through neonize
        (``send_image`` / ``send_video`` / ``send_audio`` /
        ``send_document``).

        ✅ Images, video, audio AND documents all flow through here.
        """
        t = getattr(part, "type", None)
        logger.info(
            "whatsapp: send_media called, type=%s to=%s",
            t,
            to_handle,
        )
        if not self.enabled or not self._client or not self._connected:
            logger.warning(
                "whatsapp: send_media skipped — enabled=%s client=%s connected=%s",
                self.enabled,
                bool(self._client),
                self._connected,
            )
            return
        meta = meta or {}
        chat_jid_str = meta.get("chat_jid") or to_handle
        jid = _str_to_jid(chat_jid_str)

        # Extract the local file path from the content part
        raw_path = None
        if t == ContentType.IMAGE:
            raw_path = getattr(part, "image_url", None)
        elif t == ContentType.VIDEO:
            raw_path = getattr(part, "video_url", None)
        elif t == ContentType.FILE:
            raw_path = getattr(part, "file_url", None) or getattr(
                part,
                "file_id",
                None,
            )
        elif t == ContentType.AUDIO:
            raw_path = getattr(part, "data", None)

        if not raw_path:
            logger.warning("whatsapp: send_media missing path for type=%s", t)
            return
        # Strip file:// scheme (LLM tools often emit file:///path form)
        file_path = (
            raw_path.replace("file://", "")
            if isinstance(raw_path, str) and raw_path.startswith("file://")
            else raw_path
        )
        exists = os.path.isfile(file_path) if file_path else False
        logger.info(
            "whatsapp: send_media file_path=%s exists=%s",
            file_path,
            exists,
        )
        if not exists:
            logger.warning("whatsapp: media file not found: %s", file_path)
            return
        try:
            if t == ContentType.IMAGE:
                # Filename convention: `.sticker.webp` → send as sticker
                # (explicit, controllable, no guessing). Applies only to
                # WhatsApp sticker format (.webp with .sticker. marker).
                if isinstance(file_path, str) and file_path.lower().endswith(
                    ".sticker.webp",
                ):
                    logger.info(
                        "whatsapp: send_media → sticker path=%s",
                        file_path,
                    )
                    await self._client.send_sticker(jid, file_path)
                else:
                    await self._client.send_image(jid, file_path)
            elif t == ContentType.VIDEO:
                await self._client.send_video(jid, file_path)
            elif t == ContentType.AUDIO:
                await self._client.send_audio(jid, file_path, ptt=True)
            else:  # FILE
                # Extract filename from path to fix the "Untitled" issue on WhatsApp
                filename = os.path.basename(file_path)
                await self._client.send_document(
                    jid,
                    file_path,
                    filename=filename,
                )
            logger.info(
                "whatsapp: sent media to %s (type=%s, size=%d bytes)",
                to_handle,
                t,
                os.path.getsize(file_path),
            )
        except Exception as e:
            logger.error(
                "whatsapp: send_media FAILED to=%s type=%s path=%s: %s",
                to_handle,
                t,
                file_path,
                e,
            )

    # ── Text chunking ─────────────────────────────────────────────────

    def _chunk_text(self, text: str) -> list[str]:
        if not text or len(text) <= self._text_chunk_limit:
            return [text] if text else []
        chunks: list[str] = []
        rest = text
        while rest:
            if len(rest) <= self._text_chunk_limit:
                chunks.append(rest)
                break
            chunk = rest[: self._text_chunk_limit]
            last_nl = chunk.rfind("\n")
            if last_nl > self._text_chunk_limit // 2:
                chunk = rest[:last_nl]
            chunks.append(chunk)
            rest = rest[len(chunk) :]
        return chunks

    # ── Typing indicator loop ──────────────────────────────────────────

    async def _typing_loop(self, client, typing_jid, interval: float = 4.0):
        """Re-send typing indicator every `interval` seconds until cancelled.

        WhatsApp typing indicators expire after ~5s, so we need to keep
        re-sending during the entire response generation.
        """
        try:
            while True:
                try:
                    _jb = typing_jid.SerializeToString()
                    await client._NewAClient__client.SendChatPresence(
                        client.uuid,
                        _jb,
                        len(_jb),
                        0,
                        0,
                    )
                except Exception:
                    pass
                await asyncio.sleep(interval)
        except asyncio.CancelledError:
            # Note: presence type 2 (paused) causes neonize Go panic
            # (index out of range [2] with length 2 in SendChatPresence).
            # WhatsApp auto-clears typing after ~5s, so we just let it expire.
            pass

    # ── Reactions ─────────────────────────────────────────────────────

    async def _send_reaction(
        self,
        client,
        chat_jid,
        sender_jid,
        msg_id: str,
        emoji: str,
    ) -> None:
        """Build + send a reaction to a specific message.

        neonize exposes ``build_reaction(chat, sender, message_id, emoji)``
        which returns a Message protobuf; we then push it through
        ``send_message(to=chat_jid, message=...)``. Pass emoji="" to
        remove an existing reaction.
        """
        try:
            reaction_msg = await client.build_reaction(
                chat_jid,
                sender_jid,
                msg_id,
                emoji or "",
            )
            await client.send_message(chat_jid, reaction_msg)
            logger.debug(
                "whatsapp: reaction %r sent on msg=%s",
                emoji,
                msg_id,
            )
        except Exception as e:
            logger.warning("whatsapp: reaction %r failed: %s", emoji, e)

    # ── Process loop override ─────────────────────────────────────────

    async def _stream_with_tracker(self, payload):
        """Override base to handle QwenPaw event format for WhatsApp."""
        import json as _json

        request = self._payload_to_request(payload)
        send_meta = getattr(request, "channel_meta", None) or {}
        to_handle = self.get_to_handle_from_request(request)
        await self._before_consume_process(request)

        text_parts = []
        message_completed = False
        process_iterator = None
        typing_task = None
        try:
            # Start persistent typing loop (re-sends every 4s until cancelled)
            # Typing info stored on request object (not channel_meta) to avoid
            # JSON serialization issues with JID/client objects.
            typing_jid = getattr(request, "_wa_typing_jid", None)
            typing_client = getattr(request, "_wa_typing_client", None)
            # Store raw message for reply-to quoting in send()
            raw_msg = getattr(request, "_wa_raw_message", None)
            if raw_msg and self._reply_to_trigger:
                chat_key = (getattr(request, "channel_meta", None) or {}).get(
                    "chat_jid",
                    "",
                )
                if chat_key:
                    self._pending_quote_msgs[chat_key] = raw_msg
            if typing_jid and typing_client:
                typing_task = asyncio.create_task(
                    self._typing_loop(typing_client, typing_jid),
                )

            _runner_health = getattr(self._process, "__self__", None)
            if _runner_health:
                _h = getattr(_runner_health, "_health", "unknown")
                logger.warning(
                    "whatsapp: _process runner id=%s health=%s",
                    id(_runner_health),
                    _h,
                )
            process_iterator = self._process(request)
            async for event in process_iterator:
                if hasattr(event, "model_dump_json"):
                    data = event.model_dump_json()
                elif hasattr(event, "json"):
                    data = event.json()
                else:
                    data = _json.dumps({"text": str(event)})
                yield f"data: {data}\n\n"

                obj = getattr(event, "object", None)
                status = getattr(event, "status", None)

                if obj == "message" and status == RunStatus.Completed:
                    await self.on_event_message_completed(
                        request,
                        to_handle,
                        event,
                        send_meta,
                    )
                    message_completed = True

                # Fallback text collection
                for part in getattr(event, "content", []) or []:
                    txt = getattr(part, "text", None)
                    if not txt or txt in text_parts:
                        continue
                    if self._filter_thinking:
                        from agentscope_runtime.engine.schemas.agent_schemas import (
                            MessageType,
                        )

                        if (
                            getattr(event, "type", None)
                            == MessageType.REASONING
                        ):
                            continue
                    text_parts.append(txt)

            if text_parts and not message_completed:
                reply = chr(10).join(text_parts)
                logger.info(
                    "whatsapp: sending reply (%d chars) to %s",
                    len(reply),
                    to_handle,
                )
                await self.send(to_handle, reply.strip(), send_meta)

            if self._on_reply_sent:
                args = self.get_on_reply_sent_args(request, to_handle)
                self._on_reply_sent(self.channel, *args)

            # Pick the right closing reaction:
            # - done  when the agent actually produced some reply
            # - error when nothing was sent (agent crashed silently or
            #   produced no content) — user would otherwise see the
            #   thinking emoji stuck forever.
            produced_reply = message_completed or bool(text_parts)
            ack_emoji = (
                self._ack_reaction_done
                if produced_reply
                else self._ack_reaction_error
            )
            if ack_emoji:
                ack_chat = getattr(request, "_wa_ack_chat_jid", None)
                ack_sender = getattr(request, "_wa_ack_sender_jid", None)
                ack_msg_id = getattr(request, "_wa_ack_msg_id", None)
                ack_client = getattr(request, "_wa_typing_client", None)
                if ack_chat and ack_sender and ack_msg_id and ack_client:
                    await self._send_reaction(
                        ack_client,
                        ack_chat,
                        ack_sender,
                        ack_msg_id,
                        ack_emoji,
                    )

        except asyncio.CancelledError:
            if process_iterator:
                await process_iterator.aclose()
            raise
        except Exception:
            logger.exception("whatsapp: _stream_with_tracker failed")
            # Flip thinking → error so user knows the request died
            if self._ack_reaction_error:
                ack_chat = getattr(request, "_wa_ack_chat_jid", None)
                ack_sender = getattr(request, "_wa_ack_sender_jid", None)
                ack_msg_id = getattr(request, "_wa_ack_msg_id", None)
                ack_client = getattr(request, "_wa_typing_client", None)
                if ack_chat and ack_sender and ack_msg_id and ack_client:
                    try:
                        await self._send_reaction(
                            ack_client,
                            ack_chat,
                            ack_sender,
                            ack_msg_id,
                            self._ack_reaction_error,
                        )
                    except Exception:
                        pass
        finally:
            if typing_task and not typing_task.done():
                typing_task.cancel()

    # ── Session / routing ─────────────────────────────────────────────

    def build_agent_request_from_native(self, native_payload: Any) -> Any:
        """Build AgentRequest from WhatsApp native dict payload.

        WhatsApp's main inbound path constructs the AgentRequest inline in
        `_dispatch_message()` with full envelope/history/quote handling,
        but this hook is required by BaseChannel so that any code routing
        through `_payload_to_request()` (e.g. re-delivery, replay, testing)
        still gets a valid AgentRequest with `user_id` + `channel_meta`
        set correctly.
        """
        payload = native_payload if isinstance(native_payload, dict) else {}
        channel_id = payload.get("channel_id") or self.channel
        sender_id = payload.get("sender_id") or ""
        content_parts = payload.get("content_parts") or []
        meta = payload.get("meta") or {}
        session_id = self.resolve_session_id(sender_id, meta)
        user_id = str(meta.get("user_id") or sender_id)
        request = self.build_agent_request_from_user_content(
            channel_id=channel_id,
            sender_id=sender_id,
            session_id=session_id,
            content_parts=content_parts,
            channel_meta=meta,
        )
        request.user_id = user_id
        request.channel_meta = meta
        return request

    def resolve_session_id(
        self,
        sender_id: str,
        channel_meta: Optional[Dict[str, Any]] = None,
    ) -> str:
        meta = channel_meta or {}
        chat_jid = meta.get("chat_jid")
        is_group = meta.get("is_group", False)
        if is_group and chat_jid:
            return f"whatsapp:group:{chat_jid}"
        return f"whatsapp:{sender_id}"

    def get_to_handle_from_request(self, request) -> str:
        meta = getattr(request, "channel_meta", None) or {}
        return meta.get("chat_jid") or getattr(request, "user_id", "") or ""
