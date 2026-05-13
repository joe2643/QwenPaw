# -*- coding: utf-8 -*-
# pylint: disable=too-many-arguments,too-many-locals,too-many-branches
# pylint: disable=too-many-statements,too-many-instance-attributes
"""Signal channel: signal-cli subprocess via stdin/stdout JSON-RPC.

One ``signal-cli`` child process per channel instance, lifecycle owned
here. Replaces the previous HTTP + WebSocket coupling to
``bbernhard/signal-cli-rest-api`` — no port to expose, no Docker required;
CoPaw owns respawn on crash.

Feature set (markdown→Signal text-style, mention expansion, sender-name
cache, group history, envelope prefix, bot-identity hint, quote
extraction, ack reactions, continuous typing loop) is preserved from the
prior implementation — only the transport layer changed.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from agentscope_runtime.engine.schemas.agent_schemas import (
    AudioContent,
    ContentType,
    FileContent,
    ImageContent,
    RunStatus,
    TextContent,
    VideoContent,
)

from ....config.config import SignalConfig
from ....constant import WORKING_DIR
from ..base import (
    BaseChannel,
    OnReplySent,
    OutgoingContentPart,
    ProcessHandler,
)
from ..media_utils import resolve_media_url
from .._format import format_local_timestamp as _format_local_timestamp
from .subprocess_client import SignalSubprocessClient

logger = logging.getLogger(__name__)

SIGNAL_MAX_TEXT_LENGTH = 4000
_MEDIA_DIR = WORKING_DIR / "media" / "signal"

# Default data_dir: WORKING_DIR/credentials/signal/default when no
# workspace_dir is passed. When a workspace_dir IS passed (agent-scoped
# install), the default becomes workspace_dir/credentials/signal/default
# so each agent gets its own signal-cli account store. Explicit ``data_dir``
# in the channel config overrides both.
_DEFAULT_DATA_DIR = WORKING_DIR / "credentials" / "signal" / "default"
_LEGACY_DATA_DIR = Path.home() / ".local" / "share" / "signal-cli"
_LEGACY_WARNED = False


def _resolve_signal_data_dir(
    explicit_data_dir: str,
    workspace_dir: Optional[Path] = None,
) -> Path:
    """Compute signal-cli data-dir path consistently across channel + router.

    Priority:
      1. ``explicit_data_dir`` from channel config (expanduser applied)
      2. ``workspace_dir/credentials/signal/default`` when workspace is set
      3. ``WORKING_DIR/credentials/signal/default``

    If the resolved path does not yet exist but signal-cli's global default
    (``~/.local/share/signal-cli``) does, log a one-line warning — the user
    may want to migrate their existing account data manually.
    """
    if explicit_data_dir:
        resolved = Path(explicit_data_dir).expanduser()
    elif workspace_dir is not None:
        resolved = (
            Path(workspace_dir).expanduser()
            / "credentials"
            / "signal"
            / "default"
        )
    else:
        resolved = _DEFAULT_DATA_DIR
    global _LEGACY_WARNED
    if (
        not _LEGACY_WARNED
        and not resolved.exists()
        and _LEGACY_DATA_DIR.exists()
    ):
        logger.warning(
            "signal: legacy data dir found at %s; consider moving its "
            "contents to %s for per-agent isolation",
            _LEGACY_DATA_DIR,
            resolved,
        )
        _LEGACY_WARNED = True
    return resolved


_UUID_LIKE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    re.IGNORECASE,
)


def _looks_like_uuid(s: str) -> bool:
    """True if the string looks like a Signal account UUID."""
    return bool(s) and bool(_UUID_LIKE.match(s))


# ── File type detection by magic bytes ───────────────────────────────────

_MAGIC_MAP = [
    (b"\xff\xd8\xff", "image/jpeg", "jpg"),
    (b"\x89PNG\r\n\x1a\n", "image/png", "png"),
    (b"GIF87a", "image/gif", "gif"),
    (b"GIF89a", "image/gif", "gif"),
    (b"RIFF", "image/webp", "webp"),
    (b"\x1a\x45\xdf\xa3", "video/webm", "webm"),
    (b"OggS", "audio/ogg", "ogg"),
    (b"fLaC", "audio/flac", "flac"),
    (b"ID3", "audio/mpeg", "mp3"),
    (b"\xff\xfb", "audio/mpeg", "mp3"),
    (b"%PDF", "application/pdf", "pdf"),
]


def _detect_mime(data: bytes) -> str:
    """Detect MIME type from the first few bytes."""
    for magic, mime, _ in _MAGIC_MAP:
        if data[: len(magic)] == magic:
            if magic == b"RIFF" and data[8:12] != b"WEBP":
                continue
            return mime
    if len(data) >= 8 and data[4:8] == b"ftyp":
        return "video/mp4"
    return ""


# ── Markdown → Signal text-style ─────────────────────────────────────────


def _markdown_to_signal(text: str) -> tuple[str, List[Dict[str, Any]]]:
    """Convert a limited markdown subset to plain text + Signal text-style
    ranges (BOLD, ITALIC, MONOSPACE, STRIKETHROUGH)."""
    text = re.sub(r"^#{1,6}\s+(.+)$", r"**\1**", text, flags=re.MULTILINE)

    patterns = [
        (re.compile(r"```(?:\w*\n)?(.*?)```", re.DOTALL), "MONOSPACE"),
        (re.compile(r"`([^`]+)`"), "MONOSPACE"),
        (re.compile(r"\*\*(.+?)\*\*"), "BOLD"),
        (re.compile(r"__(.+?)__"), "BOLD"),
        (re.compile(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)"), "ITALIC"),
        (re.compile(r"(?<!_)_(?!_)(.+?)(?<!_)_(?!_)"), "ITALIC"),
        (re.compile(r"~~(.+?)~~"), "STRIKETHROUGH"),
    ]
    all_matches: List[tuple[int, int, str, str]] = []
    for pat, style in patterns:
        for m in pat.finditer(text):
            all_matches.append((m.start(), m.end(), m.group(1), style))
    all_matches.sort(key=lambda x: x[0])

    filtered: List[tuple[int, int, str, str]] = []
    for s, e, inner, style in all_matches:
        if filtered and s < filtered[-1][1]:
            continue
        filtered.append((s, e, inner, style))

    parts: List[str] = []
    styles: List[Dict[str, Any]] = []
    cursor = 0
    offset = 0
    for s, e, inner, style in filtered:
        before = text[cursor:s]
        parts.append(before)
        offset += len(before)
        styles.append({"start": offset, "length": len(inner), "style": style})
        parts.append(inner)
        offset += len(inner)
        cursor = e
    parts.append(text[cursor:])
    return "".join(parts), styles


# ═══════════════════════════════════════════════════════════════════════
#  SignalChannel
# ═══════════════════════════════════════════════════════════════════════


class SignalChannel(BaseChannel):
    """Signal channel backed by a signal-cli subprocess (stdin/stdout JSON-RPC)."""

    channel = "signal"
    uses_manager_queue = True
    # Signal protocol is hard one-process-per-account: two signal-cli
    # daemons on the same account will fight for the SQLite/lock file
    # under ``~/.local/share/signal-cli/data/``.  The default
    # zero-downtime ``replace_channel`` path (start NEW first, stop OLD
    # second) creates a brief overlap during which both daemons are
    # alive, the new one stuck with ``Config file is in use by another
    # instance, waiting…`` until the old one exits — and any RPCs
    # routed to the new client during that window time out (we hit
    # this concretely with ``uploadStickerPack`` returning empty
    # ``signal: uploadStickerPack failed:``).  Sequential restart
    # (stop old → start new) trades a few seconds of downtime for
    # never having two daemons alive at once.
    requires_sequential_restart = True

    def __init__(
        self,
        process: ProcessHandler,
        enabled: bool = False,
        account: str = "",
        signal_cli_path: str = "signal-cli",
        data_dir: str = "",
        extra_args: Optional[List[str]] = None,
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
        show_typing: Optional[bool] = True,
        text_chunk_limit: int = SIGNAL_MAX_TEXT_LENGTH,
        ack_reaction_thinking: str = "🤔",
        ack_reaction_done: str = "👀",
        ack_reaction_error: str = "⚠️",
        workspace_dir: Optional[Path] = None,
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
        self._account = account
        self._send_read_receipts = send_read_receipts
        self._show_typing = show_typing if show_typing is not None else True
        self._text_chunk_limit = text_chunk_limit
        self._ack_reaction_thinking = ack_reaction_thinking or ""
        self._ack_reaction_done = ack_reaction_done or ""
        self._ack_reaction_error = ack_reaction_error or ""
        self._groups: List[str] = kwargs.get("groups") or []
        self._group_allow_from: List[str] = (
            kwargs.get("group_allow_from") or []
        )
        self._reply_to_trigger: bool = kwargs.get("reply_to_trigger", True)
        self._account_uuid: str = kwargs.get("account_uuid") or ""
        self._media_dir = _MEDIA_DIR
        self._group_history: Dict[str, list] = {}
        self._group_history_limit = 50
        self._sender_names: Dict[str, str] = {}
        # 8-char UUID prefix → full UUID. Populated every time we see a
        # sender in a group so the outbound mention compiler can recover
        # full UUIDs from truncated ``uuid:abc12345`` forms that still
        # appear in older session state.
        self._uuid_prefix_lookup: Dict[str, str] = {}
        self._workspace_dir = workspace_dir
        self._data_dir: Path = _resolve_signal_data_dir(
            data_dir,
            workspace_dir,
        )

        # Mention detection regression guard: signal-cli emits structured
        # mentions with the bot's ACI (``uuid`` field) but usually NOT the
        # bot's phone number (privacy). If ``account_uuid`` is unset,
        # ``_is_bot_mentioned`` cannot match ANY Signal-UI @bot tap — it
        # falls back to plain-text regex which only catches manually typed
        # ``@+<number>`` forms. Auto-populate from signal-cli's own
        # ``<data_dir>/accounts.json`` so linked accounts always have a
        # usable UUID.
        if not self._account_uuid and account:
            self._account_uuid = self._auto_discover_account_uuid(
                account,
                self._data_dir,
            )

        self.client = SignalSubprocessClient(
            account=account,
            signal_cli_path=signal_cli_path,
            extra_args=extra_args,
            data_dir=self._data_dir,
        )

        if self.enabled:
            logger.info(
                "signal: initialized (account=%s, uuid=%s, signal_cli=%s, "
                "data_dir=%s)",
                account,
                self._account_uuid[:8] + "…"
                if self._account_uuid
                else "<unset>",
                signal_cli_path,
                self._data_dir,
            )

    @staticmethod
    def _auto_discover_account_uuid(account: str, data_dir: Path) -> str:
        """Look up the bot's UUID in signal-cli's accounts.json.

        Returns empty string if the file is missing, unreadable, or the
        account is not listed — the caller will fall back to plain-text
        mention detection, which still works for manually typed
        ``@+<number>`` mentions.
        """
        try:
            import json as _json

            accounts_path = Path(data_dir) / "data" / "accounts.json"
            if not accounts_path.is_file():
                return ""
            data = _json.loads(accounts_path.read_text(encoding="utf-8"))
            for entry in data.get("accounts") or []:
                if (
                    isinstance(entry, dict)
                    and entry.get("number") == account
                    and entry.get("uuid")
                ):
                    uuid_value = str(entry["uuid"]).strip().lower()
                    logger.info(
                        "signal: auto-discovered account_uuid=%s… for %s "
                        "(from %s)",
                        uuid_value[:8],
                        account,
                        accounts_path,
                    )
                    return uuid_value
        except Exception as e:  # pragma: no cover - best-effort discovery
            logger.debug(
                "signal: auto-discover account_uuid failed: %s",
                e,
            )
        return ""

    @classmethod
    def from_config(
        cls,
        process: ProcessHandler,
        config: Union[SignalConfig, dict],
        on_reply_sent: OnReplySent = None,
        show_tool_details: bool = True,
        filter_tool_messages: bool = False,
        filter_thinking: bool = False,
        workspace_dir: Optional[Path] = None,
        **kwargs,
    ) -> "SignalChannel":
        if isinstance(config, dict):
            c = config
        elif hasattr(config, "model_dump"):
            c = config.model_dump()
        else:
            c = vars(config)
        return cls(
            process=process,
            enabled=bool(c.get("enabled", False)),
            account=c.get("account") or "",
            signal_cli_path=c.get("signal_cli_path") or "signal-cli",
            data_dir=c.get("data_dir") or "",
            workspace_dir=workspace_dir,
            extra_args=c.get("extra_args") or [],
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
            show_typing=c.get("show_typing"),
            text_chunk_limit=c.get("text_chunk_limit", SIGNAL_MAX_TEXT_LENGTH),
            ack_reaction_thinking=c.get("ack_reaction_thinking", "🤔"),
            ack_reaction_done=c.get("ack_reaction_done", "👀"),
            ack_reaction_error=c.get("ack_reaction_error", "⚠️"),
            groups=c.get("groups") or [],
            group_allow_from=c.get("group_allow_from") or [],
            account_uuid=c.get("account_uuid") or c.get("accountUuid") or "",
            reply_to_trigger=c.get("reply_to_trigger", True),
        )

    async def update_config(self, config) -> bool:
        """Patch config in-place without restarting signal-cli.

        Signal config splits into two buckets:

        * **Hard fields** — baked into the signal-cli spawn command
          (``account``, ``signal_cli_path``, ``data_dir``,
          ``extra_args``) or determine whether the daemon should
          even be running (``enabled``).  Changing any of these
          must trigger a real channel restart; we ``return False``
          and let ``ChannelManager.replace_channel`` swap us out.
        * **Soft fields** — read at request time by Python
          (``send_read_receipts``, ``show_typing``, ``text_chunk_limit``,
          ack reactions, ``groups`` / ``group_allow_from`` /
          ``allow_from`` / ``dm_policy`` / ``group_policy`` /
          ``require_mention`` / ``reply_to_trigger`` /
          ``account_uuid`` / ``deny_message``).  Patching these
          on the live channel is safe and avoids any signal-cli
          subprocess restart, which would otherwise contest the
          account file lock for several seconds.
        """
        if isinstance(config, dict):
            c = config
        elif hasattr(config, "model_dump"):
            c = config.model_dump()
        else:
            c = vars(config) if hasattr(config, "__dict__") else dict(config)

        # ── Hard-field guard ────────────────────────────────────────
        new_enabled = bool(c.get("enabled", False))
        if new_enabled != self.enabled:
            logger.info(
                "signal: update_config: enabled changed (%s → %s), "
                "needs full restart",
                self.enabled,
                new_enabled,
            )
            return False

        new_account = (c.get("account") or "").strip()
        if new_account != (self._account or ""):
            logger.info(
                "signal: update_config: account changed, needs restart",
            )
            return False

        new_signal_cli_path = c.get("signal_cli_path") or "signal-cli"
        if new_signal_cli_path != self.client._signal_cli_path:
            logger.info(
                "signal: update_config: signal_cli_path changed, needs restart",
            )
            return False

        new_data_dir = _resolve_signal_data_dir(
            c.get("data_dir") or "",
            self._workspace_dir,
        )
        if new_data_dir != self._data_dir:
            logger.info(
                "signal: update_config: data_dir changed, needs restart",
            )
            return False

        new_extra_args = list(c.get("extra_args") or [])
        if new_extra_args != list(self.client._extra_args):
            logger.info(
                "signal: update_config: extra_args changed, needs restart",
            )
            return False

        # If signal-cli isn't actually connected, an in-place patch
        # would leave a dead subprocess; surface the broken state to
        # the caller so a full restart re-spawns the daemon.
        if not self.client.connected:
            logger.info(
                "signal: update_config: subprocess not connected — "
                "triggering full restart so the daemon respawns cleanly",
            )
            return False

        # ── Soft-patchable fields ──────────────────────────────────
        self._send_read_receipts = c.get("send_read_receipts", True)
        self._show_typing = c.get("show_typing", True)
        self._text_chunk_limit = c.get(
            "text_chunk_limit",
            SIGNAL_MAX_TEXT_LENGTH,
        )
        self._ack_reaction_thinking = c.get("ack_reaction_thinking", "🤔") or ""
        self._ack_reaction_done = c.get("ack_reaction_done", "👀") or ""
        self._ack_reaction_error = c.get("ack_reaction_error", "⚠️") or ""
        self._groups = list(c.get("groups") or [])
        self._group_allow_from = list(c.get("group_allow_from") or [])
        self._reply_to_trigger = c.get("reply_to_trigger", True)
        new_uuid = c.get("account_uuid") or c.get("accountUuid") or ""
        if new_uuid:
            self._account_uuid = new_uuid

        # BaseChannel access-control fields
        self.dm_policy = c.get("dm_policy") or "open"
        self.group_policy = c.get("group_policy") or "open"
        self.allow_from = list(c.get("allow_from") or [])
        self.deny_message = c.get("deny_message") or ""
        self.require_mention = c.get("require_mention", False)

        logger.info(
            "signal: config patched in-place (signal-cli daemon preserved)",
        )
        return True

    # ── Lifecycle ────────────────────────────────────────────────────

    async def start(self) -> None:
        if not self.enabled:
            return
        ok = await self.client.connect(self._on_notification)
        if not ok:
            logger.error("signal: failed to start subprocess")
            return
        logger.info("signal: channel started (subprocess JSON-RPC active)")

    async def stop(self) -> None:
        if not self.enabled:
            return
        await self.client.disconnect()
        logger.info("signal: channel stopped")

    # ── Inbound ──────────────────────────────────────────────────────

    async def _on_notification(self, params: Dict[str, Any]) -> None:
        """Handle an inbound `receive` JSON-RPC notification."""
        try:
            envelope = params.get("envelope", params)
            source = (
                envelope.get("sourceNumber") or envelope.get("source") or ""
            )
            source_uuid = envelope.get("sourceUuid") or ""
            source_name = envelope.get("sourceName") or ""
            timestamp = envelope.get("timestamp", 0)

            if source_name:
                self._remember_sender(source, source_uuid, source_name)

            reaction_msg = envelope.get("reactionMessage")
            if reaction_msg:
                await self._handle_inbound_reaction(
                    source,
                    source_uuid,
                    reaction_msg,
                    envelope,
                )
                return

            data_message = envelope.get("dataMessage") or {}
            body = data_message.get("message") or ""

            msg_mentions = data_message.get("mentions") or []
            for m in msg_mentions:
                self._remember_sender(
                    m.get("number") or "",
                    m.get("uuid") or "",
                    m.get("name") or "",
                )
            body = self._expand_mentions(body, msg_mentions)

            group_info = data_message.get("groupInfo") or {}
            group_id = group_info.get("groupId") or ""

            attachments_raw = data_message.get("attachments") or []
            if attachments_raw:
                logger.info(
                    "signal: attachments found: %s",
                    json.dumps(attachments_raw)[:500],
                )
            if (
                not body
                and not attachments_raw
                and not data_message.get("sticker")
            ):
                return

            downloaded_media: List[Dict[str, str]] = []
            for att in attachments_raw:
                att_id = att.get("id") or ""
                if not att_id:
                    continue
                ct = att.get("contentType") or ""
                local = await self.client.download_attachment(
                    att_id,
                    self._media_dir,
                )
                if local:
                    downloaded_media.append({"path": str(local), "type": ct})

            # Sticker: dataMessage.sticker is Signal's in-band sticker
            # reference ({packId, packKey, stickerId, emoji}) rather
            # than a plain attachment.  Fetch the webp via
            # ``getSticker`` (which auto-installs the pack when we
            # have the pack key), feed it in as image/webp, and
            # remember the emoji so downstream can include it in the
            # "[sticker]" hint.  Marking ``is_sticker=True`` lets the
            # content-parts builder emit the hint block without
            # disturbing plain-image attachments.
            sticker_info = data_message.get("sticker") or {}
            if sticker_info:
                sticker_path = await self._download_sticker(sticker_info)
                if sticker_path:
                    downloaded_media.append(
                        {
                            "path": str(sticker_path),
                            "type": "image/webp",
                            "is_sticker": "1",
                            "emoji": str(sticker_info.get("emoji") or ""),
                        },
                    )

            # ── Access control ────────────────────────────────────
            if group_id:
                if self.group_policy == "allowlist":
                    if not self._groups or group_id not in self._groups:
                        # Elevated from DEBUG: at INFO level this is
                        # how operators diagnose "I messaged the bot
                        # in a group and nothing happened".  Default
                        # root log level is INFO so DEBUG was invisible.
                        logger.info(
                            "signal: blocked group %s (not in allowlist of "
                            "%d entries)",
                            group_id[:20],
                            len(self._groups or []),
                        )
                        return
                if self._group_allow_from:
                    sender_id = source or source_uuid
                    if not (
                        "*" in self._group_allow_from
                        or sender_id in self._group_allow_from
                        or source in self._group_allow_from
                        or source_uuid in self._group_allow_from
                        or f"uuid:{source_uuid}" in self._group_allow_from
                    ):
                        logger.info(
                            "signal: blocked sender %s in group %s by "
                            "group_allow_from",
                            sender_id,
                            group_id[:20],
                        )
                        return
                # Compute mention status once so channel_meta can reflect
                # reality regardless of require_mention. In groups we use
                # the real check; DMs are always implicitly addressed to
                # the bot so they fall through to True below.
                bot_mentioned_actual = self._is_bot_mentioned(
                    data_message,
                    body,
                )
                if self.require_mention:
                    if not bot_mentioned_actual:
                        # Silent-drop until now.  Log what we know so
                        # the operator can tell whether the structured
                        # picker-mention reached us at all.
                        mentions_raw = data_message.get("mentions") or []
                        mention_summary = [
                            {
                                "uuid": (m.get("uuid") or "")[:8],
                                "number": m.get("number"),
                            }
                            for m in mentions_raw
                        ]
                        logger.info(
                            "signal: require_mention drop (group %s, "
                            "sender %s) — my_uuid=%s my_phone=%s "
                            "body=%r mentions=%s",
                            group_id[:20],
                            (source or source_uuid)[:16],
                            (self._account_uuid or "")[:8],
                            self._account,
                            body[:80],
                            mention_summary,
                        )
                        if body or downloaded_media:
                            media_paths = [m["path"] for m in downloaded_media]
                            sender_label = self._format_sender_display(
                                source,
                                source_uuid,
                            )
                            history = self._group_history.setdefault(
                                group_id,
                                [],
                            )
                            history.append(
                                {
                                    "sender": sender_label,
                                    "body": body or "[media]",
                                    "ts": timestamp,
                                    "media": media_paths,
                                },
                            )
                            if len(history) > self._group_history_limit:
                                self._group_history[group_id] = history[
                                    -self._group_history_limit :
                                ]
                        return
            else:
                # DM: implicitly addressed to the bot.
                bot_mentioned_actual = True
                if self.dm_policy == "allowlist":
                    # Empty allowlist in allowlist mode ⇒ reject everyone.
                    # (Dropping the `and self.allow_from` guard that previously
                    # short-circuited this check for empty list, which caused
                    # an empty-allowlist + allowlist mode to silently allow
                    # every DM through.)
                    if not self._is_source_allowed(source, source_uuid):
                        # INFO level (not DEBUG) so operators can diagnose
                        # "why didn't my DM go through" without flipping
                        # the whole app's log level. Low-volume by nature
                        # — only fires once per blocked DM.
                        logger.info(
                            "signal: blocked DM from %s (dm_policy=allowlist, "  # noqa: E501
                            "allow_from=%d entries)",
                            source or source_uuid,
                            len(self.allow_from or []),
                        )
                        return

            logger.info(
                "signal: from %s%s: %s",
                source or source_uuid[:12],
                " (group)" if group_id else "",
                body[:80]
                if body
                else f"[{len(attachments_raw)} attachment(s)]",
            )

            # ── Build content parts ───────────────────────────────
            content_parts: List[Any] = []
            if body:
                content_parts.append(
                    TextContent(type=ContentType.TEXT, text=body),
                )

            quote_parts = await self._extract_quote_content(data_message)
            if quote_parts:
                content_parts = quote_parts + content_parts

            for m in downloaded_media:
                ct = m["type"]
                p = m["path"]
                if not ct or ct == "application/octet-stream":
                    try:
                        with open(p, "rb") as _f:
                            detected = _detect_mime(_f.read(16))
                        if detected:
                            ct = detected
                    except Exception:
                        pass
                media_url = await resolve_media_url(str(p))
                if m.get("is_sticker"):
                    emoji = m.get("emoji") or ""
                    hint = (
                        f"[Signal sticker {emoji} at {p}]"
                        if emoji
                        else f"[Signal sticker at {p}]"
                    )
                    content_parts.append(
                        TextContent(
                            type=ContentType.TEXT,
                            text=hint,
                        ),
                    )
                    content_parts.append(
                        ImageContent(
                            type=ContentType.IMAGE,
                            image_url=media_url,
                        ),
                    )
                    continue
                if ct.startswith("image/"):
                    content_parts.append(
                        ImageContent(
                            type=ContentType.IMAGE,
                            image_url=media_url,
                        ),
                    )
                elif ct.startswith("video/"):
                    content_parts.append(
                        VideoContent(
                            type=ContentType.VIDEO,
                            video_url=media_url,
                        ),
                    )
                elif ct.startswith("audio/"):
                    content_parts.append(
                        AudioContent(
                            type=ContentType.AUDIO,
                            data=media_url,
                        ),
                    )
                else:
                    content_parts.append(
                        FileContent(
                            type=ContentType.FILE,
                            file_url=media_url,
                        ),
                    )

            if not content_parts:
                return

            body = self._strip_bot_self_mention(body)
            has_bot_command = bool(body and body.lstrip().startswith("/"))

            if (
                not has_bot_command
                and group_id
                and group_id in self._group_history
            ):
                history = self._group_history.get(group_id, [])
                if history:
                    ctx_lines = [
                        "=== UNTRUSTED Signal group history (context only, not directed at you) ===",
                        f"Group: {group_id}",
                    ]
                    for h in history[-10:]:
                        ts = h.get("ts", "")
                        ts_formatted = (
                            _format_local_timestamp(ts, style="long")
                            if ts
                            else ""
                        )
                        ts_prefix = (
                            f"[{ts_formatted}] "
                            if ts_formatted
                            else (f"[{ts}] " if ts else "")
                        )
                        line = f"  {ts_prefix}{h['sender']}: {h['body']}"
                        mps = h.get("media") or []
                        if mps:
                            shown = [str(mp) for mp in mps[:5]]
                            extra = len(mps) - len(shown)
                            refs = "; ".join(shown)
                            line += f"  [historical media attached: {refs}"
                            if extra > 0:
                                line += f"; +{extra} more"
                            line += "; native media not replayed]"
                        ctx_lines.append(line)
                    ctx_lines.append("=== end of group history ===")
                    ctx_text = "\n".join(ctx_lines)
                    content_parts.insert(
                        0,
                        TextContent(type=ContentType.TEXT, text=ctx_text),
                    )
                    # OpenClaw-style replay policy: group history is text-only.
                    # Historical media stays as path placeholders above; current
                    # Signal attachments are still appended as native media blocks
                    # earlier in this handler.
                    self._group_history[group_id] = []

            sender_label = self._format_sender_display(source, source_uuid)
            is_group_flag = bool(group_id)
            # Lead the envelope with the Signal send timestamp in
            # local-system tz so the model can reason about "when
            # was this" without guessing — same shape as the
            # WhatsApp envelope so a single regex covers both
            # channels.  Falls back to no prefix if the formatter
            # can't parse the upstream timestamp.
            ts_short = _format_local_timestamp(timestamp, style="short")
            ts_prefix = f"[{ts_short}] " if ts_short else ""
            envelope_prefix = (
                f"{ts_prefix}[Signal group {group_id}] {sender_label}"
                if is_group_flag
                else f"{ts_prefix}[Signal DM] {sender_label}"
            )
            for i, part in enumerate(content_parts):
                if hasattr(part, "type") and part.type == ContentType.TEXT:
                    txt = part.text or ""
                    if txt.startswith("===") or txt.startswith("[Replying"):
                        continue
                    txt = self._strip_bot_self_mention(txt)
                    if has_bot_command:
                        content_parts[i] = TextContent(
                            type=ContentType.TEXT,
                            text=txt,
                        )
                    else:
                        content_parts[i] = TextContent(
                            type=ContentType.TEXT,
                            text=f"{envelope_prefix}: {txt}",
                        )
                    break
            else:
                content_parts.insert(
                    0,
                    TextContent(
                        type=ContentType.TEXT,
                        text=f"{envelope_prefix}: [media]",
                    ),
                )

            # Emit in both DMs and groups so the bot always knows how to mention.
            # Skip for slash commands (they bypass the agent entirely).
            if not has_bot_command:
                bot_id = self._account or (
                    f"uuid:{self._account_uuid[:8]}"
                    if self._account_uuid
                    else ""
                )
                hint_line = (
                    f"[Signal bot {bot_id}. "
                    f"To mention someone, write @+phone (name) or @uuid:xxxxxxxx (name) "
                    f"(e.g. @+85251159218 (Joe) or @uuid:82e0393a (Joe)). "
                    f"The (name) part is optional but helpful for readability.]"
                )
                content_parts.insert(
                    0,
                    TextContent(
                        type=ContentType.TEXT,
                        text=hint_line,
                    ),
                )

            channel_meta: Dict[str, Any] = {
                "platform": "signal",
                "account": self._account,
                "timestamp": timestamp,
                "group_id": group_id,
                "source": source or source_uuid,
                "source_uuid": source_uuid,
                "quote_timestamp": timestamp,
                "quote_author": source or source_uuid,
                "has_bot_command": has_bot_command,
                # Reflects actual @mention detection, not "we passed the
                # mention gate". DMs are implicitly True; groups use
                # _is_bot_mentioned(). Avoids the round-8 WhatsApp bug
                # where downstream consumers got hard-coded True.
                "bot_mentioned": bot_mentioned_actual,
            }
            session_id = self.resolve_session_id(
                source or source_uuid,
                channel_meta,
            )
            effective_sender = (
                f"group:{group_id}" if group_id else (source or source_uuid)
            )
            request = self.build_agent_request_from_user_content(
                channel_id=self.channel,
                sender_id=effective_sender,
                session_id=session_id,
                content_parts=content_parts,
                channel_meta=channel_meta,
            )
            request.channel_meta = channel_meta

            is_group = bool(group_id)
            typing_target = group_id if is_group else (source or source_uuid)
            channel_meta["_typing_target"] = typing_target
            channel_meta["_typing_is_group"] = is_group
            channel_meta["_ack_target"] = typing_target
            channel_meta["_ack_author"] = source or source_uuid
            channel_meta["_ack_timestamp"] = timestamp

            if self._ack_reaction_thinking:
                asyncio.create_task(
                    self.client.send_reaction(
                        typing_target,
                        self._ack_reaction_thinking,
                        target_author=source or source_uuid,
                        target_timestamp=timestamp,
                        is_group=is_group,
                    ),
                )

            if self._enqueue is not None:
                self._enqueue(request)
            else:
                await self.consume_one(request)

        except Exception:
            logger.exception("signal: error processing inbound notification")

    async def _download_sticker(
        self,
        sticker: Dict[str, Any],
    ) -> Optional[Path]:
        """Resolve a ``dataMessage.sticker`` reference into a local webp.

        signal-cli reports stickers as ``{packId, packKey, stickerId,
        emoji, ...}`` without an in-band attachment blob — the bytes
        live on Signal's sticker-pack CDN keyed by pack.  We delegate
        to :meth:`SignalSubprocessClient.get_sticker` which handles
        the install-then-fetch dance when the pack isn't locally
        cached yet.
        """
        pack_id = str(sticker.get("packId") or "").strip()
        raw_sticker_id = sticker.get("stickerId")
        if raw_sticker_id is None:
            return None
        try:
            sticker_id = int(raw_sticker_id)
        except (TypeError, ValueError):
            logger.warning(
                "signal: sticker field has non-int stickerId: %r",
                raw_sticker_id,
            )
            return None
        if not pack_id:
            return None
        pack_key = sticker.get("packKey")
        try:
            return await self.client.get_sticker(
                pack_id,
                sticker_id,
                self._media_dir,
                pack_key=pack_key if isinstance(pack_key, str) else None,
            )
        except Exception as e:
            logger.warning(
                "signal: sticker fetch failed pack=%s id=%s: %s",
                pack_id[:12],
                sticker_id,
                e,
            )
            return None

    async def _handle_inbound_reaction(
        self,
        source: str,
        source_uuid: str,
        reaction: Dict[str, Any],
        envelope: Dict[str, Any],
    ) -> None:
        emoji = reaction.get("emoji") or ""
        is_remove = reaction.get("isRemove", False)
        target_ts = reaction.get("targetSentTimestamp") or 0
        group_info = reaction.get("groupInfo") or {}
        group_id = group_info.get("groupId") or ""
        logger.info(
            "signal: reaction %s%s from %s on msg %d%s",
            emoji,
            " (remove)" if is_remove else "",
            source or source_uuid[:12],
            target_ts,
            " (group)" if group_id else "",
        )

    # ── Access control / naming helpers ──────────────────────────────

    def _is_bot_mentioned(
        self,
        data_message: Dict[str, Any],
        body: str,
    ) -> bool:
        # Structured mentions are the reliable signal (set by signal-cli
        # from the wire protocol). Accept both uuid and phone matches.
        mentions = data_message.get("mentions") or []
        for m in mentions:
            if m.get("uuid") == self._account_uuid:
                return True
            if m.get("number") == self._account:
                return True
        # Reply-to-bot counts as an implicit mention.
        quote = data_message.get("quote")
        if quote:
            qa = quote.get("author") or quote.get("authorUuid") or ""
            if qa == self._account or qa == self._account_uuid:
                return True
        # Plain-text fallback. Use a trailing non-digit / non-word guard so
        # account "+123" doesn't falsely match body "@+12345 hello"
        # (prefix-collision bug; identical to the WhatsApp round-3 fix).
        if self._account and re.search(
            rf"@\+?{re.escape(self._account.lstrip('+'))}(?!\d)",
            body,
        ):
            return True
        if self._account_uuid and re.search(
            rf"@{re.escape(self._account_uuid)}(?!\w)",
            body,
        ):
            return True
        return False

    def _is_source_allowed(self, source: str, source_uuid: str) -> bool:
        for entry in self.allow_from:
            if entry.startswith("uuid:"):
                if source_uuid == entry[5:]:
                    return True
            elif entry.startswith("+"):
                if source == entry:
                    return True
            elif source == entry or source_uuid == entry:
                return True
        return False

    def _remember_sender(
        self,
        source: str,
        source_uuid: str,
        name: str,
    ) -> None:
        # Name map is populated only when we know a human-friendly name, but
        # UUID prefix → full UUID MUST be recorded unconditionally so that
        # outbound mentions can resolve short-form uuids from older history.
        if source_uuid:
            self._uuid_prefix_lookup[
                source_uuid[:8].lower()
            ] = source_uuid.lower()
        if not name or _looks_like_uuid(name):
            return
        if source:
            self._sender_names[source] = name
        if source_uuid:
            self._sender_names[source_uuid] = name

    def _resolve_full_uuid(self, raw: str) -> str:
        """Return a full Signal ACI for *raw* if it looks like a short prefix.

        ``_expand_mentions`` used to truncate UUIDs to 8 chars when building
        the text form the bot sees (``uuid:abc12345``), which meant the
        outbound parser had only 8 chars to work with — too short for
        signal-cli's ACI lookup, so the mention was silently dropped and
        the recipient saw the raw UUID text instead of a styled @-mention.

        We now emit the full UUID in the bot's view and keep this resolver
        around for backward compatibility with historical session state
        that still contains the truncated form.
        """
        key = (raw or "").strip().lower()
        if not key:
            return key
        # Already looks full (has at least one hyphen and ≥ 36 chars)
        if len(key) >= 36 and "-" in key:
            return key
        return self._uuid_prefix_lookup.get(key[:8], key)

    def _strip_bot_self_mention(self, text: str) -> str:
        if not text:
            return text
        ids: List[str] = []
        if self._account:
            ids.append(re.escape(self._account.lstrip("+")))
        if self._account_uuid:
            ids.append(re.escape(self._account_uuid))
            ids.append(re.escape(self._account_uuid[:8]))
        if not ids:
            return text
        id_alt = "|".join(ids)
        id_core = rf"(?:\+?(?:{id_alt})|uuid:(?:{id_alt}))"
        patterns = [
            re.compile(rf"^\s*@[^\s()]+\s*\({id_core}\)\s*"),
            re.compile(rf"^\s*@{id_core}\s*"),
        ]
        for pat in patterns:
            m = pat.match(text)
            if m:
                return text[m.end() :].lstrip()
        return text

    def _format_sender_display(self, source: str, source_uuid: str) -> str:
        name = (
            self._sender_names.get(source)
            or self._sender_names.get(source_uuid)
            or ""
        )
        if _looks_like_uuid(name):
            name = ""
        phone = source or ""
        if name and phone:
            return f"{name} ({phone})"
        if name and source_uuid:
            return f"{name} (uuid:{source_uuid[:8]})"
        if name:
            return name
        if phone:
            return phone
        if source_uuid:
            return f"uuid:{source_uuid[:8]}"
        return "unknown"

    def _compile_outbound_mentions(
        self,
        text: str,
    ) -> tuple[str, List[Dict[str, Any]]]:
        pat = re.compile(
            r"@(?:[^@\s()]+\s*)?"
            r"(?:"
            r"\(\+(\d{7,15})\)"
            r"|\(uuid:([0-9a-f]{8}[0-9a-f-]*)\)"
            r"|\+(\d{7,15})"
            r"|uuid:([0-9a-f]{8}[0-9a-f-]*)"
            r")",
            # UUIDs from Signal can be lowercase or uppercase hex; make
            # the whole pattern case-insensitive so Mixed/UPPER uuid
            # mentions also compile to structured Signal mentions.
            re.IGNORECASE,
        )
        out: List[str] = []
        mentions: List[Dict[str, Any]] = []
        cursor = 0
        for m in pat.finditer(text):
            out.append(text[cursor : m.start()])
            phone = m.group(1) or m.group(3) or ""
            uuid_v = m.group(2) or m.group(4) or ""
            if phone or uuid_v:
                start = sum(len(p) for p in out)
                entry: Dict[str, Any] = {"start": start, "length": 1}
                if phone:
                    entry["number"] = f"+{phone}"
                else:
                    # signal-cli requires a full ACI (36-char UUID).
                    # Older session history may emit 8-char prefix
                    # (uuid:abc12345) — resolve via per-channel map so
                    # the mention actually renders in recipient client.
                    entry["uuid"] = self._resolve_full_uuid(uuid_v)
                mentions.append(entry)
                out.append("\ufffc")
            else:
                out.append(m.group(0))
            cursor = m.end()
        out.append(text[cursor:])
        return "".join(out), mentions

    def _expand_mentions(
        self,
        body: str,
        mentions: List[Dict[str, Any]],
    ) -> str:
        if not body or not mentions:
            return body
        sorted_mentions = sorted(
            mentions,
            key=lambda m: m.get("start", 0),
            reverse=True,
        )
        result = body
        for m in sorted_mentions:
            start = m.get("start")
            length = m.get("length") or 1
            if start is None or start < 0 or start + length > len(result):
                continue
            name = m.get("name") or ""
            number = m.get("number") or ""
            uuid_v = m.get("uuid") or ""
            if _looks_like_uuid(name):
                name = ""
            if not name and number:
                name = self._sender_names.get(number, "")
            if not name and uuid_v:
                name = self._sender_names.get(uuid_v, "")
            if _looks_like_uuid(name):
                name = ""
            # Format: @ID (Name) — ID-first so bot learns the mention syntax.
            # Prefer phone over uuid for the id. Emit the FULL UUID (not an
            # 8-char prefix) so the outbound parser can round-trip it back
            # into a structured Signal mention: signal-cli needs a complete
            # ACI to look up the contact, and a truncated prefix ended up
            # as raw text in the recipient's view.
            phone_str = (
                number
                if number.startswith("+")
                else f"+{number}"
                if number
                else ""
            )
            id_str = (
                phone_str
                if phone_str
                else (f"uuid:{uuid_v}" if uuid_v else "")
            )
            if id_str and name:
                token = f"@{id_str} ({name})"
            elif id_str:
                token = f"@{id_str}"
            elif name:
                token = f"@{name}"
            else:
                token = "@someone"
            result = result[:start] + token + result[start + length :]
        return result

    # ── Quote / reply-to extraction ──────────────────────────────────

    async def _extract_quote_content(
        self,
        data_message: Dict[str, Any],
    ) -> List[Any]:
        quote = data_message.get("quote")
        if not quote:
            return []

        parts: List[Any] = []
        quote_text = quote.get("text") or ""
        quote_author_number = quote.get("author") or ""
        quote_author_uuid = quote.get("authorUuid") or ""
        quote_mentions = quote.get("mentions") or []
        if quote_mentions:
            for m in quote_mentions:
                self._remember_sender(
                    m.get("number") or "",
                    m.get("uuid") or "",
                    m.get("name") or "",
                )
            quote_text = self._expand_mentions(quote_text, quote_mentions)
        quote_id = quote.get("id") or ""

        quote_attachments = quote.get("attachments") or []
        media_labels: List[str] = []
        for att in quote_attachments:
            att_ct = att.get("contentType") or ""
            att_fname = att.get("fileName") or ""
            att_id = att.get("id") or ""
            if att_id:
                local = await self.client.download_attachment(
                    att_id,
                    self._media_dir,
                )
                if local:
                    if not att_ct or att_ct == "application/octet-stream":
                        try:
                            with open(str(local), "rb") as _qf:
                                detected = _detect_mime(_qf.read(16))
                            if detected:
                                att_ct = detected
                        except Exception:
                            pass
                    att_media_url = await resolve_media_url(str(local))
                    # Inline the absolute local path into the
                    # ``Media:`` label so the agent can pass it to
                    # tools (codex image i2i, view_video,
                    # transcribe) — mirrors the WhatsApp channel's
                    # reply-to formatting.  Without the path the
                    # text block is dead weight; the agent would
                    # have to scan the sibling parts and guess
                    # which attachment is which.
                    local_path_str = str(local)
                    if att_ct.startswith("image/"):
                        parts.append(
                            ImageContent(
                                type=ContentType.IMAGE,
                                image_url=att_media_url,
                            ),
                        )
                        media_labels.append(f"image: {local_path_str}")
                    elif att_ct.startswith("video/"):
                        parts.append(
                            VideoContent(
                                type=ContentType.VIDEO,
                                video_url=att_media_url,
                            ),
                        )
                        media_labels.append(f"video: {local_path_str}")
                    elif att_ct.startswith("audio/"):
                        parts.append(
                            AudioContent(
                                type=ContentType.AUDIO,
                                data=att_media_url,
                            ),
                        )
                        media_labels.append(f"audio: {local_path_str}")
                    else:
                        base = f"file: {att_fname}" if att_fname else "file"
                        parts.append(
                            FileContent(
                                type=ContentType.FILE,
                                file_url=att_media_url,
                            ),
                        )
                        media_labels.append(f"{base} ({local_path_str})")
                    continue
            media_labels.append(att_fname or att_ct or "attachment")

        # Signal stores stickers on quoted messages in a separate
        # ``sticker`` field rather than the ``attachments`` array —
        # handle it the same way the direct-inbound path does so
        # the agent sees ``sticker: <path>`` in the reply-context
        # label and gets the webp inlined for visual reference.
        quote_sticker = quote.get("sticker") or {}
        if quote_sticker:
            sticker_path = await self._download_sticker(quote_sticker)
            if sticker_path:
                emoji = str(quote_sticker.get("emoji") or "").strip()
                label = (
                    f"sticker {emoji} ({sticker_path})"
                    if emoji
                    else f"sticker ({sticker_path})"
                )
                media_labels.append(label)
                parts.append(
                    ImageContent(
                        type=ContentType.IMAGE,
                        image_url=await resolve_media_url(str(sticker_path)),
                    ),
                )
            else:
                emoji = str(quote_sticker.get("emoji") or "").strip()
                media_labels.append(
                    f"sticker {emoji} (fetch failed)"
                    if emoji
                    else "sticker (fetch failed)",
                )

        if quote_text or media_labels:
            lines = [
                "=== UNTRUSTED reply-to (this message quotes an earlier one) ===",
            ]
            author_str = self._format_sender_display(
                quote_author_number,
                quote_author_uuid,
            )
            lines.append(f"From: {author_str}")
            if quote_id:
                lines.append(f"Quoted message id: {quote_id}")
            if quote_text:
                lines.append(f"Message: {quote_text[:400]}")
            if media_labels:
                lines.append(f"Media: {', '.join(media_labels)}")
            lines.append("=== end of reply-to ===")
            parts.insert(
                0,
                TextContent(type=ContentType.TEXT, text="\n".join(lines)),
            )
        return parts

    # ── Outbound send ────────────────────────────────────────────────

    async def send(
        self,
        to_handle: str,
        text: str,
        meta: Optional[dict] = None,
    ) -> None:
        if not self.enabled or not self.client.connected:
            return
        if not text:
            return

        meta = meta or {}
        group_id = meta.get("group_id") or ""
        is_group = bool(group_id) or (
            to_handle.endswith("=") and not to_handle.startswith("+")
        )
        if is_group and group_id:
            to_handle = group_id

        img_re = re.compile(r"\[Image: (file:///[^\]]+|/[^\]]+)\]")
        safe_dir = self._media_dir.resolve()
        att_paths: List[str] = []
        for m in img_re.findall(text):
            p = m.replace("file://", "") if m.startswith("file://") else m
            try:
                resolved = Path(p).resolve()
            except Exception:
                continue
            # Use is_relative_to for proper path containment — str.startswith
            # would let /media/signal2/foo masquerade as being under /media/signal/.
            try:
                is_contained = resolved.is_relative_to(safe_dir)
            except (ValueError, AttributeError):
                # Python <3.9 or cross-drive paths: fall back to commonpath.
                try:
                    is_contained = os.path.commonpath(
                        [str(resolved), str(safe_dir)],
                    ) == str(safe_dir)
                except ValueError:
                    is_contained = False
            if is_contained and resolved.is_file():
                att_paths.append(str(resolved))
                text = text.replace(f"[Image: {m}]", "").strip()
                logger.info("signal: extracted image attachment: %s", resolved)
            elif os.path.isfile(p):
                logger.warning(
                    "signal: blocked send of %s — outside media dir",
                    p,
                )
                text = text.replace(f"[Image: {m}]", "").strip()

        # Order here matters. signal-cli wants style ranges AND mention
        # ranges to index into the exact plain-text string we hand over.
        # The old order was: strip markdown → compile mentions, but
        # compile-mentions replaces each "@+phone (Name)" (variable length)
        # with a single U+FFFC, shifting every style start offset computed
        # against the pre-rewrite text.
        #
        # New order: compile mentions first so U+FFFC placeholders are in
        # place, then strip markdown. Markdown stripping only *removes*
        # characters, so U+FFFC positions survive verbatim in the output.
        # Finally, recompute mention start offsets by scanning the final
        # plain_text for U+FFFC positions — one per placeholder, in order.
        text_with_ffc, tentative_mentions = self._compile_outbound_mentions(
            text,
        )
        plain_text, styles = _markdown_to_signal(text_with_ffc)
        text_style_params = (
            [f"{s['start']}:{s['length']}:{s['style']}" for s in styles]
            if styles
            else None
        )
        ffc_positions = [i for i, c in enumerate(plain_text) if c == "\ufffc"]
        mention_dicts: List[Dict[str, Any]] = []
        for pos, entry in zip(ffc_positions, tentative_mentions):
            entry = dict(entry)
            entry["start"] = pos
            mention_dicts.append(entry)
        text = plain_text

        chunks = self._chunk_text(text) if text.strip() else [""]
        for i, chunk in enumerate(chunks):
            atts = att_paths if i == 0 and att_paths else None

            chunk_mention_strs: Optional[List[str]] = None
            if mention_dicts and "\ufffc" in chunk:
                chunk_positions = [
                    idx for idx, ch in enumerate(chunk) if ch == "\ufffc"
                ]
                consumed_before = sum(
                    1 for c in "".join(chunks[:i]) if c == "\ufffc"
                )
                mentions_in_chunk = mention_dicts[
                    consumed_before : consumed_before + len(chunk_positions)
                ]
                chunk_mention_strs = []
                for pos, orig in zip(chunk_positions, mentions_in_chunk):
                    target = orig.get("number") or orig.get("uuid") or ""
                    if target:
                        chunk_mention_strs.append(f"{pos}:1:{target}")
                if not chunk_mention_strs:
                    chunk_mention_strs = None

            if not (chunk.strip() or atts):
                continue

            qt = (
                meta.get("quote_timestamp", 0) if self._reply_to_trigger else 0
            )
            qa = meta.get("quote_author", "") if self._reply_to_trigger else ""
            # Diagnostic: snapshot outbound text + mention params so we
            # can diff the "what the bot wrote" vs "what signal-cli got"
            # when mentions fail to render in the recipient's client.
            if chunk_mention_strs or "￼" in chunk:
                logger.info(
                    "signal: outbound chunk to %s chunk=%r mentions=%r "
                    "ffc_count=%d prefix_map_size=%d",
                    to_handle,
                    chunk,
                    chunk_mention_strs,
                    chunk.count("￼"),
                    len(self._uuid_prefix_lookup),
                )
            await self.client.send_message(
                to_handle,
                chunk,
                is_group=is_group,
                attachments=atts,
                text_style=text_style_params if i == 0 else None,
                mentions=chunk_mention_strs,
                quote_timestamp=qt if i == 0 else 0,
                quote_author=qa if i == 0 else "",
            )

    async def send_media(
        self,
        to_handle: str,
        part: OutgoingContentPart,
        meta: Optional[dict] = None,
    ) -> None:
        t = getattr(part, "type", None)
        if not self.enabled or not self.client.connected:
            return
        meta = meta or {}
        group_id = meta.get("group_id") or ""
        is_group = bool(group_id) or (
            to_handle.endswith("=") and not to_handle.startswith("+")
        )
        if is_group and group_id:
            to_handle = group_id

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
            return
        file_path = (
            raw_path.replace("file://", "")
            if isinstance(raw_path, str) and raw_path.startswith("file://")
            else raw_path
        )
        if not file_path or not os.path.isfile(file_path):
            logger.warning("signal: media file not found: %s", file_path)
            return
        await self.client.send_message(
            to_handle,
            "",
            is_group=is_group,
            attachments=[file_path],
        )

    async def send_reaction_to(
        self,
        to_handle: str,
        emoji: str,
        target_author: str,
        target_timestamp: int,
        is_group: bool = False,
    ) -> bool:
        return await self.client.send_reaction(
            to_handle,
            emoji,
            target_author,
            target_timestamp,
            is_group=is_group,
        )

    # ── Text chunking ────────────────────────────────────────────────

    def _chunk_text(self, text: str) -> List[str]:
        if not text or len(text) <= self._text_chunk_limit:
            return [text] if text else []
        chunks: List[str] = []
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

    # ── Typing indicator loop ────────────────────────────────────────

    # ── Typing indicator: public hooks for off-pipeline callers ────────
    # ``consume()`` already starts a typing loop tied to the inbound
    # request; these wrappers expose the same machinery to off-pipeline
    # dispatchers (Listen, cron, future proactive nudge) so they can
    # show a composing indicator while their background agent runs.

    async def start_typing(self, to_handle, meta=None):
        """Begin a composing loop for an off-pipeline dispatch.

        Resolves ``(target, is_group)`` from ``meta`` first, falling
        back to ``to_handle`` for DMs.  Returns the task handle (or
        None when typing is disabled / client missing) so callers can
        stop it in ``finally:``.
        """
        if not getattr(self, "_show_typing", True):
            return None
        client = getattr(self, "client", None)
        if client is None:
            return None
        meta = meta or {}
        target = (
            meta.get("_typing_target")
            or meta.get("group_id")
            or meta.get("source")
            or meta.get("source_uuid")
            or to_handle
        )
        if not target:
            return None
        is_group = bool(
            meta.get("_typing_is_group")
            or meta.get("is_group")
            or (meta.get("group_id") and not meta.get("source")),
        )
        try:
            return asyncio.create_task(self._typing_loop(target, is_group))
        except Exception:  # pylint: disable=broad-exception-caught
            return None

    async def _typing_loop(
        self,
        target: str,
        is_group: bool,
        interval: float = 4.0,
    ) -> None:
        try:
            while True:
                await self.client.send_typing(
                    target,
                    start=True,
                    is_group=is_group,
                )
                await asyncio.sleep(interval)
        except asyncio.CancelledError:
            await self.client.send_typing(
                target,
                start=False,
                is_group=is_group,
            )

    # ── Process-loop override ────────────────────────────────────────

    async def _stream_with_tracker(self, payload):
        request = self._payload_to_request(payload)
        send_meta = getattr(request, "channel_meta", None) or {}
        to_handle = self.get_to_handle_from_request(request)
        await self._before_consume_process(request)

        text_parts: List[str] = []
        message_completed = False
        process_iterator = None
        typing_task = None
        try:
            typing_target = send_meta.get("_typing_target")
            typing_is_group = send_meta.get("_typing_is_group", False)
            if typing_target and self._show_typing:
                typing_task = asyncio.create_task(
                    self._typing_loop(typing_target, typing_is_group),
                )

            process_iterator = self._process(request)
            async for event in process_iterator:
                if hasattr(event, "model_dump_json"):
                    data = event.model_dump_json()
                elif hasattr(event, "json"):
                    data = event.json()
                else:
                    data = json.dumps({"text": str(event)})
                yield f"data: {data}\n\n"

                obj = getattr(event, "object", None)
                status = getattr(event, "status", None)

                if obj == "message" and status == RunStatus.Completed:
                    # Drop REASONING-type messages from the
                    # channel send (Codex commentary, Claude
                    # thinking, Qwen / Mimo / DeepSeek
                    # ``reasoning_content``).  Console UI still
                    # receives the SSE yield above for thinking-
                    # pane rendering.
                    from agentscope_runtime.engine.schemas.agent_schemas import (  # noqa: E501
                        MessageType,
                    )

                    if getattr(event, "type", None) == MessageType.REASONING:
                        continue
                    logger.info(
                        "signal: message_completed, sending to %s",
                        to_handle,
                    )
                    await self.on_event_message_completed(
                        request,
                        to_handle,
                        event,
                        send_meta,
                    )
                    message_completed = True

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
                        pt = str(getattr(part, "type", ""))
                        if "thinking" in pt.lower():
                            continue
                    text_parts.append(txt)

            if text_parts and not message_completed:
                reply = "\n".join(text_parts)
                logger.info(
                    "signal: fallback sending reply (%d chars) to %s",
                    len(reply),
                    to_handle,
                )
                await self.send(to_handle, reply.strip(), send_meta)

            if self._on_reply_sent:
                args = self.get_on_reply_sent_args(request, to_handle)
                self._on_reply_sent(self.channel, *args)

            produced_reply = message_completed or bool(text_parts)
            ack_emoji = (
                self._ack_reaction_done
                if produced_reply
                else self._ack_reaction_error
            )
            if ack_emoji:
                await self._send_ack_reaction(send_meta, ack_emoji)

        except asyncio.CancelledError:
            if process_iterator:
                await process_iterator.aclose()
            raise
        except Exception:
            logger.exception("signal: _stream_with_tracker failed")
            if self._ack_reaction_error:
                await self._send_ack_reaction(
                    send_meta,
                    self._ack_reaction_error,
                )
            raise
        finally:
            if typing_task and not typing_task.done():
                typing_task.cancel()

    async def _send_ack_reaction(
        self,
        send_meta: Dict[str, Any],
        emoji: str,
    ) -> None:
        ack_target = send_meta.get("_ack_target")
        ack_author = send_meta.get("_ack_author")
        ack_ts = send_meta.get("_ack_timestamp")
        if ack_target and ack_author and ack_ts:
            try:
                await self.client.send_reaction(
                    ack_target,
                    emoji,
                    target_author=ack_author,
                    target_timestamp=int(ack_ts),
                    is_group=bool(send_meta.get("_typing_is_group")),
                )
            except Exception as e:
                logger.debug("signal: close reaction failed: %s", e)

    # ── Session / routing ────────────────────────────────────────────

    def resolve_session_id(
        self,
        sender_id: str,
        channel_meta: Optional[Dict[str, Any]] = None,
    ) -> str:
        meta = channel_meta or {}
        group_id = meta.get("group_id")
        if group_id:
            return f"signal:group:{group_id}"
        return f"signal:{sender_id}"

    def get_to_handle_from_request(self, request) -> str:
        meta = getattr(request, "channel_meta", None) or {}
        group_id = meta.get("group_id")
        if group_id:
            return group_id
        return meta.get("source") or getattr(request, "user_id", "") or ""
