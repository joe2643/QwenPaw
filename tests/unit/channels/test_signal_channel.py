# -*- coding: utf-8 -*-
# pylint: disable=protected-access
"""Unit tests for the Signal channel (subprocess JSON-RPC transport).

The transport (SignalSubprocessClient) is replaced with a fake that
records calls and lets tests push notifications — no real signal-cli
subprocess is spawned.
"""
from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock

import pytest

from qwenpaw.app.channels.signal.channel import (
    SignalChannel,
    _markdown_to_signal,
)


# ───────────────────────────── fakes ─────────────────────────────────


class FakeClient:
    """Records outbound calls, exposes `connected` flag, pushes
    notifications through the registered callback."""

    def __init__(self, *_args, **_kwargs):
        self.connected = False
        # Mirror the real SignalSubprocessClient init args that
        # SignalChannel.update_config compares against — without
        # these attributes the hard-field check raises
        # ``AttributeError: 'FakeClient' object has no attribute
        # '_signal_cli_path'`` instead of returning a clean True/False.
        self._signal_cli_path = _kwargs.get("signal_cli_path") or "signal-cli"
        self._extra_args: List[str] = list(_kwargs.get("extra_args") or [])
        self.sent: List[Dict[str, Any]] = []
        self.reactions: List[Dict[str, Any]] = []
        self.typing: List[Dict[str, Any]] = []
        self._on_notify = None

    async def connect(self, on_notify) -> bool:
        self._on_notify = on_notify
        self.connected = True
        return True

    async def disconnect(self) -> None:
        self.connected = False

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
        self.sent.append(
            {
                "target": target,
                "text": text,
                "is_group": is_group,
                "quote_timestamp": quote_timestamp,
                "quote_author": quote_author,
                "attachments": list(attachments or []),
                "text_style": list(text_style or []),
                "mentions": list(mentions or []),
            },
        )
        return 1_700_000_000

    async def send_reaction(self, *args, **kwargs) -> bool:
        self.reactions.append({"args": args, "kwargs": kwargs})
        return True

    async def send_typing(self, *args, **kwargs) -> None:
        self.typing.append({"args": args, "kwargs": kwargs})

    async def download_attachment(self, attachment_id, dest_dir):
        return None

    async def get_sticker(
        self,
        pack_id,
        sticker_id,
        dest_dir,
        *,
        pack_key=None,
    ):
        """Default behaviour: write a stub webp into dest_dir and
        return the path.  Tests that want to exercise failure paths
        override this attribute directly on the instance.
        """
        dest_dir.mkdir(parents=True, exist_ok=True)
        p = dest_dir / f"signal_sticker_{pack_id[:8]}_{sticker_id}.webp"
        p.write_bytes(b"RIFF\x00\x00\x00\x00WEBP")
        return p


# ───────────────────────────── helpers ───────────────────────────────


def _make_channel(**overrides: Any) -> SignalChannel:
    """Build a SignalChannel with the subprocess client swapped for a fake."""

    async def _noop_process(_request):
        if False:
            yield None  # pragma: no cover

    defaults: Dict[str, Any] = {
        "process": _noop_process,
        "enabled": True,
        "account": "+85251159218",
        "account_uuid": "82e0393a-1f09-4a0a-b000-000000000000",
    }
    defaults.update(overrides)
    ch = SignalChannel(**defaults)
    # Replace the real client with the fake
    ch.client = FakeClient()
    return ch


# ───────────────────────────── markdown ──────────────────────────────


def test_markdown_bold_produces_text_style_range() -> None:
    plain, styles = _markdown_to_signal("hello **world** foo")
    assert plain == "hello world foo"
    assert len(styles) == 1
    assert styles[0] == {"start": 6, "length": 5, "style": "BOLD"}


def test_markdown_header_becomes_bold() -> None:
    plain, styles = _markdown_to_signal("## Heading\nbody")
    assert plain == "Heading\nbody"
    assert styles and styles[0]["style"] == "BOLD"
    assert styles[0]["start"] == 0
    assert styles[0]["length"] == len("Heading")


def test_markdown_monospace_and_italic() -> None:
    plain, styles = _markdown_to_signal("run `foo` then *go*")
    assert plain == "run foo then go"
    kinds = sorted(s["style"] for s in styles)
    assert kinds == ["ITALIC", "MONOSPACE"]


# ───────────────────────────── reply-to (quote) ──────────────────────


async def test_quote_with_image_attachment_inlines_local_path(
    tmp_path,
) -> None:
    # Mirrors the WhatsApp channel fix: when the quoted message had an
    # image attachment and we successfully downloaded it, the
    # ``Media:`` label in the reply-to block must include the absolute
    # local path.  Without it the agent sees "Media: image" and has
    # nothing to hand to tools like codex image i2i / view_image.
    ch = _make_channel()
    ch._media_dir = tmp_path

    fake_local = tmp_path / "sig_quote_123.jpg"
    fake_local.write_bytes(b"\xff\xd8\xff" + b"x" * 64)

    # Override the download to return the path we just wrote.
    ch.client.download_attachment = AsyncMock(return_value=fake_local)

    data_message = {
        "quote": {
            "text": "earlier text",
            "author": "+85298765432",
            "authorUuid": "",
            "id": "quote-id-1",
            "mentions": [],
            "attachments": [
                {
                    "id": "att-123",
                    "contentType": "image/jpeg",
                    "fileName": "sig_quote_123.jpg",
                },
            ],
        },
    }

    parts = await ch._extract_quote_content(data_message)
    text_parts = [p for p in parts if hasattr(p, "text")]
    assert text_parts, "expected a reply-to text block"
    joined = " ".join(p.text for p in text_parts)
    assert "=== UNTRUSTED reply-to" in joined
    assert "Media: image:" in joined
    assert str(fake_local) in joined


async def test_quote_with_video_attachment_inlines_local_path(
    tmp_path,
) -> None:
    # Generalises the guarantee to video attachments — same reason:
    # the agent needs the path to invoke view_video downstream.
    ch = _make_channel()
    ch._media_dir = tmp_path
    fake_local = tmp_path / "sig_quote_vid.mp4"
    fake_local.write_bytes(b"x" * 1024)
    ch.client.download_attachment = AsyncMock(return_value=fake_local)

    data_message = {
        "quote": {
            "text": "check this clip",
            "author": "+85298765432",
            "authorUuid": "",
            "id": "quote-id-2",
            "mentions": [],
            "attachments": [
                {
                    "id": "att-vid",
                    "contentType": "video/mp4",
                    "fileName": "clip.mp4",
                },
            ],
        },
    }

    parts = await ch._extract_quote_content(data_message)
    text = next(p.text for p in parts if hasattr(p, "text"))
    assert "Media: video:" in text
    assert str(fake_local) in text


async def test_quote_with_sticker_inlines_image_and_emoji(tmp_path) -> None:
    """Quoted messages that reference a sticker must surface both
    the webp path (so the agent can reuse it with
    ``signal_send_sticker`` / view it) and the emoji (so the agent
    can grasp the tone of what the user is quoting).
    """
    ch = _make_channel()
    ch._media_dir = tmp_path

    data_message = {
        "quote": {
            "text": "",
            "author": "+85298765432",
            "authorUuid": "",
            "id": "quote-sticker-1",
            "sticker": {
                "packId": "PACKID" * 8,
                "packKey": "PACKKEY" * 8,
                "stickerId": 4,
                "emoji": "🦀",
            },
        },
    }

    parts = await ch._extract_quote_content(data_message)
    # Reply-to block should name the sticker with emoji + path.
    text = next(p.text for p in parts if hasattr(p, "text"))
    assert "Media: sticker 🦀" in text
    # An ImageContent block should have been appended for vision.
    images = [p for p in parts if getattr(p, "type", None) == "image"]
    assert len(images) == 1
    # The file written by FakeClient.get_sticker must be under
    # the channel's media_dir.
    sticker_files = list(tmp_path.glob("signal_sticker_*.webp"))
    assert len(sticker_files) == 1
    assert str(sticker_files[0]) in text


async def test_quote_with_sticker_fetch_failure_labels_as_failed(
    tmp_path,
) -> None:
    """When the sticker pack is unreachable (no key + RPC errors)
    we still surface the fact that the quoted message was a
    sticker, just without the path — better than silently
    dropping the quote context."""
    ch = _make_channel()
    ch._media_dir = tmp_path

    async def _fail(*_a, **_kw):
        return None

    ch.client.get_sticker = _fail

    parts = await ch._extract_quote_content(
        {
            "quote": {
                "text": "",
                "author": "+85298765432",
                "sticker": {
                    "packId": "PID" * 10,
                    "stickerId": 0,
                    "emoji": "😀",
                },
            },
        },
    )
    text = next(p.text for p in parts if hasattr(p, "text"))
    assert "Media: sticker 😀 (fetch failed)" in text


# ───────────────────────────── outbound mentions ─────────────────────


def test_compile_outbound_mentions_phone_bare() -> None:
    ch = _make_channel()
    text, mentions = ch._compile_outbound_mentions("Ping @+85298765432 now")
    assert text == "Ping \ufffc now"
    assert mentions == [{"start": 5, "length": 1, "number": "+85298765432"}]


def test_compile_outbound_mentions_uuid_bare() -> None:
    ch = _make_channel()
    text, mentions = ch._compile_outbound_mentions(
        "Hi @uuid:abc12345 there",
    )
    assert "\ufffc" in text
    # Unknown short-prefix passes through unchanged (resolver falls back
    # to the raw value when no prefix map entry exists).
    assert mentions and mentions[0]["uuid"].startswith("abc12345")


def test_compile_outbound_mentions_name_with_phone_parens() -> None:
    ch = _make_channel()
    text, mentions = ch._compile_outbound_mentions(
        "See @Joe (+85251159218)!",
    )
    assert text.count("\ufffc") == 1
    assert mentions == [{"start": 4, "length": 1, "number": "+85251159218"}]


def test_compile_outbound_mentions_resolves_short_uuid_to_full() -> None:
    """Short uuid:<8-char> prefix expands to full ACI via the channel's
    prefix lookup, populated as senders appear in a group. Without this
    outgoing mentions would ship to signal-cli with an 8-char ACI —
    not a valid contact, so the mention silently fails and the
    recipient sees raw UUID text."""
    ch = _make_channel()
    full = "82e0393a-4c79-4905-b84d-986298f4f8c5"
    ch._remember_sender("", full, "Alice")
    text, mentions = ch._compile_outbound_mentions(
        f"Tag @Alice (uuid:{full[:8]}) please",
    )
    assert mentions and mentions[0]["uuid"] == full


def test_compile_outbound_mentions_keeps_full_uuid_as_is() -> None:
    """Full UUID already in text → pass through without resolver lookup."""
    ch = _make_channel()
    full = "1a2b3c4d-1111-2222-3333-444455556666"
    text, mentions = ch._compile_outbound_mentions(
        f"Heads-up @Bob (uuid:{full})",
    )
    assert mentions and mentions[0]["uuid"] == full


# ───────────────────────────── outbound send ─────────────────────────


@pytest.mark.asyncio
async def test_send_sets_target_and_style_and_mention_params() -> None:
    ch = _make_channel()
    ch.client.connected = True  # pretend the subprocess is up

    await ch.send(
        "+85298765432",
        "Hey @+85298349370, please run **this**",
        meta={"group_id": "", "quote_timestamp": 0, "quote_author": ""},
    )
    assert len(ch.client.sent) == 1
    frame = ch.client.sent[0]
    assert frame["target"] == "+85298765432"
    assert frame["is_group"] is False
    # Placeholder present in the sent text (U+FFFC)
    assert "\ufffc" in frame["text"]
    # Bold style in `text_style`
    assert any(s.endswith(":BOLD") for s in frame["text_style"])
    # Mention compiled to "start:1:+number" form
    assert any(m.split(":")[-1] == "+85298349370" for m in frame["mentions"])


@pytest.mark.asyncio
async def test_send_group_uses_group_id_as_target() -> None:
    ch = _make_channel()
    ch.client.connected = True
    await ch.send(
        "somehandle",
        "hello",
        meta={"group_id": "ZW5jb2RlZC1ncm91cC1pZA==", "quote_timestamp": 0},
    )
    assert ch.client.sent[0]["target"] == "ZW5jb2RlZC1ncm91cC1pZA=="
    assert ch.client.sent[0]["is_group"] is True


@pytest.mark.asyncio
async def test_send_includes_quote_when_reply_to_trigger_on() -> None:
    ch = _make_channel(reply_to_trigger=True)
    ch.client.connected = True
    await ch.send(
        "+85298765432",
        "ack",
        meta={
            "group_id": "",
            "quote_timestamp": 1_700_000_000,
            "quote_author": "+85298765432",
        },
    )
    sent = ch.client.sent[0]
    assert sent["quote_timestamp"] == 1_700_000_000
    assert sent["quote_author"] == "+85298765432"


@pytest.mark.asyncio
async def test_send_disabled_channel_is_noop() -> None:
    ch = _make_channel(enabled=False)
    ch.client.connected = True
    await ch.send("+85298765432", "should not send", meta={})
    assert ch.client.sent == []


# ───────────────────────────── inbound ───────────────────────────────


@pytest.mark.asyncio
# ───────────────────────────── update_config ─────────────────────────


def test_signal_channel_requests_sequential_restart() -> None:
    """Two signal-cli daemons cannot share an account — the default
    zero-downtime ``replace_channel`` (start-new-then-stop-old)
    contests the lock and stalls RPCs in the overlap window.  This
    flag pins the channel to the safe stop-old-then-start-new path."""
    from qwenpaw.app.channels.signal.channel import SignalChannel

    assert SignalChannel.requires_sequential_restart is True


@pytest.mark.asyncio
async def test_update_config_soft_patches_runtime_fields() -> None:
    """Fields that Python reads at request time (read-receipts,
    chunk limit, allowlists, ack reactions, etc.) get patched into
    the live channel WITHOUT bouncing the signal-cli subprocess.
    Before this, every Console save triggered a daemon respawn +
    file-lock contest just to flip ``send_read_receipts``."""
    ch = _make_channel(send_read_receipts=True, text_chunk_limit=4000)
    ch.client.connected = True

    res = await ch.update_config(
        {
            "enabled": True,
            "account": "+85251159218",  # same as fixture default
            "signal_cli_path": "signal-cli",
            "send_read_receipts": False,
            "text_chunk_limit": 1234,
            "ack_reaction_thinking": "✨",
            "groups": ["GROUPABC=="],
            "group_allow_from": ["+85299999999"],
            "dm_policy": "allowlist",
            "group_policy": "allowlist",
            "allow_from": ["+85211111111"],
            "require_mention": True,
            "reply_to_trigger": False,
        },
    )
    assert res is True, "soft-patchable fields must NOT request a restart"
    assert ch._send_read_receipts is False
    assert ch._text_chunk_limit == 1234
    assert ch._ack_reaction_thinking == "✨"
    assert ch._groups == ["GROUPABC=="]
    assert ch._group_allow_from == ["+85299999999"]
    assert ch.dm_policy == "allowlist"
    assert ch.allow_from == ["+85211111111"]
    assert ch.require_mention is True
    assert ch._reply_to_trigger is False


@pytest.mark.parametrize(
    "hard_field,changed_value",
    [
        ("enabled", False),
        ("account", "+85299999999"),
        ("signal_cli_path", "/opt/different/signal-cli"),
        ("extra_args", ["--trust-new-identities", "always"]),
    ],
)
@pytest.mark.asyncio
async def test_update_config_returns_false_on_hard_field_change(
    hard_field,
    changed_value,
) -> None:
    """Hard fields are baked into the signal-cli spawn cmdline (or
    determine whether to spawn at all) — patching them in-place
    would leave a stale daemon talking to the wrong account.  The
    channel must fall through to a full restart in this case."""
    ch = _make_channel()
    ch.client.connected = True

    cfg = {
        "enabled": True,
        "account": "+85251159218",
        "signal_cli_path": "signal-cli",
        "extra_args": [],
    }
    cfg[hard_field] = changed_value
    res = await ch.update_config(cfg)
    assert (
        res is False
    ), f"changing hard field {hard_field!r} must trigger restart"


@pytest.mark.asyncio
async def test_update_config_returns_false_when_subprocess_dead() -> None:
    """If signal-cli isn't connected, in-place patch would just
    perpetuate a broken channel.  Force a full restart so the
    daemon respawns and the file lock is re-acquired cleanly."""
    ch = _make_channel()
    ch.client.connected = False  # subprocess crashed or never connected

    res = await ch.update_config(
        {
            "enabled": True,
            "account": "+85251159218",
            "signal_cli_path": "signal-cli",
        },
    )
    assert res is False


# ───────────────────────────── inbound flow ──────────────────────────


async def test_inbound_dm_enqueues_agent_request() -> None:
    enqueue_calls: List[Any] = []

    ch = _make_channel()
    ch._enqueue = enqueue_calls.append  # attach fake enqueue
    ch.client.connected = True

    notification = {
        "envelope": {
            "sourceNumber": "+85298765432",
            "sourceUuid": "abcd1234-0000-0000-0000-000000000000",
            "sourceName": "Alice",
            "timestamp": 1_700_000_000,
            "dataMessage": {"message": "hi bot"},
        },
    }
    await ch._on_notification(notification)

    assert len(enqueue_calls) == 1
    req = enqueue_calls[0]
    meta = req.channel_meta
    assert meta["platform"] == "signal"
    assert meta["group_id"] == ""
    assert meta["source"] == "+85298765432"


@pytest.mark.asyncio
async def test_group_without_mention_buffers_into_history() -> None:
    ch = _make_channel(require_mention=True, group_policy="open")
    ch._enqueue = lambda _r: (_ for _ in ()).throw(
        AssertionError("should NOT enqueue when unmentioned"),
    )
    ch.client.connected = True

    group_id = "GID=="
    await ch._on_notification(
        {
            "envelope": {
                "sourceNumber": "+85211111111",
                "sourceName": "Bob",
                "timestamp": 1,
                "dataMessage": {
                    "message": "casual chat",
                    "groupInfo": {"groupId": group_id},
                },
            },
        },
    )
    # Nothing sent, but history now has the line
    assert group_id in ch._group_history
    assert ch._group_history[group_id][0]["body"] == "casual chat"


@pytest.mark.asyncio
async def test_group_with_mention_enqueues_and_injects_history() -> None:
    enqueue_calls: List[Any] = []
    ch = _make_channel(require_mention=True)
    ch._enqueue = enqueue_calls.append
    ch.client.connected = True

    group_id = "GID=="
    # Prior unmentioned chatter → captured as history
    await ch._on_notification(
        {
            "envelope": {
                "sourceNumber": "+85211111111",
                "sourceName": "Bob",
                "timestamp": 1,
                "dataMessage": {
                    "message": "weather is nice",
                    "groupInfo": {"groupId": group_id},
                },
            },
        },
    )
    # Now a mention triggers the bot
    await ch._on_notification(
        {
            "envelope": {
                "sourceNumber": "+85222222222",
                "sourceName": "Carol",
                "timestamp": 2,
                "dataMessage": {
                    "message": "@+85251159218 summarise",
                    "mentions": [
                        {"number": "+85251159218", "start": 0, "length": 1},
                    ],
                    "groupInfo": {"groupId": group_id},
                },
            },
        },
    )
    assert len(enqueue_calls) == 1
    # Collect text chunks from the enqueued request and assert that the
    # group-history context snippet actually made it into the prompt.
    texts = (
        [
            p.text
            for p in enqueue_calls[0].input[-1].content
            if hasattr(p, "text") and p.text
        ]
        if hasattr(enqueue_calls[0], "input")
        else []
    )
    assert any(
        "weather is nice" in t for t in texts
    ), "group-history context should have been injected into the prompt"
    # History should be drained after injection
    assert ch._group_history.get(group_id) == []


# ───────────────────────────── inbound sticker ───────────────────────


@pytest.mark.asyncio
async def test_inbound_sticker_only_message_enqueues_with_hint(
    tmp_path,
) -> None:
    """Sticker-only envelope (no body, no attachments) is enqueued
    with both a `[Signal sticker … at <path>]` hint TextContent and
    the actual ImageContent block — mirrors how the WhatsApp
    channel surfaces stickers so the agent can decide to reply in
    kind.
    """
    enqueue_calls: List[Any] = []

    ch = _make_channel()
    ch._media_dir = tmp_path / "media"
    ch._enqueue = enqueue_calls.append
    ch.client.connected = True

    await ch._on_notification(
        {
            "envelope": {
                "sourceNumber": "+85298765432",
                "sourceUuid": "abcd1234-0000-0000-0000-000000000000",
                "sourceName": "Alice",
                "timestamp": 1_700_000_000,
                "dataMessage": {
                    "sticker": {
                        "packId": "abcdef0123456789" * 2,
                        "packKey": "fedcba9876543210" * 4,
                        "stickerId": 7,
                        "emoji": "🔥",
                    },
                },
            },
        },
    )

    assert len(enqueue_calls) == 1
    req = enqueue_calls[0]
    # Find the TextContent + ImageContent pair the channel builds
    # from the sticker reference.  The envelope-prefix hint
    # rewrites the first TextContent to prefix the sender, which
    # is why we scan instead of hard-indexing.
    content = req.input[-1].content
    texts = [c.text for c in content if getattr(c, "type", None) == "text"]
    images = [c for c in content if getattr(c, "type", None) == "image"]
    assert any("Signal sticker" in t for t in texts), texts
    assert any("🔥" in t for t in texts), "emoji should survive"
    # Path in the hint should point at the file we wrote into the
    # channel's media_dir.
    sticker_files = list((tmp_path / "media").glob("signal_sticker_*.webp"))
    assert len(sticker_files) == 1
    assert any(str(sticker_files[0]) in t for t in texts)
    assert len(images) == 1


@pytest.mark.asyncio
async def test_inbound_sticker_without_emoji_omits_emoji_token(
    tmp_path,
) -> None:
    """No emoji → hint reads `[Signal sticker at <path>]`."""
    enqueue_calls: List[Any] = []
    ch = _make_channel()
    ch._media_dir = tmp_path / "media"
    ch._enqueue = enqueue_calls.append
    ch.client.connected = True

    await ch._on_notification(
        {
            "envelope": {
                "sourceNumber": "+85298765432",
                "timestamp": 1_700_000_000,
                "dataMessage": {
                    "sticker": {
                        "packId": "abc" * 10,
                        "packKey": "def" * 20,
                        "stickerId": 0,
                    },
                },
            },
        },
    )

    req = enqueue_calls[0]
    texts = [
        c.text
        for c in req.input[-1].content
        if getattr(c, "type", None) == "text"
    ]
    hint = next(t for t in texts if "Signal sticker" in t)
    # No emoji glyph between the word and "at"
    assert " at " in hint
    assert "🔥" not in hint


@pytest.mark.asyncio
async def test_inbound_sticker_fetch_failure_drops_sticker_silently(
    tmp_path,
) -> None:
    """When get_sticker returns None the message continues with
    whatever body/attachments exist — we don't want a single bad
    sticker to black-hole the text the user typed alongside it.
    """
    enqueue_calls: List[Any] = []
    ch = _make_channel()
    ch._media_dir = tmp_path / "media"
    ch._enqueue = enqueue_calls.append
    ch.client.connected = True

    async def _fail_sticker(*_args, **_kwargs):
        return None

    ch.client.get_sticker = _fail_sticker

    await ch._on_notification(
        {
            "envelope": {
                "sourceNumber": "+85298765432",
                "timestamp": 1_700_000_000,
                "dataMessage": {
                    "message": "check this out",
                    "sticker": {
                        "packId": "abc" * 10,
                        "packKey": "def" * 20,
                        "stickerId": 0,
                    },
                },
            },
        },
    )

    assert len(enqueue_calls) == 1
    texts = [
        c.text
        for c in enqueue_calls[0].input[-1].content
        if getattr(c, "type", None) == "text"
    ]
    assert any("check this out" in t for t in texts)
    assert not any("Signal sticker" in t for t in texts)


# ───────────────────────────── allowlist ─────────────────────────────


def test_is_source_allowed_phone() -> None:
    ch = _make_channel(dm_policy="allowlist", allow_from=["+85298765432"])
    assert ch._is_source_allowed("+85298765432", "")
    assert not ch._is_source_allowed("+85299999999", "")


def test_is_source_allowed_uuid_prefix() -> None:
    ch = _make_channel(
        dm_policy="allowlist",
        allow_from=["uuid:abcd1234-0000-0000-0000-000000000000"],
    )
    assert ch._is_source_allowed("", "abcd1234-0000-0000-0000-000000000000")
    assert not ch._is_source_allowed(
        "",
        "11111111-0000-0000-0000-000000000000",
    )


@pytest.mark.asyncio
async def test_dm_allowlist_empty_rejects_all() -> None:
    """Regression: DM `allowlist` policy + empty `allow_from` blocks all.

    Earlier the guard was
    ``if self.dm_policy == "allowlist" and self.allow_from``, which
    short-circuited on an empty list — so setting policy to allowlist
    and clearing the list silently let every DM through. Ensure the
    check now runs unconditionally for allowlist mode.
    """
    enqueue_calls: List[Any] = []
    ch = _make_channel(dm_policy="allowlist", allow_from=[])
    ch._enqueue = enqueue_calls.append
    ch.client.connected = True

    notification = {
        "envelope": {
            "sourceNumber": "+85299999999",
            "sourceUuid": "ffffffff-0000-0000-0000-000000000000",
            "sourceName": "Stranger",
            "timestamp": 1_700_000_000,
            "dataMessage": {"message": "hi bot"},
        },
    }
    await ch._on_notification(notification)

    # Message should be dropped — nothing enqueued to the agent runner.
    assert enqueue_calls == [], (
        "DM from non-allowlisted sender was not blocked when allow_from "
        "is empty + dm_policy=allowlist"
    )


# ───────────────────────────── bot self-mention strip ────────────────


def test_strip_bot_self_mention_plain_phone() -> None:
    ch = _make_channel()
    stripped = ch._strip_bot_self_mention("@+85251159218 /stop now")
    assert stripped == "/stop now"


def test_strip_bot_self_mention_name_with_id() -> None:
    ch = _make_channel()
    stripped = ch._strip_bot_self_mention("@Bot (+85251159218) hello")
    assert stripped == "hello"


# ───────────────────────────── data_dir resolution ───────────────────


def test_data_dir_resolves_from_explicit(tmp_path) -> None:
    from qwenpaw.app.channels.signal.channel import _resolve_signal_data_dir

    custom = tmp_path / "custom-dir"
    resolved = _resolve_signal_data_dir(str(custom), None)
    assert resolved == custom


def test_data_dir_resolves_from_workspace_dir(tmp_path) -> None:
    from qwenpaw.app.channels.signal.channel import _resolve_signal_data_dir

    ws = tmp_path / "agent-ws"
    resolved = _resolve_signal_data_dir("", ws)
    assert resolved == ws / "credentials" / "signal" / "default"


def test_data_dir_resolves_from_working_dir_fallback() -> None:
    """With no workspace and no explicit path, falls back to WORKING_DIR."""
    from qwenpaw.app.channels.signal.channel import (
        _resolve_signal_data_dir,
        _DEFAULT_DATA_DIR,
    )

    resolved = _resolve_signal_data_dir("", None)
    assert resolved == _DEFAULT_DATA_DIR


def test_data_dir_explicit_expands_tilde(tmp_path) -> None:
    """Explicit ``~/foo`` is expanded to the user's home."""
    from pathlib import Path as _P
    from qwenpaw.app.channels.signal.channel import _resolve_signal_data_dir

    resolved = _resolve_signal_data_dir("~/foo", None)
    assert str(resolved).startswith(str(_P.home()))
    assert resolved.name == "foo"


def test_channel_stores_resolved_data_dir(tmp_path) -> None:
    """SignalChannel(workspace_dir=...) picks up the workspace-scoped default."""
    ws = tmp_path / "ws"
    ch = _make_channel(workspace_dir=ws, account="+85251159218")
    assert ch._data_dir == ws / "credentials" / "signal" / "default"


# ───────────────────────────── subprocess -c flag ────────────────────


def test_subprocess_cmd_includes_config_flag(tmp_path) -> None:
    """data_dir string/Path produces `-c <path>` before `-a`."""
    from qwenpaw.app.channels.signal.subprocess_client import (
        SignalSubprocessClient,
    )

    target = tmp_path / "signal-data"
    client = SignalSubprocessClient(
        account="+85251159218",
        data_dir=target,
    )
    cmd = client._build_cmd()
    assert "-c" in cmd
    c_idx = cmd.index("-c")
    assert cmd[c_idx + 1] == str(target)
    # -c must come before -a (signal-cli CLI requires it that way).
    assert c_idx < cmd.index("-a")


def test_subprocess_cmd_no_config_flag_when_unset() -> None:
    """When data_dir is None or empty, -c flag is omitted (backward compat)."""
    from qwenpaw.app.channels.signal.subprocess_client import (
        SignalSubprocessClient,
    )

    for v in (None, ""):
        client = SignalSubprocessClient(account="+1", data_dir=v)
        cmd = client._build_cmd()
        assert "-c" not in cmd, f"-c should not appear for data_dir={v!r}"


# ───────────────────────────── download_attachment dest_dir ─────────


async def test_download_attachment_copies_into_dest_dir(
    tmp_path,
    monkeypatch,
) -> None:
    # Without this copy the returned path sits under
    # ~/.local/share/signal-cli/attachments/, which isn't in the
    # media server's allowed_dirs — /sign 403s, resolve_media_url
    # falls back to the raw path, and remote LLMs can't reach the
    # file.  This test locks the invariant: returned path must be
    # under dest_dir (signed-URL-eligible) when signal-cli has
    # autosaved the file.
    from qwenpaw.app.channels.signal.subprocess_client import (
        SignalSubprocessClient,
    )

    # Spoof the signal-cli autosave location with a temp dir we
    # actually own (the real code reads ``Path.home() / .local ...``).
    fake_autosave = tmp_path / "signal_autosave"
    fake_autosave.mkdir()
    att_id = "AtXxXzQq123.jpg"
    (fake_autosave / att_id).write_bytes(b"\xff\xd8\xff" + b"x" * 64)
    monkeypatch.setattr(
        "qwenpaw.app.channels.signal.subprocess_client.Path.home",
        lambda: tmp_path,
    )
    # Path.home() / .local / share / signal-cli / attachments →
    # tmp_path / .local / share / signal-cli / attachments.  Make it
    # point at our seeded autosave dir.
    real_dir = tmp_path / ".local" / "share" / "signal-cli" / "attachments"
    real_dir.mkdir(parents=True, exist_ok=True)
    (real_dir / att_id).write_bytes((fake_autosave / att_id).read_bytes())

    client = SignalSubprocessClient(account="+1")
    dest_dir = tmp_path / "copaw_media"

    result = await client.download_attachment(att_id, dest_dir)
    assert result is not None
    # The returned path MUST be under dest_dir, not the autosave dir.
    assert str(result).startswith(str(dest_dir))
    assert result.is_file()
    # Same bytes — not a zero-length stub.
    assert result.read_bytes() == b"\xff\xd8\xff" + b"x" * 64


# ───────────────────────────── subprocess client: get_sticker ───────


async def test_get_sticker_writes_under_dest_dir(tmp_path) -> None:
    """Happy path: RPC returns base64 bytes, get_sticker persists
    them under dest_dir with a deterministic filename and the pack
    key is NOT needed for the first attempt.
    """
    import base64 as _b64
    from qwenpaw.app.channels.signal.subprocess_client import (
        SignalSubprocessClient,
    )

    client = SignalSubprocessClient(account="+1")
    # Bypass the "connected" gate by stubbing .call directly.
    rpc_calls: List[Dict[str, Any]] = []

    async def _fake_call(method, params=None, timeout=None):
        rpc_calls.append({"method": method, "params": params})
        if method == "getSticker":
            return {"data": _b64.b64encode(b"WEBPBYTES").decode("ascii")}
        raise AssertionError(f"unexpected method {method}")

    client.call = _fake_call  # type: ignore[method-assign]

    dest_dir = tmp_path / "m"
    result = await client.get_sticker(
        "PACKHEXID" * 4,
        3,
        dest_dir,
    )
    assert result is not None
    assert result.parent == dest_dir
    assert result.name.endswith("_3.webp")
    assert result.read_bytes() == b"WEBPBYTES"
    # Only one RPC — pack was already local, no addStickerPack fallback.
    assert [c["method"] for c in rpc_calls] == ["getSticker"]


async def test_get_sticker_falls_back_to_add_pack_then_retries(
    tmp_path,
) -> None:
    """getSticker raises (pack unknown) → addStickerPack → getSticker
    succeeds.  Uses the fragment-URI format Signal Desktop / signal-cli
    expect (``pack_id=...&pack_key=...`` in the URL hash).
    """
    import base64 as _b64
    from qwenpaw.app.channels.signal.subprocess_client import (
        SignalSubprocessClient,
    )

    client = SignalSubprocessClient(account="+1")
    calls: List[Dict[str, Any]] = []
    attempt = {"n": 0}

    async def _fake_call(method, params=None, timeout=None):
        calls.append({"method": method, "params": params})
        if method == "getSticker":
            attempt["n"] += 1
            if attempt["n"] == 1:
                raise RuntimeError("StickerPackNotFoundException")
            return _b64.b64encode(b"OK").decode("ascii")
        if method == "addStickerPack":
            return None
        raise AssertionError(f"unexpected method {method}")

    client.call = _fake_call  # type: ignore[method-assign]

    result = await client.get_sticker(
        "PACKID" * 10,
        0,
        tmp_path,
        pack_key="PACKKEY" * 10,
    )
    assert result is not None
    assert result.read_bytes() == b"OK"
    # Order: getSticker (fail) → addStickerPack → getSticker (ok)
    assert [c["method"] for c in calls] == [
        "getSticker",
        "addStickerPack",
        "getSticker",
    ]
    uri = calls[1]["params"]["uri"]
    assert uri.startswith("https://signal.art/addstickers/#")
    assert "pack_id=PACKID" in uri
    assert "pack_key=PACKKEY" in uri


async def test_get_sticker_cache_hit_skips_rpc(tmp_path) -> None:
    """Pre-existing webp at the deterministic cache path must be
    returned immediately without any ``getSticker`` RPC — Signal
    packs are protocol-immutable so cached bytes are always
    current.  Without this short-circuit every preview burns
    50-300 KB of base64 + a round-trip."""
    from qwenpaw.app.channels.signal.subprocess_client import (
        SignalSubprocessClient,
    )

    client = SignalSubprocessClient(account="+1")

    # Seed the cache file with the exact name ``get_sticker`` uses.
    pack_id = "ABCDEF1234567890" * 2
    cached = tmp_path / f"signal_sticker_{pack_id[:8]}_7.webp"
    cached.write_bytes(b"cached-webp-bytes")

    calls: List[str] = []

    async def _fake_call(method, params=None, timeout=None):
        calls.append(method)
        raise AssertionError("RPC should not fire on cache hit")

    client.call = _fake_call  # type: ignore[method-assign]

    result = await client.get_sticker(pack_id, 7, tmp_path)
    assert result == cached
    assert calls == [], f"expected zero RPCs, got {calls}"


async def test_get_sticker_refresh_bypasses_cache(tmp_path) -> None:
    """``refresh=True`` forces a re-fetch even when the cache file
    exists — used when a previous write was truncated (e.g.
    ``readline()`` overflow before we bumped the buffer)."""
    import base64 as _b64
    from qwenpaw.app.channels.signal.subprocess_client import (
        SignalSubprocessClient,
    )

    client = SignalSubprocessClient(account="+1")
    pack_id = "FFFFFFFF" + "0" * 24
    cached = tmp_path / f"signal_sticker_{pack_id[:8]}_0.webp"
    cached.write_bytes(b"stale")

    async def _fake_call(method, params=None, timeout=None):
        assert method == "getSticker"
        return {"data": _b64.b64encode(b"fresh-bytes").decode("ascii")}

    client.call = _fake_call  # type: ignore[method-assign]

    result = await client.get_sticker(
        pack_id,
        0,
        tmp_path,
        refresh=True,
    )
    assert result == cached
    assert cached.read_bytes() == b"fresh-bytes"


async def test_get_sticker_cache_miss_ignores_empty_file(tmp_path) -> None:
    """A zero-byte cache file (e.g. partial write from a previous
    crash) must not count as cached — re-fetch and overwrite."""
    import base64 as _b64
    from qwenpaw.app.channels.signal.subprocess_client import (
        SignalSubprocessClient,
    )

    client = SignalSubprocessClient(account="+1")
    pack_id = "1234" + "0" * 28
    cached = tmp_path / f"signal_sticker_{pack_id[:8]}_2.webp"
    cached.write_bytes(b"")  # truncated

    async def _fake_call(method, params=None, timeout=None):
        return {"data": _b64.b64encode(b"fresh-bytes").decode("ascii")}

    client.call = _fake_call  # type: ignore[method-assign]

    result = await client.get_sticker(pack_id, 2, tmp_path)
    assert result == cached
    assert cached.read_bytes() == b"fresh-bytes"


async def test_get_sticker_returns_none_without_key_when_fetch_fails(
    tmp_path,
) -> None:
    """No pack_key → we can't install the pack, so the single
    getSticker failure is terminal (caller should fall through to
    its own handling — e.g. drop the sticker block).
    """
    from qwenpaw.app.channels.signal.subprocess_client import (
        SignalSubprocessClient,
    )

    client = SignalSubprocessClient(account="+1")

    async def _fake_call(method, params=None, timeout=None):
        raise RuntimeError("nope")

    client.call = _fake_call  # type: ignore[method-assign]

    result = await client.get_sticker("PID" * 10, 0, tmp_path)
    assert result is None


# ───────────────────────── subprocess client: orphan reap ──────────


async def test_reap_account_orphans_sigterms_matches_and_noops_otherwise(
    monkeypatch,
) -> None:
    """The pre-spawn reap step must target ONLY signal-cli pids that
    match our account AND aren't the current self._proc — killing
    anything else would be a footgun for devs running parallel
    signal-cli on other accounts."""
    import sys

    if not sys.platform.startswith("linux"):
        pytest.skip("orphan reap is Linux-only")

    from qwenpaw.app.channels.signal import subprocess_client as sc

    client = sc.SignalSubprocessClient(account="+1")
    client._proc = None

    kills: List[Dict[str, Any]] = []

    def _fake_kill(pid, sig):
        kills.append({"pid": pid, "sig": int(sig)})

    monkeypatch.setattr(sc.os, "kill", _fake_kill)

    # First probe returns an orphan; second probe (after grace) says
    # it's gone — simulating SIGTERM taking effect quickly.
    state = {"calls": 0}

    def _fake_iter(account):
        state["calls"] += 1
        if state["calls"] == 1:
            return [9999]
        return []

    monkeypatch.setattr(sc, "_iter_signal_cli_pids", _fake_iter)

    await client._reap_account_orphans()

    assert len(kills) == 1
    assert kills[0]["pid"] == 9999
    # First call should be SIGTERM, not SIGKILL.
    import signal as _sig

    assert kills[0]["sig"] == int(_sig.SIGTERM)


async def test_reap_account_orphans_escalates_to_sigkill(monkeypatch) -> None:
    """An orphan that shrugs off SIGTERM must get SIGKILL — otherwise
    the next signal-cli stalls on the file lock indefinitely."""
    import sys

    if not sys.platform.startswith("linux"):
        pytest.skip("orphan reap is Linux-only")

    from qwenpaw.app.channels.signal import subprocess_client as sc

    client = sc.SignalSubprocessClient(account="+1")
    client._proc = None

    kills: List[Dict[str, Any]] = []

    def _fake_kill(pid, sig):
        kills.append({"pid": pid, "sig": int(sig)})

    monkeypatch.setattr(sc.os, "kill", _fake_kill)

    # Orphan survives all probes — must end in SIGKILL.
    monkeypatch.setattr(
        sc,
        "_iter_signal_cli_pids",
        lambda _acct: [4242],
    )

    # Speed up the wait loop so the test isn't slow.
    async def _fast_sleep(_):
        return None

    monkeypatch.setattr(sc.asyncio, "sleep", _fast_sleep)

    await client._reap_account_orphans()

    import signal as _sig

    sigs = [k["sig"] for k in kills]
    assert int(_sig.SIGTERM) in sigs
    assert int(_sig.SIGKILL) in sigs


async def test_reap_ignores_self_child(monkeypatch) -> None:
    """Our own live child (self._proc.pid) must never be treated as
    an orphan — that would just kill the signal-cli we want to
    keep."""
    import sys

    if not sys.platform.startswith("linux"):
        pytest.skip("orphan reap is Linux-only")

    from qwenpaw.app.channels.signal import subprocess_client as sc

    client = sc.SignalSubprocessClient(account="+1")

    class _FakeProc:
        pid = 12345

    client._proc = _FakeProc()  # type: ignore[assignment]

    kills: List[int] = []
    monkeypatch.setattr(sc.os, "kill", lambda pid, sig: kills.append(pid))
    monkeypatch.setattr(
        sc,
        "_iter_signal_cli_pids",
        lambda _acct: [12345, 67890],
    )

    async def _fast_sleep(_):
        return None

    monkeypatch.setattr(sc.asyncio, "sleep", _fast_sleep)
    # Also keep the post-SIGTERM probe deterministic.
    call = {"n": 0}

    def _iter(_acct):
        call["n"] += 1
        if call["n"] == 1:
            return [12345, 67890]
        return [12345]  # our child still running; orphan gone

    monkeypatch.setattr(sc, "_iter_signal_cli_pids", _iter)

    await client._reap_account_orphans()
    assert kills == [
        67890,
    ], f"expected only the orphan to be signalled, got {kills}"


async def test_iter_signal_cli_pids_filters_own_children_via_ppid(
    tmp_path,
    monkeypatch,
) -> None:
    """The crash loop in prod was caused by ``_iter_signal_cli_pids``
    returning every signal-cli matching the account, including
    children of our own copaw process (i.e. spawns from a parallel
    supervise task during agent hot-reload).  The reaper then
    SIGTERM'd those, killing live signal-cli daemons every <1s.

    Real fix: filter by ``PPid`` from ``/proc/<pid>/status``.  A
    process whose parent IS the current copaw is never an orphan.
    Only init-adopted (PPid==1) or other-parent processes qualify
    for reaping.
    """
    import os
    import sys

    if not sys.platform.startswith("linux"):
        pytest.skip("/proc-based reap is Linux-only")

    from qwenpaw.app.channels.signal import subprocess_client as sc

    fake_proc = tmp_path / "proc"
    fake_proc.mkdir()
    my_pid = os.getpid()
    my_uid = os.getuid()

    def _make_pid(pid: int, ppid: int, account: str) -> None:
        d = fake_proc / str(pid)
        d.mkdir()
        (d / "status").write_text(
            f"Name:\tjava\nUid:\t{my_uid}\t{my_uid}\t{my_uid}\t{my_uid}\n"
            f"PPid:\t{ppid}\n",
            encoding="utf-8",
        )
        # cmdline is null-separated argv.
        argv = [
            "signal-cli",
            "-c",
            "/data",
            "-a",
            account,
            "--output=json",
            "jsonRpc",
        ]
        (d / "cmdline").write_bytes(b"\x00".join(a.encode() for a in argv))

    # Three candidates on the same account:
    #   our own child  (PPid = copaw)        → must NOT be reaped
    #   reparented orphan (PPid = 1)         → must be reaped
    #   live foreign signal-cli (PPid = 999) → must be reaped
    _make_pid(11111, my_pid, "+8521")  # our child
    _make_pid(22222, 1, "+8521")  # init-adopted orphan
    _make_pid(33333, 999, "+8521")  # foreign-parent
    _make_pid(44444, my_pid, "+8522")  # different account, our child
    _make_pid(55555, 1, "+8522")  # different account orphan

    hits = sc._iter_signal_cli_pids(
        "+8521",
        _proc_dir_override=fake_proc,
    )
    # Our own child must be excluded; orphan + foreign-parent included.
    assert sorted(hits) == [
        22222,
        33333,
    ], f"expected only orphans on +8521, got {sorted(hits)}"
    # Different-account scan must skip the wrong-account orphan +
    # our own child, returning only the init-adopted orphan on the
    # second account.
    hits_other = sc._iter_signal_cli_pids(
        "+8522",
        _proc_dir_override=fake_proc,
    )
    assert hits_other == [
        55555,
    ], f"expected only init-adopted orphan on +8522, got {hits_other}"


# ───────────────────────── subprocess client: pack RPCs ─────────────


async def test_list_sticker_packs_returns_array_on_list_result(
    tmp_path,
) -> None:
    from qwenpaw.app.channels.signal.subprocess_client import (
        SignalSubprocessClient,
    )

    client = SignalSubprocessClient(account="+1")

    async def _fake_call(method, params=None, timeout=None):
        assert method == "listStickerPacks"
        return [{"packId": "A" * 32, "title": "x"}]

    client.call = _fake_call  # type: ignore[method-assign]
    packs = await client.list_sticker_packs()
    assert len(packs) == 1
    assert packs[0]["title"] == "x"


async def test_list_sticker_packs_tolerates_packs_key_wrapper() -> None:
    """signal-cli older versions wrap the array in ``{packs: [...]}``.
    The client should unwrap that shape transparently."""
    from qwenpaw.app.channels.signal.subprocess_client import (
        SignalSubprocessClient,
    )

    client = SignalSubprocessClient(account="+1")

    async def _fake_call(method, params=None, timeout=None):
        return {"packs": [{"packId": "B", "title": "y"}]}

    client.call = _fake_call  # type: ignore[method-assign]
    packs = await client.list_sticker_packs()
    assert packs == [{"packId": "B", "title": "y"}]


async def test_list_sticker_packs_returns_empty_on_rpc_error() -> None:
    """Read-only RPC: failures surface as ``[]`` instead of a raise,
    so callers treat them as "no packs" — the tool layer already
    returns an empty array in that case."""
    from qwenpaw.app.channels.signal.subprocess_client import (
        SignalSubprocessClient,
    )

    client = SignalSubprocessClient(account="+1")

    async def _fake_call(method, params=None, timeout=None):
        raise RuntimeError("boom")

    client.call = _fake_call  # type: ignore[method-assign]
    assert await client.list_sticker_packs() == []


async def test_add_sticker_pack_uses_signal_art_uri() -> None:
    from qwenpaw.app.channels.signal.subprocess_client import (
        SignalSubprocessClient,
    )

    client = SignalSubprocessClient(account="+1")
    calls: List[Dict[str, Any]] = []

    async def _fake_call(method, params=None, timeout=None):
        calls.append({"method": method, "params": params})
        return None

    client.call = _fake_call  # type: ignore[method-assign]

    ok = await client.add_sticker_pack("PACKID", "PACKKEY")
    assert ok is True
    assert calls == [
        {
            "method": "addStickerPack",
            "params": {
                "uri": (
                    "https://signal.art/addstickers/"
                    "#pack_id=PACKID&pack_key=PACKKEY"
                ),
            },
        },
    ]


async def test_upload_sticker_pack_handles_plain_url_string() -> None:
    from qwenpaw.app.channels.signal.subprocess_client import (
        SignalSubprocessClient,
    )

    client = SignalSubprocessClient(account="+1")

    async def _fake_call(method, params=None, timeout=None):
        assert method == "uploadStickerPack"
        assert params == {"path": "/tmp/pack/manifest.json"}
        return "https://signal.art/addstickers/#pack_id=X&pack_key=Y"

    client.call = _fake_call  # type: ignore[method-assign]

    url = await client.upload_sticker_pack("/tmp/pack/manifest.json")
    assert url == "https://signal.art/addstickers/#pack_id=X&pack_key=Y"


async def test_upload_sticker_pack_handles_dict_url_field() -> None:
    from qwenpaw.app.channels.signal.subprocess_client import (
        SignalSubprocessClient,
    )

    client = SignalSubprocessClient(account="+1")

    async def _fake_call(method, params=None, timeout=None):
        return {"url": "https://signal.art/addstickers/#pack_id=Z&pack_key=W"}

    client.call = _fake_call  # type: ignore[method-assign]

    url = await client.upload_sticker_pack("/x.json")
    assert url == "https://signal.art/addstickers/#pack_id=Z&pack_key=W"


async def test_send_sticker_message_wires_sticker_param() -> None:
    """send with sticker param must NOT carry ``message`` — signal-cli
    routes the send purely on the sticker reference.  Bonus check:
    group vs DM target shape."""
    from qwenpaw.app.channels.signal.subprocess_client import (
        SignalSubprocessClient,
    )

    client = SignalSubprocessClient(account="+85251159218")
    calls: List[Dict[str, Any]] = []

    async def _fake_call(method, params=None, timeout=None):
        calls.append({"method": method, "params": params})
        return {"timestamp": 42}

    client.call = _fake_call  # type: ignore[method-assign]

    # DM
    ts = await client.send_sticker_message(
        "+85298765432",
        "PACKID",
        3,
        is_group=False,
    )
    assert ts == 42
    dm_params = calls[0]["params"]
    assert dm_params["sticker"] == "PACKID:3"
    # signal-cli 0.14.x's documented JSON-RPC schema for ``send``
    # uses singular ``recipient`` (still accepts plural
    # ``recipients`` as a compat fallback, but using the canonical
    # field avoids any version-specific routing quirks for sticker
    # sends — observed in prod where ``recipients`` made 0.14.3
    # render as a non-sticker payload on the receiver).
    assert dm_params["recipient"] == ["+85298765432"]
    assert "recipients" not in dm_params
    assert "groupId" not in dm_params

    # Group
    ts2 = await client.send_sticker_message(
        "GROUPBASE64==",
        "PACKID",
        0,
        is_group=True,
    )
    assert ts2 == 42
    grp_params = calls[1]["params"]
    assert grp_params["groupId"] == "GROUPBASE64=="
    assert "recipient" not in grp_params
    assert "recipients" not in grp_params


# ───────────────────────────── mention prefix collision ──────────────


def test_is_mentioned_plain_text_phone() -> None:
    """Account +85298349370 in body '@+85298349370 hello' → mentioned."""
    ch = _make_channel(account="+85298349370", account_uuid="")
    assert ch._is_bot_mentioned({}, "@+85298349370 hello") is True


def test_is_mentioned_phone_prefix_no_false_positive() -> None:
    """Account +123 should NOT match body '@+12345 hi' (prefix collision)."""
    ch = _make_channel(account="+123", account_uuid="")
    assert ch._is_bot_mentioned({}, "@+12345 hi") is False


def test_is_mentioned_uuid_word_boundary() -> None:
    """UUID prefix requires a non-word char boundary (no substring match)."""
    ch = _make_channel(account="", account_uuid="447e")
    assert ch._is_bot_mentioned({}, "@447efoo") is False
    assert ch._is_bot_mentioned({}, "@447e hello") is True


def test_is_mentioned_via_structured_mention() -> None:
    """Structured mentions dict (from signal-cli) marks bot mentioned."""
    ch = _make_channel(
        account="+85251159218",
        account_uuid="82e0393a-1f09-4a0a-b000-000000000000",
    )
    mentions = [{"uuid": "82e0393a-1f09-4a0a-b000-000000000000"}]
    assert ch._is_bot_mentioned({"mentions": mentions}, "") is True


# ───────────────────────────── link-endpoint helpers ─────────────────


def test_read_signal_accounts_missing_file(tmp_path) -> None:
    from qwenpaw.app.routers.config import _read_signal_accounts

    res = _read_signal_accounts(tmp_path / "does-not-exist")
    assert res == {"accounts": []}


def test_read_signal_accounts_valid_file(tmp_path) -> None:
    import json as _json
    from qwenpaw.app.routers.config import _read_signal_accounts

    data_dir = tmp_path / "sig"
    (data_dir / "data").mkdir(parents=True)
    (data_dir / "data" / "accounts.json").write_text(
        _json.dumps(
            {
                "accounts": [
                    {
                        "number": "+85298349370",
                        "uuid": "447e962a-0000-0000-0000-000000000000",
                        "path": "some/path",
                    },
                ],
            },
        ),
    )
    res = _read_signal_accounts(data_dir)
    assert res["accounts"][0]["number"] == "+85298349370"
    assert res["accounts"][0]["uuid"].startswith("447e962a")


def test_read_signal_accounts_malformed_file(tmp_path) -> None:
    """Garbage JSON → empty accounts (treated as not-linked)."""
    from qwenpaw.app.routers.config import _read_signal_accounts

    data_dir = tmp_path / "sig"
    (data_dir / "data").mkdir(parents=True)
    (data_dir / "data" / "accounts.json").write_text("not json {{")
    assert _read_signal_accounts(data_dir) == {"accounts": []}


def test_get_signal_link_state_idempotent() -> None:
    from qwenpaw.app.routers.config import _get_signal_link_state

    s1 = _get_signal_link_state("test-agent-zzz")
    s2 = _get_signal_link_state("test-agent-zzz")
    # Same dict object — idempotent and preserves in-flight state.
    assert s1 is s2
    assert s1["status"] == "idle"


# ───────────────────────────── auto-discover account_uuid ────────────
# Signal-UI @bot taps emit structured mentions carrying only the bot's
# ACI ``uuid`` (no phone, for privacy). If ``account_uuid`` is blank
# ``_is_bot_mentioned`` can't match ANY Signal-UI tap — only manually
# typed ``@+<number>`` forms. The channel now auto-populates
# ``account_uuid`` from signal-cli's own ``accounts.json`` on startup.


def test_auto_discover_account_uuid_from_accounts_json(tmp_path) -> None:
    import json as _json

    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "accounts.json").write_text(
        _json.dumps(
            {
                "accounts": [
                    {
                        "number": "+85298349370",
                        "uuid": "447e962a-1f09-4a21-aef6-79617d8e8ad0",
                        "path": "750890",
                    },
                ],
            },
        ),
    )
    uuid = SignalChannel._auto_discover_account_uuid(
        "+85298349370",
        tmp_path,
    )
    assert uuid == "447e962a-1f09-4a21-aef6-79617d8e8ad0"


def test_auto_discover_account_uuid_lowercased(tmp_path) -> None:
    """Mixed-case UUIDs from signal-cli are normalized to lowercase so the
    equality check against structured mentions is case-insensitive."""
    import json as _json

    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "accounts.json").write_text(
        _json.dumps(
            {
                "accounts": [{"number": "+1", "uuid": "AB-CD-EF"}],
            },
        ),
    )
    assert (
        SignalChannel._auto_discover_account_uuid("+1", tmp_path) == "ab-cd-ef"
    )


def test_auto_discover_account_uuid_missing_file(tmp_path) -> None:
    """Missing accounts.json → empty string (channel boots without UUID,
    falls back to plain-text mention detection)."""
    assert SignalChannel._auto_discover_account_uuid("+1", tmp_path) == ""


def test_auto_discover_account_uuid_wrong_account(tmp_path) -> None:
    """Account not listed → empty string (don't pick someone else's UUID)."""
    import json as _json

    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "accounts.json").write_text(
        _json.dumps(
            {
                "accounts": [{"number": "+111", "uuid": "not-ours"}],
            },
        ),
    )
    assert SignalChannel._auto_discover_account_uuid("+999", tmp_path) == ""


def test_auto_discover_account_uuid_malformed_json(tmp_path) -> None:
    """Corrupt accounts.json → empty string, no exception."""
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "accounts.json").write_text("garbage {{")
    assert SignalChannel._auto_discover_account_uuid("+1", tmp_path) == ""


def test_channel_auto_populates_account_uuid_when_unset(tmp_path) -> None:
    """End-to-end: constructing a SignalChannel with account_uuid=""
    pulls the UUID from data_dir/data/accounts.json."""
    import json as _json

    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "accounts.json").write_text(
        _json.dumps(
            {
                "accounts": [
                    {
                        "number": "+85298349370",
                        "uuid": "447e962a-0000-0000-0000-000000000000",
                    },
                ],
            },
        ),
    )
    ch = _make_channel(
        account="+85298349370",
        account_uuid="",
        data_dir=str(tmp_path),
    )
    assert ch._account_uuid == "447e962a-0000-0000-0000-000000000000"


def test_channel_respects_explicit_account_uuid(tmp_path) -> None:
    """If caller provides account_uuid, discovery does NOT override it."""
    import json as _json

    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "accounts.json").write_text(
        _json.dumps(
            {
                "accounts": [{"number": "+1", "uuid": "from-file"}],
            },
        ),
    )
    ch = _make_channel(
        account="+1",
        account_uuid="from-config",
        data_dir=str(tmp_path),
    )
    assert ch._account_uuid == "from-config"


def test_mention_detection_after_auto_discover_fixes_ui_tap(tmp_path) -> None:
    """Regression: Signal-UI @bot tap (structured mention, UUID-only)
    is now correctly recognised as a bot mention after auto-discovery.
    Before the fix, ``account_uuid=""`` made the structured check
    always fall through to plain-text, which missed UUID-only mentions
    entirely."""
    import json as _json

    bot_uuid = "447e962a-1f09-4a21-aef6-79617d8e8ad0"
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "accounts.json").write_text(
        _json.dumps(
            {
                "accounts": [{"number": "+85298349370", "uuid": bot_uuid}],
            },
        ),
    )
    ch = _make_channel(
        account="+85298349370",
        account_uuid="",
        data_dir=str(tmp_path),
    )
    # Typical UI-tap payload: structured mention carries ACI only.
    data_message = {"mentions": [{"uuid": bot_uuid}]}
    assert ch._is_bot_mentioned(data_message, "hello bot") is True


# ───────────────────────────── _expand_mentions round-trip ───────────
# _expand_mentions used to truncate UUIDs to 8 chars (``uuid:abc12345``)
# for the bot's context — minor token saving but fatal on outbound:
# the 8-char prefix is not a valid Signal ACI, so signal-cli could not
# resolve the contact and the mention was dropped, leaving raw UUID text
# in the recipient's view. We now emit the full UUID so the outbound
# parser can round-trip the mention back into a proper structured form.


def test_expand_mentions_uuid_only_emits_full_uuid() -> None:
    """When signal-cli gives us a UUID-only mention (privacy: no phone),
    ``_expand_mentions`` now emits ``uuid:<full>`` instead of the
    truncated ``uuid:<8char>`` form, so outbound round-trip works."""
    ch = _make_channel()
    full = "82e0393a-4c79-4905-b84d-986298f4f8c5"
    mentions = [
        {"start": 0, "length": 1, "uuid": full, "name": "Alice"},
    ]
    body = "￼ hey"
    expanded = ch._expand_mentions(body, mentions)
    # Must contain the full UUID, not just the 8-char prefix.
    assert full in expanded
    # Format: "@uuid:<full> (Name) hey"
    assert expanded.startswith(f"@uuid:{full}")


def test_expand_mentions_prefers_phone_over_uuid() -> None:
    """When a mention has both, the text form uses the phone number
    (shorter and rounds-trips via the phone-bare regex)."""
    ch = _make_channel()
    mentions = [
        {
            "start": 0,
            "length": 1,
            "number": "+85251159218",
            "uuid": "82e0393a-4c79-4905-b84d-986298f4f8c5",
            "name": "Alice",
        },
    ]
    expanded = ch._expand_mentions("￼ hi", mentions)
    assert "+85251159218" in expanded
    # UUID must NOT leak into the bot's view when phone is available.
    assert "uuid:" not in expanded


# ===================================================================
# TestLocalTimestampShared
# ===================================================================


def test_signal_imports_shared_timestamp_helper():
    """Signal must use the same ``_format_local_timestamp`` helper
    as WhatsApp so the envelope timestamp shape stays consistent
    across channels — agents can rely on a single regex to extract
    the time."""
    from qwenpaw.app.channels.signal import channel as sig_mod
    from qwenpaw.app.channels._format import format_local_timestamp

    # Same callable, not a re-implementation.
    assert sig_mod._format_local_timestamp is format_local_timestamp


def test_signal_timestamp_renders_in_local_tz():
    """Smoke test: the helper produces a parseable local-tz
    string for the kind of epoch values Signal hands us."""
    from qwenpaw.app.channels.signal import channel as sig_mod

    out = sig_mod._format_local_timestamp(1777106276, style="short")
    # Format: YYYY-MM-DD HH:MM <ZONE>
    assert "-04-" in out
    assert ":" in out
