# -*- coding: utf-8 -*-
"""Normalization helpers for provider chat payloads.

The persisted session history remains AgentScope ``Msg`` objects. For
provider requests we build a normalized copy before formatting so
request-time repair and multimodal downgrade logic does not mutate the
stored conversation state.
"""

from __future__ import annotations

from copy import deepcopy

from agentscope.message import Msg

from ...constant import MEDIA_UNSUPPORTED_PLACEHOLDER
from .tool_message_utils import _sanitize_tool_messages

# Formerly the all-or-nothing ``supports_multimodal`` decision
# stripped every media block regardless of the model's actual
# per-type capabilities.  That mis-handles the common case of a
# vision-only model (Claude, ChatGPT-OAuth) receiving a VideoBlock:
# the model can't process video, the normalizer left the block in,
# the Anthropic-family formatter passed it through verbatim, and
# Anthropic's API rejected the request (observed in production as
# a 413 Request Too Large on Claude OAuth).
#
# Per-type flags let the normalizer keep image blocks for an
# image-capable model while stripping video / audio that would
# otherwise reach an endpoint that doesn't understand them.
# Stripped blocks become a TextBlock that preserves the file path,
# so the agent can still reason about "there is a video at X" and
# call other tools (ffmpeg frames + view_image, transcribe) against
# the same file.
_MEDIA_BLOCK_TYPES = {"image", "audio", "video", "file"}

# Fields that are provider-specific and should not leak across families.
# Gemini: extra_content carries thought_signature.
# AgentScope internal: raw_input is a stream-parsing artefact.
_PROVIDER_ONLY_TOOL_USE_FIELDS = frozenset({"extra_content", "raw_input"})

# The subset that is preserved when the target is its native family.
_GEMINI_NATIVE_FIELDS = frozenset({"extra_content"})


def _clean_provider_specific_fields(
    msgs: list[Msg],
    target_family: str,
) -> None:
    """Remove provider-specific fields that may leak from a previous provider.

    Operates **in-place** on already-cloned messages so the stored
    conversation history is never mutated.

    Current rules
    ~~~~~~~~~~~~~
    * ``extra_content`` – Gemini-specific (``thought_signature``).
      Kept only when *target_family* is ``"gemini"``.
    * ``raw_input`` – AgentScope stream-parsing artefact.
      Stripped unconditionally; some providers reject unknown fields.
    """
    preserve = (
        _GEMINI_NATIVE_FIELDS if target_family == "gemini" else frozenset()
    )
    strip_fields = _PROVIDER_ONLY_TOOL_USE_FIELDS - preserve

    if not strip_fields:
        return

    for msg in msgs:
        if not isinstance(msg.content, list):
            continue
        for block in msg.content:
            if not isinstance(block, dict):
                continue
            if block.get("type") != "tool_use":
                continue
            for field in strip_fields:
                block.pop(field, None)


def _strip_unsigned_thinking_for_anthropic(msgs: list[Msg]) -> None:
    """Drop thinking blocks that lack a non-empty ``signature``.

    Anthropic requires ``thinking.signature`` on every thinking block in the
    request. Blocks carried over from other providers (OpenAI/Qwen reasoning,
    Gemini thoughts, etc.) have no signature and would 400 the request. Native
    Claude thinking blocks always carry one, so they survive untouched.
    """
    for msg in msgs:
        if not isinstance(msg.content, list):
            continue
        msg.content = [
            block
            for block in msg.content
            if not (
                isinstance(block, dict)
                and block.get("type") == "thinking"
                and not block.get("signature")
            )
        ]


def _clone_msg(msg: Msg) -> Msg:
    """Return a deep copy of an AgentScope message."""
    return Msg.from_dict(deepcopy(msg.to_dict()))


def _clone_messages(msgs: list[Msg]) -> list[Msg]:
    """Return deep-copied messages suitable for request-time normalization."""
    return [_clone_msg(msg) for msg in msgs]


def _extract_media_path(block: dict) -> str | None:
    """Best-effort recovery of the file path / URL a media block
    refers to, so the path-preserving placeholder can keep pointing
    the agent at the file it just lost.
    """
    source = block.get("source")
    if isinstance(source, dict):
        # Prefer durable local paths over signed/remote URLs.  Chat-log
        # serialisation may enrich signed media-server URLs with
        # ``source.file_path`` so historical placeholders remain useful
        # after the URL expires.
        u = source.get("file_path") or source.get("path") or source.get("url")
        if isinstance(u, str) and u:
            return u
    for key in (
        "image_url",
        "video_url",
        "audio_url",
        "file_url",
        "url",
        "file_path",
    ):
        v = block.get(key)
        if isinstance(v, str) and v:
            return v
        if isinstance(v, dict):
            vu = v.get("url")
            if isinstance(vu, str) and vu:
                return vu
    return None


def _path_preserving_placeholder(block_type: str, path: str | None) -> dict:
    """Text block that replaces a stripped media block because the
    current target model cannot decode that media type.  Includes
    the original path when we could recover one — the agent can
    then invoke other tools on the same file instead of blindly
    apologising to the user.

    **Wording is load-bearing.**  An earlier version led with
    ``"removed — this model cannot process video"`` which agents
    (Claude in particular) parsed as "the tool failed" and
    quoted verbatim in their reply, even when sibling content
    blocks carried a full description from a fallback model.
    The neutral phrasing below reads as an *informational note*
    and lets any actual description in adjacent blocks win.
    """
    if path:
        return {
            "type": "text",
            "text": (
                f"[Note: raw {block_type} file at {path} isn't inlined "
                f"in this turn — the current model doesn't decode "
                f"{block_type} directly.  If another block carries a "
                f"description or transcription, trust that; the file "
                f"itself is still at the path and can be reopened with "
                f"a compatible tool / fallback model.]"
            ),
        }
    return {"type": "text", "text": MEDIA_UNSUPPORTED_PLACEHOLDER}


def _historical_media_placeholder(block_type: str, path: str | None) -> dict:
    """Text block replacing a media block from a previous conversation
    turn.  Unlike ``_path_preserving_placeholder`` this is **not** a
    capability downgrade: even image-capable models should not receive
    old screenshots / uploads / tool media over and over.

    The raw session memory is left untouched; this placeholder only
    exists in the cloned provider request, preserving restart/error
    recovery while avoiding historical native-media replay.
    """
    label = "media file" if block_type == "file" else f"{block_type} file"
    if path:
        return {
            "type": "text",
            "text": (
                f"[Note: historical raw {label} is available "
                f"at {path}, but it is not re-inlined in this request "
                f"to avoid replaying old media. Reopen that path with "
                f"a compatible tool if the current user explicitly asks "
                f"about it.]"
            ),
        }
    return {
        "type": "text",
        "text": (
            f"[Note: historical raw {block_type} content was present "
            f"in session history but is not re-inlined in this request.]"
        ),
    }


def _iter_text_fragments(value) -> list[str]:
    """Collect text strings from a Msg content value or nested blocks."""
    texts: list[str] = []
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        if value.get("type") == "text" and isinstance(value.get("text"), str):
            texts.append(value["text"])
        for child in value.values():
            texts.extend(_iter_text_fragments(child))
    elif isinstance(value, list):
        for child in value:
            texts.extend(_iter_text_fragments(child))
    return texts


def _is_system_hint_message(msg: Msg) -> bool:
    """Best-effort detection for one-shot ReAct hint messages.

    AgentScope stores HINT marks separately from Msg objects, so the
    provider normalizer only sees a plain user/system message.  If a
    ``<system-hint>`` message is appended after the real user input,
    treating it as the current-turn boundary would incorrectly mark the
    user's freshly uploaded image as historical.
    """
    texts = [t.strip() for t in _iter_text_fragments(msg.content) if t.strip()]
    return bool(texts) and all(
        t.startswith("<system-hint>") or t.startswith("<system-note:hint>")
        for t in texts
    )


def _is_real_user_message(msg: Msg) -> bool:
    """Return True for user input messages that should start a turn."""
    if getattr(msg, "role", None) != "user":
        return False
    return not _is_system_hint_message(msg)


def _find_current_turn_start(msgs: list[Msg]) -> int:
    """Find the first message of the current user turn.

    ReAct stores the active user message, then assistant tool_use and
    system tool_result messages for the same turn.  Therefore everything
    from the last *real* user message onward is current-turn context and
    may keep native media (subject to model capability).  Everything
    before it is historical context and should keep only path-preserving
    placeholders.
    """
    for idx in range(len(msgs) - 1, -1, -1):
        if _is_real_user_message(msgs[idx]):
            # ``reply(msg=[...])`` can append multiple user messages as one
            # turn.  Keep adjacent trailing user inputs together instead of
            # treating all but the last as historical.
            start = idx
            while start > 0 and (
                _is_real_user_message(msgs[start - 1])
                or _is_system_hint_message(msgs[start - 1])
            ):
                start -= 1
            return start
    # No active user turn (rare internal calls): fail safe by treating all
    # existing memory as historical.
    return len(msgs)


def _should_strip(block_type: str, support: "_MediaSupport") -> bool:
    if block_type == "image":
        return not support.image
    if block_type == "video":
        return not support.video
    if block_type == "audio":
        return not support.audio
    return False


class _MediaSupport:
    """Compact view of per-type multimodal capability.

    Any of ``image`` / ``video`` / ``audio`` defaulting to the
    legacy ``supports_multimodal`` flag preserves the old
    all-or-nothing behaviour for callers that haven't been updated
    yet.
    """

    __slots__ = ("image", "video", "audio")

    def __init__(
        self,
        *,
        supports_multimodal: bool,
        supports_image: bool | None = None,
        supports_video: bool | None = None,
        supports_audio: bool | None = None,
    ) -> None:
        fallback = supports_multimodal
        self.image = fallback if supports_image is None else supports_image
        self.video = fallback if supports_video is None else supports_video
        self.audio = fallback if supports_audio is None else supports_audio

    @property
    def all_unsupported(self) -> bool:
        return not (self.image or self.video or self.audio)


_STRIP_ALL = None  # sentinel — see default below


def _replace_media_in_tool_result_output(
    output: list,
    *,
    historical: bool,
    support: "_MediaSupport | None" = None,
) -> tuple[list, int]:
    """Replace media blocks inside a tool_result output list.

    ``historical=True`` always placeholders native media because old tool
    results should not be re-sent as raw image/video/audio.  Otherwise,
    media is only replaced when unsupported by the current model.
    """
    new_output: list = []
    replaced = 0
    for item in output:
        if isinstance(item, dict) and item.get("type") in _MEDIA_BLOCK_TYPES:
            block_type = item["type"]
            if historical:
                new_output.append(
                    _historical_media_placeholder(
                        block_type,
                        _extract_media_path(item),
                    ),
                )
                replaced += 1
                continue
            if support is not None and _should_strip(block_type, support):
                new_output.append(
                    _path_preserving_placeholder(
                        block_type,
                        _extract_media_path(item),
                    ),
                )
                replaced += 1
                continue
        new_output.append(item)
    return new_output, replaced


def _strip_historical_media_blocks_in_place(msgs: list[Msg]) -> int:
    """Replace native media blocks before the current user turn.

    This is the core replay guard: session memory and chat logs keep
    the original blocks for restart/error recovery and UI rendering,
    but provider requests should not keep re-sending old uploads,
    screenshots, stickers, videos, or view_media tool outputs.

    Current-turn media remains untouched here, so a freshly uploaded
    image or a just-called ``view_image`` tool can still be consumed by
    a vision-capable model.
    """
    current_turn_start = _find_current_turn_start(msgs)
    total_replaced = 0

    for idx, msg in enumerate(msgs):
        if idx >= current_turn_start:
            break
        if not isinstance(msg.content, list):
            continue

        new_content: list = []
        replaced_this_message = 0
        for block in msg.content:
            if isinstance(block, dict) and block.get("type") in _MEDIA_BLOCK_TYPES:
                new_content.append(
                    _historical_media_placeholder(
                        block["type"],
                        _extract_media_path(block),
                    ),
                )
                total_replaced += 1
                replaced_this_message += 1
                continue

            if (
                isinstance(block, dict)
                and block.get("type") == "tool_result"
                and isinstance(block.get("output"), list)
            ):
                new_output, replaced = _replace_media_in_tool_result_output(
                    block["output"],
                    historical=True,
                )
                if replaced:
                    block["output"] = new_output
                    total_replaced += replaced
                    replaced_this_message += replaced

            new_content.append(block)

        if not new_content and replaced_this_message > 0:
            new_content.append(
                {
                    "type": "text",
                    "text": (
                        "[Note: historical raw media content was present "
                        "in session history but is not re-inlined in this "
                        "request.]"
                    ),
                },
            )

        msg.content = new_content

    return total_replaced


def _strip_media_blocks_in_place(
    msgs: list[Msg],
    support: "_MediaSupport | None" = None,
) -> int:
    """Strip only the media types the model can't process; replace
    each stripped block with a path-preserving text placeholder so
    the agent retains a reference to the source file.

    ``support=None`` defaults to strip-everything — matches the
    pre-per-type default so existing callers / tests that don't
    know about capabilities keep working and fail-safe.
    """
    if support is None:
        support = _MediaSupport(supports_multimodal=False)
    total_stripped = 0

    for msg in msgs:
        if not isinstance(msg.content, list):
            continue

        new_content = []
        stripped_this_message = 0
        for block in msg.content:
            if (
                isinstance(block, dict)
                and block.get("type") in _MEDIA_BLOCK_TYPES
                and _should_strip(block["type"], support)
            ):
                path = _extract_media_path(block)
                new_content.append(
                    _path_preserving_placeholder(block["type"], path),
                )
                total_stripped += 1
                stripped_this_message += 1
                continue

            if (
                isinstance(block, dict)
                and block.get("type") == "tool_result"
                and isinstance(block.get("output"), list)
            ):
                new_output, local_stripped = _replace_media_in_tool_result_output(
                    block["output"],
                    historical=False,
                    support=support,
                )
                total_stripped += local_stripped
                stripped_this_message += local_stripped
                if local_stripped:
                    block["output"] = new_output

            new_content.append(block)

        if not new_content and stripped_this_message > 0:
            new_content.append(
                {"type": "text", "text": MEDIA_UNSUPPORTED_PLACEHOLDER},
            )

        msg.content = new_content

    return total_stripped


def normalize_messages_for_model_request(
    msgs: list[Msg],
    *,
    supports_multimodal: bool,
    supports_image: bool | None = None,
    supports_video: bool | None = None,
    supports_audio: bool | None = None,
    target_family: str = "openai",
) -> list[Msg]:
    """Return a normalized copy for provider request formatting.

    Args:
        msgs: Source messages (will **not** be mutated).
        supports_multimodal: Catch-all flag.  Used when a per-type
            flag below is ``None`` — callers that haven't been
            updated for per-type strip get the old all-or-nothing
            behaviour.
        supports_image / supports_video / supports_audio: Per-type
            overrides.  ``True`` keeps that media type in the
            message stream; ``False`` strips it and replaces with a
            path-preserving text placeholder so the agent keeps
            the file reference.  ``None`` defers to
            ``supports_multimodal``.
        target_family: Provider family of the *current* model
            (``"openai"`` | ``"anthropic"`` | ``"gemini"``).  Used
            to strip fields that belong to other providers.
    """
    normalized = _clone_messages(msgs)
    # Sanitize first: _repair_empty_tool_inputs needs raw_input to fix
    # empty input fields.  _clean_provider_specific_fields runs after so
    # that raw_input (and other provider artefacts) are stripped only once
    # the repair has had its chance.
    normalized = _sanitize_tool_messages(normalized)
    _clean_provider_specific_fields(normalized, target_family)
    # Upstream: drop unsigned thinking blocks that Anthropic rejects.
    if target_family == "anthropic":
        _strip_unsigned_thinking_for_anthropic(normalized)

    # First enforce the turn boundary regardless of model capability:
    # historical media becomes path placeholders, while current-turn media
    # remains available for vision/multimodal models.  This keeps raw
    # session memory durable but prevents 50+ old images/videos from being
    # replayed on every request.
    _strip_historical_media_blocks_in_place(normalized)

    support = _MediaSupport(
        supports_multimodal=supports_multimodal,
        supports_image=supports_image,
        supports_video=supports_video,
        supports_audio=supports_audio,
    )
    # Skip the capability walk when every media type is supported —
    # historical media was already handled above, and current-turn media
    # can safely stay native for multimodal-capable models.
    if not (support.image and support.video and support.audio):
        _strip_media_blocks_in_place(normalized, support)
    return normalized


__all__ = [
    "normalize_messages_for_model_request",
]
