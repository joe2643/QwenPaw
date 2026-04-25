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

_MEDIA_BLOCK_TYPES = {"image", "audio", "video"}

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
        u = source.get("url") or source.get("file_path") or source.get("path")
        if isinstance(u, str) and u:
            return u
    for key in ("image_url", "video_url", "audio_url", "url", "file_path"):
        v = block.get(key)
        if isinstance(v, str) and v:
            return v
        if isinstance(v, dict):
            vu = v.get("url")
            if isinstance(vu, str) and vu:
                return vu
    return None


def _path_preserving_placeholder(block_type: str, path: str | None) -> dict:
    """Text block that replaces a stripped media block.  Includes
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
                new_output: list = []
                local_stripped = 0
                for item in block["output"]:
                    if (
                        isinstance(item, dict)
                        and item.get("type") in _MEDIA_BLOCK_TYPES
                        and _should_strip(item["type"], support)
                    ):
                        path = _extract_media_path(item)
                        new_output.append(
                            _path_preserving_placeholder(item["type"], path),
                        )
                        local_stripped += 1
                        continue
                    new_output.append(item)
                total_stripped += local_stripped
                stripped_this_message += local_stripped
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

    support = _MediaSupport(
        supports_multimodal=supports_multimodal,
        supports_image=supports_image,
        supports_video=supports_video,
        supports_audio=supports_audio,
    )
    # Skip the walk entirely when every media type is supported —
    # avoids the clone/rebuild cost for multimodal-native models.
    if not (support.image and support.video and support.audio):
        _strip_media_blocks_in_place(normalized, support)
    return normalized


__all__ = [
    "normalize_messages_for_model_request",
]
