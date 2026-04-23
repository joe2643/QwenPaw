# -*- coding: utf-8 -*-
"""Unit tests for ``message_request_normalizer``'s per-media-type
strip + path-preserving placeholder.

Regression guard for the Claude OAuth 413 observed in production:
a vision-only model (``supports_image=True``, ``supports_video=False``)
received a ``view_video`` tool_result carrying a ``VideoBlock`` with
a local path, and the Anthropic formatter forwarded it verbatim.
The old all-or-nothing strip (keyed on ``supports_multimodal``)
kept the block in, because ``supports_multimodal`` was True due to
the image flag.  This suite locks in the new per-type behaviour
and the path-preserving placeholder.
"""

from __future__ import annotations

from agentscope.message import Msg

from qwenpaw.agents.utils.message_request_normalizer import (
    normalize_messages_for_model_request,
)


# ---------------------------------------------------------------- #
# Helpers                                                          #
# ---------------------------------------------------------------- #


def _user_with_blocks(blocks: list[dict]) -> Msg:
    return Msg(
        name="user",
        role="user",
        content=blocks,
    )


def _paired_tool_result(output: list[dict]) -> list[Msg]:
    """Return an [assistant tool_use, user tool_result] pair — the
    normalizer's sanitizer drops orphaned tool_results, so tests
    that care about tool_result shape must feed a matched pair.
    """
    return [
        Msg(
            name="assistant",
            role="assistant",
            content=[
                {
                    "type": "tool_use",
                    "id": "tr1",
                    "name": "view_video",
                    "input": {},
                },
            ],
        ),
        Msg(
            name="user",
            role="user",
            content=[
                {
                    "type": "tool_result",
                    "id": "tr1",
                    "name": "view_video",
                    "output": output,
                },
            ],
        ),
    ]


def _block_types(msg: Msg) -> list:
    return [
        b.get("type") for b in msg.content if isinstance(b, dict)
    ]


def _tool_result_output_types(msg: Msg) -> list:
    for b in msg.content:
        if isinstance(b, dict) and b.get("type") == "tool_result":
            return [
                o.get("type") for o in b.get("output", [])
                if isinstance(o, dict)
            ]
    return []


# ---------------------------------------------------------------- #
# Per-type strip: image-capable model still strips video           #
# ---------------------------------------------------------------- #


def test_image_capable_model_strips_video_not_image() -> None:
    # Claude-family capability profile.
    msgs = [
        _user_with_blocks([
            {"type": "text", "text": "see both"},
            {
                "type": "image",
                "source": {"type": "url", "url": "/tmp/a.png"},
            },
            {
                "type": "video",
                "source": {"type": "url", "url": "/tmp/b.webm"},
            },
        ]),
    ]
    out = normalize_messages_for_model_request(
        msgs,
        supports_multimodal=True,
        supports_image=True,
        supports_video=False,
        supports_audio=False,
    )
    types = _block_types(out[0])
    assert "image" in types  # kept — model has vision
    assert "video" not in types  # stripped — model can't see video
    # Stripped video got replaced by a text placeholder that
    # preserves the path.
    text_blocks = [
        b for b in out[0].content
        if isinstance(b, dict) and b.get("type") == "text"
    ]
    placeholder_text = " ".join(b.get("text", "") for b in text_blocks)
    assert "/tmp/b.webm" in placeholder_text
    assert "video" in placeholder_text.lower()


def test_video_capable_model_keeps_video() -> None:
    # Gemini-family capability profile.
    msgs = [
        _user_with_blocks([
            {
                "type": "video",
                "source": {"type": "url", "url": "/tmp/b.webm"},
            },
        ]),
    ]
    out = normalize_messages_for_model_request(
        msgs,
        supports_multimodal=True,
        supports_image=True,
        supports_video=True,
        supports_audio=True,
    )
    assert _block_types(out[0]) == ["video"]


def test_text_only_model_strips_everything() -> None:
    # Strict no-media: every block stripped.
    msgs = [
        _user_with_blocks([
            {"type": "image", "source": {"type": "url", "url": "/tmp/x.png"}},
            {"type": "video", "source": {"type": "url", "url": "/tmp/y.mp4"}},
            {"type": "audio", "source": {"type": "url", "url": "/tmp/z.wav"}},
        ]),
    ]
    out = normalize_messages_for_model_request(
        msgs,
        supports_multimodal=False,
        supports_image=False,
        supports_video=False,
        supports_audio=False,
    )
    types = _block_types(out[0])
    # Each media block becomes exactly one text placeholder in its
    # original slot.
    assert types == ["text", "text", "text"]


def test_per_type_flag_none_defers_to_supports_multimodal() -> None:
    # Legacy behaviour: when per-type flags are None, the normalizer
    # falls back to the catch-all.  supports_multimodal=False means
    # every media type is stripped.
    msgs = [
        _user_with_blocks([
            {"type": "image", "source": {"type": "url", "url": "/tmp/x.png"}},
        ]),
    ]
    out = normalize_messages_for_model_request(
        msgs,
        supports_multimodal=False,
    )
    assert _block_types(out[0]) == ["text"]


# ---------------------------------------------------------------- #
# tool_result output shape is the actual regression site           #
# ---------------------------------------------------------------- #


def test_tool_result_output_strips_video_preserves_image() -> None:
    # The precise shape that blew up on Claude OAuth in production.
    msgs = _paired_tool_result([
        {
            "type": "video",
            "source": {"type": "url", "url": "/tmp/clip.webm"},
        },
        {"type": "text", "text": "view_video loaded the clip"},
        {
            "type": "image",
            "source": {"type": "url", "url": "/tmp/thumb.jpg"},
        },
    ])
    out = normalize_messages_for_model_request(
        msgs,
        supports_multimodal=True,
        supports_image=True,
        supports_video=False,
        supports_audio=False,
    )
    # The tool_result message is the second output message.
    tr_msg = out[1]
    output_types = _tool_result_output_types(tr_msg)
    # video → placeholder text, original text preserved, image kept.
    assert output_types == ["text", "text", "image"]
    tool_result = next(
        b for b in tr_msg.content
        if isinstance(b, dict) and b.get("type") == "tool_result"
    )
    all_text = " ".join(
        o.get("text", "") for o in tool_result["output"]
        if isinstance(o, dict) and o.get("type") == "text"
    )
    assert "/tmp/clip.webm" in all_text


def test_tool_result_with_non_list_output_untouched() -> None:
    # tool_result.output can be a plain str (no media), and the
    # normalizer must leave it alone.
    msgs = [
        Msg(
            name="assistant", role="assistant",
            content=[{"type": "tool_use", "id": "tr1", "name": "echo",
                      "input": {}}],
        ),
        Msg(
            name="user", role="user",
            content=[{
                "type": "tool_result",
                "id": "tr1",
                "name": "echo",
                "output": "just text",
            }],
        ),
    ]
    out = normalize_messages_for_model_request(
        msgs,
        supports_multimodal=False,
    )
    tool_result = out[1].content[0]
    assert tool_result["output"] == "just text"


# ---------------------------------------------------------------- #
# Path preservation detail                                         #
# ---------------------------------------------------------------- #


def test_path_preserved_from_different_source_shapes() -> None:
    # Blocks can carry the path under several keys depending on how
    # the tool built them; the extractor walks the common ones.
    cases = [
        {"type": "video", "source": {"type": "url", "url": "/p1"}},
        {"type": "video", "source": {"type": "file_path", "path": "/p2"}},
        {"type": "audio", "audio_url": "/p3"},
        {"type": "image", "image_url": {"url": "/p4"}},
    ]
    for block in cases:
        out = normalize_messages_for_model_request(
            [_user_with_blocks([block])],
            supports_multimodal=False,
        )
        text = out[0].content[0]["text"]
        expected = {
            "/tmp/video": "/p1",
            "file_path": "/p2",
            "audio_url": "/p3",
            "image_url_dict": "/p4",
        }
        for p in expected.values():
            if p in text:
                break
        else:
            raise AssertionError(f"path not preserved in {text!r}")


def test_no_path_falls_back_to_generic_placeholder() -> None:
    # No source / no URL ⇒ falls back to the old
    # MEDIA_UNSUPPORTED_PLACEHOLDER string.
    msgs = [_user_with_blocks([{"type": "video"}])]
    out = normalize_messages_for_model_request(
        msgs,
        supports_multimodal=False,
    )
    text = out[0].content[0]["text"]
    assert "Media content removed" in text
