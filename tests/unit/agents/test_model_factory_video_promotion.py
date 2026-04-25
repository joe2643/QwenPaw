# -*- coding: utf-8 -*-
"""Tests for video promotion dedup + expiration handling in model_factory."""

# pylint: disable=protected-access
import time

from agentscope.message import Msg, ToolResultBlock

from qwenpaw.agents import model_factory


FUTURE = int(time.time()) + 3600
PAST = int(time.time()) - 3600


def _video_url(token: str, exp: int) -> str:
    return (
        f"https://media.joe2643.work/media?"
        f"t={token}&exp={exp}&sig=deadbeefsig"
    )


def _view_video_history(pairs: list[tuple[str, str]]) -> list[Msg]:
    """Build a history of (tool_call_id, video_url) view_video turns."""
    msgs: list[Msg] = []
    for call_id, url in pairs:
        msgs.append(
            Msg(
                name="assistant",
                role="assistant",
                content=[
                    {
                        "type": "tool_use",
                        "id": call_id,
                        "name": "view_video",
                        "input": {"video_path": f"/tmp/{call_id}.mp4"},
                    },
                ],
            ),
        )
        msgs.append(
            Msg(
                name="system",
                role="system",
                content=[
                    ToolResultBlock(
                        type="tool_result",
                        id=call_id,
                        name="view_video",
                        output=[
                            {
                                "type": "video",
                                "source": {"type": "url", "url": url},
                            },
                        ],
                    ),
                ],
            ),
        )
    return msgs


def _formatted_tool_message(call_id: str) -> dict:
    """Mimic the tool-role message that agentscope's OpenAI formatter emits."""
    return {
        "role": "tool",
        "tool_call_id": call_id,
        "content": "[video placeholder]",
    }


def _promoted_texts(promoted_msg: dict) -> list[str]:
    return [
        item.get("text", "")
        for item in promoted_msg.get("content", [])
        if isinstance(item, dict) and item.get("type") == "text"
    ]


def _promoted_has_video_url(promoted_msg: dict, url: str) -> bool:
    for item in promoted_msg.get("content", []):
        if (
            isinstance(item, dict)
            and item.get("type") == "video_url"
            and item.get("video_url", {}).get("url") == url
        ):
            return True
    return False


# ---------------------------------------------------------------------------
# _video_url_expired
# ---------------------------------------------------------------------------


def test_video_url_expired_detects_past_exp() -> None:
    assert model_factory._video_url_expired(_video_url("tok", PAST)) is True


def test_video_url_expired_accepts_future_exp() -> None:
    assert model_factory._video_url_expired(_video_url("tok", FUTURE)) is False


def test_video_url_expired_no_exp_treated_valid() -> None:
    """URLs without exp= (e.g. public CDN) should not be flagged expired."""
    assert (
        model_factory._video_url_expired(
            "https://example.com/video.mp4",
        )
        is False
    )


def test_video_url_expired_malformed_does_not_raise() -> None:
    assert (
        model_factory._video_url_expired(
            "https://x/?exp=not-a-number",
        )
        is False
    )


# ---------------------------------------------------------------------------
# _video_dedup_key
# ---------------------------------------------------------------------------


def test_dedup_key_same_token_same_key() -> None:
    """Same video (same t=) with different exp/sig → same dedup key."""
    k1 = model_factory._video_dedup_key(_video_url("SAME", PAST))
    k2 = model_factory._video_dedup_key(_video_url("SAME", FUTURE))
    assert k1 == k2


def test_dedup_key_different_token_different_key() -> None:
    k1 = model_factory._video_dedup_key(_video_url("A", FUTURE))
    k2 = model_factory._video_dedup_key(_video_url("B", FUTURE))
    assert k1 != k2


# ---------------------------------------------------------------------------
# _promote_tool_result_videos
# ---------------------------------------------------------------------------


def test_promote_includes_valid_video_block() -> None:
    url = _video_url("fresh", FUTURE)
    msgs = _view_video_history([("call_1", url)])
    formatted = [_formatted_tool_message("call_1")]

    result = model_factory._promote_tool_result_videos(msgs, formatted)

    # tool message + promoted user message
    assert len(result) == 2
    promoted = result[1]
    assert promoted["role"] == "user"
    assert _promoted_has_video_url(promoted, url)


def test_promote_replaces_expired_video_with_placeholder() -> None:
    url = _video_url("stale", PAST)
    msgs = _view_video_history([("call_1", url)])
    formatted = [_formatted_tool_message("call_1")]

    result = model_factory._promote_tool_result_videos(msgs, formatted)

    promoted = result[1]
    assert not _promoted_has_video_url(
        promoted,
        url,
    ), "expired video_url must not leak into the API request"
    assert any("expired" in t for t in _promoted_texts(promoted))


def test_promote_dedups_repeated_video() -> None:
    """Same video referenced by two view_video turns → only one video_url block."""
    # Different exp/sig but same t= → same underlying video
    url_a = _video_url("dup", FUTURE)
    url_b = _video_url("dup", FUTURE + 100)
    msgs = _view_video_history([("call_1", url_a), ("call_2", url_b)])
    formatted = [
        _formatted_tool_message("call_1"),
        _formatted_tool_message("call_2"),
    ]

    result = model_factory._promote_tool_result_videos(msgs, formatted)

    video_url_count = sum(
        1
        for msg in result
        if msg.get("role") == "user"
        for item in msg.get("content", [])
        if isinstance(item, dict) and item.get("type") == "video_url"
    )
    assert video_url_count == 1

    # Second promotion should carry the "omitted" placeholder
    second_promoted = result[-1]
    assert any(
        "omitted" in t or "already visible" in t
        for t in _promoted_texts(second_promoted)
    )


def test_promote_mixed_expired_and_fresh_keeps_only_fresh() -> None:
    """The core bug: one expired URL must not poison fresh ones in the batch."""
    fresh = _video_url("fresh", FUTURE)
    stale = _video_url("stale", PAST)
    msgs = _view_video_history([("call_old", stale), ("call_new", fresh)])
    formatted = [
        _formatted_tool_message("call_old"),
        _formatted_tool_message("call_new"),
    ]

    result = model_factory._promote_tool_result_videos(msgs, formatted)

    # Collect every video_url across the result
    urls = [
        item.get("video_url", {}).get("url")
        for msg in result
        if msg.get("role") == "user"
        for item in msg.get("content", [])
        if isinstance(item, dict) and item.get("type") == "video_url"
    ]
    assert stale not in urls
    assert fresh in urls


def test_promote_empty_when_no_video_tool_results() -> None:
    msgs = [
        Msg(name="user", role="user", content="hello"),
        Msg(name="assistant", role="assistant", content="hi"),
    ]
    formatted = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
    ]

    result = model_factory._promote_tool_result_videos(msgs, formatted)

    # No promotions → formatted list returned unchanged
    assert result is formatted
