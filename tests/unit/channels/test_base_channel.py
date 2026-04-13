"""Tests for BaseChannel shared helpers."""

from qwenpaw.app.channels.base import BaseChannel


def _new_channel() -> BaseChannel:
    """Instantiate without running __init__ (no deps needed)."""
    return BaseChannel.__new__(BaseChannel)


class TestExtractQueryFromPayload:
    """`_extract_query_from_payload` must return the user's real text
    even when channels prepend context parts (group history, reply
    blocks) at index 0 — otherwise slash commands like ``/new`` would
    be shadowed by the history text and never reach the command
    registry."""

    def test_plain_text_first_part(self):
        c = _new_channel()
        payload = {"content_parts": [{"type": "text", "text": "hello"}]}
        assert c._extract_query_from_payload(payload) == "hello"

    def test_skips_group_history_context(self):
        c = _new_channel()
        payload = {
            "content_parts": [
                {
                    "type": "text",
                    "text": "=== UNTRUSTED group history ===\n"
                    "  +85251159218: hi\n=== end ===",
                },
                {"type": "text", "text": "/new"},
            ],
        }
        assert c._extract_query_from_payload(payload) == "/new"

    def test_skips_reply_to_block(self):
        c = _new_channel()
        payload = {
            "content_parts": [
                {"type": "text", "text": "[Replying to Alice: earlier msg]"},
                {"type": "text", "text": "/stop"},
            ],
        }
        assert c._extract_query_from_payload(payload) == "/stop"

    def test_skips_both_history_and_reply(self):
        c = _new_channel()
        payload = {
            "content_parts": [
                {"type": "text", "text": "=== UNTRUSTED history ===\n=== end ==="},
                {"type": "text", "text": "[Replying to Bob: hey]"},
                {"type": "text", "text": "/clear"},
            ],
        }
        assert c._extract_query_from_payload(payload) == "/clear"

    def test_fallback_when_only_context_parts(self):
        """If every part is a context marker, fall back to first one
        (edge case — at least return something non-empty)."""
        c = _new_channel()
        payload = {
            "content_parts": [
                {"type": "text", "text": "=== UNTRUSTED history ==="},
                {"type": "text", "text": "[Replying to X]"},
            ],
        }
        assert c._extract_query_from_payload(payload).startswith("===")

    def test_empty_parts(self):
        c = _new_channel()
        assert c._extract_query_from_payload({"content_parts": []}) == ""
        assert c._extract_query_from_payload({}) == ""

    def test_ignores_non_text_parts(self):
        c = _new_channel()
        payload = {
            "content_parts": [
                {"type": "image", "image_url": "file:///tmp/x.png"},
                {"type": "text", "text": "/stop"},
            ],
        }
        assert c._extract_query_from_payload(payload) == "/stop"
