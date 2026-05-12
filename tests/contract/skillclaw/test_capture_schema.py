# -*- coding: utf-8 -*-
"""Schema contract: every record our SkillClaw capture hook appends to
``conversations.jsonl`` must satisfy SkillClaw's ingest expectations.

If SkillClaw upstream changes the record shape (renames a field,
makes an "optional" field required, changes ``timestamp`` format,
etc.), this test fails — telling us to update
:mod:`qwenpaw.agents.hooks.skillclaw_capture` BEFORE the next
production run silently writes incompatible jsonl that evolve_server
then chokes on.

Pinned against SkillClaw v0.4.0 / commit ``4a6b444`` (2026-04-24).
When upstream is bumped:

1. Re-capture ``UPSTREAM_FIXTURE`` from a real proxy run.
2. Update ``REQUIRED_FIELDS`` / ``TOLERATED_EXTRAS`` if the schema
   evolved.
3. Update :func:`SkillClawCaptureHook._msg_to_openai_dict` if message
   blocks need a new flatten rule.
"""
from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path

import pytest

from qwenpaw.agents.hooks.skillclaw_capture import SkillClawCaptureHook


# Schema invariants — mirror what evolve_server's summarizer expects.
REQUIRED_FIELDS = {"session_id", "turn", "timestamp", "messages"}

# Fields proxy writes today but evolve treats as optional.  Listed so
# we *know* about them — surfacing in tests so we can opt in to
# emitting any of them later if quality suffers without them.
TOLERATED_EXTRAS = {
    "instruction_text",
    "prompt_text",
    "response_text",
    "tool_calls",
    "tool_results",
    "tool_observations",
    "tool_errors",
    "reasoning_content",
    "prm_score",
    "next_state",
    # Phase-1 skill-attribution fields emitted by SkillClawCaptureHook
    # so evolve_server's summarizer can compute ``_skills_referenced``.
    "injected_skills",
    "read_skills",
    "modified_skills",
}

ALLOWED_ROLES = {"user", "assistant", "system", "tool"}

# A trimmed real record captured from upstream SkillClaw v0.4.0 proxy.
# Content snippets are placeholders to keep the fixture small and avoid
# committing private workspace data — shape, not text, is what matters
# to a contract test.
UPSTREAM_FIXTURE: dict = {
    "session_id": "tui-qwen3.6-plus-243b4255",
    "turn": 1,
    "timestamp": "2026-04-22 14:38:02",
    "messages": [
        {"role": "system", "content": "<system prompt>"},
        {"role": "user", "content": "<user message>"},
    ],
    "instruction_text": "<system prompt>",
    "prompt_text": "<user message>",
    "response_text": "",
    "tool_calls": [],
    "next_state": "init",
}


def _validate_schema(rec: dict) -> None:
    """Assert a record conforms to our pinned contract.  Raises
    AssertionError with a precise diff so failures are actionable."""
    missing = REQUIRED_FIELDS - rec.keys()
    assert not missing, (
        f"record missing required fields: {sorted(missing)} "
        f"(got: {sorted(rec.keys())})"
    )

    # Detect previously-tolerated fields graduating to required, or
    # entirely new unknown fields appearing — both are signals that
    # SkillClaw upstream changed shape.
    unknown = rec.keys() - REQUIRED_FIELDS - TOLERATED_EXTRAS
    assert not unknown, (
        f"unrecognised fields in record: {sorted(unknown)}.  "
        f"SkillClaw schema may have evolved — re-pin contract test."
    )

    assert isinstance(rec["session_id"], str) and rec["session_id"]
    assert isinstance(rec["turn"], int) and rec["turn"] >= 1
    assert isinstance(rec["timestamp"], str) and re.fullmatch(
        r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}",
        rec["timestamp"],
    ), f"timestamp shape changed: {rec['timestamp']!r}"

    msgs = rec["messages"]
    assert isinstance(msgs, list) and msgs, "messages must be non-empty list"
    for i, m in enumerate(msgs):
        assert isinstance(m, dict), f"messages[{i}] not a dict"
        assert m.keys() >= {
            "role",
            "content",
        }, f"messages[{i}] missing role/content: {sorted(m.keys())}"
        assert (
            m["role"] in ALLOWED_ROLES
        ), f"messages[{i}].role={m['role']!r} not in {ALLOWED_ROLES}"
        # Content tolerated as str (proxy default) or list (multimodal)
        assert isinstance(
            m["content"],
            (str, list),
        ), f"messages[{i}].content type={type(m['content']).__name__}"


def test_upstream_fixture_satisfies_contract() -> None:
    """Real SkillClaw proxy output must satisfy our pinned contract —
    if this fails we mis-described what upstream produces."""
    _validate_schema(UPSTREAM_FIXTURE)


def test_hook_output_satisfies_contract(tmp_path: Path) -> None:
    """Our hook's record must satisfy the same contract.  This is the
    invariant evolve_server's summarizer pipeline relies on."""

    class _FakeMsg:
        def __init__(self, role: str, content):
            self.role = role
            self.content = content

    class _FakeMemory:
        async def get_memory(self):
            return [
                _FakeMsg("system", "You are helpful."),
                _FakeMsg("user", "Hi"),
                _FakeMsg(
                    "assistant",
                    [
                        {"type": "text", "text": "let me search"},
                        {
                            "type": "tool_use",
                            "name": "memory_search",
                            "input": {"query": "Hi"},
                        },
                    ],
                ),
            ]

    class _FakeAgent:
        memory = _FakeMemory()

    hook = SkillClawCaptureHook(
        records_dir=tmp_path,
        session_id="contract-test-001",
    )
    asyncio.run(hook(_FakeAgent(), {}))

    line = (tmp_path / "conversations.jsonl").read_text().strip()
    rec = json.loads(line)
    _validate_schema(rec)


@pytest.mark.parametrize(
    "tier,expected_role",
    [
        ("system", "system"),
        ("user", "user"),
        ("assistant", "assistant"),
    ],
)
def test_each_role_round_trips(tier: str, expected_role: str) -> None:
    """Every role we'd realistically emit must pass through unchanged.
    Catches a regression like silently rewriting ``"system"`` →
    ``"developer"`` (some OpenAI shapes do this) which would break
    SkillClaw's role-based summarizer heuristics."""
    from qwenpaw.agents.hooks.skillclaw_capture import _msg_to_openai_dict

    class _M:
        def __init__(self, role: str, content: str):
            self.role = role
            self.content = content

    out = _msg_to_openai_dict(_M(tier, "x"))
    assert out["role"] == expected_role
    assert out["content"] == "x"
