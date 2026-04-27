# -*- coding: utf-8 -*-
"""Unit tests for the acpx Claude Code session registry — drift
detection, hash stability, LRU eviction, per-key locking.

The registry is the brain of the cache-hit thesis: if drift detection
false-positives (compaction triggers reseed) the cache benefit
collapses, and if false-negatives (real divergence reused a stale
session) Claude replies are wrong.  These tests pin both directions.
"""

from __future__ import annotations

import asyncio

import pytest

from qwenpaw.providers.claude_acpx_session_registry import (
    AcpxSessionEntry,
    Registry,
    SessionKey,
    env_hash,
    history_hash,
    make_session_name,
    msg_signature,
)


# ---------------------------------------------------------------- #
# msg_signature — stable across whitespace/b64/tool-call reserialize
# ---------------------------------------------------------------- #


class TestMsgSignature:
    def test_text_str_content(self) -> None:
        assert msg_signature({"role": "user", "content": "hello"}) == "user|hello"

    def test_text_block_list_content(self) -> None:
        s = msg_signature(
            {"role": "user", "content": [{"type": "text", "text": "hi"}]},
        )
        assert s == "user|hi"

    def test_str_and_block_list_match_when_equivalent(self) -> None:
        a = msg_signature({"role": "user", "content": "hi"})
        b = msg_signature(
            {"role": "user", "content": [{"type": "text", "text": "hi"}]},
        )
        # Both shapes carry the same text — must hash identically so
        # an upstream formatter swap doesn't trip drift.
        assert a == b

    def test_image_b64_payload_stripped(self) -> None:
        # If we hashed the actual base64, re-encoding (different MIME
        # casing, padding, whatever) would trigger drift.  We collapse
        # all images to a marker.
        a = msg_signature(
            {"role": "user", "content": [
                {"type": "input_image", "image_url": "data:image/png;base64,AAAA"},
            ]},
        )
        b = msg_signature(
            {"role": "user", "content": [
                {"type": "input_image", "image_url": "data:image/png;base64,ZZZZ"},
            ]},
        )
        assert a == b == "user|[IMG]"

    def test_tool_calls_contribute_to_signature(self) -> None:
        s = msg_signature({
            "role": "assistant",
            "content": "ok",
            "tool_calls": [
                {"id": "call_1", "function": {"name": "read", "arguments": '{"p":"a"}'}},
            ],
        })
        assert "call|read|" in s
        assert '{"p":"a"}' in s

    def test_tool_role_includes_tool_call_id(self) -> None:
        s = msg_signature({
            "role": "tool",
            "content": "result",
            "tool_call_id": "call_xyz",
        })
        assert "tcid|call_xyz" in s

    def test_assistant_message_does_not_include_tool_call_id(self) -> None:
        # agentscope sometimes re-mints assistant tool_call ids on
        # memory replay.  Including id here would force false drift.
        a = msg_signature({
            "role": "assistant",
            "tool_calls": [
                {"id": "call_v1", "function": {"name": "f", "arguments": "{}"}},
            ],
        })
        b = msg_signature({
            "role": "assistant",
            "tool_calls": [
                {"id": "call_v2", "function": {"name": "f", "arguments": "{}"}},
            ],
        })
        assert a == b


# ---------------------------------------------------------------- #
# history_hash — chained, stable
# ---------------------------------------------------------------- #


class TestHistoryHash:
    def test_empty_history(self) -> None:
        h = history_hash([], 0)
        assert isinstance(h, str) and len(h) == 40  # SHA-1 hex

    def test_zero_idx_is_empty_hash(self) -> None:
        # up_to_idx = 0 means "hash of nothing", regardless of how
        # long the messages list is.
        h0 = history_hash([], 0)
        h_full_but_zero_idx = history_hash(
            [{"role": "user", "content": "anything"}],
            0,
        )
        assert h0 == h_full_but_zero_idx

    def test_changing_one_message_changes_hash(self) -> None:
        a = history_hash(
            [{"role": "user", "content": "a"}],
            1,
        )
        b = history_hash(
            [{"role": "user", "content": "b"}],
            1,
        )
        assert a != b

    def test_extending_history_does_not_alter_prefix_hash(self) -> None:
        # The whole point: ship_tail keeps the prefix hash stable.
        msgs_short = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "u1"},
        ]
        msgs_long = msgs_short + [
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "u2"},
        ]
        # Hashing first 2 of each should match.
        assert history_hash(msgs_short, 2) == history_hash(msgs_long, 2)

    def test_compaction_invariant(self) -> None:
        """If agentscope re-formats user content from str to
        block-list with same text, the hash MUST stay stable.
        Otherwise every memory load triggers reseed.
        """
        before = [{"role": "user", "content": "hello"}]
        after = [
            {"role": "user", "content": [{"type": "text", "text": "hello"}]},
        ]
        assert history_hash(before, 1) == history_hash(after, 1)


# ---------------------------------------------------------------- #
# env_hash — system+tools+cwd+perm+gen_kwargs sensitivity
# ---------------------------------------------------------------- #


class TestEnvHash:
    def test_identical_inputs_match(self) -> None:
        a = env_hash(
            system_prompt="s",
            tool_names=["t1", "t2"],
            cwd="/repo",
            permission_mode="default",
            generate_kwargs={"thinking": {"effort": "low"}},
        )
        b = env_hash(
            system_prompt="s",
            tool_names=["t2", "t1"],  # order shouldn't matter
            cwd="/repo",
            permission_mode="default",
            generate_kwargs={"thinking": {"effort": "low"}},
        )
        assert a == b

    def test_system_prompt_change_invalidates(self) -> None:
        a = env_hash(
            system_prompt="v1",
            tool_names=[],
            cwd="/r",
            permission_mode="default",
            generate_kwargs=None,
        )
        b = env_hash(
            system_prompt="v2",
            tool_names=[],
            cwd="/r",
            permission_mode="default",
            generate_kwargs=None,
        )
        assert a != b

    def test_cwd_change_invalidates(self) -> None:
        a = env_hash(
            system_prompt="s",
            tool_names=[],
            cwd="/repo/A",
            permission_mode="default",
            generate_kwargs=None,
        )
        b = env_hash(
            system_prompt="s",
            tool_names=[],
            cwd="/repo/B",
            permission_mode="default",
            generate_kwargs=None,
        )
        assert a != b

    def test_tool_names_change_invalidates(self) -> None:
        # Adding/removing a tool changes the agent capability surface
        # Claude was prompted with — must mint a new session.
        a = env_hash(
            system_prompt="s",
            tool_names=["t1"],
            cwd="/r",
            permission_mode="default",
            generate_kwargs=None,
        )
        b = env_hash(
            system_prompt="s",
            tool_names=["t1", "t2"],
            cwd="/r",
            permission_mode="default",
            generate_kwargs=None,
        )
        assert a != b

    def test_session_mutable_fields_excluded_from_env_hash(self) -> None:
        """``reasoning_effort``/``reasoning``/``thinking`` round-trip
        through ``acpx claude set effort`` instead of identifying a
        new session.  Including them in env_hash would mint a fresh
        session on every effort change, defeating the cache thesis.
        Multi-turn QA 2026-04-27 caught this.
        """
        # Effort change alone must NOT change env_hash.
        a = env_hash(
            system_prompt="s",
            tool_names=[],
            cwd="/r",
            permission_mode="default",
            generate_kwargs={"max_tokens": 1024, "reasoning_effort": "low"},
        )
        b = env_hash(
            system_prompt="s",
            tool_names=[],
            cwd="/r",
            permission_mode="default",
            generate_kwargs={"max_tokens": 1024, "reasoning_effort": "high"},
        )
        assert a == b
        # Same for reasoning dict and Anthropic-style thinking dict.
        c = env_hash(
            system_prompt="s",
            tool_names=[],
            cwd="/r",
            permission_mode="default",
            generate_kwargs={
                "max_tokens": 1024,
                "reasoning": {"effort": "high"},
                "thinking": {"budget_tokens": 4096},
            },
        )
        assert a == c

    def test_other_generate_kwargs_still_invalidate(self) -> None:
        # Sanity: max_tokens IS part of identity (changes shipped tokens
        # accounting and can affect Claude Code's behavior).
        a = env_hash(
            system_prompt="s",
            tool_names=[],
            cwd="/r",
            permission_mode="default",
            generate_kwargs={"max_tokens": 1024},
        )
        b = env_hash(
            system_prompt="s",
            tool_names=[],
            cwd="/r",
            permission_mode="default",
            generate_kwargs={"max_tokens": 4096},
        )
        assert a != b


# ---------------------------------------------------------------- #
# make_session_name — stable, includes host+pid
# ---------------------------------------------------------------- #


class TestSessionName:
    def test_format(self) -> None:
        name = make_session_name(
            agent_id="agent01",
            session_id="telegram:12345678",
            model="claude-sonnet-4-6",
        )
        assert name.startswith("copaw-")
        # Splits: copaw, host, pid, agent[:8], session[:12], model_short
        parts = name.split("-")
        assert len(parts) >= 6
        # agent[:8]
        assert "agent01" in name
        # session[:12]: ":" survives
        assert "telegram:123" in name
        # model_short for sonnet-4-6 → s4.6 (then random suffix)
        assert "-s4.6-" in name

    def test_different_models_distinct_names(self) -> None:
        a = make_session_name(
            agent_id="a",
            session_id="s",
            model="claude-sonnet-4-6",
        )
        b = make_session_name(
            agent_id="a",
            session_id="s",
            model="claude-opus-4-7",
        )
        assert a != b


# ---------------------------------------------------------------- #
# Registry — plan_turn + commit_turn + drift detection
# ---------------------------------------------------------------- #


@pytest.fixture
def reg() -> Registry:
    return Registry()


class TestRegistryFirstTurn:
    @pytest.mark.asyncio
    async def test_cold_lookup_mints_seed_full(self, reg: Registry) -> None:
        plan = await reg.plan_turn(
            agent_id="a1",
            session_id="s1",
            model="claude-sonnet-4-6",
            env_hash_value="env1",
            messages=[{"role": "user", "content": "hi"}],
        )
        assert plan.mode == "seed_full"
        assert plan.from_idx == 0
        assert plan.session_name.startswith("copaw-")
        assert len(reg) == 1

    @pytest.mark.asyncio
    async def test_missing_agent_id_raises(self, reg: Registry) -> None:
        with pytest.raises(RuntimeError, match="ContextVars"):
            await reg.plan_turn(
                agent_id="",
                session_id="s1",
                model="m",
                env_hash_value="e",
                messages=[],
            )

    @pytest.mark.asyncio
    async def test_missing_session_id_raises(self, reg: Registry) -> None:
        with pytest.raises(RuntimeError, match="ContextVars"):
            await reg.plan_turn(
                agent_id="a",
                session_id="",
                model="m",
                env_hash_value="e",
                messages=[],
            )


class TestRegistryShipTail:
    @pytest.mark.asyncio
    async def test_second_turn_ships_tail(self, reg: Registry) -> None:
        msgs1 = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "u1"},
        ]
        plan1 = await reg.plan_turn(
            agent_id="a", session_id="s", model="m", env_hash_value="e",
            messages=msgs1,
        )
        assert plan1.mode == "seed_full"

        # Caller commits after successful ship.
        await reg.commit_turn(
            plan1.entry, new_shipped_idx=2, messages=msgs1,
        )

        # Second turn: assistant reply landed + new user msg.
        msgs2 = msgs1 + [
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "u2"},
        ]
        plan2 = await reg.plan_turn(
            agent_id="a", session_id="s", model="m", env_hash_value="e",
            messages=msgs2,
        )
        assert plan2.mode == "ship_tail"
        assert plan2.from_idx == 2
        assert plan2.session_name == plan1.session_name


class TestRegistryDrift:
    @pytest.mark.asyncio
    async def test_history_shorter_triggers_reseed(self, reg: Registry) -> None:
        msgs1 = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "u1"},
            {"role": "assistant", "content": "a1"},
        ]
        plan1 = await reg.plan_turn(
            agent_id="a", session_id="s", model="m", env_hash_value="e",
            messages=msgs1,
        )
        await reg.commit_turn(plan1.entry, new_shipped_idx=3, messages=msgs1)

        # User /clear: history truncated.
        msgs2 = [{"role": "user", "content": "fresh"}]
        plan2 = await reg.plan_turn(
            agent_id="a", session_id="s", model="m", env_hash_value="e",
            messages=msgs2,
        )
        assert plan2.mode == "seed_full"
        assert plan2.from_idx == 0
        assert plan2.session_name != plan1.session_name

    @pytest.mark.asyncio
    async def test_hash_mismatch_triggers_reseed(self, reg: Registry) -> None:
        msgs1 = [{"role": "user", "content": "u1"}]
        plan1 = await reg.plan_turn(
            agent_id="a", session_id="s", model="m", env_hash_value="e",
            messages=msgs1,
        )
        await reg.commit_turn(plan1.entry, new_shipped_idx=1, messages=msgs1)

        # User edited the past message.  Hash chain breaks.
        edited = [{"role": "user", "content": "u1-EDITED"}]
        plan2 = await reg.plan_turn(
            agent_id="a", session_id="s", model="m", env_hash_value="e",
            messages=edited,
        )
        assert plan2.mode == "seed_full"
        assert plan2.session_name != plan1.session_name

    @pytest.mark.asyncio
    async def test_compaction_does_not_trigger_drift(self, reg: Registry) -> None:
        # Regression R1 from test plan: agentscope memory may
        # re-format messages (str → block-list) without changing
        # semantics.  This MUST NOT trigger reseed.
        msgs1 = [{"role": "user", "content": "u1"}]
        plan1 = await reg.plan_turn(
            agent_id="a", session_id="s", model="m", env_hash_value="e",
            messages=msgs1,
        )
        await reg.commit_turn(plan1.entry, new_shipped_idx=1, messages=msgs1)

        # Simulate compaction: same content, different shape.
        reformatted = [
            {"role": "user", "content": [{"type": "text", "text": "u1"}]},
            {"role": "user", "content": "u2"},  # new turn
        ]
        plan2 = await reg.plan_turn(
            agent_id="a", session_id="s", model="m", env_hash_value="e",
            messages=reformatted,
        )
        assert plan2.mode == "ship_tail", (
            "Compaction MUST NOT false-trigger drift — "
            "msg_signature should be format-stable"
        )
        assert plan2.from_idx == 1


class TestRegistryEnvChange:
    @pytest.mark.asyncio
    async def test_different_env_hash_mints_separate_session(
        self,
        reg: Registry,
    ) -> None:
        # Same conversation, different env (e.g. cwd switched).
        # New env_hash → new SessionKey → fresh session.
        plan_a = await reg.plan_turn(
            agent_id="a", session_id="s", model="m", env_hash_value="env_a",
            messages=[{"role": "user", "content": "hi"}],
        )
        plan_b = await reg.plan_turn(
            agent_id="a", session_id="s", model="m", env_hash_value="env_b",
            messages=[{"role": "user", "content": "hi"}],
        )
        assert plan_a.session_name != plan_b.session_name


class TestRegistryTearDown:
    @pytest.mark.asyncio
    async def test_reseed_invokes_tear_down(self) -> None:
        torn: list[str] = []

        async def cb(name: str) -> None:
            torn.append(name)

        r = Registry(tear_down_cb=cb)

        msgs1 = [{"role": "user", "content": "u1"}]
        p1 = await r.plan_turn(
            agent_id="a", session_id="s", model="m", env_hash_value="e",
            messages=msgs1,
        )
        await r.commit_turn(p1.entry, new_shipped_idx=1, messages=msgs1)

        # Force reseed via hash mismatch.
        edited = [{"role": "user", "content": "edited"}]
        await r.plan_turn(
            agent_id="a", session_id="s", model="m", env_hash_value="e",
            messages=edited,
        )

        assert torn == [p1.session_name]


class TestRegistryLRU:
    @pytest.mark.asyncio
    async def test_evicts_oldest_when_over_cap(self) -> None:
        torn: list[str] = []

        async def cb(name: str) -> None:
            torn.append(name)

        r = Registry(cap=2, tear_down_cb=cb)

        for i in range(3):
            await r.plan_turn(
                agent_id=f"a{i}", session_id="s", model="m", env_hash_value="e",
                messages=[{"role": "user", "content": f"u{i}"}],
            )
        # Give pending tear-down task a chance to run.
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        assert len(r) == 2
        # Oldest (a0) torn down.
        assert any("a0" in n for n in torn), torn


class TestRegistryEffortTracking:
    @pytest.mark.asyncio
    async def test_update_effort_records_value(self, reg: Registry) -> None:
        plan = await reg.plan_turn(
            agent_id="a", session_id="s", model="m", env_hash_value="e",
            messages=[{"role": "user", "content": "u"}],
        )
        assert plan.entry.last_effort is None
        await reg.update_effort(plan.entry, "high")
        assert plan.entry.last_effort == "high"
