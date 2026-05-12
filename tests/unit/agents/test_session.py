# -*- coding: utf-8 -*-
"""Tests for SafeJSONSession JSON corruption resilience."""
# pylint: disable=redefined-outer-name
import json
import os
import pathlib
import tempfile

import pytest

from qwenpaw.app.runner.session import SafeJSONSession


@pytest.fixture
def tmp_session_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def sess(tmp_session_dir):
    return SafeJSONSession(save_dir=tmp_session_dir)


def _corrupt_file(path, valid_json, tail_garbage):
    """Write a valid JSON object followed by garbage bytes."""
    with open(path, "w", encoding="utf-8") as f:
        f.write(valid_json + tail_garbage)


class FakeModule:
    """Minimal state module mock for testing."""

    def __init__(self):
        self.data = None

    def state_dict(self):
        return self.data

    def load_state_dict(self, d):
        self.data = d


# ── load_session_state ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_load_valid_json(sess, tmp_session_dir):
    """Normal case: valid JSON loads without error."""
    path = os.path.join(tmp_session_dir, "test--session.json")
    data = {"memory": {"content": ["hello"], "_compressed_summary": ""}}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)

    mod = FakeModule()
    await sess.load_session_state(
        "test:session",
        user_id="",
        memory=mod,
    )
    assert mod.data == data["memory"]


@pytest.mark.asyncio
async def test_load_corrupted_json_extra_data(sess, tmp_session_dir):
    """Corrupted file with extra data after valid JSON should recover."""
    path = os.path.join(tmp_session_dir, "test--session.json")
    valid = json.dumps(
        {"memory": {"content": [], "_compressed_summary": ""}},
        ensure_ascii=False,
    )
    garbage = '=============="}}'
    _corrupt_file(path, valid, garbage)

    mod = FakeModule()
    await sess.load_session_state(
        "test:session",
        user_id="",
        memory=mod,
    )
    assert mod.data == {"content": [], "_compressed_summary": ""}


@pytest.mark.asyncio
async def test_load_corrupted_json_real_world_tail(sess, tmp_session_dir):
    """Real-world corruption pattern from QQ session (203-char tail)."""
    path = os.path.join(tmp_session_dir, "test--session.json")
    valid = json.dumps(
        {
            "memory": {"content": [], "_compressed_summary": ""},
            "toolkit": {"active_groups": []},
        },
        ensure_ascii=False,
    )
    # Actual garbage observed in production
    garbage = (
        "perform actions. A response without a tool call indicates "
        "the task is complete. To continue a task, you must generate "
        "a tool call or provide useful feedback if you are blocked."
        '\\n\\n===================="}}'
    )
    _corrupt_file(path, valid, garbage)

    mod_mem = FakeModule()
    mod_tool = FakeModule()
    await sess.load_session_state(
        "test:session",
        user_id="",
        memory=mod_mem,
        toolkit=mod_tool,
    )
    assert mod_mem.data == {"content": [], "_compressed_summary": ""}
    assert mod_tool.data == {"active_groups": []}


# ── get_session_state_dict ──────────────────────────────────────────


@pytest.mark.asyncio
async def test_get_corrupted_json(sess, tmp_session_dir):
    """get_session_state_dict should recover from corrupted files."""
    path = os.path.join(tmp_session_dir, "test--session.json")
    valid = json.dumps(
        {"memory": {"content": ["x"], "_compressed_summary": ""}},
        ensure_ascii=False,
    )
    _corrupt_file(path, valid, "GARBAGE}}")

    result = await sess.get_session_state_dict(
        "test:session",
        user_id="",
    )
    assert "memory" in result
    assert result["memory"]["content"] == ["x"]


# ── update_session_state ────────────────────────────────────────────


@pytest.mark.asyncio
async def test_update_corrupted_json(sess, tmp_session_dir):
    """update_session_state should recover corrupted file then write clean."""
    path = os.path.join(tmp_session_dir, "test--session.json")
    valid = json.dumps(
        {"memory": {"content": [], "_compressed_summary": ""}},
        ensure_ascii=False,
    )
    _corrupt_file(path, valid, "EXTRA")

    await sess.update_session_state(
        "test:session",
        key="memory.content",
        value=["updated"],
        user_id="",
        channel="",
    )

    # Verify the file is now clean JSON
    with open(path, encoding="utf-8") as f:
        result = json.load(f)
    assert result["memory"]["content"] == ["updated"]


# ── non-existent session ────────────────────────────────────────────


@pytest.mark.asyncio
async def test_load_nonexistent(sess):
    """Non-existent session should not raise when allow_not_exist=True."""
    await sess.load_session_state(
        "no:exist",
        user_id="",
        memory=FakeModule(),
    )


@pytest.mark.asyncio
async def test_get_nonexistent(sess):
    """Non-existent session should return empty dict."""
    result = await sess.get_session_state_dict(
        "no:exist",
        user_id="",
    )
    assert result == {}


# ── completely corrupted file ──────────────────────────────────────


@pytest.mark.asyncio
async def test_load_completely_corrupted(sess, tmp_session_dir):
    """File with no valid JSON at all should not crash (returns empty)."""
    path = os.path.join(tmp_session_dir, "test--session.json")
    with open(path, "w", encoding="utf-8") as f:
        f.write("{{{THIS IS NOT JSON AT ALL!!!")

    mod = FakeModule()
    await sess.load_session_state(
        "test:session",
        user_id="",
        memory=mod,
    )
    # memory key not in recovered (empty) dict → data stays None
    assert mod.data is None


@pytest.mark.asyncio
async def test_get_completely_corrupted(sess, tmp_session_dir):
    """get_session_state_dict returns empty dict for totally garbled file."""
    path = os.path.join(tmp_session_dir, "test--session.json")
    with open(path, "w", encoding="utf-8") as f:
        f.write("NOT JSON {{{{")

    result = await sess.get_session_state_dict(
        "test:session",
        user_id="",
    )
    assert result == {}


@pytest.mark.asyncio
async def test_update_completely_corrupted(sess, tmp_session_dir):
    """update_session_state recovers from total corruption
    by starting fresh."""
    path = os.path.join(tmp_session_dir, "test--session.json")
    with open(path, "w", encoding="utf-8") as f:
        f.write("GARBAGE DATA !!!")

    await sess.update_session_state(
        "test:session",
        key="memory.content",
        value=["recovered"],
        user_id="",
    )

    with open(path, encoding="utf-8") as f:
        result = json.load(f)
    assert result["memory"]["content"] == ["recovered"]


# ── edge-case: empty file ─────────────────────────────────────────


@pytest.mark.asyncio
async def test_load_empty_file(sess, tmp_session_dir):
    """Zero-byte file should recover as empty dict without crash."""
    path = os.path.join(tmp_session_dir, "test--session.json")
    pathlib.Path(path).touch()  # create empty file

    mod = FakeModule()
    await sess.load_session_state(
        "test:session",
        user_id="",
        memory=mod,
    )
    assert mod.data is None  # "memory" key absent from empty dict


@pytest.mark.asyncio
async def test_get_empty_file(sess, tmp_session_dir):
    """Zero-byte file returns empty dict via get_session_state_dict."""
    path = os.path.join(tmp_session_dir, "test--session.json")
    pathlib.Path(path).touch()

    result = await sess.get_session_state_dict("test:session", user_id="")
    assert result == {}


@pytest.mark.asyncio
async def test_update_empty_file(sess, tmp_session_dir):
    """update_session_state on empty file creates clean structure."""
    path = os.path.join(tmp_session_dir, "test--session.json")
    with open(path, "w", encoding="utf-8") as f:
        pass

    await sess.update_session_state(
        "test:session",
        key="memory.content",
        value=["fresh"],
        user_id="",
    )

    with open(path, encoding="utf-8") as f:
        result = json.load(f)
    assert result["memory"]["content"] == ["fresh"]


# ── edge-case: binary / null bytes ────────────────────────────────


@pytest.mark.asyncio
async def test_load_null_bytes(sess, tmp_session_dir):
    """File filled with null bytes should not crash."""
    path = os.path.join(tmp_session_dir, "test--session.json")
    with open(path, "wb") as f:
        f.write(b"\x00" * 256)

    mod = FakeModule()
    await sess.load_session_state("test:session", user_id="", memory=mod)
    assert mod.data is None


# ── edge-case: multiple concatenated JSON objects ──────────────────


@pytest.mark.asyncio
async def test_load_double_write_overlap(sess, tmp_session_dir):
    """Simulates race condition: two full JSON objects concatenated."""
    path = os.path.join(tmp_session_dir, "test--session.json")
    obj1 = json.dumps(
        {"memory": {"content": ["first"], "_compressed_summary": ""}},
    )
    obj2 = json.dumps(
        {"memory": {"content": ["second"], "_compressed_summary": ""}},
    )
    with open(path, "w", encoding="utf-8") as f:
        f.write(obj1 + obj2)

    mod = FakeModule()
    await sess.load_session_state("test:session", user_id="", memory=mod)
    # Should recover the FIRST object (raw_decode behavior)
    assert mod.data == {"content": ["first"], "_compressed_summary": ""}


# ── edge-case: only whitespace ────────────────────────────────────


@pytest.mark.asyncio
async def test_load_whitespace_only(sess, tmp_session_dir):
    """File with only whitespace should not crash."""
    path = os.path.join(tmp_session_dir, "test--session.json")
    with open(path, "w", encoding="utf-8") as f:
        f.write("   \n\n\t  ")

    mod = FakeModule()
    await sess.load_session_state("test:session", user_id="", memory=mod)
    assert mod.data is None


# =========================================================================
# Restart-behaviour invariants
#
# These tests pin down what actually survives a CoPaw restart and where
# the gaps are.  Each test is named against a concrete user-visible
# symptom so a regression here can be triaged without re-reading the
# runner.
# =========================================================================


@pytest.mark.asyncio
async def test_round_trip_preserves_session_state(sess, tmp_session_dir):
    """Save-then-load must restore the exact same state dict — this
    is the invariant "the whatsapp context file is readable after a
    restart".  A break here means even a graceful restart would drop
    memory.
    """
    mod = FakeModule()
    mod.data = {
        "content": [
            [{"id": "a", "role": "user", "content": "hi"}, []],
            [{"id": "b", "role": "assistant", "content": "hello"}, []],
        ],
        "_compressed_summary": "",
    }
    await sess.save_session_state(
        "whatsapp:+85251159218",
        user_id="+85251159218",
        memory=mod,
    )

    # Fresh module — simulates a new process loading the file.
    fresh = FakeModule()
    await sess.load_session_state(
        "whatsapp:+85251159218",
        user_id="+85251159218",
        memory=fresh,
    )
    assert fresh.data == mod.data


@pytest.mark.asyncio
async def test_dm_and_group_sessions_do_not_collide(sess, tmp_session_dir):
    """Same user_id across DM vs group must write to different files.
    A collision here causes the "one channel's restart nukes another"
    bug mode the WAL fix earlier solved at the WAL layer — we want
    the same guarantee at the session-state layer.
    """
    dm_mod = FakeModule()
    dm_mod.data = {
        "content": [[{"role": "user", "content": "dm"}, []]],
        "_compressed_summary": "",
    }
    group_mod = FakeModule()
    group_mod.data = {
        "content": [[{"role": "user", "content": "group"}, []]],
        "_compressed_summary": "",
    }

    await sess.save_session_state(
        "whatsapp:+85251159218",
        user_id="+85251159218",
        memory=dm_mod,
    )
    await sess.save_session_state(
        "whatsapp:group:120363421135228220@g.us",
        user_id="group--120363421135228220@g.us",
        memory=group_mod,
    )

    loaded_dm = FakeModule()
    loaded_group = FakeModule()
    await sess.load_session_state(
        "whatsapp:+85251159218",
        user_id="+85251159218",
        memory=loaded_dm,
    )
    await sess.load_session_state(
        "whatsapp:group:120363421135228220@g.us",
        user_id="group--120363421135228220@g.us",
        memory=loaded_group,
    )
    assert loaded_dm.data == dm_mod.data
    assert loaded_group.data == group_mod.data
    # Two distinct files must exist on disk — if one nuked the other
    # the load would surface the wrong payload or a missing file.
    files = sorted(os.listdir(tmp_session_dir))
    assert len(files) == 2, f"expected 2 files, got {files}"


@pytest.mark.asyncio
async def test_overwrite_same_session_replaces_not_appends(
    sess,
    tmp_session_dir,
):
    """Successive saves to the same session must fully replace the
    prior content (not merge / not append a second JSON object).
    If this breaks, reloading after a restart either picks the wrong
    turn or trips the 'extra data' decoder path.
    """
    mod = FakeModule()
    mod.data = {
        "content": [[{"role": "user", "content": "old"}, []]],
        "_compressed_summary": "",
    }
    await sess.save_session_state("s", user_id="u", memory=mod)

    mod.data = {
        "content": [[{"role": "user", "content": "new"}, []]],
        "_compressed_summary": "summary",
    }
    await sess.save_session_state("s", user_id="u", memory=mod)

    fresh = FakeModule()
    await sess.load_session_state("s", user_id="u", memory=fresh)
    assert fresh.data["content"][0][0]["content"] == "new"
    assert fresh.data["_compressed_summary"] == "summary"


@pytest.mark.asyncio
async def test_merge_concurrent_saves_preserves_sibling_turns(
    sess,
    tmp_session_dir,
):
    """Parallel same-session saves must not lose a sibling run's messages."""
    base = [[{"id": "base", "role": "user", "content": "base"}, []]]
    initial = FakeModule()
    initial.data = {
        "memory": {"content": base, "_compressed_summary": ""},
        "toolkit": {"active_groups": []},
    }
    await sess.save_session_state("s", user_id="u", agent=initial)

    run_one = FakeModule()
    run_one.data = {
        "memory": {
            "content": base
            + [[{"id": "run-one", "role": "assistant", "content": "one"}, []]],
            "_compressed_summary": "",
        },
        "toolkit": {"active_groups": ["one"]},
    }
    run_two = FakeModule()
    run_two.data = {
        "memory": {
            "content": base
            + [[{"id": "run-two", "role": "assistant", "content": "two"}, []]],
            "_compressed_summary": "",
        },
        "toolkit": {"active_groups": ["two"]},
    }

    await sess.save_session_state(
        "s",
        user_id="u",
        merge_concurrent=True,
        agent=run_one,
    )
    await sess.save_session_state(
        "s",
        user_id="u",
        merge_concurrent=True,
        agent=run_two,
    )

    fresh = FakeModule()
    await sess.load_session_state("s", user_id="u", agent=fresh)
    content = fresh.data["memory"]["content"]
    assert [item[0]["id"] for item in content] == ["base", "run-one", "run-two"]
    # Non-memory state still follows the latest completing run.
    assert fresh.data["toolkit"]["active_groups"] == ["two"]


@pytest.mark.asyncio
async def test_merge_caps_compressed_msg_id_tombstones_fifo(
    sess,
    tmp_session_dir,
):
    """Tombstone list FIFO-evicts past _TOMBSTONE_CAP so disk stays bounded."""
    from qwenpaw.agents.context.agent_context import _TOMBSTONE_CAP
    from qwenpaw.app.runner.session import _merge_memory_dict

    existing = {
        "content": [],
        "_compressed_summary": "",
        "_compressed_msg_ids": [f"old-{i}" for i in range(_TOMBSTONE_CAP)],
    }
    incoming = {
        "content": [
            [{"id": "fresh-content", "role": "user", "content": "hi"}, []],
        ],
        "_compressed_summary": "summary",
        "_compressed_msg_ids": [f"new-{i}" for i in range(5)],
    }

    merged = _merge_memory_dict(existing, incoming)
    tombs = merged["_compressed_msg_ids"]
    assert len(tombs) == _TOMBSTONE_CAP
    # Newest 5 survive; oldest 5 evicted.
    assert tombs[-5:] == [f"new-{i}" for i in range(5)]
    for i in range(5):
        assert f"old-{i}" not in tombs
    content_ids = [item[0]["id"] for item in merged["content"]]
    assert "fresh-content" in content_ids


@pytest.mark.asyncio
async def test_agent_context_trims_compressed_msg_ids_above_cap():
    """AgentContext.mark_messages_compressed evicts oldest above cap."""
    from agentscope.message import Msg

    from qwenpaw.agents.context.agent_context import (
        AgentContext,
        _TOMBSTONE_CAP,
    )
    from qwenpaw.agents.utils.estimate_token_counter import (
        EstimatedTokenCounter,
    )

    ctx = AgentContext(token_counter=EstimatedTokenCounter())
    # Pre-load with CAP-1 tombstones, then add 3 more via mark — only 2 oldest
    # should be evicted to stay at exactly CAP.
    ctx._compressed_msg_ids = {
        f"old-{i}": None for i in range(_TOMBSTONE_CAP - 1)
    }
    fresh = [Msg("user", f"m{i}", "user") for i in range(3)]
    fresh_ids = [m.id for m in fresh]
    await ctx.mark_messages_compressed(fresh)

    assert len(ctx._compressed_msg_ids) == _TOMBSTONE_CAP
    for fid in fresh_ids:
        assert fid in ctx._compressed_msg_ids
    # Two oldest evicted.
    assert "old-0" not in ctx._compressed_msg_ids
    assert "old-1" not in ctx._compressed_msg_ids
    # The CAP-3 most-recent originals survive.
    assert f"old-{_TOMBSTONE_CAP - 2}" in ctx._compressed_msg_ids


@pytest.mark.asyncio
async def test_mark_messages_compressed_preserves_batch_order():
    """Tombstone insertion order matches input batch order so FIFO eviction
    later evicts the earliest-compacted ids first."""
    from agentscope.message import Msg

    from qwenpaw.agents.context.agent_context import AgentContext
    from qwenpaw.agents.utils.estimate_token_counter import (
        EstimatedTokenCounter,
    )

    ctx = AgentContext(token_counter=EstimatedTokenCounter())
    msgs = [Msg("user", f"m{i}", "user") for i in range(8)]
    expected_order = [m.id for m in msgs]

    await ctx.mark_messages_compressed(msgs)

    actual_order = list(ctx._compressed_msg_ids.keys())
    assert actual_order == expected_order, (
        "tombstone insertion order must follow the input batch order; "
        "set-based iteration would scramble it"
    )


@pytest.mark.asyncio
async def test_trim_logs_warning_on_first_eviction(monkeypatch):
    """First eviction emits a WARNING so the residual eviction-then-
    resurrection scenario is visible in production logs.

    Spies on the module logger's ``warning`` directly because CoPaw
    installs custom handlers that bypass pytest's caplog plumbing.
    """
    from qwenpaw.agents.context import agent_context as ac_mod
    from qwenpaw.agents.context.agent_context import (
        AgentContext,
        _TOMBSTONE_CAP,
    )
    from qwenpaw.agents.utils.estimate_token_counter import (
        EstimatedTokenCounter,
    )

    captured: list[str] = []

    def _spy(msg, *args, **_kwargs):
        captured.append(msg % args if args else str(msg))

    monkeypatch.setattr(ac_mod.logger, "warning", _spy)

    ctx = AgentContext(token_counter=EstimatedTokenCounter())
    # Pre-load above the cap; the first trim call should fire the warning
    # and bump the counter from zero.
    ctx._compressed_msg_ids = {
        f"old-{i}": None for i in range(_TOMBSTONE_CAP + 5)
    }
    assert ctx._compressed_msg_evicted_count == 0

    ctx._trim_compressed_msg_ids()

    assert ctx._compressed_msg_evicted_count == 5
    assert any(
        "Tombstone cap reached" in m for m in captured
    ), "expected WARNING when tombstones first evict"

    # Second trim past cap should NOT re-warn (counter is now non-zero).
    captured.clear()
    ctx._compressed_msg_ids.update({f"more-{i}": None for i in range(3)})
    ctx._trim_compressed_msg_ids()
    assert ctx._compressed_msg_evicted_count == 8
    assert not any(
        "Tombstone cap reached" in m for m in captured
    ), "warning must fire only on first eviction per session"


@pytest.mark.asyncio
async def test_state_dict_persists_eviction_count_round_trip():
    """The eviction counter survives state_dict → load_state_dict."""
    from qwenpaw.agents.context.agent_context import AgentContext
    from qwenpaw.agents.utils.estimate_token_counter import (
        EstimatedTokenCounter,
    )

    ctx = AgentContext(token_counter=EstimatedTokenCounter())
    ctx._compressed_msg_evicted_count = 42
    state = ctx.state_dict()
    assert state["_compressed_msg_evicted_count"] == 42

    ctx2 = AgentContext(token_counter=EstimatedTokenCounter())
    ctx2.load_state_dict(state)
    assert ctx2._compressed_msg_evicted_count == 42


@pytest.mark.asyncio
async def test_merge_propagates_eviction_count_max():
    """Session merge takes max(existing, incoming) for the eviction counter
    so a stale sibling save with a smaller counter cannot roll it back."""
    from qwenpaw.app.runner.session import _merge_memory_dict

    existing = {
        "content": [],
        "_compressed_summary": "",
        "_compressed_msg_ids": [],
        "_compressed_msg_evicted_count": 7,
    }
    incoming = {
        "content": [],
        "_compressed_summary": "",
        "_compressed_msg_ids": [],
        "_compressed_msg_evicted_count": 3,
    }

    merged = _merge_memory_dict(existing, incoming)
    assert merged["_compressed_msg_evicted_count"] == 7

    # Reverse direction: incoming has the larger counter.
    merged2 = _merge_memory_dict(incoming, existing)
    assert merged2["_compressed_msg_evicted_count"] == 7


@pytest.mark.asyncio
async def test_merge_counts_merge_time_evictions(monkeypatch):
    """When the tombstone union exceeds the cap only at merge time, the
    persisted counter must reflect those evictions and the first-eviction
    warning must fire from the merge path too."""
    from qwenpaw.agents.context.agent_context import _TOMBSTONE_CAP
    from qwenpaw.app.runner import session as session_mod
    from qwenpaw.app.runner.session import _merge_memory_dict

    captured: list[str] = []

    def _spy(msg, *args, **_kwargs):
        captured.append(msg % args if args else str(msg))

    monkeypatch.setattr(session_mod.logger, "warning", _spy)

    # Each side under-cap on its own; union exceeds it by 100.
    half = _TOMBSTONE_CAP // 2 + 50
    existing = {
        "content": [],
        "_compressed_summary": "",
        "_compressed_msg_ids": [f"e-{i}" for i in range(half)],
        # Both sides start at zero pre-merge — the warning must fire
        # because of the merge-time eviction itself.
        "_compressed_msg_evicted_count": 0,
    }
    incoming = {
        "content": [],
        "_compressed_summary": "",
        "_compressed_msg_ids": [f"i-{i}" for i in range(half)],
        "_compressed_msg_evicted_count": 0,
    }

    merged = _merge_memory_dict(existing, incoming)

    # Union size 2*half = CAP + 100 → 100 evicted at merge.
    assert merged["_compressed_msg_evicted_count"] == 100
    assert len(merged["_compressed_msg_ids"]) == _TOMBSTONE_CAP
    assert any(
        "Tombstone cap reached at merge" in m for m in captured
    ), "expected merge-time first-eviction WARNING"

    # Second merge with the now-non-zero counter should NOT re-warn.
    captured.clear()
    bumped = dict(existing)
    bumped["_compressed_msg_evicted_count"] = 100
    merged2 = _merge_memory_dict(bumped, incoming)
    # max(100, 0) + merge_evicted (still 100 here)
    assert merged2["_compressed_msg_evicted_count"] >= 100
    assert not any(
        "Tombstone cap reached at merge" in m for m in captured
    ), "merge warning must fire only on the first merge-eviction"


@pytest.mark.asyncio
async def test_merge_concurrent_honors_compressed_msg_id_tombstones(
    sess,
    tmp_session_dir,
):
    """Auto-compaction in one concurrent run must not be undone when
    a sibling run's later save is merged with the stale on-disk state.

    Repro: group session, two parallel agent runs A and B both load
    the same baseline (msgs 1..3). A's pre_reasoning compacts msgs
    1..2 (mark_messages_compressed → tombstones {1,2}, in-memory
    keeps msg 3 + a new reply 4). When A saves with merge_concurrent,
    the existing on-disk content still has 1..3 — without tombstones
    the merge resurrects 1..2, defeating the compaction.
    """
    base = [
        [{"id": "m1", "role": "user", "content": "first"}, []],
        [{"id": "m2", "role": "assistant", "content": "second"}, []],
        [{"id": "m3", "role": "user", "content": "third"}, []],
    ]
    initial = FakeModule()
    initial.data = {
        "memory": {
            "content": base,
            "_compressed_summary": "",
            "_compressed_msg_ids": [],
        },
    }
    await sess.save_session_state("s", user_id="u", agent=initial)

    # Run A: compacts m1, m2 (drops them from in-memory, records
    # tombstones), keeps m3 and adds its own reply m4.
    run_a = FakeModule()
    run_a.data = {
        "memory": {
            "content": [
                base[2],
                [{"id": "m4", "role": "assistant", "content": "reply-A"}, []],
            ],
            "_compressed_summary": "summary-from-A",
            "_compressed_msg_ids": ["m1", "m2"],
        },
    }
    await sess.save_session_state(
        "s",
        user_id="u",
        merge_concurrent=True,
        agent=run_a,
    )

    fresh = FakeModule()
    await sess.load_session_state("s", user_id="u", agent=fresh)
    content_ids = [item[0]["id"] for item in fresh.data["memory"]["content"]]
    assert "m1" not in content_ids
    assert "m2" not in content_ids
    assert content_ids == ["m3", "m4"]
    assert sorted(fresh.data["memory"]["_compressed_msg_ids"]) == ["m1", "m2"]
    assert fresh.data["memory"]["_compressed_summary"] == "summary-from-A"

    # Run B started before A's compaction — its in-memory still holds
    # the pre-compaction baseline plus its own reply m5. After A's
    # save, B's save must still honor the persisted tombstones so m1,
    # m2 stay gone, while m5 (a genuinely new sibling reply) survives.
    run_b = FakeModule()
    run_b.data = {
        "memory": {
            "content": base
            + [[{"id": "m5", "role": "assistant", "content": "reply-B"}, []]],
            "_compressed_summary": "",
            "_compressed_msg_ids": [],
        },
    }
    await sess.save_session_state(
        "s",
        user_id="u",
        merge_concurrent=True,
        agent=run_b,
    )

    fresh2 = FakeModule()
    await sess.load_session_state("s", user_id="u", agent=fresh2)
    final_ids = [item[0]["id"] for item in fresh2.data["memory"]["content"]]
    assert "m1" not in final_ids
    assert "m2" not in final_ids
    assert final_ids == ["m3", "m4", "m5"]
    assert sorted(fresh2.data["memory"]["_compressed_msg_ids"]) == ["m1", "m2"]


class _FakeTaskTracker:
    """Minimal stand-in for ``TaskTracker`` — just enough surface
    for ``Runner.shutdown_handler`` to exercise its drain logic
    without pulling in the real broker.
    """

    def __init__(self, active_keys: list[str] | None = None) -> None:
        self._active = list(active_keys or [])
        self.wait_all_done_calls: list[float] = []
        self._drain_delay = 0.0

    async def list_active_tasks(self) -> list[str]:
        return list(self._active)

    async def has_active_tasks(self) -> bool:
        return bool(self._active)

    async def wait_all_done(self, timeout: float) -> bool:
        self.wait_all_done_calls.append(timeout)
        # Simulate tasks completing after ``_drain_delay`` seconds
        # (or never, if > timeout).
        if self._drain_delay <= timeout:
            await asyncio.sleep(0)  # yield once
            self._active.clear()
            return True
        return False


class _FakeRunner:
    """Stand-in with just the ``_task_tracker`` attribute and the
    real ``shutdown_handler`` method bound onto it."""

    def __init__(self, agent_id: str, tracker) -> None:
        self.agent_id = agent_id
        self._task_tracker = tracker
        # Bind the real method so we test production code.
        from qwenpaw.app.runner.runner import AgentRunner

        self.shutdown_handler = AgentRunner.shutdown_handler.__get__(self)


@pytest.mark.asyncio
async def test_shutdown_handler_returns_true_when_no_active_tasks():
    """Idle shutdown must be instant — no tasks in flight means
    nothing to drain."""
    runner = _FakeRunner("default", _FakeTaskTracker(active_keys=[]))
    ok = await runner.shutdown_handler(timeout=5.0)
    assert ok is True
    assert runner._task_tracker.wait_all_done_calls == []


@pytest.mark.asyncio
async def test_shutdown_handler_waits_for_in_flight_tasks():
    """With tasks in flight, ``wait_all_done`` must be called with
    the requested timeout so their ``finally`` blocks (session save)
    run before process exit."""
    tracker = _FakeTaskTracker(active_keys=["chat-1", "chat-2"])
    runner = _FakeRunner("default", tracker)
    ok = await runner.shutdown_handler(timeout=7.5)
    assert ok is True
    assert tracker.wait_all_done_calls == [7.5]
    assert await tracker.list_active_tasks() == []


@pytest.mark.asyncio
async def test_shutdown_handler_returns_false_on_timeout():
    """If drain exceeds the timeout, the handler reports failure so
    the caller can log which runner's state wasn't flushed."""
    tracker = _FakeTaskTracker(active_keys=["chat-1"])
    tracker._drain_delay = 999.0  # simulate never-finishing task
    runner = _FakeRunner("default", tracker)
    ok = await runner.shutdown_handler(timeout=0.1)
    assert ok is False


@pytest.mark.asyncio
async def test_shutdown_handler_noop_when_tracker_absent():
    """A runner without a tracker (e.g. early-init state) must not
    crash during shutdown — just report clean."""
    from qwenpaw.app.runner.runner import AgentRunner

    class _Bare:
        agent_id = "x"
        _task_tracker = None
        shutdown_handler = AgentRunner.shutdown_handler

    ok = await _Bare().shutdown_handler(timeout=1.0)
    assert ok is True


@pytest.mark.asyncio
async def test_shutdown_all_runners_drains_each_workspace():
    """``MultiAgentManager.shutdown_all_runners`` must iterate every
    loaded workspace's runner in parallel and call its
    ``shutdown_handler`` — this is the lifespan hook that replaces
    the old empty shutdown_handler."""
    from qwenpaw.app.multi_agent_manager import MultiAgentManager

    calls: list[tuple[str, float]] = []

    class _StubRunner:
        def __init__(self, agent_id: str) -> None:
            self.agent_id = agent_id

        async def shutdown_handler(self, timeout: float = 30.0) -> bool:
            calls.append((self.agent_id, timeout))
            return True

    class _StubWorkspace:
        def __init__(self, agent_id: str) -> None:
            self.agent_id = agent_id
            self.runner = _StubRunner(agent_id)

    mgr = MultiAgentManager()
    mgr.agents = {
        "default": _StubWorkspace("default"),
        "FSkZzR": _StubWorkspace("FSkZzR"),
    }

    await mgr.shutdown_all_runners(timeout=12.0)
    # Each workspace's runner must have been drained; order unimportant
    # (parallel), only identity and timeout matter.
    assert sorted(calls) == sorted([("default", 12.0), ("FSkZzR", 12.0)])


@pytest.mark.asyncio
async def test_shutdown_all_runners_survives_runner_exception():
    """One broken runner must not block the others — its failure is
    logged and the rest still drain.  Without this invariant a single
    buggy agent could stall every restart."""
    from qwenpaw.app.multi_agent_manager import MultiAgentManager

    calls: list[str] = []

    class _GoodRunner:
        async def shutdown_handler(self, timeout: float = 30.0) -> bool:
            calls.append("good")
            return True

    class _BadRunner:
        async def shutdown_handler(self, timeout: float = 30.0) -> bool:
            raise RuntimeError("upstream service unreachable")

    class _Ws:
        def __init__(self, runner) -> None:
            self.runner = runner

    mgr = MultiAgentManager()
    mgr.agents = {
        "good": _Ws(_GoodRunner()),
        "bad": _Ws(_BadRunner()),
    }
    # Must not raise.
    await mgr.shutdown_all_runners(timeout=1.0)
    assert calls == ["good"]


@pytest.mark.asyncio
async def test_finally_block_saves_even_on_mid_stream_cancellation(
    sess,
    tmp_session_dir,
):
    """Async-generator contract: ``finally`` in an async generator
    runs when the consumer stops iterating (gc / aclose / cancel).
    ``query_handler`` relies on this to persist session state after
    a reply — if Python ever changed semantics (or if someone
    restructures the handler and drops the finally), the restart
    recovery silently regresses.  Lock it in.
    """
    mod = FakeModule()
    mod.data = {"content": [], "_compressed_summary": ""}

    saved = asyncio.Event()

    async def fake_reply_generator():
        try:
            for i in range(10):
                await asyncio.sleep(0)
                yield f"chunk-{i}"
        finally:
            # Mimic the real runner's finally: update + save session.
            mod.data["content"].append(
                [{"role": "assistant", "content": "partial"}, []],
            )
            await sess.save_session_state(
                "cancel-test",
                user_id="u",
                memory=mod,
            )
            saved.set()

    gen = fake_reply_generator()
    # Consume one chunk then close — simulates the downstream channel
    # dropping the generator when the user hits /stop.
    first = await gen.__anext__()
    assert first == "chunk-0"
    await gen.aclose()

    assert saved.is_set(), "finally must run on aclose"
    # And the save must have landed on disk.
    fresh = FakeModule()
    await sess.load_session_state("cancel-test", user_id="u", memory=fresh)
    assert fresh.data["content"][0][0]["content"] == "partial"


import asyncio  # noqa: E402 — late import keeps earlier tests hermetic


# =========================================================================
# Atomic write + .prev backup recovery
# =========================================================================


@pytest.mark.asyncio
async def test_save_creates_prev_backup_after_first_overwrite(
    sess,
    tmp_session_dir,
):
    """Every save after the first must leave a ``.prev`` sibling
    with the previous content — that's the safety net the load-side
    fallback relies on."""
    mod = FakeModule()
    mod.data = {
        "content": [[{"role": "user", "content": "turn-1"}, []]],
        "_compressed_summary": "",
    }
    await sess.save_session_state("s", user_id="u", memory=mod)

    # First save — no backup yet.
    path = os.path.join(tmp_session_dir, "u_s.json")
    assert os.path.exists(path)
    assert not os.path.exists(path + ".prev")

    # Second save rotates the first one to .prev.
    mod.data = {
        "content": [[{"role": "user", "content": "turn-2"}, []]],
        "_compressed_summary": "",
    }
    await sess.save_session_state("s", user_id="u", memory=mod)

    assert os.path.exists(path)
    assert os.path.exists(path + ".prev")
    with open(path + ".prev") as fp:
        prev = json.load(fp)
    assert prev["memory"]["content"][0][0]["content"] == "turn-1"


@pytest.mark.asyncio
async def test_save_is_atomic_no_tmp_file_left_behind(
    sess,
    tmp_session_dir,
):
    """After a successful save, ``<path>.tmp`` must not linger —
    if it does, a later ``save`` could pick up a stale tmp file
    mid-replace and race.  ``os.replace`` already renames atomically
    so this is really an implementation smoke test."""
    mod = FakeModule()
    mod.data = {"content": [], "_compressed_summary": ""}
    await sess.save_session_state("s", user_id="u", memory=mod)

    path = os.path.join(tmp_session_dir, "u_s.json")
    assert os.path.exists(path)
    assert not os.path.exists(path + ".tmp")


@pytest.mark.asyncio
async def test_save_never_leaves_primary_missing_between_steps(
    sess,
    tmp_session_dir,
):
    """Regression guard: on 2026-04-24 a SIGKILL hit between the
    ``os.replace(primary → .prev)`` and the following write, leaving
    the primary gone and ``.prev`` one revision behind.  The current
    ordering (write tmp first, copy-to-prev, then replace) must
    guarantee primary is referenced at every instant.

    Simulates the crash by interposing on ``os.replace``: after any
    call, the primary file path MUST exist on disk.
    """
    import os as _os
    import unittest.mock as _mock

    mod = FakeModule()
    mod.data = {
        "content": [[{"role": "user", "content": "turn-1"}, []]],
        "_compressed_summary": "",
    }
    await sess.save_session_state("s", user_id="u", memory=mod)
    path = os.path.join(tmp_session_dir, "u_s.json")
    assert os.path.exists(path)

    # Second save — wrap os.replace to verify primary never disappears.
    mod.data = {
        "content": [[{"role": "user", "content": "turn-2"}, []]],
        "_compressed_summary": "",
    }

    original_replace = _os.replace
    primary_missing_seen = False

    def _checked_replace(src, dst):
        nonlocal primary_missing_seen
        original_replace(src, dst)
        # After ANY replace call, primary must still be reachable.
        if not _os.path.exists(path):
            primary_missing_seen = True

    with _mock.patch("os.replace", side_effect=_checked_replace):
        await sess.save_session_state("s", user_id="u", memory=mod)

    assert not primary_missing_seen, (
        "primary disappeared at some point during save_session_state "
        "— SIGKILL in that window = full context loss."
    )


@pytest.mark.asyncio
async def test_load_recovers_from_prev_when_primary_missing(
    sess,
    tmp_session_dir,
):
    """The real-world symptom: primary session file went missing
    between a successful save and the next load (root cause still
    unknown).  The ``.prev`` sibling must serve as a fallback so
    the agent's memory doesn't blank out on restart."""
    mod = FakeModule()
    mod.data = {
        "content": [[{"role": "user", "content": "important ctx"}, []]],
        "_compressed_summary": "",
    }
    await sess.save_session_state("s", user_id="u", memory=mod)
    # Second save to create a .prev.
    mod.data = {
        "content": [[{"role": "user", "content": "newer"}, []]],
        "_compressed_summary": "",
    }
    await sess.save_session_state("s", user_id="u", memory=mod)

    path = os.path.join(tmp_session_dir, "u_s.json")
    # Simulate the mystery deleter — wipe the primary but leave .prev.
    os.remove(path)
    assert not os.path.exists(path)
    assert os.path.exists(path + ".prev")

    fresh = FakeModule()
    await sess.load_session_state("s", user_id="u", memory=fresh)
    # Got the OLDER content (from the .prev rotation) — that's the
    # correct failure mode: up to one turn stale, but non-empty.
    assert fresh.data is not None
    assert fresh.data["content"][0][0]["content"] == "important ctx"


@pytest.mark.asyncio
async def test_load_skips_when_both_primary_and_prev_missing(
    sess,
    tmp_session_dir,
):
    """No file + no backup → fall back to the historic "skip"
    behaviour; don't crash, don't populate the module."""
    mod = FakeModule()
    await sess.load_session_state("no:such:session", user_id="u", memory=mod)
    assert mod.data is None
