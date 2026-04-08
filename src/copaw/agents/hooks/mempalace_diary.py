# -*- coding: utf-8 -*-
"""MemPalace Knowledge Hooks v5 — BgSave (direct LLM extraction).

All hooks use DirectModel (DashScope API) for extraction instead of
prompt injection. Proven pattern from post_mine_reclassify.py.

Hooks:
- pre_reasoning: MemPalacePreCompactHook (awaited BgSave when context >= threshold)
- post_reasoning: MemPalaceIntervalHook (background BgSave every N messages)
- pre_reply: MemPalacePreReplyHook (safety diary on /new or /clear)
"""
import asyncio
import hashlib
import json
import logging
import sqlite3
from typing import Any
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)

# -- Dedicated log file --
_MP_LOG_PATH = Path.home() / ".mempalace" / "hook.log"


def _mp_log(msg: str, level: str = "INFO"):
    try:
        _MP_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(_MP_LOG_PATH, "a") as f:
            f.write(f"{ts} | {level} | {msg}\n")
    except Exception:
        pass


def _extract_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(
            item.get("text", "") if isinstance(item, dict) else str(item)
            for item in content
        )
    return str(content) if content else ""


def _count_user_messages(messages) -> int:
    count = 0
    for m in messages:
        role = getattr(m, "role", None) or (m.get("role") if isinstance(m, dict) else None)
        if role == "user":
            count += 1
    return count


def _estimate_tokens(messages) -> int:
    total = 0
    for m in messages:
        content = getattr(m, "content", None) or (m.get("content", "") if isinstance(m, dict) else "")
        total += len(_extract_text(content))
    return total // 4


# -- DirectModel: DashScope API, no agentscope dependency --

_direct_model = None


def _get_direct_model():
    global _direct_model
    if _direct_model is not None:
        return _direct_model

    import requests

    secret_path = Path.home() / ".copaw.secret" / "providers" / "custom" / "bailian.json"
    try:
        cfg = json.loads(secret_path.read_text())
        api_key = cfg.get("api_key", "")
        base_url = cfg.get("base_url", "").rstrip("/")
        if not api_key or not base_url:
            _mp_log("DirectModel: no API key or base_url in bailian.json", "ERROR")
            return None

        class DirectModel:
            def __init__(self, key, base):
                self.key = key
                self.base = base
                self.model = "qwen3.5-plus"

            def call_sync(self, messages):
                resp = requests.post(
                    f"{self.base}/chat/completions",
                    headers={"Authorization": f"Bearer {self.key}", "Content-Type": "application/json"},
                    json={"model": self.model, "messages": messages, "temperature": 0.1, "enable_thinking": False},
                    timeout=60,
                )
                data = resp.json()
                return data.get("choices", [{}])[0].get("message", {}).get("content", "")

            async def call_async(self, messages):
                return await asyncio.to_thread(self.call_sync, messages)

        _direct_model = DirectModel(api_key, base_url)
        return _direct_model
    except Exception as e:
        _mp_log(f"DirectModel init failed: {e}", "ERROR")
        return None


# -- Extraction prompt (same as reclassifier) --

EXTRACT_PROMPT = """你係 knowledge extractor。分析以下對話，提取重要知識。

{content}

Task 1 - 分類 wing/room/hall:
- wing: wing_{entity} format — each project/person/domain gets own wing
  e.g. wing_openclaw, wing_copaw, wing_joe, wing_infra, wing_ai, wing_tools, wing_vesper, wing_claude
- room: specific subtopic (AVOID 'general' — use: architecture, config, bugs, api, whatsapp, typescript, deployment, performance, preferences, diary, incidents, milestones)
- hall: hall_facts | hall_events | hall_discoveries | hall_preferences | hall_advice | hall_diary
- 新 wing/room OK (lowercase, wing_ prefix)

Task 2 - Extract entity relationships (如有):
- format: [subject, predicate, object]
- 冇就返空 array []

Task 3 - 提煉 knowledge items (最多 3 條):
- 只提取有長期價值嘅知識
- 唔好存 raw transcript

返回 JSON (只返回 JSON):
{{"items": [{{"content": "...", "wing": "...", "room": "...", "hall": "..."}}], "triples": [["subject", "predicate", "object"]]}}"""


# -- Core BgSave function --

async def _bg_save_from_messages(messages: list, source: str = "hook") -> bool:
    """Extract knowledge from messages via DirectModel and write to ChromaDB + KG."""
    import sys
    sys.path.insert(0, str(Path.home() / ".local" / "lib" / "python3.13" / "site-packages"))

    model = _get_direct_model()
    if model is None:
        _mp_log(f"BgSave({source}): no model, falling back to diary")
        await _write_diary(messages, source)
        return False

    # Build conversation text from messages
    parts = []
    for m in messages:
        role = getattr(m, "role", None) or (m.get("role") if isinstance(m, dict) else None)
        if role not in ("user", "assistant"):
            continue
        raw = getattr(m, "content", "") or (m.get("content", "") if isinstance(m, dict) else "")
        text = _extract_text(raw).strip()
        if not text or len(text) < 20 or text.startswith("{") or text.startswith("[{"):
            continue
        parts.append(f"[{role}] {text[:200]}")

    convo = "\n".join(parts[-15:])[:3000]
    if not convo.strip():
        _mp_log(f"BgSave({source}): empty conversation")
        return False

    # Call LLM
    # Fetch existing rooms so LLM classifies accurately
    _rooms_hint = "openclaw, copaw, mempalace, joe, infra, ai, tools, vesper"
    try:
        from collections import defaultdict as _dd
        _c2 = get_collection(palace_path=MempalaceConfig().palace_path)
        _m2 = _c2.get(include=["metadatas"])
        _wr = _dd(set)
        for _mm in (_m2.get("metadatas") or []):
            _ww, _rr = (_mm or {}).get("wing", ""), (_mm or {}).get("room", "")
            if _ww and _rr: _wr[_ww].add(_rr)
        _rooms_hint = ", ".join(w + ": " + "/".join(sorted(rs)) for w, rs in sorted(_wr.items()))
    except Exception:
        pass
    prompt = EXTRACT_PROMPT.format(content=convo).replace(
        "openclaw, copaw, mempalace, tianyuan, joe, contacts, infra, ai, tools, cooking, vesper, claude",
        _rooms_hint,
    )
    try:
        _mp_log(f"BgSave({source}): calling LLM, {len(convo)} chars")
        response_text = await model.call_async([
            {"role": "system", "content": "Only respond with valid JSON."},
            {"role": "user", "content": prompt},
        ])
        _mp_log(f"BgSave({source}): response length={len(response_text)}")

        if not response_text.strip():
            _mp_log(f"BgSave({source}): empty response, diary fallback")
            await _write_diary(messages, source)
            return False

        # Parse JSON
        text = response_text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1] if "\n" in text else text[3:]
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()

        result = json.loads(text)
    except json.JSONDecodeError as e:
        _mp_log(f"BgSave({source}): JSON parse failed ({e}), preview={response_text[:100]!r}")
        await _write_diary(messages, source)
        return False
    except Exception as e:
        _mp_log(f"BgSave({source}): LLM call failed ({e})")
        await _write_diary(messages, source)
        return False

    # Write knowledge items to ChromaDB
    items = result.get("items", [])
    triples = result.get("triples", [])

    from mempalace.chroma_helper import get_collection
    from mempalace.config import MempalaceConfig
    cfg = MempalaceConfig()
    col = get_collection(palace_path=cfg.palace_path)
    now = datetime.now()
    saved = 0

    for item in items[:5]:
        try:
            content = item.get("content", "")
            wing = item.get("wing", "knowledge")
            room = item.get("room", "general")
            hall = item.get("hall", "hall_facts")
            if not content or len(content) < 10:
                continue
            # Duplicate check (match MCP server's tool_add_drawer behavior)
            try:
                dup_results = col.query(query_texts=[content], n_results=1, include=["distances"])
                if dup_results["ids"] and dup_results["ids"][0]:
                    similarity = round(1 - dup_results["distances"][0][0], 3)
                    if similarity >= 0.9:
                        _mp_log(f"BgSave({source}): skipping duplicate (similarity={similarity})")
                        continue
            except Exception:
                pass  # If dup check fails, proceed with write

            entry_id = f"drawer_{wing}_{room}_{hashlib.md5((content[:100] + now.isoformat()).encode()).hexdigest()[:16]}"
            col.add(
                ids=[entry_id],
                documents=[content],
                metadatas=[{
                    "wing": wing, "room": room, "hall": hall,
                    "type": "knowledge", "filed_at": now.isoformat(),
                    "date": now.strftime("%Y-%m-%d"), "added_by": f"bgsave_{source}",
                }],
            )
            saved += 1
        except Exception as e:
            _mp_log(f"BgSave({source}): write failed: {e}")

    # Write KG triples
    kg_added = 0
    if triples:
        try:
            db = sqlite3.connect(str(Path.home() / ".mempalace" / "knowledge_graph.sqlite3"))
            cur = db.cursor()
            for triple in triples[:5]:
                if isinstance(triple, list) and len(triple) >= 3:
                    subj, pred, obj = str(triple[0]).strip(), str(triple[1]).strip(), str(triple[2]).strip()
                    if subj and pred and obj and len(subj) > 1:
                        cur.execute("SELECT id FROM triples WHERE subject=? AND predicate=? AND object=?", (subj, pred, obj))
                        if not cur.fetchone():
                            cur.execute(
                                "INSERT INTO triples (subject, predicate, object, valid_from, confidence, source_closet, extracted_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                                (subj, pred, obj, now.strftime("%Y-%m-%d"), 0.8, f"bgsave_{source}", now.isoformat()),
                            )
                            kg_added += 1
                            for entity in [subj, obj]:
                                cur.execute("INSERT OR IGNORE INTO entities (name, type, created_at) VALUES (?, ?, ?)",
                                           (entity, "auto", now.isoformat()))
            if kg_added:
                db.commit()
            db.close()
        except Exception as e:
            _mp_log(f"BgSave({source}): KG write failed: {e}")

    _mp_log(f"BgSave({source}): {saved} items + {kg_added} triples saved")

    # Also write diary
    await _write_diary(messages, source, note=f"extracted:{saved}items+{kg_added}triples")
    return saved > 0


async def _write_diary(messages: list, source: str = "hook", note: str = "") -> None:
    """Write a quick diary entry (no LLM needed)."""
    import sys
    sys.path.insert(0, str(Path.home() / ".local" / "lib" / "python3.13" / "site-packages"))
    try:
        from mempalace.chroma_helper import get_collection
        from mempalace.config import MempalaceConfig
        cfg = MempalaceConfig()
        col = get_collection(palace_path=cfg.palace_path)
        now = datetime.now()
        user_count = _count_user_messages(messages)
        entry = f"{now.strftime('%Y-%m-%d')}|session.{source}|{user_count}msgs|auto|*neutral*|★★★"
        if note:
            entry += f"|{note}"
        agent_name = "copaw"
        wing = "wing_copaw"
        room = "diary"
        entry_id = f"diary_{wing}_{room}_{now.strftime('%Y%m%d_%H%M%S')}_{hashlib.md5(entry[:50].encode()).hexdigest()[:8]}"
        col.add(
            ids=[entry_id], documents=[entry],
            metadatas=[{
                "wing": wing, "room": room, "hall": "hall_diary",
                "topic": source, "type": "diary_entry",
                "agent": agent_name, "filed_at": now.isoformat(),
                "date": now.strftime("%Y-%m-%d"), "added_by": f"bgsave_{source}",
            }],
        )
        _mp_log(f"Diary({source}): {entry_id} -> {wing}/diary")
    except Exception as e:
        _mp_log(f"Diary({source}): FAILED {e}", "ERROR")


# =====================================================================
# HOOKS
# =====================================================================


class MemPalacePreCompactHook:
    """pre_reasoning: BgSave when context near capacity (awaited — urgent)."""

    def __init__(self, compact_threshold: float = 0.75):
        self.compact_threshold = compact_threshold
        self._saved_this_cycle = False

    async def __call__(self, agent, kwargs: dict[str, Any]) -> dict[str, Any] | None:
        try:
            if self._saved_this_cycle:
                return None

            memory = agent.memory
            messages = await memory.get_memory()
            if not messages:
                return None

            est_tokens = _estimate_tokens(messages)
            max_tokens = getattr(agent, "max_input_length", 128000)
            if max_tokens <= 0:
                return None
            usage = est_tokens / max_tokens
            if usage < self.compact_threshold:
                self._saved_this_cycle = False
                return None

            _mp_log(f"PreCompact: context {usage:.0%} full, running BgSave")
            self._saved_this_cycle = True
            await _bg_save_from_messages(messages, source="precompact")

        except Exception as e:
            _mp_log(f"PreCompact: ERROR {e}", "ERROR")
        return None


class MemPalaceIntervalHook:
    """post_reasoning: BgSave every N user messages (background task)."""

    def __init__(self, working_dir: Path, write_interval: int = 15,
                 state_file: str = ".mempalace_hook_state.json"):
        self.working_dir = working_dir
        self.write_interval = write_interval
        self.state_file = working_dir / state_file
        self._load_state()

    def _load_state(self):
        try:
            if self.state_file.exists():
                state = json.loads(self.state_file.read_text())
                self.last_write_count = state.get("last_write_count", 0)
            else:
                self.last_write_count = 0
        except Exception:
            self.last_write_count = 0

    def _save_state(self):
        try:
            self.state_file.parent.mkdir(parents=True, exist_ok=True)
            self.state_file.write_text(json.dumps({
                "last_write_count": self.last_write_count,
                "last_updated": datetime.now().isoformat(),
            }, indent=2))
        except Exception:
            pass

    async def __call__(self, agent, kwargs: dict[str, Any], output: Any = None) -> Any:
        try:
            memory = agent.memory
            messages = await memory.get_memory()
            if not messages:
                return output

            user_count = _count_user_messages(messages)

            # Session restart detection
            if user_count < self.last_write_count:
                _mp_log(f"Interval: session restart ({user_count} < {self.last_write_count})")
                self.last_write_count = 0

            if (user_count - self.last_write_count) < self.write_interval:
                return output

            _mp_log(f"Interval: {user_count} msgs (last_write={self.last_write_count}), launching BgSave")
            self.last_write_count = user_count
            self._save_state()

            # Background — don't block the response
            asyncio.create_task(_bg_save_from_messages(messages, source="interval"))

        except Exception as e:
            _mp_log(f"Interval: ERROR {e}", "ERROR")
        return output


class MemPalacePreReplyHook:
    """pre_reply: safety diary before /new or /clear."""

    def __init__(self, working_dir: Path, state_file: str = ".mempalace_hook_state.json"):
        self.working_dir = working_dir
        self.state_file = working_dir / state_file

    def _get_last_write_count(self) -> int:
        try:
            if self.state_file.exists():
                return json.loads(self.state_file.read_text()).get("last_write_count", 0)
        except Exception:
            pass
        return 0

    async def __call__(self, agent, kwargs: dict[str, Any]) -> dict[str, Any] | None:
        try:
            msg = kwargs.get("msg")
            if msg is None:
                return None
            last_msg = msg[-1] if isinstance(msg, list) else msg
            query = _extract_text(getattr(last_msg, "content", "") if hasattr(last_msg, "content") else "").strip().lower()

            if not (query.startswith("/new") or query.startswith("/clear")):
                return None

            memory = agent.memory
            messages = await memory.get_memory()
            user_count = _count_user_messages(messages)
            unsaved = user_count - self._get_last_write_count()

            if unsaved <= 0:
                return None

            _mp_log(f"PreReply: /{query.split()[0]} detected, {unsaved} unsaved msgs, writing diary")
            await _write_diary(messages, source="pre_reply")

        except Exception as e:
            _mp_log(f"PreReply: ERROR {e}", "ERROR")
        return None


# -- Background save for /new command (called from command_handler.py) --

async def _bg_mempalace_save(command_handler, messages: list) -> None:
    """Background LLM extraction on /new. Called via asyncio.create_task."""
    _mp_log(f"BgSave(/new): STARTED with {len(messages)} messages")
    await _bg_save_from_messages(messages, source="new_command")
