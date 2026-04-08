# -*- coding: utf-8 -*-
"""MemPalace Knowledge Hooks v4 — split pre/post reasoning.

pre_reasoning  → PreCompact save (emergency, before AI reasons)
post_reasoning → Interval save (after AI responds, inject for next turn, then cleanup)

Key design:
- pre_reasoning: if context near capacity → inject PRECOMPACT_PROMPT, AI saves before running out
- post_reasoning: every N user msgs → inject SAVE_PROMPT into memory, cleaned up after AI acts on it
"""
import logging
import json
from typing import Any
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)

# Dedicated mempalace hook log file for easy monitoring
_mp_log_path = Path.home() / ".mempalace" / "hook.log"
_mp_handler = None
def _mp_log(msg: str, level: str = "INFO"):
    """Write to dedicated mempalace hook log file + standard logger."""
    global _mp_handler
    try:
        _mp_log_path.parent.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(_mp_log_path, "a") as f:
            f.write(f"{ts} | {level} | {msg}\n")
    except Exception:
        pass
    # Removed standard logger call to prevent double logging

# Mark for cleanup — matches agentscope's _MemoryMark pattern
_MEMPALACE_SAVE_MARK = "__mempalace_save_pending__"

SAVE_PROMPT = (
    "🧠 MemPalace auto-save: please save important knowledge from recent conversation.\n\n"
    "1. **mempalace_add_drawer** — facts/decisions/preferences → correct wing/room/hall\n"
    "   Wings: projects / people / knowledge / events / agents\n   New rooms OK: use lowercase, no spaces (e.g. projects/comfyui, people/sarah)\n"
    "   Halls: hall_facts / hall_events / hall_discoveries / hall_preferences / hall_advice / hall_diary\n"
    "2. **mempalace_kg_add** — entity relationships (subject, predicate, object, valid_from)\n"
    "3. **mempalace_diary_write** — AAAK session diary (agent_name, entry, topic)\n\n"
    "If nothing important since last save, just write a short diary. Then continue normally."
)

PRECOMPACT_PROMPT = (
    "🚨 Context approaching capacity — save ALL knowledge NOW before compaction.\n\n"
    "After compaction, detailed context will be lost. Be thorough:\n"
    "1. **mempalace_add_drawer** — ALL topics, decisions, facts, preferences (wing/room/hall)\n"
    "2. **mempalace_kg_add** — ALL entity relationships discovered\n"
    "3. **mempalace_diary_write** — comprehensive session summary\n\n"
    "Wings: projects/people/knowledge/events/agents. New rooms OK (lowercase, no spaces).. Be thorough."
)


def _count_user_messages(messages) -> int:
    count = 0
    for m in messages:
        role = getattr(m, 'role', None) or (m.get('role') if isinstance(m, dict) else None)
        if role == 'user':
            count += 1
    return count


def _estimate_tokens(messages) -> int:
    total = 0
    for m in messages:
        content = getattr(m, 'content', None) or (m.get('content', '') if isinstance(m, dict) else '')
        if isinstance(content, str):
            total += len(content)
        elif isinstance(content, list):
            for item in content:
                text = getattr(item, 'text', '') or (item.get('text', '') if isinstance(item, dict) else '')
                total += len(str(text))
    return total // 4


def _msg_has_mark(msg, mark: str) -> bool:
    """Check if a message has our mark in metadata."""
    metadata = getattr(msg, 'metadata', None) or (msg.get('metadata') if isinstance(msg, dict) else None)
    if isinstance(metadata, dict):
        return metadata.get('_mark') == mark
    return False


class MemPalacePreCompactHook:
    """pre_reasoning: emergency save when context near capacity."""

    def __init__(self, compact_threshold: float = 0.75):
        self.compact_threshold = compact_threshold
        self._already_injected = False

    async def __call__(self, agent, kwargs: dict[str, Any]) -> dict[str, Any] | None:
        try:
            _mp_log("PreCompact: called")
            # Reset flag if AI has responded since injection
            if self._already_injected:
                memory = agent.memory
                messages = await memory.get_memory()
                if messages:
                    last_role = getattr(messages[-1], 'role', None) or (
                        messages[-1].get('role') if isinstance(messages[-1], dict) else None
                    )
                    if last_role == 'assistant':
                        self._already_injected = False
                return None

            memory = agent.memory
            messages = await memory.get_memory()
            if not messages:
                return None

            est_tokens = _estimate_tokens(messages)
            max_tokens = getattr(agent, 'max_input_length', 128000)
            if max_tokens <= 0:
                return None

            usage = est_tokens / max_tokens
            if usage < self.compact_threshold:
                return None

            # Emergency: inject precompact save
            _mp_log(f"PreCompact: context {usage:.0%} full, injecting save prompt")
            try:
                from agentscope.message import Msg
                save_msg = Msg(name="system", content=PRECOMPACT_PROMPT, role="system")
                await memory.add(save_msg)
                self._already_injected = True
            except Exception as e:
                _mp_log(f"PreCompact inject FAILED: {e}")

        except Exception as e:
            _mp_log(f"PreCompact hook ERROR: {e}")

        return None


class MemPalaceIntervalHook:
    """post_reasoning: interval save after AI responds, with cleanup."""

    def __init__(
        self,
        working_dir: Path,
        write_interval: int = 15,
        state_file: str = ".mempalace_hook_state.json",
    ):
        self.working_dir = working_dir
        self.write_interval = write_interval
        self.state_file = working_dir / state_file
        self._load_state()
        self._pending_cleanup = False
        self._injected_msg_id = None

    def _load_state(self):
        try:
            if self.state_file.exists():
                state = json.loads(self.state_file.read_text())
                self.last_write_count = state.get('last_write_count', 0)
            else:
                self.last_write_count = 0
        except Exception:
            self.last_write_count = 0

    def _save_state(self):
        try:
            self.state_file.parent.mkdir(parents=True, exist_ok=True)
            self.state_file.write_text(json.dumps({
                'last_write_count': self.last_write_count,
                'last_updated': datetime.now().isoformat(),
            }, indent=2))
        except Exception as e:
            logger.warning(f"Failed to save hook state: {e}")

    async def __call__(self, agent, kwargs: dict[str, Any], output: Any = None) -> Any:
        """post_reasoning hook. Receives (agent, kwargs, output).
        
        Two phases:
        1. If _pending_cleanup: AI just processed our save prompt → clean it up, update state
        2. If user_count threshold reached: inject save prompt for next turn
        """
        try:
            memory = agent.memory
            messages = await memory.get_memory()
            if not messages:
                return output

            user_count = _count_user_messages(messages)
            _mp_log(f"Interval: called, {user_count} user msgs, last_write={self.last_write_count}, pending_cleanup={self._pending_cleanup}")

            # Session restart detection
            if user_count < self.last_write_count:
                _mp_log(f"Interval: session restart ({user_count} < {self.last_write_count})")
                self.last_write_count = 0
                self._pending_cleanup = False

            # Phase 1: Cleanup — AI has responded to our save prompt
            if self._pending_cleanup:
                # Try to remove the injected save message from memory
                try:
                    # Find and remove messages with our mark
                    all_msgs = await memory.get_memory()
                    for i, msg in enumerate(all_msgs):
                        if _msg_has_mark(msg, _MEMPALACE_SAVE_MARK):
                            msg_id = getattr(msg, 'id', None)
                            if msg_id and hasattr(memory, 'delete'):
                                await memory.delete(msg_id)
                                logger.debug("MemPalace: cleaned up save prompt from memory")
                except Exception as e:
                    logger.debug(f"MemPalace cleanup: {e} (non-critical)")

                self._pending_cleanup = False
                self.last_write_count = user_count
                self._save_state()
                _mp_log(f"Interval: save cycle complete, next at {user_count + self.write_interval}")
                return output

            # Phase 2: Check if we should inject save prompt
            if (user_count - self.last_write_count) < self.write_interval:
                return output

            # Inject save prompt — AI will see it on next _reasoning call
            _mp_log(f"Interval: {user_count} msgs, injecting save prompt")
            try:
                from agentscope.message import Msg
                save_msg = Msg(
                    name="system",
                    content=SAVE_PROMPT,
                    role="system",
                    metadata={"_mark": _MEMPALACE_SAVE_MARK},
                )
                await memory.add(save_msg)
                self._pending_cleanup = True
            except Exception as e:
                _mp_log(f"Interval inject FAILED: {e}")

        except Exception as e:
            _mp_log(f"Interval hook ERROR: {e}")

        return output


class MemPalacePreReplyHook:
    """pre_reply: safety-net save before /new or /clear wipes memory.

    Does a quick mempalace_diary_write with session summary.
    Not full extraction — just prevents total loss on /new.
    """

    def __init__(self, working_dir: Path, state_file: str = ".mempalace_hook_state.json"):
        self.working_dir = working_dir
        self.state_file = working_dir / state_file

    def _get_last_write_count(self) -> int:
        try:
            if self.state_file.exists():
                return json.loads(self.state_file.read_text()).get('last_write_count', 0)
        except Exception:
            pass
        return 0

    async def __call__(self, agent, kwargs: dict[str, Any]) -> dict[str, Any] | None:
        try:
            msg = kwargs.get('msg')
            if msg is None:
                return None

            # Extract text from msg
            last_msg = msg[-1] if isinstance(msg, list) else msg
            query = getattr(last_msg, 'content', '') if hasattr(last_msg, 'content') else ''
            if isinstance(query, list):
                query = ' '.join(
                    item.get('text', '') if isinstance(item, dict) else str(item)
                    for item in query
                )
            query = str(query).strip().lower()

            _mp_log(f"PreReply: called, query={query[:30]}")
            # Only trigger on /new or /clear
            if not (query.startswith('/new') or query.startswith('/clear')):
                return None

            memory = agent.memory
            messages = await memory.get_memory()
            user_count = _count_user_messages(messages)
            last_write = self._get_last_write_count()

            # Only save if there's unsaved knowledge (user_count > last_write)
            unsaved = user_count - last_write
            if unsaved <= 0:
                logger.debug("MemPalace pre_reply: /new but nothing unsaved, skipping")
                return None

            _mp_log(f"PreReply: /{query.split()[0]} detected, {unsaved} unsaved msgs, writing safety diary")

            # Quick safety-net diary write via direct ChromaDB
            try:
                import sys, hashlib
                sys.path.insert(0, str(Path.home() / '.local' / 'lib' / 'python3.13' / 'site-packages'))
                from mempalace.chroma_helper import get_collection
                from mempalace.mcp_server import _config

                now = datetime.now()
                col = get_collection(palace_path=_config.palace_path)

                # Build quick summary from last N messages
                recent = messages[-min(unsaved * 2, 20):]
                summary_parts = []
                for m in recent:
                    role = getattr(m, 'role', None) or (m.get('role') if isinstance(m, dict) else '?')
                    content = getattr(m, 'content', '') or (m.get('content', '') if isinstance(m, dict) else '')
                    if isinstance(content, list):
                        content = ' '.join(
                            item.get('text', '') if isinstance(item, dict) else str(item)
                            for item in content
                        )
                    if content and role in ('user', 'assistant'):
                        summary_parts.append(f"[{role}] {str(content)[:150]}")

                summary = '\n'.join(summary_parts[-10:])  # Last 10 exchanges max
                entry = f"{now.strftime('%Y-%m-%d')}|session.pre-clear|{unsaved}msgs.unsaved|auto-save|*neutral*|★★★"

                agent_name = getattr(agent, 'name', 'default') or 'default'
                room = agent_name.lower().replace(' ', '_')
                entry_id = f"diary_agents_{room}_{now.strftime('%Y%m%d_%H%M%S')}_{hashlib.md5(entry.encode()).hexdigest()[:8]}"

                col.add(
                    ids=[entry_id],
                    documents=[f"{entry}\n---\n{summary}"],
                    metadatas=[{
                        "wing": "agents",
                        "room": room,
                        "hall": "hall_diary",
                        "topic": "pre-clear-safety",
                        "type": "diary_entry",
                        "agent": agent_name,
                        "filed_at": now.isoformat(),
                        "date": now.strftime("%Y-%m-%d"),
                        "added_by": "MemPalacePreReplyHook",
                    }],
                )
                _mp_log(f"PreReply: safety diary saved ({entry_id})")

            except Exception as e:
                _mp_log(f"PreReply: safety save FAILED: {e}")

        except Exception as e:
            _mp_log(f"PreReply hook ERROR: {e}")

        return None


async def _bg_mempalace_save(command_handler, messages: list) -> None:
    """Background task: LLM extracts knowledge from messages, saves to MemPalace.

    Called by /new before memory clear. Runs asynchronously — does not block.
    Uses the agent's own model for extraction, then writes via direct ChromaDB.
    """
    _mp_log(f"BgSave: STARTED with {len(messages)} messages")
    try:
        import sys, hashlib
        sys.path.insert(0, str(Path.home() / '.local' / 'lib' / 'python3.13' / 'site-packages'))
        from agentscope.message import Msg

        # Get agent's model for LLM extraction
        agent = command_handler
        model = getattr(agent, 'model', None)
        if model is None:
            # Try memory_manager's model
            mm = getattr(agent, 'memory_manager', None)
            if mm and hasattr(mm, '_prepare_model_formatter'):
                mm._prepare_model_formatter()
                model = mm.chat_model
            if model is None:
                _mp_log("BgSave: no model available, falling back to direct diary")
                await _bg_direct_diary(messages)
                return

        # Build conversation text for LLM (last 30 messages max)
        # Filter to user+assistant only, skip system/tool/empty
        parts = []
        for m in messages:
            role = getattr(m, 'role', None) or (m.get('role') if isinstance(m, dict) else None)
            if role not in ('user', 'assistant'):
                continue
            raw = getattr(m, 'content', '') or (m.get('content', '') if isinstance(m, dict) else '')
            if isinstance(raw, list):
                raw = ' '.join(
                    item.get('text', '') if isinstance(item, dict) else str(item)
                    for item in raw
                )
            text = str(raw).strip()
            # Skip short/empty, tool results, system envelopes
            if not text or len(text) < 20:
                continue
            if text.startswith('{') or text.startswith('[{'):
                continue  # skip JSON blobs
            parts.append(f"[{role}] {text[:200]}")

        # Take last 15 meaningful exchanges, cap total at 3000 chars
        convo_text = '\n'.join(parts[-15:])[:3000]
        if not convo_text.strip():
            logger.debug("MemPalace bg save: empty conversation, skipping")
            return

        # Ask LLM to extract knowledge
        extract_prompt = f"""以下係一段對話記錄。請提取重要知識並用 JSON 格式返回。

{convo_text[:3000]}

返回格式 (JSON array):
[
  {{
    "content": "提煉過嘅知識（唔係 raw transcript）",
    "wing": "projects|people|knowledge|events|agents",
    "room": "具體名稱 (openclaw/copaw/joe/infra/ai/vesper...)",
    "hall": "hall_facts|hall_events|hall_discoveries|hall_preferences|hall_advice"
  }}
]

規則：
- 只提取有長期價值嘅知識，skip 日常寒暄
- wing/room/hall 按 PALACE_SCHEMA.md 分類
- 如果冇重要嘢，返回空 array []
- 最多 5 條
- 只返回 JSON，唔使解釋"""

        try:
            from agentscope.message import Msg as AgMsg
            formatter = getattr(agent, 'formatter', None)
            if formatter is None:
                mm = getattr(agent, 'memory_manager', None)
                if mm:
                    mm._prepare_model_formatter()
                    formatter = mm.formatter

            # Call model with DashScope-compatible format (bypass formatter)
            # ChatResponse is dict-like: response["content"] = list of TextBlock dicts
            formatted_messages = [
                {"role": "system", "content": "You are a knowledge extractor. Always respond with valid JSON only."},
                {"role": "user", "content": extract_prompt},
            ]
            _mp_log(f"BgSave: calling model type={type(model).__name__}, prompt={len(extract_prompt)} chars")
            response = await model(formatted_messages)

            # Extract text from ChatResponse content blocks
            response_text = ""
            try:
                content_blocks = response["content"] if isinstance(response, dict) else response.content
                _mp_log(f"BgSave: response content_blocks type={type(content_blocks).__name__}, len={len(content_blocks) if content_blocks else 0}")
                if isinstance(content_blocks, list):
                    text_parts = []
                    for block in content_blocks:
                        if isinstance(block, dict) and block.get("type") == "text":
                            text_parts.append(block.get("text", ""))
                        elif isinstance(block, str):
                            text_parts.append(block)
                    response_text = "".join(text_parts).strip()
                elif isinstance(content_blocks, str):
                    response_text = content_blocks.strip()
            except Exception as e:
                _mp_log(f"BgSave: response extraction error: {e}, raw response keys={list(response.keys()) if hasattr(response, 'keys') else dir(response)[:10]}")
                if isinstance(response_text, list):
                    response_text = ' '.join(
                        item.get('text', '') if isinstance(item, dict) else str(item)
                        for item in response_text
                    )

            # Parse JSON response
            response_text = response_text.strip()
            _mp_log(f"BgSave: LLM response length={len(response_text)}, preview={response_text[:100]!r}")

            # Handle markdown code blocks
            if response_text.startswith('```'):
                response_text = response_text.split('\n', 1)[1] if '\n' in response_text else response_text[3:]
                if response_text.endswith('```'):
                    response_text = response_text[:-3]
                response_text = response_text.strip()

            if not response_text:
                _mp_log("BgSave: LLM returned empty response, falling back to diary")
                await _bg_direct_diary(messages, note="llm_empty_response")
                return

            items = json.loads(response_text)
            if not isinstance(items, list):
                items = [items]

        except json.JSONDecodeError as e:
            _mp_log(f"BgSave: JSON parse failed ({e}), response was: {response_text[:200]!r}")
            # Retry once with simpler prompt
            try:
                _mp_log("BgSave: retrying with simpler prompt...")
                retry_prompt = f"Extract 1-3 key facts from this conversation as JSON array. Each item: {{\"content\": \"fact\", \"wing\": \"knowledge\", \"room\": \"general\", \"hall\": \"hall_facts\"}}. ONLY return JSON array, nothing else.\n\n{convo_text[:1500]}"
                retry_msgs = [AgMsg(name="user", role="user", content=retry_prompt)]
                if formatter:
                    formatted = await formatter.format(msgs=retry_msgs)
                    response2 = await model(formatted)
                else:
                    response2 = await model.generate(messages=retry_msgs)
                rt2 = _get_text(getattr(response2, 'text', None) or getattr(response2, 'content', ''))
                rt2 = rt2.strip()
                if rt2.startswith('```'):
                    rt2 = rt2.split('\n', 1)[1] if '\n' in rt2 else rt2[3:]
                    if rt2.endswith('```'): rt2 = rt2[:-3]
                    rt2 = rt2.strip()
                items = json.loads(rt2)
                if not isinstance(items, list): items = [items]
                _mp_log(f"BgSave: retry succeeded, {len(items)} items")
            except Exception as e2:
                _mp_log(f"BgSave: retry also failed ({e2}), falling back to diary")
                await _bg_direct_diary(messages, note=f"llm_json_fail:{str(e)[:50]}")
                return
        except Exception as e:
            _mp_log(f"BgSave: LLM call failed ({e}), falling back to diary")
            await _bg_direct_diary(messages, note=f"llm_error:{str(e)[:50]}")
            return

        # Write extracted knowledge to ChromaDB
        if not items:
            logger.debug("MemPalace bg save: LLM returned empty, writing diary only")
            await _bg_direct_diary(messages)
            return

        from mempalace.chroma_helper import get_collection
        from mempalace.mcp_server import _config

        col = get_collection(palace_path=_config.palace_path)
        now = datetime.now()
        saved = 0

        for item in items[:5]:
            try:
                content = item.get('content', '')
                wing = item.get('wing', 'knowledge')
                room = item.get('room', 'general')
                hall = item.get('hall', 'hall_facts')

                if not content or len(content) < 10:
                    continue

                entry_id = f"drawer_{wing}_{room}_{now.strftime('%Y%m%d_%H%M%S')}_{hashlib.md5(content[:50].encode()).hexdigest()[:8]}"
                col.add(
                    ids=[entry_id],
                    documents=[content],
                    metadatas=[{
                        "wing": wing,
                        "room": room,
                        "hall": hall,
                        "type": "knowledge",
                        "filed_at": now.isoformat(),
                        "date": now.strftime("%Y-%m-%d"),
                        "added_by": "bg_mempalace_save",
                    }],
                )
                saved += 1
            except Exception as e:
                logger.warning(f"MemPalace bg save: failed to write item: {e}")

        _mp_log(f"BgSave: {saved}/{len(items)} knowledge items saved")

        # Also write a diary entry
        await _bg_direct_diary(messages, note=f"bg_extracted:{saved}_items")

    except Exception as e:
        _mp_log(f"BgSave ERROR: {e}")


async def _bg_direct_diary(messages: list, note: str = "") -> None:
    """Fallback: write a quick diary entry without LLM."""
    try:
        import sys, hashlib
        sys.path.insert(0, str(Path.home() / '.local' / 'lib' / 'python3.13' / 'site-packages'))
        from mempalace.chroma_helper import get_collection
        from mempalace.mcp_server import _config

        now = datetime.now()
        col = get_collection(palace_path=_config.palace_path)
        user_count = _count_user_messages(messages)

        entry = f"{now.strftime('%Y-%m-%d')}|session.pre-new|{user_count}msgs|auto|*neutral*|★★★"
        if note:
            entry += f"|{note}"

        entry_id = f"diary_agents_auto_{now.strftime('%Y%m%d_%H%M%S')}_{hashlib.md5(entry.encode()).hexdigest()[:8]}"
        col.add(
            ids=[entry_id],
            documents=[entry],
            metadatas=[{
                "wing": "agents",
                "room": "vesper",
                "hall": "hall_diary",
                "type": "diary_entry",
                "filed_at": now.isoformat(),
                "date": now.strftime("%Y-%m-%d"),
                "added_by": "bg_mempalace_save",
            }],
        )
        _mp_log(f"BgDiary: {entry_id}")
    except Exception as e:
        _mp_log(f"BgDiary FAILED: {e}")
