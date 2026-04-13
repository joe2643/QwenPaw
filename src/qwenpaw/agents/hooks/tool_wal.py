# -*- coding: utf-8 -*-
"""Session Event Log (WAL) — survives process crashes.

Logs 4 event types to {working_dir}/.session_wal.jsonl:
1. reasoning  — AI's decision/plan (post_reasoning)
2. sent       — outbound message to user (post_acting on generate_response)
3. tool_start — tool call about to execute (pre_acting)
4. tool_done  — tool call completed (post_acting)

On crash: tool_start without tool_done = crashed mid-action.
Bootstrap reads WAL and injects recovery context.
"""
import json
import logging
from typing import Any
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)

_DANGEROUS_PATTERNS = [
    "restart", "reboot", "kill -", "pkill",
    "systemctl stop", "systemctl restart",
    "qwenpaw", "shutdown",
    "supervisorctl", "docker restart", "docker stop",
]

_WAL_MAX_LINES = 200  # rotate after 200 entries


def _truncate(s: str, n: int = 200) -> str:
    return s[:n] + "..." if len(s) > n else s


def _get_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                # Tool result blocks, text blocks, etc
                text = item.get("text", item.get("output", ""))
                if isinstance(text, list):
                    text = " ".join(str(t) for t in text)
                parts.append(str(text))
            else:
                parts.append(str(item))
        return " ".join(parts)
    return str(content) if content else ""


class SessionWAL:
    """Append-only event log that survives crashes."""

    def __init__(self, working_dir: Path, wal_file: str = ".session_wal.jsonl"):
        self.wal_path = working_dir / wal_file
        self.wal_path.parent.mkdir(parents=True, exist_ok=True)

    def _append(self, entry: dict):
        try:
            with open(self.wal_path, "a") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            # Rotate if too long
            self._maybe_rotate()
        except Exception as e:
            logger.debug(f"WAL append failed: {e}")

    def _maybe_rotate(self):
        try:
            lines = self.wal_path.read_text().strip().split("\n")
            if len(lines) > _WAL_MAX_LINES:
                # Keep last half
                keep = lines[len(lines)//2:]
                self.wal_path.write_text("\n".join(keep) + "\n")
        except Exception:
            pass

    def _update_last_matching(self, match_type: str, updates: dict):
        """Update the last entry of a given type."""
        try:
            lines = self.wal_path.read_text().strip().split("\n")
            for i in range(len(lines) - 1, -1, -1):
                entry = json.loads(lines[i])
                if entry.get("type") == match_type and entry.get("status") == "pending":
                    entry.update(updates)
                    lines[i] = json.dumps(entry, ensure_ascii=False)
                    self.wal_path.write_text("\n".join(lines) + "\n")
                    return
        except Exception:
            pass

    def log_reasoning(self, content: str):
        """Log AI reasoning output (post_reasoning)."""
        self._append({
            "ts": datetime.now().isoformat(),
            "type": "reasoning",
            "content": _truncate(content, 300),
        })

    def log_sent_message(self, channel: str, to: str, content: str):
        """Log outbound message to user."""
        self._append({
            "ts": datetime.now().isoformat(),
            "type": "sent",
            "channel": channel,
            "to": _truncate(to, 50),
            "content": _truncate(content, 300),
        })

    def log_tool_start(self, tool_name: str, args_summary: str):
        """Log tool call about to execute (pre_acting)."""
        dangerous = any(p in f"{tool_name} {args_summary}".lower() for p in _DANGEROUS_PATTERNS)
        self._append({
            "ts": datetime.now().isoformat(),
            "type": "tool_start",
            "tool": tool_name,
            "args": _truncate(args_summary, 200),
            "dangerous": dangerous,
            "status": "pending",
        })
        if dangerous:
            logger.warning(f"WAL: DANGEROUS tool: {tool_name}({_truncate(args_summary, 80)})")

    def log_tool_done(self, tool_name: str = ""):
        """Mark last pending tool as completed (post_acting)."""
        self._update_last_matching("tool_start", {
            "status": "done",
            "completed_at": datetime.now().isoformat(),
        })

    @staticmethod
    def get_crash_report(working_dir: Path, wal_file: str = ".session_wal.jsonl") -> str | None:
        """Read WAL on startup. If last tool_start is pending = crashed mid-action.
        
        Returns context string for the agent, or None if clean shutdown.
        """
        wal_path = working_dir / wal_file
        try:
            if not wal_path.exists():
                return None
            lines = wal_path.read_text().strip().split("\n")
            if not lines:
                return None

            # Find last tool_start with status=pending
            crashed_tool = None
            for line in reversed(lines):
                try:
                    entry = json.loads(line)
                    if entry.get("type") == "tool_start" and entry.get("status") == "pending":
                        crashed_tool = entry
                        break
                except json.JSONDecodeError:
                    continue

            if not crashed_tool:
                return None

            # Mark as crashed
            for i in range(len(lines) - 1, -1, -1):
                try:
                    entry = json.loads(lines[i])
                    if entry.get("type") == "tool_start" and entry.get("status") == "pending":
                        entry["status"] = "crashed"
                        entry["crash_detected_at"] = datetime.now().isoformat()
                        lines[i] = json.dumps(entry, ensure_ascii=False)
                        break
                except json.JSONDecodeError:
                    continue
            wal_path.write_text("\n".join(lines) + "\n")

            # Build recovery context from last N entries
            recent = []
            for line in lines[-15:]:
                try:
                    recent.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

            parts = ["⚠️ CRASH RECOVERY — your last session crashed mid-action.\n"]
            parts.append("Recent session events before crash:")

            for entry in recent:
                etype = entry.get("type", "?")
                ts = entry.get("ts", "?")[:19]
                if etype == "reasoning":
                    parts.append(f"  [{ts}] 🧠 Thought: {entry.get('content', '')[:150]}")
                elif etype == "sent":
                    parts.append(f"  [{ts}] 📤 Sent to {entry.get('to','?')}: {entry.get('content','')[:100]}")
                elif etype == "tool_start":
                    status = entry.get("status", "?")
                    marker = "💀" if status == "crashed" else "🔧"
                    parts.append(f"  [{ts}] {marker} Tool: {entry.get('tool','')}({entry.get('args','')[:80]}) [{status}]")

            tool = crashed_tool.get("tool", "unknown")
            args = crashed_tool.get("args", "")[:150]
            dangerous = crashed_tool.get("dangerous", False)

            if dangerous:
                parts.append(f"\n🚨 DANGEROUS: '{tool}({args})' likely killed your own process.")
                parts.append("DO NOT repeat without explicit user confirmation.")

            # Count total crashes
            crash_count = sum(1 for l in lines if '"crashed"' in l)
            parts.append(f"\nTotal crash events in log: {crash_count}")

            return "\n".join(parts)

        except Exception as e:
            logger.warning(f"WAL crash check failed: {e}")
            return None

    @staticmethod
    def get_recent(working_dir: Path, n: int = 10, wal_file: str = ".session_wal.jsonl") -> list[dict]:
        wal_path = working_dir / wal_file
        try:
            if not wal_path.exists():
                return []
            lines = wal_path.read_text().strip().split("\n")
            return [json.loads(l) for l in lines[-n:]]
        except Exception:
            return []


class ToolWALPreActingHook:
    """pre_acting: log tool call to WAL before execution."""

    def __init__(self, wal: SessionWAL):
        self.wal = wal

    async def __call__(self, agent, kwargs: dict[str, Any]) -> dict[str, Any] | None:
        try:
            tool_call = kwargs.get("tool_call", {})
            tool_name = tool_call.get("name", "unknown")
            args = tool_call.get("input", {})
            if isinstance(args, dict):
                args_summary = args.get("command", args.get("cmd", json.dumps(args, ensure_ascii=False)[:200]))
            else:
                args_summary = str(args)[:200]
            self.wal.log_tool_start(tool_name, str(args_summary))
        except Exception as e:
            logger.debug(f"WAL pre_acting error: {e}")
        return None


class ToolWALPostActingHook:
    """post_acting: mark tool as done + log sent messages."""

    def __init__(self, wal: SessionWAL):
        self.wal = wal

    async def __call__(self, agent, kwargs: dict[str, Any], output: Any = None) -> Any:
        try:
            tool_call = kwargs.get("tool_call", {})
            tool_name = tool_call.get("name", "")
            self.wal.log_tool_done(tool_name)

            # If tool is generate_response / finish, log the sent message
            if tool_name in ("generate_response", "finish", "reply_user"):
                content = _get_text(getattr(output, 'content', '') if output else '')
                if content:
                    channel = getattr(agent, '_last_channel', 'unknown')
                    to = getattr(agent, '_last_user', 'unknown')
                    self.wal.log_sent_message(channel, to, content)
        except Exception as e:
            logger.debug(f"WAL post_acting error: {e}")
        return output


class ReasoningWALHook:
    """post_reasoning: log AI's reasoning output."""

    def __init__(self, wal: SessionWAL):
        self.wal = wal

    async def __call__(self, agent, kwargs: dict[str, Any], output: Any = None) -> Any:
        try:
            if output is not None:
                content = _get_text(getattr(output, 'content', ''))
                if content:
                    self.wal.log_reasoning(content)
        except Exception as e:
            logger.debug(f"WAL post_reasoning error: {e}")
        return output
