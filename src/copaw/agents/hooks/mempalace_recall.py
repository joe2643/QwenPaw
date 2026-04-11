# -*- coding: utf-8 -*-
"""MemPalace L2 Room Recall Hook — auto-inject relevant context before reasoning.

pre_reasoning hook that:
1. Embeds the latest user message via BGE-M3
2. Searches mempalace semantically (ChromaDB similarity search)
3. Prepends top results as context to the user message

Uses embedding-based semantic search — no keyword matching needed.
ChromaDB's query() handles wing/room relevance naturally via cosine similarity.
"""
import hashlib
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from ._mcp_client import mcp_call

logger = logging.getLogger(__name__)

_MP_LOG_PATH = Path.home() / ".mempalace" / "hook.log"


def _mp_log(msg: str, level: str = "INFO"):
    try:
        _MP_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(_MP_LOG_PATH, "a") as f:
            f.write(f"{ts} | {level} | L2Recall: {msg}\n")
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


_SKIP_PATTERNS = re.compile(
    r"^(/new|/clear|/help|/status|/reload|ok|好|yes|no|thx|thanks|hi|hey|早|晚安)",
    re.IGNORECASE,
)

MAX_CONTEXT_CHARS = 2000
MIN_QUERY_LENGTH = 8
MIN_SIMILARITY = 0.35


class MemPalaceRecallHook:
    """pre_reasoning: L2 Room Recall — semantic search and inject relevant context."""

    def __init__(self, max_results: int = 5):
        self.max_results = max_results
        self._seen_hashes: set[str] = set()

    async def __call__(self, agent, kwargs: dict[str, Any]) -> dict[str, Any] | None:
        try:
            _mp_log("HOOK CALLED")
            messages = await agent.memory.get_memory()
            if not messages:
                _mp_log("SKIP: no messages")
                return None

            # Find last user message
            last_user_text = ""
            last_user_msg = None
            for msg in reversed(messages):
                role = getattr(msg, "role", None)
                if role == "user":
                    last_user_text = _extract_text(
                        getattr(msg, "content", "")
                    ).strip()
                    last_user_msg = msg
                    break

            if not last_user_text or len(last_user_text) < MIN_QUERY_LENGTH:
                _mp_log(f"SKIP: msg too short ({len(last_user_text)} chars)")
                return None

            if _SKIP_PATTERNS.match(last_user_text):
                _mp_log(f"SKIP: command/greeting")
                return None

            # Skip if already injected by previous pre_reasoning call
            if "[MemPalace L2 Context]" in last_user_text:
                _mp_log("SKIP: already injected")
                return None

            msg_hash = hashlib.md5(last_user_text[:200].encode()).hexdigest()[:12]
            if msg_hash in self._seen_hashes:
                _mp_log(f"SKIP: dedup hash={msg_hash}")
                return None

            query = last_user_text[:200].strip()
            _mp_log(f"SEARCHING: query={query[:80]!r}")

            result = mcp_call("mempalace_search", {
                "query": query,
                "limit": self.max_results * 2,  # over-fetch to compensate for wing_general filter
            })

            _mp_log(f"MCP result keys={list(result.keys()) if isinstance(result, dict) else type(result)}, preview={str(result)[:200]}")

            results_list = result.get("results", [])
            if not results_list:
                _mp_log("SKIP: no results from MCP")
                return None

            # Format context
            context_lines = ["[MemPalace L2 Context]"]
            total_chars = 0
            included = 0

            for item in results_list:
                score = item.get("similarity", item.get("score", 0))
                if score < MIN_SIMILARITY:
                    continue
                # Skip wing_general — mostly unclassified raw messages
                if item.get("wing") == "wing_general":
                    continue
                content = item.get("content", item.get("text", item.get("document", "")))
                wing = item.get("wing", "?")
                room = item.get("room", "?")
                snippet = content.strip().replace("\n", " ")
                if len(snippet) > 300:
                    snippet = snippet[:297] + "..."
                line = f"- [{wing}/{room} sim={score:.2f}] {snippet}"
                if total_chars + len(line) > MAX_CONTEXT_CHARS:
                    break
                context_lines.append(line)
                total_chars += len(line)
                included += 1

            if included == 0:
                _mp_log(f"SKIP: {len(results_list)} results but all below similarity {MIN_SIMILARITY}")
                return None

            context_text = "\n".join(context_lines)

            from ..utils.message_processing import prepend_to_message_content
            prepend_to_message_content(last_user_msg, context_text)

            self._seen_hashes.add(msg_hash)

            _mp_log(f"INJECTED {included} results (top_sim={results_list[0].get('similarity', 0):.2f}) for query={query[:60]!r}")
            return None

        except Exception as e:
            import traceback
            _mp_log(f"ERROR: {e}\n{traceback.format_exc()}", "ERROR")
            return None
