# -*- coding: utf-8 -*-
"""In-process SkillClaw session capture.

SkillClaw's ``evolve_server`` pipeline feeds on ``conversations.jsonl``
(one JSON object per turn, OpenAI-chat shape) that its client proxy
captures by MITM-ing ``/v1/chat/completions`` traffic.  Running the
proxy means every CoPaw agent has to point ``OPENAI_BASE_URL`` at it,
which loses CoPaw's Codex OAuth in-process translation and doesn't see
non-OpenAI-compat flows (Anthropic messages, DingTalk channel turns,
etc.) at all.

This hook replaces the proxy: fires on every ``pre_reasoning``, reads
``agent.memory.get_memory()`` (which is the exact message list about
to go to the LLM), serialises to SkillClaw's schema, and appends one
line to ``conversations.jsonl``.  No extra port, no auth middleman,
and every channel gets captured.

Schema — per ``skillclaw/api_server.py:1187`` (the proxy's own writer):

    {
      "session_id": "<str>",
      "turn": <int, 1-based, per session>,
      "timestamp": "YYYY-MM-DD HH:MM:SS",
      "messages": [ {"role": "...", "content": "..."}, ... ]
    }

Downstream consumers (``evolve_server/pipeline/summarizer.py``) group
lines by ``session_id``, pick the highest ``turn`` as the canonical
message list for that session, and hand it to the summarizer LLM.
"""
from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import httpx

logger = logging.getLogger(__name__)


class SkillClawCaptureHook:
    """``pre_reasoning`` hook that publishes each turn to SkillClaw in
    the schema its ``evolve_server`` summarizer expects.  Two
    transports — ``mode="file"`` appends to a local jsonl, ``mode="http"``
    POSTs to a SkillClaw ingest endpoint.  HTTP mode falls back to
    file on transport error so capture never breaks the agent loop.
    """

    def __init__(
        self,
        records_dir: str | Path,
        session_id: str,
        session_id_prefix: str = "",
        mode: Literal["file", "http"] = "file",
        ingest_url: str = "",
        ingest_api_key: str = "",
        workspace_dir: str | Path = "",
        channel_name: str = "all",
    ) -> None:
        resolved = (
            Path(records_dir).expanduser()
            if records_dir
            else (Path.home() / ".skillclaw" / "records")
        )
        resolved.mkdir(parents=True, exist_ok=True)
        self._path = resolved / "conversations.jsonl"
        self._session_id = (
            f"{session_id_prefix}{session_id}"
            if session_id_prefix
            else session_id
        )
        self._turn = 0
        self._mode = mode
        self._ingest_url = ingest_url
        self._ingest_api_key = ingest_api_key
        # Skill attribution context — needed to populate
        # ``injected_skills`` per turn so evolve_server's summarizer
        # can compute ``_skills_referenced`` and run
        # ``evolve_skill_from_sessions`` (otherwise sessions land in
        # the ``NO_SKILL_KEY`` group and evolve only ever creates
        # brand-new skills, never improves existing ones).
        self._workspace_dir = (
            Path(workspace_dir).expanduser() if workspace_dir else None
        )
        self._channel_name = channel_name or "all"
        # Lazy httpx client — created on first http POST so file-mode
        # users don't pay for connection-pool init.  Reused across
        # turns; closed when CoPaw shuts down (we hand off to httpx's
        # GC since the hook lifetime == agent lifetime).
        self._client: httpx.AsyncClient | None = None
        # Guard concurrent appends from overlapping agent invocations
        # sharing the same hook instance.  ``asyncio.Lock`` is enough
        # because agent reasoning is single-task per agent — but belt
        # and suspenders: the file append itself is O_APPEND so even
        # without the lock lines won't interleave mid-write, only
        # ``turn`` counter updates need guarding.
        self._lock = asyncio.Lock()

    async def __call__(
        self,
        agent: Any,
        kwargs: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Capture the current turn's message list and publish a record.

        Runs before the LLM call so the snapshot matches exactly what
        the model will see, identical to what a proxy would record on
        the outbound wire.  Hook never mutates ``kwargs``.
        """
        try:
            messages = await agent.memory.get_memory()
            serialised = [_msg_to_openai_dict(m) for m in messages]

            # Skill attribution — emit fields ``evolve_server.summarizer``
            # uses to compute ``_skills_referenced`` per session.
            injected = self._resolve_injected_skills()
            read, modified = _scan_messages_for_skill_io(messages)

            async with self._lock:
                self._turn += 1
                record = {
                    "session_id": self._session_id,
                    "turn": self._turn,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "messages": serialised,
                    # New: skill attribution context
                    "injected_skills": [{"skill_name": n} for n in injected],
                    "read_skills": [{"skill_name": n} for n in read],
                    "modified_skills": [
                        {"skill_name": n, "action": "modify"} for n in modified
                    ],
                }

                if self._mode == "http" and self._ingest_url:
                    delivered = await self._post_record(record)
                    if delivered:
                        return None
                    # HTTP failed — fall through to file so we don't
                    # silently drop turns when SkillClaw is down.
                    logger.info(
                        "SkillClaw http ingest failed, falling back "
                        "to file (session=%s turn=%d)",
                        self._session_id,
                        self._turn,
                    )

                # File mode (also used as http fallback).  O_APPEND so
                # multiple writers don't tear lines (matches the proxy's
                # own writer).
                with self._path.open("a", encoding="utf-8") as f:
                    f.write(
                        json.dumps(record, ensure_ascii=False) + "\n",
                    )
        except Exception as e:  # pylint: disable=broad-exception-caught
            # Capture must never break the agent loop — log and move on.
            logger.warning(
                "SkillClaw capture failed (session=%s turn=%d): %s",
                self._session_id,
                self._turn,
                e,
            )
        return None

    def _resolve_injected_skills(self) -> list[str]:
        """Return the workspace skills enabled for this hook's channel.

        Mirrors what the ReAct agent's prompt builder injects, so the
        capture record matches the actual prompt-time skill set.
        Lookup failures (missing workspace_dir, broken manifest) are
        silently absorbed — the hook stays best-effort.
        """
        if self._workspace_dir is None:
            return []
        try:
            from ...agents.skills_manager import resolve_effective_skills

            return list(
                resolve_effective_skills(
                    self._workspace_dir,
                    self._channel_name,
                ),
            )
        except Exception:  # pylint: disable=broad-exception-caught
            return []

    async def _post_record(self, record: dict[str, Any]) -> bool:
        """POST a record to the SkillClaw ingest endpoint.  Returns
        ``True`` on success (HTTP 2xx), ``False`` on any failure so
        the caller can fall back to file mode.  Never raises — logs
        and absorbs all transport errors."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(10, connect=5),
            )
        headers = {"Content-Type": "application/json"}
        if self._ingest_api_key:
            headers["Authorization"] = f"Bearer {self._ingest_api_key}"
        try:
            r = await self._client.post(
                self._ingest_url,
                json=record,
                headers=headers,
            )
            if 200 <= r.status_code < 300:
                return True
            logger.warning(
                "SkillClaw ingest %s returned HTTP %d: %s",
                self._ingest_url,
                r.status_code,
                r.text[:200],
            )
            return False
        except (httpx.HTTPError, OSError) as e:
            logger.warning(
                "SkillClaw ingest POST failed: %s",
                e,
            )
            return False


_SKILL_PATH_PATTERNS = (
    "/.copaw/skill_pool/",
    "/.copaw/workspaces/",
    "/.skillclaw/skills/",
    "/.skillclaw/local-share/",
)


def _extract_skill_name_from_path(path: str) -> str | None:
    """Extract the skill name from a tool-call path argument.

    Expected shapes:
      ``~/.copaw/skill_pool/<name>/SKILL.md``
      ``~/.copaw/workspaces/<ws>/skills/<name>/...``
      ``~/.skillclaw/skills/<name>/...``

    Returns the segment immediately following ``skill_pool/`` /
    ``skills/`` / ``local-share/qwenpaw/skills/``.  Returns ``None``
    when the path doesn't look like a skill path at all.
    """
    if not isinstance(path, str) or not any(
        p in path for p in _SKILL_PATH_PATTERNS
    ):
        return None
    # Try most specific markers first.
    for marker in (
        "/skill_pool/",
        "/local-share/qwenpaw/skills/",
    ):
        if marker in path:
            tail = path.split(marker, 1)[1]
            seg = tail.split("/", 1)[0].strip()
            if seg and seg not in {"skill.json", "skill_stats.json"}:
                return seg
    if "/skills/" in path:
        tail = path.split("/skills/", 1)[1]
        seg = tail.split("/", 1)[0].strip()
        if seg and seg not in {"skill.json", "skill_stats.json"}:
            return seg
    return None


def _scan_messages_for_skill_io(
    messages: list[Any],
) -> tuple[list[str], list[str]]:
    """Walk every assistant tool-use block in ``messages`` and bucket
    skill-touching ones into ``read_skills`` / ``modified_skills``.

    Read tools: ``read_file``, ``view_file``, ``cat`` (anything that
    reads a SKILL.md or anything under a skill dir).
    Write tools: ``write_file``, ``edit_file`` (mutate skill content).

    Returns deduped, sorted lists.  Over-attributes by intention — a
    cumulative scan across the message history means each turn's
    record reports every skill the LLM has touched in the session so
    far, but evolve_server's summarizer aggregates as a set anyway.
    """
    READ_TOOLS = {"read_file", "view_file", "cat", "read_skill"}
    WRITE_TOOLS = {"write_file", "edit_file", "write_skill"}
    read: set[str] = set()
    modified: set[str] = set()
    for msg in messages or []:
        content = getattr(msg, "content", None)
        if not isinstance(content, list):
            continue
        for blk in content:
            if not isinstance(blk, dict):
                continue
            if blk.get("type") != "tool_use":
                continue
            tool_name = str(blk.get("name", "") or "").strip()
            inp = blk.get("input") or blk.get("arguments") or {}
            path = ""
            if isinstance(inp, dict):
                path = str(
                    inp.get("path")
                    or inp.get("file_path")
                    or inp.get("file")
                    or "",
                )
            skill = _extract_skill_name_from_path(path)
            if not skill:
                continue
            if tool_name in READ_TOOLS:
                read.add(skill)
            elif tool_name in WRITE_TOOLS:
                modified.add(skill)
    return sorted(read), sorted(modified)


def _msg_to_openai_dict(msg: Any) -> dict[str, Any]:
    """Flatten an agentscope ``Msg`` into the ``{role, content}`` dict
    SkillClaw expects.

    Content can be a str OR a list of typed blocks.  The evolve
    pipeline treats content as a text corpus, so we collapse blocks
    to a text representation:

    - ``TextBlock`` → raw text
    - ``ThinkingBlock`` → ``[thinking: ...]`` marker (kept for signal)
    - ``ToolUseBlock`` → ``[tool_call: name({args})]``
    - ``ToolResultBlock`` → ``[tool_result: ...]``
    - ``Image/Audio/VideoBlock`` → placeholder with source hint

    This matches what the proxy's on-wire capture would see for a
    non-vision text-only conversation, and is lossy-but-meaningful
    for multimodal turns (the evolve pipeline primarily reasons over
    text anyway).
    """
    role = getattr(msg, "role", "user")
    content = getattr(msg, "content", "")

    if isinstance(content, str):
        return {"role": role, "content": content}

    if not isinstance(content, list):
        return {"role": role, "content": str(content or "")}

    parts: list[str] = []
    for blk in content:
        if not isinstance(blk, dict):
            parts.append(str(blk))
            continue
        t = blk.get("type")
        if t == "text":
            parts.append(str(blk.get("text", "")))
        elif t == "thinking":
            think = str(blk.get("thinking", "") or blk.get("text", ""))
            if think:
                parts.append(f"[thinking: {think}]")
        elif t == "tool_use":
            name = blk.get("name", "")
            inp = blk.get("input", blk.get("arguments", ""))
            try:
                inp_s = (
                    json.dumps(inp, ensure_ascii=False)
                    if not isinstance(inp, str)
                    else inp
                )
            except Exception:
                inp_s = str(inp)
            parts.append(f"[tool_call: {name}({inp_s})]")
        elif t == "tool_result":
            output = blk.get("output", blk.get("content", ""))
            # tool_result.output can itself be a list of sub-blocks —
            # flatten to text for corpus purposes.
            if isinstance(output, list):
                sub = []
                for o in output:
                    if isinstance(o, dict) and o.get("type") == "text":
                        sub.append(str(o.get("text", "")))
                    else:
                        sub.append(str(o))
                output = "".join(sub)
            parts.append(f"[tool_result: {output}]")
        elif t == "image":
            parts.append(
                f"[image: {blk.get('source', {}).get('type', 'inline')}]",
            )
        elif t in ("audio", "video"):
            parts.append(f"[{t}]")
        else:
            parts.append(str(blk))

    return {"role": role, "content": "".join(parts)}
