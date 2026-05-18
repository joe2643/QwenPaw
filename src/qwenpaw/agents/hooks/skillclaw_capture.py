# -*- coding: utf-8 -*-
"""In-process SkillClaw session capture.

SkillClaw's ``evolve_server`` pipeline feeds on ``conversations.jsonl``
(one JSON object per turn, OpenAI-chat shape) that its client proxy
captures by MITM-ing ``/v1/chat/completions`` traffic.  Running the
proxy means every CoPaw agent has to point ``OPENAI_BASE_URL`` at it,
which loses CoPaw's Codex OAuth in-process translation and doesn't see
non-OpenAI-compat flows (Anthropic messages, DingTalk channel turns,
etc.) at all.

This hook replaces the proxy in-process.  Its ``pre_reasoning`` hook
injects the same OpenClaw/SkillClaw XML skill catalog into the system
prompt and snapshots the outbound prompt.  Its ``post_reasoning`` hook
serialises the model response, tool calls, skill attribution, and later
tool-result/error patches into SkillClaw's ingest schema.  No extra
model-facing proxy is required, and every channel gets captured.

Schema — per ``skillclaw/api_server.py:1187`` (the proxy's own writer):

    {
      "session_id": "<str>",
      "turn": <int, 1-based, per session>,
      "timestamp": "YYYY-MM-DD HH:MM:SS",
    "messages": [ {"role": "...", "content": "..."}, ... ],
    "prompt_text": "...",
    "response_text": "...",
    "tool_calls": [...]
    }

Downstream consumers (``evolve_server/pipeline/summarizer.py``) group
lines by ``session_id``, pick the highest ``turn`` as the canonical
message list for that session, and hand it to the summarizer LLM.
"""
from __future__ import annotations

import asyncio
import glob
import hashlib
import json
import logging
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import httpx
import yaml

logger = logging.getLogger(__name__)

_READ_TOOL_NAMES = {"read", "file_read", "read_file", "readfile"}
_SKILL_WRITE_TOOL_NAMES = {
    "write",
    "file_write",
    "write_file",
    "writefile",
    "create_file",
    "edit",
    "edit_file",
    "replace",
    "replace_in_file",
    "append",
    "append_file",
    "patch",
    "apply_patch",
    "move",
    "rename",
    "mv",
}
_SHELL_TOOL_NAMES = {"shell", "exec", "bash", "terminal", "exec_command"}
_PATCH_PATH_RE = re.compile(
    r"^\*\*\* (?:Add|Update|Delete) File: (.+)$",
    re.MULTILINE,
)
_SHELL_SKILL_PATH_RE = re.compile(
    r"([~./A-Za-z0-9_\-][^\n\"'`]*?"
    r"(?:SKILL\.md|references/[^\s\"'`]+|scripts/[^\s\"'`]+|assets/[^\s\"'`]+|history/[^\s\"'`]+))",
)
_TOOL_RESULT_CONTENT_MAX_CHARS = 4_000
_ERROR_PATTERNS: list[tuple[re.Pattern, str]] = [
    (
        re.compile(
            r"exited with code (?!0\b)\d+|exit code (?!0\b)\d+|exit status (?!0\b)\d+",
            re.IGNORECASE,
        ),
        "exit_code",
    ),
    (
        re.compile(
            r"Traceback \(most recent call last\)|\.py\", line \d+",
            re.IGNORECASE,
        ),
        "traceback",
    ),
    (
        re.compile(r"Permission denied|EACCES|PermissionError", re.IGNORECASE),
        "permission",
    ),
    (
        re.compile(
            r"No such file|FileNotFoundError|ENOENT|not found",
            re.IGNORECASE,
        ),
        "not_found",
    ),
    (
        re.compile(
            r"command not found|not recognized as|is not recognized",
            re.IGNORECASE,
        ),
        "command_not_found",
    ),
    (
        re.compile(r"timed?\s*out|TimeoutError|ETIMEDOUT", re.IGNORECASE),
        "timeout",
    ),
    (
        re.compile(r"(?:^|\W)(?:Error|Exception):\s", re.MULTILINE),
        "generic_error",
    ),
]


class _PromptMsg:
    def __init__(self, role: str, content: Any) -> None:
        self.role = role
        self.content = content


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
        inject_catalog: bool = True,
        skills_dir: str | Path = "",
        skills_public_root: str | Path = "",
        max_skills_prompt_chars: int = 30_000,
        read_tool_name: str = "read_file",
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
        self._inject_catalog = bool(inject_catalog)
        cfg_skills_dir, cfg_public_root = _read_skillclaw_skill_config()
        self._skills_dir = Path(
            skills_dir
            or cfg_skills_dir
            or Path.home() / ".copaw" / "skill_pool",
        ).expanduser()
        self._skills_public_root = (
            str(Path(skills_public_root).expanduser())
            if skills_public_root
            else str(cfg_public_root or "")
        )
        self._max_skills_prompt_chars = int(max_skills_prompt_chars or 30_000)
        self._read_tool_name = (
            str(read_tool_name or "read_file").strip() or "read_file"
        )
        self._catalog_fingerprint: tuple[tuple[str, int, int], ...] = ()
        self._catalog_skills: list[dict[str, Any]] = []
        self._skill_path_map: dict[str, dict[str, str]] = {}
        self._last_applied_injection = ""
        self._pending_prompt_messages: list[Any] = []
        self._pending_injected_skills: list[str] = []
        self._last_record: dict[str, Any] | None = None
        self._last_patched_turn = 0
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
        """Backward-compatible pre-reasoning capture entry point.

        Production registration uses ``pre_reasoning`` + ``post_reasoning``
        so records are emitted after the model response, matching the
        SkillClaw proxy.  Some tests and older configs call the hook
        directly; keep that path best-effort and pre-response.
        """
        await self.pre_reasoning(agent, kwargs)
        try:
            messages = (
                self._pending_prompt_messages
                or await self._prompt_messages(agent)
            )
            read, modified = _scan_messages_for_skill_io(messages)
            injected = self._pending_injected_skills or (
                self._catalog_skill_names()
                if self._inject_catalog
                else self._resolve_injected_skills()
            )
            record = self._build_record(
                messages=messages,
                injected=injected,
                response_text="",
                tool_calls=[],
                read_skills=[{"skill_name": n} for n in read],
                modified_skills=[
                    {"skill_name": n, "action": "modify"} for n in modified
                ],
            )
            await self._publish_new_record(record)
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.warning(
                "SkillClaw capture failed (session=%s turn=%d): %s",
                self._session_id,
                self._turn,
                e,
            )
        return None

    async def pre_reasoning(
        self,
        agent: Any,
        kwargs: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Inject SkillClaw catalog and snapshot the prompt before LLM call."""
        try:
            await self._patch_previous_turn_with_tool_results(agent)
            injected: list[str]
            if self._inject_catalog:
                skill_text, injected = self._skillclaw_injection_prompt()
                self._apply_skillclaw_injection(agent, skill_text)
            else:
                injected = self._resolve_injected_skills()
            self._pending_prompt_messages = await self._prompt_messages(agent)
            self._pending_injected_skills = injected
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.warning(
                "SkillClaw pre_reasoning failed (session=%s turn=%d): %s",
                self._session_id,
                self._turn,
                e,
            )
        return None

    async def post_reasoning(
        self,
        agent: Any,
        kwargs: dict[str, Any],
        output: Any,
    ) -> Any:
        """Publish a proxy-style record after the model response."""
        try:
            messages = (
                self._pending_prompt_messages
                or await self._prompt_messages(agent)
            )
            injected = self._pending_injected_skills or (
                self._catalog_skill_names()
                if self._inject_catalog
                else self._resolve_injected_skills()
            )
            tool_calls = _extract_tool_calls_from_msg(output)
            response_text = _response_text_from_msg(output, tool_calls)
            reasoning = _reasoning_content_from_msg(output)
            (
                read_skills,
                modified_skills,
            ) = self._extract_skills_from_tool_calls(
                tool_calls,
            )
            record = self._build_record(
                messages=messages,
                injected=injected,
                response_text=response_text,
                tool_calls=tool_calls,
                read_skills=read_skills,
                modified_skills=modified_skills,
                reasoning_content=reasoning,
            )
            await self._publish_new_record(record)
            self._last_record = json.loads(
                json.dumps(record, ensure_ascii=False),
            )
            self._last_patched_turn = 0
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.warning(
                "SkillClaw post_reasoning failed (session=%s turn=%d): %s",
                self._session_id,
                self._turn,
                e,
            )
        finally:
            self._pending_prompt_messages = []
            self._pending_injected_skills = []
        return None

    async def post_acting(
        self,
        agent: Any,
        kwargs: dict[str, Any],
        output: Any,
    ) -> Any:
        """Patch the previous turn with tool result/error signal."""
        try:
            await self._patch_previous_turn_with_tool_results(agent)
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.warning(
                "SkillClaw post_acting patch failed (session=%s turn=%d): %s",
                self._session_id,
                self._turn,
                e,
            )
        return None

    async def _prompt_messages(self, agent: Any) -> list[Any]:
        messages = await agent.memory.get_memory()
        system_prompt = str(getattr(agent, "sys_prompt", "") or "")
        if system_prompt:
            return [_PromptMsg("system", system_prompt), *messages]
        return list(messages or [])

    def _build_record(
        self,
        *,
        messages: list[Any],
        injected: list[str],
        response_text: str,
        tool_calls: list[dict[str, Any]],
        read_skills: list[dict[str, Any]],
        modified_skills: list[dict[str, Any]],
        reasoning_content: str = "",
    ) -> dict[str, Any]:
        serialised = [_msg_to_openai_dict(m) for m in messages]
        prompt_text = "\n".join(
            f"{m.get('role', '?')}: {m.get('content', '')}" for m in serialised
        )
        record: dict[str, Any] = {
            "session_id": self._session_id,
            "turn": 0,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "messages": serialised,
            "instruction_text": _last_user_instruction(serialised),
            "prompt_text": prompt_text,
            "response_text": response_text,
            "tool_calls": tool_calls or None,
            "tool_results": _build_tool_summaries(tool_calls),
            "tool_observations": [],
            "tool_errors": [],
            "injected_skills": [{"skill_name": n} for n in injected],
            "read_skills": read_skills,
            "modified_skills": modified_skills,
            "prm_score": None,
        }
        if reasoning_content:
            record["reasoning_content"] = reasoning_content
        return record

    async def _publish_new_record(self, record: dict[str, Any]) -> None:
        async with self._lock:
            self._turn += 1
            record["turn"] = self._turn
            await self._publish_record(record)

    async def _publish_record(self, record: dict[str, Any]) -> None:
        if self._mode == "http" and self._ingest_url:
            delivered = await self._post_record(record)
            if delivered:
                return
            logger.info(
                "SkillClaw http ingest failed, falling back to file "
                "(session=%s turn=%d)",
                self._session_id,
                int(record.get("turn") or 0),
            )

        with self._path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    async def _patch_previous_turn_with_tool_results(self, agent: Any) -> None:
        record = self._last_record
        if not record or not record.get("tool_calls"):
            return
        turn_num = int(record.get("turn") or 0)
        if not turn_num or self._last_patched_turn == turn_num:
            return
        messages = await agent.memory.get_memory()
        results = _extract_recent_tool_results(messages)
        if not results:
            return
        _merge_tool_results(record, results)
        async with self._lock:
            await self._publish_record(record)
        self._last_patched_turn = turn_num

    def _apply_skillclaw_injection(self, agent: Any, skill_text: str) -> None:
        if not hasattr(agent, "_sys_prompt"):
            return
        base = str(getattr(agent, "_sys_prompt", "") or "")
        if self._last_applied_injection:
            suffix = "\n\n" + self._last_applied_injection
            if base.endswith(suffix):
                base = base[: -len(suffix)]
        if skill_text:
            setattr(agent, "_sys_prompt", base + "\n\n" + skill_text)
            self._last_applied_injection = skill_text
        else:
            setattr(agent, "_sys_prompt", base)
            self._last_applied_injection = ""

    def _skillclaw_injection_prompt(self) -> tuple[str, list[str]]:
        self._refresh_catalog_if_changed()
        skills = self._catalog_skills
        if not skills:
            return "", []
        full_prompt = _format_skills_for_prompt(
            skills,
            self._skills_public_root,
        )
        if len(full_prompt) <= self._max_skills_prompt_chars:
            catalog = full_prompt
        else:
            catalog = _format_skills_compact(skills, self._skills_public_root)
        skill_text = _build_skills_section(catalog, self._read_tool_name)
        return skill_text, self._catalog_skill_names()

    def _catalog_skill_names(self) -> list[str]:
        return [
            str(s.get("name", "") or "").strip()
            for s in self._catalog_skills
            if str(s.get("name", "") or "").strip()
        ]

    def _refresh_catalog_if_changed(self) -> None:
        fingerprint = _skill_catalog_fingerprint(self._skills_dir)
        if fingerprint == self._catalog_fingerprint:
            return
        self._catalog_fingerprint = fingerprint
        self._catalog_skills = _load_skill_catalog(self._skills_dir)
        self._skill_path_map = _build_skill_path_map(
            self._catalog_skills,
            self._skills_public_root,
        )

    def _extract_skills_from_tool_calls(
        self,
        tool_calls: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        self._refresh_catalog_if_changed()
        return (
            _extract_read_skills_from_tool_calls(
                tool_calls,
                self._skill_path_map,
            ),
            _extract_modified_skills_from_tool_calls(
                tool_calls,
                self._skill_path_map,
            ),
        )

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


def _read_skillclaw_skill_config() -> tuple[str, str]:
    path = Path.home() / ".skillclaw" / "config.yaml"
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return "", ""
    if not isinstance(data, dict):
        return "", ""
    skills = data.get("skills") or {}
    if not isinstance(skills, dict):
        return "", ""
    return (
        str(skills.get("dir") or "").strip(),
        str(skills.get("public_root") or "").strip(),
    )


def _parse_skill_md(path: str) -> dict[str, Any] | None:
    try:
        raw = Path(path).read_text(encoding="utf-8")
    except OSError:
        return None
    if not raw.startswith("---"):
        return None
    end_idx = raw.find("\n---", 3)
    if end_idx == -1:
        return None
    try:
        frontmatter = yaml.safe_load(raw[3:end_idx].strip()) or {}
    except yaml.YAMLError:
        return None
    if not isinstance(frontmatter, dict):
        return None
    name = str(frontmatter.get("name", "") or "").strip()
    description = str(frontmatter.get("description", "") or "").strip()
    if not name or not description:
        return None
    extra = {
        k: v
        for k, v in frontmatter.items()
        if k not in {"name", "description", "metadata", "category"}
    }
    skill: dict[str, Any] = {
        "id": hashlib.sha256(name.encode()).hexdigest()[:12],
        "name": name,
        "description": description,
        "file_path": os.path.realpath(path),
    }
    if extra:
        skill["_extra_frontmatter"] = extra
    return skill


def _skill_md_paths(skills_dir: Path) -> list[str]:
    root = str(skills_dir.expanduser())
    if os.path.realpath(root) == os.path.realpath(
        os.path.join(os.path.expanduser("~"), ".hermes", "skills"),
    ):
        return sorted(
            glob.glob(os.path.join(root, "**", "SKILL.md"), recursive=True),
        )
    return sorted(glob.glob(os.path.join(root, "*", "SKILL.md")))


def _skill_catalog_fingerprint(
    skills_dir: Path,
) -> tuple[tuple[str, int, int], ...]:
    out: list[tuple[str, int, int]] = []
    for path in _skill_md_paths(skills_dir):
        try:
            stat = os.stat(path)
        except OSError:
            continue
        out.append(
            (os.path.realpath(path), int(stat.st_mtime_ns), int(stat.st_size)),
        )
    return tuple(out)


def _load_skill_catalog(skills_dir: Path) -> list[dict[str, Any]]:
    skills: list[dict[str, Any]] = []
    for path in _skill_md_paths(skills_dir):
        skill = _parse_skill_md(path)
        if skill is None:
            continue
        if skill.get("_extra_frontmatter", {}).get(
            "disable-model-invocation",
            False,
        ):
            continue
        skills.append(skill)
    return skills


def _escape_xml(text: Any) -> str:
    return (
        str(text or "")
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
    )


def _public_skill_path(skill: dict[str, Any], public_root: str) -> str:
    root = str(public_root or "").strip()
    name = str(skill.get("name", "") or "").strip()
    if not root or not name:
        return ""
    return os.path.join(root, name, "SKILL.md")


def _format_skills_for_prompt(
    skills: list[dict[str, Any]],
    public_root: str,
) -> str:
    if not skills:
        return ""
    lines = [
        "\n\nThe following skills provide specialized instructions for specific tasks.",
        "Use the read tool to load a skill's file when the task matches its description.",
        "When a skill file references a relative path, resolve it against the skill "
        "directory (parent of SKILL.md / dirname of the path) and use that absolute "
        "path in tool commands.",
        "",
        "<available_skills>",
    ]
    for skill in skills:
        lines.append("  <skill>")
        lines.append(f"    <name>{_escape_xml(skill.get('name', ''))}</name>")
        lines.append(
            f"    <description>{_escape_xml(skill.get('description', ''))}</description>",
        )
        location = _public_skill_path(skill, public_root) or skill.get(
            "file_path",
            "",
        )
        lines.append(f"    <location>{_escape_xml(location)}</location>")
        lines.append("  </skill>")
    lines.append("</available_skills>")
    return "\n".join(lines)


def _format_skills_compact(
    skills: list[dict[str, Any]],
    public_root: str,
) -> str:
    if not skills:
        return ""
    lines = [
        "\n\nThe following skills provide specialized instructions for specific tasks.",
        "Use the read tool to load a skill's file when the task matches its name.",
        "When a skill file references a relative path, resolve it against the skill "
        "directory (parent of SKILL.md / dirname of the path) and use that absolute "
        "path in tool commands.",
        "",
        "<available_skills>",
    ]
    for skill in skills:
        lines.append("  <skill>")
        lines.append(f"    <name>{_escape_xml(skill.get('name', ''))}</name>")
        location = _public_skill_path(skill, public_root) or skill.get(
            "file_path",
            "",
        )
        lines.append(f"    <location>{_escape_xml(location)}</location>")
        lines.append("  </skill>")
    lines.append("</available_skills>")
    return "\n".join(lines)


def _build_skills_section(skills_prompt: str, read_tool_name: str) -> str:
    trimmed = skills_prompt.strip()
    if not trimmed:
        return ""
    return "\n".join(
        [
            "## Skills (mandatory)",
            "Before replying: scan <available_skills> <description> entries.",
            f"- If exactly one skill clearly applies: read its SKILL.md at "
            f"<location> with `{read_tool_name}`, then follow it.",
            "- If multiple could apply: choose the most specific one, then read/follow it.",
            "- If none clearly apply: do not read any SKILL.md.",
            "Constraints: never read more than one skill up front; only read after selecting.",
            "- When a skill drives external API writes, assume rate limits: prefer fewer "
            "larger writes, avoid tight one-item loops, serialize bursts when possible, "
            "and respect 429/Retry-After.",
            trimmed,
            "",
        ],
    )


def _list_skill_bundle_paths(skill_dir: str) -> list[str]:
    root = Path(skill_dir)
    if not root.is_dir():
        return []
    rels = ["SKILL.md"] if (root / "SKILL.md").is_file() else []
    for folder in ("scripts", "references", "assets", "history"):
        base = root / folder
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*")):
            if path.is_file():
                rels.append(path.relative_to(root).as_posix())
    return rels


def _build_skill_path_map(
    skills: list[dict[str, Any]],
    public_root: str,
) -> dict[str, dict[str, str]]:
    path_map: dict[str, dict[str, str]] = {}
    for skill in skills:
        skill_dir = os.path.dirname(str(skill.get("file_path", "") or ""))
        bundle_paths = _list_skill_bundle_paths(skill_dir) or ["SKILL.md"]
        public_dir = os.path.dirname(_public_skill_path(skill, public_root))
        for rel_path in bundle_paths:
            locations = []
            if skill_dir:
                locations.append(
                    os.path.realpath(os.path.join(skill_dir, rel_path)),
                )
            if public_dir:
                locations.append(
                    os.path.realpath(os.path.join(public_dir, rel_path)),
                )
            for file_path in locations:
                path_map[file_path] = {
                    "skill_id": str(skill.get("id", "") or ""),
                    "skill_name": str(skill.get("name", "") or ""),
                }
    return path_map


def _normalize_tool_call_name(raw_name: Any) -> str:
    name = str(raw_name or "").strip()
    if name.startswith("functions."):
        return name.split(".", 1)[1]
    return name


def _tool_call_args(tool_call: dict[str, Any]) -> dict[str, Any]:
    func = tool_call.get("function", {}) if isinstance(tool_call, dict) else {}
    args_raw = func.get("arguments", "{}")
    if isinstance(args_raw, dict):
        return args_raw
    if not isinstance(args_raw, str):
        return {}
    try:
        loaded = json.loads(args_raw)
    except Exception:
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _looks_like_path(value: str) -> bool:
    text = str(value or "").strip()
    if not text or text in {".", ".."}:
        return False
    return (
        "/" in text
        or "\\" in text
        or text.startswith("~")
        or text.endswith("SKILL.md")
    )


def _deduplicate_paths(paths: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for path in paths:
        clean = str(path or "").strip()
        if not clean or clean in seen:
            continue
        seen.add(clean)
        out.append(clean)
    return out


def _extract_skill_paths_from_args_dict(args: dict[str, Any]) -> list[str]:
    paths: list[str] = []
    for key in (
        "path",
        "file",
        "file_path",
        "target",
        "destination",
        "dest",
        "to",
        "source",
        "src",
        "old_path",
        "new_path",
    ):
        value = args.get(key)
        if isinstance(value, str) and _looks_like_path(value):
            paths.append(value.strip())
    raw_paths = args.get("paths")
    if isinstance(raw_paths, list):
        paths.extend(
            item.strip()
            for item in raw_paths
            if isinstance(item, str) and _looks_like_path(item)
        )
    return _deduplicate_paths(paths)


def _extract_skill_paths_from_tool_call(
    tool_call: dict[str, Any],
) -> tuple[str, list[str]]:
    func = tool_call.get("function", {}) if isinstance(tool_call, dict) else {}
    tool_name = _normalize_tool_call_name(func.get("name") or "")
    args_raw = func.get("arguments", "{}")
    args = _tool_call_args(tool_call)
    paths = _extract_skill_paths_from_args_dict(args)
    if tool_name.lower() in _SHELL_TOOL_NAMES:
        command = str(args.get("command") or args.get("cmd") or args_raw or "")
        paths.extend(
            match.group(1).strip()
            for match in _SHELL_SKILL_PATH_RE.finditer(command)
            if match.group(1).strip()
        )
    if tool_name.lower() in {"apply_patch", "patch"}:
        text = str(args_raw or "")
        paths.extend(
            match.group(1).strip()
            for match in _PATCH_PATH_RE.finditer(text)
            if match.group(1).strip()
        )
    return tool_name, _deduplicate_paths(paths)


def _resolve_skill_reference(
    path: str,
    skill_path_map: dict[str, dict[str, str]],
) -> dict[str, str]:
    expanded = os.path.expanduser(str(path or "").strip())
    real_path = os.path.realpath(expanded) if expanded else ""
    skill_info = (
        skill_path_map.get(real_path)
        or skill_path_map.get(expanded)
        or skill_path_map.get(str(path or "").strip())
    )
    if skill_info:
        return {
            "skill_id": str(skill_info.get("skill_id", "") or ""),
            "skill_name": str(skill_info.get("skill_name", "") or ""),
            "path": str(path or "").strip(),
        }
    return {"skill_id": "", "skill_name": "", "path": str(path or "").strip()}


def _extract_read_skills_from_tool_calls(
    tool_calls: list[dict[str, Any]],
    skill_path_map: dict[str, dict[str, str]],
) -> list[dict[str, Any]]:
    read_skills: list[dict[str, Any]] = []
    seen: set[str] = set()
    for tc in tool_calls:
        tool_name, skill_paths = _extract_skill_paths_from_tool_call(tc)
        if tool_name.lower() not in _READ_TOOL_NAMES:
            continue
        for path in skill_paths:
            skill_ref = _resolve_skill_reference(path, skill_path_map)
            dedupe = skill_ref.get("skill_id") or skill_ref.get("skill_name")
            if not dedupe or dedupe in seen:
                continue
            read_skills.append(skill_ref)
            seen.add(dedupe)
    return read_skills


def _extract_modified_skills_from_tool_calls(
    tool_calls: list[dict[str, Any]],
    skill_path_map: dict[str, dict[str, str]],
) -> list[dict[str, Any]]:
    modified_skills: list[dict[str, Any]] = []
    seen: set[str] = set()
    for tc in tool_calls:
        tool_name, skill_paths = _extract_skill_paths_from_tool_call(tc)
        normalized = tool_name.lower()
        if normalized in _READ_TOOL_NAMES:
            continue
        if (
            normalized not in _SKILL_WRITE_TOOL_NAMES
            and normalized not in _SHELL_TOOL_NAMES
        ):
            continue
        for path in skill_paths:
            skill_ref = _resolve_skill_reference(path, skill_path_map)
            dedupe = skill_ref.get("skill_id") or skill_ref.get("skill_name")
            if not dedupe or dedupe in seen:
                continue
            modified_skills.append(
                {
                    **skill_ref,
                    "action": "shell"
                    if normalized in _SHELL_TOOL_NAMES
                    else normalized,
                },
            )
            seen.add(dedupe)
    return modified_skills


def _tool_use_block_to_openai(
    block: dict[str, Any],
    index: int,
) -> dict[str, Any]:
    args = block.get("input", block.get("arguments", {}))
    if isinstance(args, str):
        args_s = args
    else:
        try:
            args_s = json.dumps(args or {}, ensure_ascii=False)
        except Exception:
            args_s = "{}"
    return {
        "id": str(block.get("id") or f"call_{index}"),
        "type": "function",
        "function": {
            "name": _normalize_tool_call_name(
                block.get("name") or "unknown_tool",
            ),
            "arguments": args_s,
        },
    }


def _extract_tool_calls_from_msg(msg: Any) -> list[dict[str, Any]]:
    content = getattr(msg, "content", None)
    if not isinstance(content, list):
        return []
    tool_calls: list[dict[str, Any]] = []
    for block in content:
        if isinstance(block, dict) and block.get("type") == "tool_use":
            tool_calls.append(
                _tool_use_block_to_openai(block, len(tool_calls)),
            )
    return tool_calls


def _text_from_blocks(blocks: list[Any]) -> str:
    parts: list[str] = []
    for block in blocks:
        if not isinstance(block, dict):
            parts.append(str(block))
            continue
        if block.get("type") == "text":
            parts.append(str(block.get("text", "") or ""))
    return "".join(parts).strip()


def _response_text_from_msg(msg: Any, tool_calls: list[dict[str, Any]]) -> str:
    content = getattr(msg, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text = _text_from_blocks(content)
        if text:
            return text
    return json.dumps(tool_calls, ensure_ascii=False) if tool_calls else ""


def _reasoning_content_from_msg(msg: Any) -> str:
    content = getattr(msg, "content", None)
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for block in content:
        if isinstance(block, dict) and block.get("type") == "thinking":
            parts.append(str(block.get("thinking") or block.get("text") or ""))
    return "\n".join(part for part in parts if part).strip()


def _flatten_tool_output(output: Any) -> str:
    if isinstance(output, str):
        return output
    if isinstance(output, list):
        parts: list[str] = []
        for item in output:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "") or ""))
            else:
                parts.append(str(item))
        return "".join(parts)
    return str(output or "")


def _classify_tool_error(content: str) -> tuple[bool, str | None]:
    for pattern, error_type in _ERROR_PATTERNS:
        if pattern.search(content):
            return True, error_type
    return False, None


def _extract_recent_tool_results(messages: list[Any]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for msg in reversed(messages or []):
        content = getattr(msg, "content", None)
        blocks = content if isinstance(content, list) else []
        tool_blocks = [
            block
            for block in blocks
            if isinstance(block, dict) and block.get("type") == "tool_result"
        ]
        if tool_blocks:
            for block in tool_blocks:
                output = _flatten_tool_output(
                    block.get("output", block.get("content", "")),
                )
                has_error, error_type = _classify_tool_error(output)
                results.append(
                    {
                        "tool_name": str(block.get("name") or "unknown"),
                        "tool_call_id": str(block.get("id") or ""),
                        "content": output[:_TOOL_RESULT_CONTENT_MAX_CHARS],
                        "has_error": has_error,
                        "error_type": error_type,
                    },
                )
            continue
        role = getattr(msg, "role", "")
        if role == "user":
            continue
        break
    results.reverse()
    return results


def _build_tool_summaries(
    tool_calls: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for tc in tool_calls:
        func = tc.get("function", {})
        name = _normalize_tool_call_name(func.get("name") or "unknown")
        args_raw = func.get("arguments", "{}")
        args = _tool_call_args(tc)
        _, skill_paths = _extract_skill_paths_from_tool_call(tc)
        summary: dict[str, Any] = {
            "tool_name": name,
            "tool_call_id": str(tc.get("id") or ""),
            "arguments": str(args_raw)[:400],
            "has_error": False,
        }
        if name.lower() in _SHELL_TOOL_NAMES:
            command = str(args.get("command") or args.get("cmd") or "")
            if command:
                summary["command"] = command[:400]
        path = str(
            args.get("path")
            or args.get("file")
            or args.get("file_path")
            or "",
        )
        if path:
            summary["path"] = path
        elif skill_paths:
            summary["path"] = skill_paths[0]
        summaries.append(summary)
    return summaries


def _merge_tool_results(
    turn_record: dict[str, Any],
    tool_results: list[dict[str, Any]],
) -> None:
    summaries = turn_record.get("tool_results") or []
    observations: list[dict[str, Any]] = []
    for i, result in enumerate(tool_results):
        obs = {
            "tool_name": result.get("tool_name", "unknown"),
            "tool_call_id": result.get("tool_call_id", ""),
            "has_error": bool(result.get("has_error", False)),
        }
        if result.get("error_type"):
            obs["error_type"] = result["error_type"]
        if result.get("content"):
            obs["content"] = str(result["content"])[
                :_TOOL_RESULT_CONTENT_MAX_CHARS
            ]
        observations.append(obs)
        if i < len(summaries):
            summaries[i].update(obs)
        else:
            summaries.append(dict(obs))
    turn_record["tool_results"] = summaries
    turn_record["tool_observations"] = observations
    turn_record["tool_errors"] = [
        {
            "tool_name": s.get("tool_name", "unknown"),
            **(
                {"tool_call_id": s["tool_call_id"]}
                if s.get("tool_call_id")
                else {}
            ),
            **({"error_type": s["error_type"]} if s.get("error_type") else {}),
            **({"content": s["content"]} if s.get("content") else {}),
        }
        for s in summaries
        if s.get("has_error")
    ]


def _last_user_instruction(messages: list[dict[str, Any]]) -> str:
    for msg in reversed(messages):
        if isinstance(msg, dict) and msg.get("role") == "user":
            text = str(msg.get("content") or "").strip()
            if text:
                return text
    return ""


_SKILL_PATH_PATTERNS = (
    "/.copaw/skill_pool/",
    "/.copaw/workspaces/",
    "/.qwenpaw/skill_pool/",
    "/.qwenpaw/workspaces/",
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
