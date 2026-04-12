# -*- coding: utf-8 -*-
"""Core skill review logic: read WAL → LLM → SkillService.create_skill.

This module is intentionally standalone (no asyncio, no agentscope imports at module
level) so it can be invoked directly from crontab without loading the full app.
"""
import json
import logging
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

# Notification target — hardcoded, personal WhatsApp for skill review alerts.
# To change, edit here and redeploy.
NOTIFICATION_CHANNEL = "whatsapp"
NOTIFICATION_TARGET_USER = "+85251159218"
NOTIFICATION_TARGET_SESSION = "whatsapp--+85251159218"

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class SkillProposal:
    name: str
    description: str
    skill_md: str
    action: str = "create"  # "create" | "update"


# ---------------------------------------------------------------------------
# LLM prompt
# ---------------------------------------------------------------------------

import re as _re


def _load_user_context(workspace_dir: Path, max_chars: int = 2000) -> str:
    """Read user profile from PROFILE.md, stripping agent identity section.

    Fail-open: returns "" if file missing or unreadable.
    Strips ## 身份 / ## Identity section (agent persona, not user profile).
    Truncates to max_chars to bound prompt size.
    """
    profile_path = workspace_dir / "PROFILE.md"
    if not profile_path.exists():
        return ""
    try:
        text = profile_path.read_text(encoding="utf-8")
    except Exception:
        return ""

    # Strip YAML frontmatter
    if text.startswith("---"):
        parts = text.split("---", 2)
        if len(parts) >= 3:
            text = parts[2]

    # Strip ## 身份 / ## Identity section (up to next ## heading or end of text)
    text = _re.sub(
        r"^##\s*(?:身份|Identity)\b.*?(?=^##\s|\Z)",
        "",
        text,
        flags=_re.MULTILINE | _re.DOTALL,
    )

    text = text.strip()
    if len(text) > max_chars:
        text = text[:max_chars] + "\n[...truncated]"
    return text


SKILL_REVIEW_PROMPT = """你係 skill reviewer。你嘅工作係判斷 agent session 有冇 **LEARNT KNOWLEDGE** 值得固化成 skill。

重要前提：大部分 session **唔值得**建立 skill。你嘅 default 係「唔 propose」。只有 session 明確顯示真正嘅學習過程先考慮 propose。

{user_context_block}## Agent Session (WAL entries)
{wal_content}

## 現有 Skills (避免重複)
{existing_skills}

## 判斷標準 — 必須符合其中一個先值得建立或更新 skill

✅ **值得** propose 嘅情況：
1. Agent 明顯試咗多個方法 / 中途改變方向先搞掂（trial-and-error learnt approach）
2. 用戶糾正咗 Agent 嘅做法，或者 Agent 發現佢原本嘅 approach 係錯嘅
3. 用戶親身示範咗一個特定 workflow，而呢個 workflow 係 non-obvious / domain-specific 步驟組合（唔係普通 Google 到嘅嘢）

❌ **唔應該** propose 嘅情況（唔確定就唔 propose）：
- Agent 第一次就成功搞掂，冇 iteration
- 只係普通搵資料 / 搜索 / 翻譯 / 問答類型嘅 session
- 對話主要係閒聊、planning discussion、單次查詢
- 已有相似 skill 可以 cover（見上面列表）—— 除非呢個 session 搵到更好做法，可以 propose 更新（action: update）
- Skill 太 generic（e.g. "search_web", "translate", "summarize", "book_ticket"）
- 唔確定值唔值得？→ 返回 {{"propose": false}}

## 輸出格式

如果值得，返回以下 JSON；如果唔值得就返回 {{"propose": false}}：

Create 新 skill：
{{"propose": true, "action": "create", "name": "snake_case_skill_name", "description": "一句話描述 (英文或廣東話)", "skill_md": "## Purpose\\n...\\n## Steps\\n1. ...\\n2. ..."}}

Update 現有 skill（name 必須係 existing_skills 入面已有嘅）：
{{"propose": true, "action": "update", "name": "existing_skill_name", "description": "一句話描述改進點", "skill_md": "## Purpose\\n...\\n## Steps\\n1. ...\\n2. ..."}}

規則：
- skill_md 必須係完整 Markdown，包含 ## Purpose 同 ## Steps section
- name 用英文 snake_case，唔超過 40 chars，描述具體 domain（唔好太 generic）
- action: "update" 只可以用喺 existing_skills 列表入面已存在嘅 name
- action 唔填預設係 "create"
- 只係 propose 一個 skill（最值得創建或更新嗰個）

只返回 JSON，唔好返回其他文字。"""


# ---------------------------------------------------------------------------
# WAL reader
# ---------------------------------------------------------------------------

def _read_wal(workspace_dir: Path, max_entries: int = 200) -> str:
    """Read and format recent WAL entries for the review prompt."""
    wal_file = workspace_dir / ".session_wal.jsonl"
    if not wal_file.exists():
        return ""

    entries = []
    try:
        lines = wal_file.read_text(encoding="utf-8").splitlines()
        for line in lines[-max_entries:]:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                t = d.get("type", "")
                if t == "reasoning":
                    content = str(d.get("content", "")).strip()
                    if content and len(content) > 20:
                        entries.append(f"[reasoning] {content[:800]}")
                elif t == "sent":
                    content = str(d.get("content", "")).strip()
                    if content and len(content) > 10:
                        entries.append(f"[sent] {content[:600]}")
                elif t == "tool_start":
                    tool = d.get("tool", "")
                    args = str(d.get("args", ""))[:200]
                    if tool:
                        entries.append(f"[tool:{tool}] {args}")
            except Exception:
                pass
    except Exception as e:
        logger.warning("WAL read failed: %s", e)

    return "\n".join(entries[-100:])


# ---------------------------------------------------------------------------
# Skill listing (for dedup)
# ---------------------------------------------------------------------------

def _get_existing_skills(workspace_dir: Path) -> str:
    """Return a human-readable list of existing skill names + descriptions."""
    try:
        from copaw.agents.skills_manager import SkillService
        svc = SkillService(workspace_dir)
        skills = svc.list_all_skills()
        if not skills:
            return "(no existing skills)"
        return "\n".join(f"- {s.name}: {getattr(s, 'description', '')}" for s in skills)
    except Exception as e:
        logger.warning("Could not list skills: %s", e)
        return "(could not load existing skills)"


# ---------------------------------------------------------------------------
# API config loader
# ---------------------------------------------------------------------------

def _load_api_config() -> tuple[str, str]:
    """Load and decrypt DashScope API key from bailian.json.

    Returns:
        (api_key, base_url)

    Raises:
        FileNotFoundError: if bailian.json is missing
        ValueError: if api_key or base_url is empty after decryption
    """
    secret_path = Path.home() / ".copaw.secret" / "providers" / "custom" / "bailian.json"
    cfg = json.loads(secret_path.read_text())
    api_key = cfg.get("api_key", "")
    base_url = cfg.get("base_url", "").rstrip("/")

    if api_key.startswith("ENC:"):
        # Try CoPaw's decrypt() first (requires master key in keyring/file)
        try:
            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent.parent))
            from copaw.security.secret_store import decrypt
            api_key = decrypt(api_key)
        except Exception:
            pass

        # Fallback: decrypt directly from master key file
        if api_key.startswith("ENC:"):
            key_file = Path.home() / ".copaw.secret" / ".master_key"
            if key_file.exists():
                try:
                    import base64
                    from cryptography.fernet import Fernet
                    key_bytes = bytes.fromhex(key_file.read_text().strip())
                    fernet = Fernet(base64.urlsafe_b64encode(key_bytes))
                    api_key = fernet.decrypt(api_key[4:].encode()).decode()
                except Exception as e:
                    raise ValueError(f"Cannot decrypt API key: {e}") from e

    if not api_key or api_key.startswith("ENC:"):
        raise ValueError("api_key empty or still encrypted after decryption attempts")
    if not base_url:
        raise ValueError("base_url missing in bailian.json")

    return api_key, base_url


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------

def _call_llm(prompt: str, api_key: str, base_url: str, timeout: int = 120) -> str:
    """Call qwen3.6-plus with thinking enabled for higher-quality skill proposals."""
    import requests
    resp = requests.post(
        f"{base_url}/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": "qwen3.6-plus",
            "messages": [
                {"role": "system", "content": "Only respond with valid JSON."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.1,
            "enable_thinking": True,
            "max_tokens": 2000,
        },
        timeout=timeout,
    )
    resp.raise_for_status()
    data = resp.json()
    return data.get("choices", [{}])[0].get("message", {}).get("content", "")


# ---------------------------------------------------------------------------
# Notification
# ---------------------------------------------------------------------------

def _send_notification(
    skill_name: str, description: str, agent: str, action: str = "create"
) -> bool:
    """Push a Cantonese WhatsApp alert via `copaw channels send`.

    Returns True on success, False on any failure.  Never raises — cron must
    not abort just because the notification side-channel is down.
    """
    if action == "update":
        message = (
            f"🔧 skill_review 更新咗 skill\n\n"
            f"📦 {skill_name}\n"
            f"📝 {description}\n\n"
            f"原有 enabled state 保留"
        )
    else:
        message = (
            f"🔔 skill_review 建立咗新 skill（已啟用）\n\n"
            f"📦 {skill_name}\n"
            f"📝 {description}\n\n"
            f"如果唔滿意，去 ~/.copaw/workspaces/{agent}/skills.json 將 enabled 改做 false"
        )
    cmd = [
        "copaw", "channels", "send",
        "--agent-id", agent,
        "--channel", NOTIFICATION_CHANNEL,
        "--target-user", NOTIFICATION_TARGET_USER,
        "--target-session", NOTIFICATION_TARGET_SESSION,
        "--text", message,
    ]
    try:
        result = subprocess.run(cmd, timeout=30, capture_output=True, text=True)
        if result.returncode != 0:
            logger.warning(
                "skill_review: notification failed (exit %d): %s",
                result.returncode, result.stderr[:200],
            )
            return False
        logger.info("skill_review: notification sent for '%s'", skill_name)
        return True
    except subprocess.TimeoutExpired:
        logger.warning("skill_review: notification timed out for '%s'", skill_name)
        return False
    except Exception as e:
        logger.warning("skill_review: notification error for '%s': %s", skill_name, e)
        return False


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_once(
    agent_name: str,
    workspace_dir: Path,
    dry_run: bool = False,
    notification: bool = True,
) -> list[SkillProposal]:
    """Review recent WAL content and propose / create skills.

    Args:
        agent_name: Agent identifier (used for logging only).
        workspace_dir: Path to agent workspace (e.g. ~/.copaw/workspaces/default).
        dry_run: If True, propose but do not call create_skill.  Implicitly
            disables notification.
        notification: If True (default), push a WhatsApp alert after each
            successful skill creation.  Has no effect when dry_run=True.

    Returns:
        List of SkillProposal objects (empty if nothing worthy found).
    """
    workspace_dir = Path(workspace_dir).expanduser()
    logger.info("skill_review start: agent=%s workspace=%s dry_run=%s",
                agent_name, workspace_dir, dry_run)

    # 1. Read WAL
    wal_content = _read_wal(workspace_dir)
    if not wal_content.strip():
        logger.info("skill_review: empty WAL — nothing to review")
        return []

    # 2. Collect existing skills for dedup context
    existing_skills = _get_existing_skills(workspace_dir)

    # 3. Build prompt
    # Load user profile context (fail-open)
    user_ctx = _load_user_context(workspace_dir)
    user_context_block = (
        f"## User Context\n{user_ctx}\n\n"
        if user_ctx else ""
    )

    prompt = SKILL_REVIEW_PROMPT.format(
        wal_content=wal_content,
        existing_skills=existing_skills,
        user_context_block=user_context_block,
    )

    # 4. Load API config
    try:
        api_key, base_url = _load_api_config()
    except Exception as e:
        logger.error("skill_review: cannot load API config: %s", e)
        return []

    # 5. Call LLM (qwen3.6-plus + thinking=True; ~30–90s, acceptable offline)
    logger.info("skill_review: calling LLM, WAL=%d chars", len(wal_content))
    t0 = time.time()
    try:
        response = _call_llm(prompt, api_key, base_url)
    except Exception as e:
        logger.error("skill_review: LLM call failed: %s", e)
        return []
    elapsed = time.time() - t0
    logger.info("skill_review: LLM done in %.1fs, response=%d chars", elapsed, len(response))

    # 6. Parse JSON response
    try:
        text = response.strip()
        # Strip markdown code fences if present
        if text.startswith("```"):
            text = text.split("\n", 1)[1] if "\n" in text else text[3:]
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()
        result = json.loads(text)
    except Exception as e:
        logger.error("skill_review: JSON parse failed: %s, preview=%r", e, response[:100])
        return []

    if not result.get("propose", False):
        logger.info("skill_review: LLM decided no skill worth creating")
        return []

    proposal = SkillProposal(
        name=str(result.get("name", "")).strip(),
        description=str(result.get("description", "")).strip(),
        skill_md=str(result.get("skill_md", "")).strip(),
        action=str(result.get("action", "create")).strip().lower(),
    )

    if not proposal.name or not proposal.skill_md:
        logger.warning("skill_review: proposal missing name or skill_md — skipping")
        return []

    logger.info("skill_review: proposed '%s': %s", proposal.name, proposal.description)

    if dry_run:
        logger.info("skill_review: dry_run=True — skipping create_skill and notification")
        return [proposal]

    # 7. Create or update skill.
    # Hermes-style: new skills are auto-enabled so they're immediately usable.
    # For updates (overwrite=True), the manifest preserves the existing enabled
    # state — the enable=True here only acts as the fallback for brand-new entries.
    created_name = None
    is_update = proposal.action == "update"
    try:
        from copaw.agents.skills_manager import SkillService
        svc = SkillService(workspace_dir)
        created_name = svc.create_skill(
            name=proposal.name,
            content=proposal.skill_md,
            overwrite=is_update,
            enable=True,  # Auto-enable (Hermes style); existing enabled state preserved on update
            authored_by="skill_review",
        )
        if created_name:
            action_label = "updated" if is_update else "created (auto-enabled)"
            logger.info("skill_review: %s skill '%s'", action_label, created_name)
        else:
            logger.info(
                "skill_review: skill '%s' already exists — skipped (LLM should use action:update)",
                proposal.name,
            )
    except Exception as e:
        logger.error("skill_review: create_skill failed: %s", e)
        # Return proposal even on creation failure (caller can retry / inspect)

    # 8. Notify — only when creation/update succeeded and notification is enabled
    if created_name and notification:
        _send_notification(
            skill_name=proposal.name,
            description=proposal.description,
            agent=agent_name,
            action=proposal.action,
        )

    return [proposal]
