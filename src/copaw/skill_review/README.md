# skill_review — Roadmap

## Phase 1 (current): Offline cron + LLM-proposed skills

- ✓ qwen3.6-plus thinking-mode review of WAL
- ✓ Hermes-aligned prompt (trial-and-error anchored, "default = don't propose")
- ✓ Create + update paths (`action: "create"` / `action: "update"`)
- ✓ Auto-enable on create (Hermes style — skills are immediately usable)
- ✓ Update preserves existing `enabled` state (via manifest `entry.get("enabled", enable)`)
- ✓ WhatsApp notification to Joe (create vs update template)
- ✓ `authored_by: "skill_review"` in manifest for audit trail

## Phase 2 (TODO): Skill = living document

- [ ] **Usage tracking**: count how often each skill is loaded by agents; surface "unused" skills
- [ ] **In-session auto-patch**: when an agent detects a skill is outdated mid-task, patch live (mirror Hermes `skill_manage(action='patch')` fuzzy-match behavior)
- [ ] **Periodic re-review**: cron re-evaluates existing skills for relevance, consolidation, and deprecation; not just session WAL → new skill
- [ ] **Deprecation path**: mark skills unused for N days; archive rather than hard-delete
- [ ] **Version history**: preserve previous `skill_md` content on update (currently in-place overwrite with no history)
- [ ] **Cross-agent skill propagation**: skills created in one agent's workspace promoted to shared pool automatically after N successful uses
