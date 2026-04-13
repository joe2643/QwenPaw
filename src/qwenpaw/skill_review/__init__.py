# -*- coding: utf-8 -*-
"""QwenPaw Skill Review — offline WAL-based skill creation.

Run via CLI:
    python -m qwenpaw.skill_review --agent default
    python -m qwenpaw.skill_review --agent default --dry-run
"""
from .review import SkillProposal, run_once

__all__ = ["run_once", "SkillProposal"]
