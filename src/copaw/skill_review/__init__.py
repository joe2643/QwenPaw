# -*- coding: utf-8 -*-
"""CoPaw Skill Review — offline WAL-based skill creation.

Run via CLI:
    python -m copaw.skill_review --agent default
    python -m copaw.skill_review --agent default --dry-run
"""
from .review import SkillProposal, run_once

__all__ = ["run_once", "SkillProposal"]
