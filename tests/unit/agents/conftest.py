# -*- coding: utf-8 -*-
"""Pytest configuration for unit/agents tests.

Ensures heavy-dependency modules are importable for patch()-based tests even in
environments that don't have the full package installed (e.g. Mac dev machines
missing `frontmatter`, `agentscope_runtime`, etc.).
"""
import sys
import unittest.mock


def _ensure_skills_manager_importable() -> None:
    """Register a stub for copaw.agents.skills_manager if not importable.

    patch("copaw.agents.skills_manager.SkillService") requires the module to
    already be in sys.modules (because copaw.agents.__init__ uses a lazy
    __getattr__ that raises AttributeError for unknown names).  On joe-faex1
    with the full package installed the real module is loaded; on Mac dev boxes
    the stub keeps the test infra intact.
    """
    key = "copaw.agents.skills_manager"
    if key in sys.modules:
        return
    try:
        import copaw.agents.skills_manager  # noqa: F401
    except (ImportError, ModuleNotFoundError):
        stub = unittest.mock.MagicMock()
        sys.modules[key] = stub
        try:
            import copaw.agents
            copaw.agents.skills_manager = stub  # type: ignore[attr-defined]
        except Exception:
            pass


_ensure_skills_manager_importable()
