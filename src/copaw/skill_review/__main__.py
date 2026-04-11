# -*- coding: utf-8 -*-
"""CLI entry point for copaw.skill_review.

Usage:
    python -m copaw.skill_review --agent default
    python -m copaw.skill_review --agent default --workspace ~/.copaw/workspaces/default
    python -m copaw.skill_review --agent default --dry-run
    python -m copaw.skill_review --agent discord-triage --agent ai-news
"""
import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    stream=sys.stdout,
)


def _default_workspace(agent: str) -> Path:
    # TODO: adjust if CoPaw workspace root differs from ~/.copaw/workspaces/
    return Path.home() / ".copaw" / "workspaces" / agent


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m copaw.skill_review",
        description="CoPaw Skill Review — offline WAL-based skill creation",
    )
    parser.add_argument(
        "--agent",
        required=True,
        action="append",
        dest="agents",
        metavar="AGENT",
        help="Agent name (e.g. default). Repeat to review multiple agents.",
    )
    parser.add_argument(
        "--workspace",
        default=None,
        help="Workspace path override (only applies when reviewing a single agent).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Propose skills but do not call create_skill (implicitly disables notification).",
    )
    parser.add_argument(
        "--no-notification",
        action="store_true",
        help="Suppress WhatsApp notification even when a skill is created.",
    )
    args = parser.parse_args()

    if args.workspace and len(args.agents) > 1:
        parser.error("--workspace can only be used with a single --agent")

    from copaw.skill_review.review import run_once

    total_proposals = []
    for agent in args.agents:
        workspace = (
            Path(args.workspace).expanduser()
            if args.workspace
            else _default_workspace(agent)
        )
        notification = not args.dry_run and not args.no_notification
        proposals = run_once(
            agent_name=agent,
            workspace_dir=workspace,
            dry_run=args.dry_run,
            notification=notification,
        )
        total_proposals.extend(proposals)

    if total_proposals:
        status = "DRY-RUN (not created)" if args.dry_run else "created (disabled, pending review)"
        print(f"\nProposed {len(total_proposals)} skill(s) [{status}]:")
        for p in total_proposals:
            print(f"  - {p.name}: {p.description}")
    else:
        print("No skills proposed.")

    sys.exit(0)


if __name__ == "__main__":
    main()
