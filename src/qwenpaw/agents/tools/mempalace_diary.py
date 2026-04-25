# -*- coding: utf-8 -*-
"""MemPalace Diary Tool for QwenPaw.

Provides mempalace_diary_write tool for writing diary entries
via MemPalace ChromaDB integration.
"""
import logging
import hashlib
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)


def mempalace_diary_write(
    agent_name: str,
    entry: str,
    topic: str = "auto",
    wing: str = None,
) -> dict:
    """Write a diary entry to MemPalace.

    This tool writes diary entries directly to ChromaDB via mempalace.chroma_helper,
    matching the official mempalace.mcp_server.tool_diary_write format.

    Args:
        agent_name: Name of the agent (e.g., '夕慶', 'reviewer', 'architect')
        entry: Diary entry content (AAAK format recommended)
        topic: Topic tag for the entry (default: 'auto')
        wing: Optional wing override (default: auto-generated from agent_name)

    Returns:
        dict with success status and entry metadata

    Example:
        >>> mempalace_diary_write(
        ...     agent_name='夕慶',
        ...     entry='2026-04-08|MemPalace.setup|chromadb+embedding|completed|*warm*|★★★',
        ...     topic='integration'
        ... )
    """
    try:
        import sys

        sys.path.insert(
            0,
            str(
                Path.home()
                / ".local"
                / "lib"
                / "python3.13"
                / "site-packages",
            ),
        )
        from mempalace.chroma_helper import get_collection
        from mempalace.mcp_server import _config

        # Determine wing (match official logic)
        if wing is None:
            wing = "agents"

        room = agent_name.lower().replace(" ", "_")
        now = datetime.now()

        # Get collection (with correct embedding)
        col = get_collection(palace_path=_config.palace_path)

        # Create unique ID (match official format)
        entry_id = f"diary_{wing}_{now.strftime('%Y%m%d_%H%M%S')}_{hashlib.md5(entry[:50].encode()).hexdigest()[:8]}"

        # Write to ChromaDB with full metadata
        col.add(
            ids=[entry_id],
            documents=[entry],
            metadatas=[
                {
                    "wing": wing,
                    "room": room,
                    "hall": "hall_diary",
                    "topic": topic,
                    "type": "diary_entry",
                    "agent": agent_name,
                    "filed_at": now.isoformat(),
                    "date": now.strftime("%Y-%m-%d"),
                    "added_by": "mempalace_diary_write",
                },
            ],
        )

        logger.info(f"MemPalace Diary: {entry_id} → {wing}/diary/{topic}")

        return {
            "success": True,
            "entry_id": entry_id,
            "agent": agent_name,
            "topic": topic,
            "wing": wing,
            "timestamp": now.isoformat(),
        }

    except Exception as e:
        logger.error(f"Failed to write MemPalace diary: {e}")
        return {
            "success": False,
            "error": str(e),
        }
