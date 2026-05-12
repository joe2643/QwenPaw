# -*- coding: utf-8 -*-
"""API routes for MemPalace browser — drawers, wings, KG, hooks."""

from __future__ import annotations

import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

try:
    from mempalace.config import MempalaceConfig
    from mempalace.chroma_helper import get_collection

    _MEMPALACE_AVAILABLE = True
except ImportError:
    MempalaceConfig = None  # type: ignore[assignment,misc]
    get_collection = None  # type: ignore[assignment]
    _MEMPALACE_AVAILABLE = False

router = APIRouter(prefix="/mempalace", tags=["mempalace"])

if _MEMPALACE_AVAILABLE:
    _cfg = MempalaceConfig()
    _PALACE_PATH = _cfg.palace_path
else:
    _cfg = None
    _PALACE_PATH = None
_KG_DB = Path.home() / ".mempalace" / "knowledge_graph.sqlite3"
_HOOK_LOG = Path.home() / ".mempalace" / "hook.log"
_COLLECTION_NAME = "mempalace_drawers"


def _require_mempalace() -> None:
    """Return 503 when the mempalace package isn't installed in this env."""
    if not _MEMPALACE_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail=(
                "mempalace package not installed in this environment. "
                "The MemPalace router is only functional on joe-faex1."
            ),
        )


def _get_collection():
    """Return the chroma collection handle."""
    _require_mempalace()
    return get_collection(palace_path=_PALACE_PATH)


def _kg_connection() -> sqlite3.Connection:
    """Open a read-only connection to the KG sqlite database."""
    if not _KG_DB.exists():
        raise HTTPException(404, detail="Knowledge graph database not found")
    return sqlite3.connect(str(_KG_DB))


# ── Pydantic models ──────────────────────────────────────────────────


class RoomInfo(BaseModel):
    name: str
    count: int


class WingInfo(BaseModel):
    name: str
    rooms: List[RoomInfo]


class DrawerPreview(BaseModel):
    id: str
    content_preview: str = Field(default="", description="First 200 chars")
    metadata: Dict[str, Any] = Field(default_factory=dict)
    filed_at: Optional[str] = None


class DrawerFull(BaseModel):
    id: str
    content: str = ""
    metadata: Dict[str, Any] = Field(default_factory=dict)


class DrawerUpdateRequest(BaseModel):
    wing: Optional[str] = None
    room: Optional[str] = None
    hall: Optional[str] = None


class KGStats(BaseModel):
    entity_count: int = 0
    triple_count: int = 0


class KGEntity(BaseModel):
    id: Any
    name: str
    type: Optional[str] = None


class KGTriple(BaseModel):
    id: Any
    subject: str
    predicate: str
    object: str


# ── 1. GET /mempalace/status ─────────────────────────────────────────


@router.get("/status", summary="Palace statistics")
async def palace_status() -> Dict[str, Any]:
    col = _get_collection()
    total = col.count()

    # Fetch all metadata to group by wing/room
    all_data = col.get(include=["metadatas"])
    wing_room_counts: Dict[str, Dict[str, int]] = defaultdict(
        lambda: defaultdict(int),
    )
    for meta in all_data.get("metadatas") or []:
        wing = (meta or {}).get("wing", "unknown")
        room = (meta or {}).get("room", "unknown")
        wing_room_counts[wing][room] += 1

    # KG stats
    kg = {"entity_count": 0, "triple_count": 0}
    if _KG_DB.exists():
        try:
            conn = _kg_connection()
            cur = conn.cursor()
            cur.execute("SELECT count(*) FROM entities")
            kg["entity_count"] = cur.fetchone()[0]
            cur.execute("SELECT count(*) FROM triples")
            kg["triple_count"] = cur.fetchone()[0]
            conn.close()
        except Exception:
            pass

    return {
        "total_drawers": total,
        "wings": {w: dict(rooms) for w, rooms in wing_room_counts.items()},
        "kg": kg,
    }


# ── 2. GET /mempalace/wings ──────────────────────────────────────────


@router.get("/wings", summary="List wings with room counts")
async def list_wings() -> Dict[str, List[WingInfo]]:
    col = _get_collection()
    all_data = col.get(include=["metadatas"])

    wing_room_counts: Dict[str, Dict[str, int]] = defaultdict(
        lambda: defaultdict(int),
    )
    for meta in all_data.get("metadatas") or []:
        wing = (meta or {}).get("wing", "unknown")
        room = (meta or {}).get("room", "unknown")
        wing_room_counts[wing][room] += 1

    wings = []
    for wing_name in sorted(wing_room_counts):
        rooms = [
            RoomInfo(name=r, count=c)
            for r, c in sorted(wing_room_counts[wing_name].items())
        ]
        wings.append(WingInfo(name=wing_name, rooms=rooms))

    return {"wings": wings}


# ── 3. GET /mempalace/wings/{wing}/rooms/{room} ─────────────────────


@router.get(
    "/wings/{wing}/rooms/{room}",
    summary="List drawers in a room (paginated)",
)
async def list_drawers_in_room(
    wing: str,
    room: str,
    offset: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=200),
) -> Dict[str, Any]:
    col = _get_collection()
    all_data = col.get(include=["metadatas", "documents"])

    ids = all_data.get("ids") or []
    metadatas = all_data.get("metadatas") or []
    documents = all_data.get("documents") or []

    # Filter by wing+room
    matched: List[DrawerPreview] = []
    for i, meta in enumerate(metadatas):
        m = meta or {}
        if m.get("wing") == wing and m.get("room") == room:
            doc = (documents[i] if i < len(documents) else "") or ""
            matched.append(
                DrawerPreview(
                    id=ids[i],
                    content_preview=doc[:200],
                    metadata=m,
                    filed_at=m.get("filed_at"),
                ),
            )

    # Sort by filed_at descending (newest first)
    matched.sort(key=lambda d: d.filed_at or "", reverse=True)

    page = matched[offset : offset + limit]
    return {
        "wing": wing,
        "room": room,
        "total": len(matched),
        "offset": offset,
        "limit": limit,
        "drawers": [d.model_dump() for d in page],
    }


# ── 4. GET /mempalace/drawer/{drawer_id} ─────────────────────────────


@router.get("/drawer/{drawer_id}", summary="Full drawer content + metadata")
async def get_drawer(drawer_id: str) -> DrawerFull:
    col = _get_collection()
    result = col.get(ids=[drawer_id], include=["metadatas", "documents"])

    if not result.get("ids"):
        raise HTTPException(404, detail=f"Drawer '{drawer_id}' not found")

    doc = (result["documents"][0] if result.get("documents") else "") or ""
    meta = (result["metadatas"][0] if result.get("metadatas") else {}) or {}

    return DrawerFull(id=drawer_id, content=doc, metadata=meta)


# ── 5. PUT /mempalace/drawer/{drawer_id} ─────────────────────────────


@router.put("/drawer/{drawer_id}", summary="Update drawer metadata")
async def update_drawer(
    drawer_id: str,
    body: DrawerUpdateRequest,
) -> Dict[str, str]:
    col = _get_collection()

    # Verify exists
    existing = col.get(ids=[drawer_id], include=["metadatas"])
    if not existing.get("ids"):
        raise HTTPException(404, detail=f"Drawer '{drawer_id}' not found")

    meta = dict(
        (existing["metadatas"][0] if existing.get("metadatas") else {}) or {},
    )
    updates = body.model_dump(exclude_unset=True)
    meta.update(updates)

    col.update(ids=[drawer_id], metadatas=[meta])
    return {"message": f"Drawer '{drawer_id}' updated"}


# ── 6. DELETE /mempalace/drawer/{drawer_id} ──────────────────────────


@router.delete("/drawer/{drawer_id}", summary="Delete a drawer")
async def delete_drawer(drawer_id: str) -> Dict[str, str]:
    col = _get_collection()

    existing = col.get(ids=[drawer_id], include=[])
    if not existing.get("ids"):
        raise HTTPException(404, detail=f"Drawer '{drawer_id}' not found")

    col.delete(ids=[drawer_id])
    return {"message": f"Drawer '{drawer_id}' deleted"}


# ── 7. GET /mempalace/kg/stats ───────────────────────────────────────


@router.get("/kg/stats", summary="KG entity + triple counts")
async def kg_stats() -> KGStats:
    conn = _kg_connection()
    cur = conn.cursor()
    try:
        cur.execute("SELECT count(*) FROM entities")
        entities = cur.fetchone()[0]
        cur.execute("SELECT count(*) FROM triples")
        triples = cur.fetchone()[0]
    finally:
        conn.close()
    return KGStats(entity_count=entities, triple_count=triples)


# ── 8. GET /mempalace/kg/entities ────────────────────────────────────


@router.get("/kg/entities", summary="List KG entities (paginated)")
async def kg_entities(
    offset: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=500),
) -> Dict[str, Any]:
    conn = _kg_connection()
    cur = conn.cursor()
    try:
        cur.execute("SELECT count(*) FROM entities")
        total = cur.fetchone()[0]
        cur.execute(
            "SELECT id, name, type FROM entities ORDER BY id LIMIT ? OFFSET ?",
            (limit, offset),
        )
        rows = cur.fetchall()
    finally:
        conn.close()

    return {
        "total": total,
        "offset": offset,
        "limit": limit,
        "entities": [
            KGEntity(id=r[0], name=r[1], type=r[2]).model_dump() for r in rows
        ],
    }


# ── 9. GET /mempalace/kg/triples ────────────────────────────────────


@router.get("/kg/triples", summary="List KG triples (paginated)")
async def kg_triples(
    offset: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=500),
) -> Dict[str, Any]:
    conn = _kg_connection()
    cur = conn.cursor()
    try:
        cur.execute("SELECT count(*) FROM triples")
        total = cur.fetchone()[0]
        cur.execute(
            "SELECT id, subject, predicate, object FROM triples "
            "ORDER BY id LIMIT ? OFFSET ?",
            (limit, offset),
        )
        rows = cur.fetchall()
    finally:
        conn.close()

    return {
        "total": total,
        "offset": offset,
        "limit": limit,
        "triples": [
            KGTriple(
                id=r[0],
                subject=r[1],
                predicate=r[2],
                object=r[3],
            ).model_dump()
            for r in rows
        ],
    }


# ── 10. GET /mempalace/hooks/log ─────────────────────────────────────


@router.get("/hooks/log", summary="Tail hook.log")
async def hooks_log(
    lines: int = Query(50, ge=1, le=1000),
) -> Dict[str, Any]:
    if not _HOOK_LOG.exists():
        return {"lines": [], "total_lines": 0}

    all_lines = _HOOK_LOG.read_text(errors="replace").splitlines()
    tail = all_lines[-lines:]
    return {"lines": tail, "total_lines": len(all_lines)}
