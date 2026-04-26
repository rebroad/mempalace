"""
forgetting.py — lifecycle / forgetting support for MemPalace.

This module adds a small SQLite sidecar under each palace directory that tracks:

- drawer access / decay state
- tombstones for forgotten content
- maintenance timestamps

Chroma remains the verbatim drawer store. The sidecar coordinates ranking,
decay, and hard-delete forgetting without changing the storage backend
contract.
"""

from __future__ import annotations

import hashlib
import math
import os
import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from .knowledge_graph import KnowledgeGraph
from .palace import (
    build_closet_lines,
    get_closets_collection,
    get_collection,
    purge_file_closets,
    upsert_closet_lines,
)


DEFAULT_FORGETTING_CONFIG = {
    "enabled": True,
    "mode": "hard_delete",
    "profile": "conservative",
    "auto_run_hooks": True,
    "maintenance_min_interval_hours": 24,
    "initial_strength_days": 30.0,
    "decay_retrievability_threshold": 0.15,
    "decay_after_days": 180,
    "purge_after_days": 365,
    "max_strength_days": 365.0,
    "include_decayed_default": False,
}

_DB_FILENAME = "memory_lifecycle.sqlite3"
_STATE_ACTIVE = "active"
_STATE_DECAYED = "decayed"
_ORIGIN_EXPLICIT = "explicit"
_ORIGIN_MAINTENANCE = "maintenance"
_UTC = timezone.utc
_CACHE_LOCK = threading.Lock()
_STORE_CACHE: dict[str, "LifecycleStore"] = {}


def _utcnow() -> datetime:
    return datetime.now(_UTC)


def _iso_now() -> str:
    return _utcnow().isoformat()


def _parse_dt(value: Optional[str]) -> datetime:
    if not value:
        return _utcnow()
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return _utcnow()


def _content_sha256(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def forgetting_db_path(palace_path: str) -> str:
    return os.path.join(os.path.abspath(os.path.expanduser(palace_path)), _DB_FILENAME)


def kg_path_for_palace(palace_path: str) -> str:
    return os.path.join(os.path.abspath(os.path.expanduser(palace_path)), "knowledge_graph.sqlite3")


def _result_list(result, key: str) -> list:
    """Read list fields from Chroma compat objects or raw dicts."""
    if isinstance(result, dict):
        value = result.get(key)
    else:
        value = getattr(result, key, None)
    return value or []


@dataclass
class ForgetCandidate:
    drawer_id: str
    source_file: str
    content_sha256: str
    created_at: str
    state: str
    access_count: int
    strength_days: float
    importance: float
    last_accessed_at: str
    decayed_at: Optional[str]
    pinned: bool
    wing: str
    room: str


class LifecycleStore:
    def __init__(self, palace_path: str, config: dict):
        self.palace_path = os.path.abspath(os.path.expanduser(palace_path))
        self.db_path = forgetting_db_path(self.palace_path)
        self.config = dict(DEFAULT_FORGETTING_CONFIG)
        self.config.update(config or {})
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.db_path, timeout=10, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        self._init_db()

    def _init_db(self) -> None:
        with self._lock:
            self._conn.executescript(
                """
                PRAGMA journal_mode=WAL;

                CREATE TABLE IF NOT EXISTS drawer_state (
                    drawer_id TEXT PRIMARY KEY,
                    content_sha256 TEXT NOT NULL,
                    source_file TEXT,
                    wing TEXT,
                    room TEXT,
                    created_at TEXT NOT NULL,
                    last_accessed_at TEXT NOT NULL,
                    access_count INTEGER NOT NULL DEFAULT 0,
                    strength_days REAL NOT NULL,
                    importance REAL NOT NULL DEFAULT 1.0,
                    state TEXT NOT NULL DEFAULT 'active',
                    decayed_at TEXT,
                    forgotten_at TEXT,
                    pinned INTEGER NOT NULL DEFAULT 0
                );

                CREATE TABLE IF NOT EXISTS tombstones (
                    drawer_id TEXT PRIMARY KEY,
                    content_sha256 TEXT NOT NULL,
                    source_file TEXT,
                    forgotten_at TEXT NOT NULL,
                    reason TEXT,
                    origin TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS lifecycle_meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_drawer_state_source_file
                    ON drawer_state(source_file);
                CREATE INDEX IF NOT EXISTS idx_drawer_state_state
                    ON drawer_state(state);
                CREATE INDEX IF NOT EXISTS idx_tombstones_source_file
                    ON tombstones(source_file);
                """
            )
            self._conn.commit()

    def should_ingest(self, drawer_id: str, content: str, source_file: str = "") -> bool:
        row = self._conn.execute(
            "SELECT content_sha256 FROM tombstones WHERE drawer_id = ?",
            (drawer_id,),
        ).fetchone()
        if not row:
            return True
        content_hash = _content_sha256(content)
        if row["content_sha256"] == content_hash:
            return False
        with self._lock:
            self._conn.execute("DELETE FROM tombstones WHERE drawer_id = ?", (drawer_id,))
            self._conn.commit()
        return True

    def register_ingest(self, drawer_id: str, content: str, metadata: dict) -> None:
        now = metadata.get("filed_at") or _iso_now()
        strength = float(self.config["initial_strength_days"])
        importance = float(metadata.get("importance", 1.0))
        content_hash = _content_sha256(content)
        with self._lock:
            existing = self._conn.execute(
                "SELECT created_at, access_count, strength_days, pinned FROM drawer_state WHERE drawer_id = ?",
                (drawer_id,),
            ).fetchone()
            created_at = existing["created_at"] if existing else now
            access_count = int(existing["access_count"]) if existing else 0
            stored_strength = float(existing["strength_days"]) if existing else strength
            pinned = int(existing["pinned"]) if existing else 0
            self._conn.execute(
                """
                INSERT OR REPLACE INTO drawer_state (
                    drawer_id, content_sha256, source_file, wing, room,
                    created_at, last_accessed_at, access_count, strength_days,
                    importance, state, decayed_at, forgotten_at, pinned
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    drawer_id,
                    content_hash,
                    metadata.get("source_file", ""),
                    metadata.get("wing", ""),
                    metadata.get("room", ""),
                    created_at,
                    now,
                    access_count,
                    stored_strength,
                    importance,
                    _STATE_ACTIVE,
                    None,
                    None,
                    pinned,
                ),
            )
            self._conn.commit()

    def note_accesses(self, drawer_ids: list[str]) -> None:
        if not drawer_ids:
            return
        max_strength = float(self.config["max_strength_days"])
        now = _iso_now()
        with self._lock:
            rows = self._conn.execute(
                f"SELECT drawer_id, access_count, strength_days FROM drawer_state WHERE drawer_id IN ({','.join('?' for _ in drawer_ids)})",
                drawer_ids,
            ).fetchall()
            for row in rows:
                access_count = int(row["access_count"]) + 1
                strength = min(float(row["strength_days"]) * 1.2 + 1.0, max_strength)
                self._conn.execute(
                    """
                    UPDATE drawer_state
                       SET last_accessed_at = ?,
                           access_count = ?,
                           strength_days = ?,
                           state = ?,
                           decayed_at = NULL
                     WHERE drawer_id = ?
                    """,
                    (now, access_count, strength, _STATE_ACTIVE, row["drawer_id"]),
                )
            self._conn.commit()

    def retrievability(self, row: sqlite3.Row, now: Optional[datetime] = None) -> float:
        now = now or _utcnow()
        last = _parse_dt(row["last_accessed_at"])
        days = max(0.0, (now - last).total_seconds() / 86400.0)
        strength = max(float(row["strength_days"]), 0.01)
        return max(0.0, min(1.0, math.exp(-days / strength)))

    def state_for_ids(self, drawer_ids: list[str]) -> dict[str, sqlite3.Row]:
        if not drawer_ids:
            return {}
        rows = self._conn.execute(
            f"SELECT * FROM drawer_state WHERE drawer_id IN ({','.join('?' for _ in drawer_ids)})",
            drawer_ids,
        ).fetchall()
        return {row["drawer_id"]: row for row in rows}

    def apply_decay(self, now: Optional[datetime] = None) -> int:
        now = now or _utcnow()
        threshold = float(self.config["decay_retrievability_threshold"])
        stale_days = int(self.config["decay_after_days"])
        changed = 0
        with self._lock:
            rows = self._conn.execute(
                "SELECT drawer_id, last_accessed_at, strength_days, state FROM drawer_state WHERE forgotten_at IS NULL"
            ).fetchall()
            for row in rows:
                last = _parse_dt(row["last_accessed_at"])
                days = max(0.0, (now - last).total_seconds() / 86400.0)
                retr = self.retrievability(row, now)
                target = _STATE_DECAYED if (retr < threshold or days >= stale_days) else _STATE_ACTIVE
                if row["state"] != target:
                    self._conn.execute(
                        """
                        UPDATE drawer_state
                           SET state = ?, decayed_at = CASE WHEN ? = 'decayed' THEN ? ELSE NULL END
                         WHERE drawer_id = ?
                        """,
                        (target, target, now.isoformat(), row["drawer_id"]),
                    )
                    changed += 1
            self._conn.commit()
        return changed

    def purge_candidates(self, now: Optional[datetime] = None) -> list[ForgetCandidate]:
        now = now or _utcnow()
        purge_days = int(self.config["purge_after_days"])
        low_value_threshold = 1.0
        rows = self._conn.execute(
            """
            SELECT * FROM drawer_state
             WHERE state = ? AND forgotten_at IS NULL AND pinned = 0 AND access_count <= 1
            """,
            (_STATE_DECAYED,),
        ).fetchall()
        out = []
        cutoff = timedelta(days=purge_days)
        for row in rows:
            last = _parse_dt(row["last_accessed_at"])
            created = _parse_dt(row["created_at"])
            if now - last < cutoff:
                continue
            if now - created < cutoff:
                continue
            if float(row["importance"]) > low_value_threshold:
                continue
            out.append(
                ForgetCandidate(
                    drawer_id=row["drawer_id"],
                    source_file=row["source_file"] or "",
                    content_sha256=row["content_sha256"],
                    created_at=row["created_at"],
                    state=row["state"],
                    access_count=int(row["access_count"]),
                    strength_days=float(row["strength_days"]),
                    importance=float(row["importance"]),
                    last_accessed_at=row["last_accessed_at"],
                    decayed_at=row["decayed_at"],
                    pinned=bool(row["pinned"]),
                    wing=row["wing"] or "",
                    room=row["room"] or "",
                )
            )
        return out

    def has_tombstone(self, drawer_id: str) -> bool:
        row = self._conn.execute(
            "SELECT 1 FROM tombstones WHERE drawer_id = ?",
            (drawer_id,),
        ).fetchone()
        return row is not None

    def record_tombstone(
        self,
        drawer_id: str,
        content_sha256: str,
        source_file: str,
        reason: str,
        origin: str,
        forgotten_at: Optional[str] = None,
    ) -> None:
        forgotten_at = forgotten_at or _iso_now()
        with self._lock:
            self._conn.execute(
                """
                INSERT OR REPLACE INTO tombstones (
                    drawer_id, content_sha256, source_file, forgotten_at, reason, origin
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (drawer_id, content_sha256, source_file, forgotten_at, reason, origin),
            )
            self._conn.execute(
                "DELETE FROM drawer_state WHERE drawer_id = ?",
                (drawer_id,),
            )
            self._conn.commit()

    def stats(self) -> dict:
        totals = self._conn.execute(
            """
            SELECT
                COUNT(*) AS total,
                SUM(CASE WHEN state = 'active' THEN 1 ELSE 0 END) AS active,
                SUM(CASE WHEN state = 'decayed' THEN 1 ELSE 0 END) AS decayed
            FROM drawer_state
            """
        ).fetchone()
        tombstones = self._conn.execute("SELECT COUNT(*) AS c FROM tombstones").fetchone()
        return {
            "drawer_state_total": totals["total"] or 0,
            "active": totals["active"] or 0,
            "decayed": totals["decayed"] or 0,
            "tombstones": tombstones["c"] or 0,
        }

    def delete_states_for_source_file(self, source_file: str) -> None:
        if not source_file:
            return
        with self._lock:
            self._conn.execute("DELETE FROM drawer_state WHERE source_file = ?", (source_file,))
            self._conn.commit()

    def delete_state(self, drawer_id: str) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM drawer_state WHERE drawer_id = ?", (drawer_id,))
            self._conn.commit()

    def get_last_maintenance_at(self) -> Optional[str]:
        row = self._conn.execute(
            "SELECT value FROM lifecycle_meta WHERE key = 'last_maintenance_at'"
        ).fetchone()
        return row["value"] if row else None

    def set_last_maintenance_at(self, value: str) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO lifecycle_meta (key, value) VALUES ('last_maintenance_at', ?)",
                (value,),
            )
            self._conn.commit()


def get_lifecycle_store(palace_path: str, config: dict) -> LifecycleStore:
    palace_key = os.path.abspath(os.path.expanduser(palace_path))
    with _CACHE_LOCK:
        store = _STORE_CACHE.get(palace_key)
        if store is None:
            store = LifecycleStore(palace_key, config)
            _STORE_CACHE[palace_key] = store
        else:
            merged = dict(DEFAULT_FORGETTING_CONFIG)
            merged.update(config or {})
            store.config = merged
        return store


def rebuild_closets_for_source_file(palace_path: str, source_file: str) -> int:
    if not source_file:
        return 0
    drawers_col = get_collection(palace_path, create=False)
    closets_col = get_closets_collection(palace_path, create=True)
    purge_file_closets(closets_col, source_file)
    result = drawers_col.get(
        where={"source_file": source_file},
        include=["documents", "metadatas"],
    )
    docs = _result_list(result, "documents")
    metas = _result_list(result, "metadatas")
    ids = _result_list(result, "ids")
    if not docs:
        return 0
    all_lines = []
    base_meta = metas[0] if metas else {}
    wing = base_meta.get("wing", "unknown")
    room = base_meta.get("room", "general")
    for doc, meta, drawer_id in zip(docs, metas, ids):
        meta = meta or {}
        all_lines.extend(
            build_closet_lines(
                source_file,
                [drawer_id],
                doc,
                meta.get("wing", wing),
                meta.get("room", room),
            )
        )
    closet_id_base = (
        f"closet_{wing}_{room}_{hashlib.sha256(source_file.encode()).hexdigest()[:24]}"
    )
    closet_meta = {
        "wing": wing,
        "room": room,
        "source_file": source_file,
        "drawer_count": len(ids),
        "filed_at": _iso_now(),
    }
    if base_meta.get("entities"):
        closet_meta["entities"] = base_meta["entities"]
    return upsert_closet_lines(closets_col, closet_id_base, all_lines, closet_meta)


def invalidate_kg_facts_for_drawer(palace_path: str, drawer_id: str, ended: Optional[str] = None) -> int:
    kg = KnowledgeGraph(db_path=kg_path_for_palace(palace_path))
    ended = ended or _utcnow().date().isoformat()
    conn = kg._conn()
    with kg._lock:
        rows = conn.execute(
            """
            SELECT subject, predicate, object
              FROM triples
             WHERE source_drawer_id = ? AND valid_to IS NULL
            """,
            (drawer_id,),
        ).fetchall()
        count = 0
        for row in rows:
            conn.execute(
                """
                UPDATE triples
                   SET valid_to = ?
                 WHERE source_drawer_id = ? AND subject = ? AND predicate = ? AND object = ? AND valid_to IS NULL
                """,
                (ended, drawer_id, row["subject"], row["predicate"], row["object"]),
            )
            count += 1
        conn.commit()
    return count


def has_current_kg_facts_for_drawer(palace_path: str, drawer_id: str) -> bool:
    kg = KnowledgeGraph(db_path=kg_path_for_palace(palace_path))
    conn = kg._conn()
    with kg._lock:
        row = conn.execute(
            "SELECT 1 FROM triples WHERE source_drawer_id = ? AND valid_to IS NULL LIMIT 1",
            (drawer_id,),
        ).fetchone()
    return row is not None


def forget_drawer(
    palace_path: str,
    drawer_id: str,
    reason: str = "",
    origin: str = _ORIGIN_EXPLICIT,
    config: Optional[dict] = None,
    dry_run: bool = False,
) -> dict:
    store = get_lifecycle_store(palace_path, config or {})
    try:
        col = get_collection(palace_path, create=False)
    except Exception as exc:
        return {"success": False, "error": str(exc)}

    existing = col.get(ids=[drawer_id], include=["documents", "metadatas"])
    existing_ids = _result_list(existing, "ids")
    if not existing_ids:
        return {"success": False, "error": f"Drawer not found: {drawer_id}"}

    documents = _result_list(existing, "documents")
    metadatas = _result_list(existing, "metadatas")
    doc = documents[0] if documents else ""
    meta = metadatas[0] if metadatas else {}
    source_file = meta.get("source_file", "")
    content_hash = _content_sha256(doc)
    invalidated = 0 if dry_run else invalidate_kg_facts_for_drawer(palace_path, drawer_id)

    result = {
        "success": True,
        "drawer_id": drawer_id,
        "source_file": source_file,
        "origin": origin,
        "reason": reason,
        "kg_invalidated": invalidated,
        "dry_run": dry_run,
    }
    if dry_run:
        return result

    col.delete(ids=[drawer_id])
    store.record_tombstone(drawer_id, content_hash, source_file, reason, origin)
    closets_rebuilt = rebuild_closets_for_source_file(palace_path, source_file)
    result["closets_rebuilt"] = closets_rebuilt
    return result


def run_forgetting_maintenance(
    palace_path: str,
    config: dict,
    dry_run: bool = False,
    force: bool = False,
) -> dict:
    store = get_lifecycle_store(palace_path, config)
    now = _utcnow()
    if not force:
        last = store.get_last_maintenance_at()
        if last:
            hours = float(config.get("maintenance_min_interval_hours", 24))
            if now - _parse_dt(last) < timedelta(hours=hours):
                return {
                    "success": True,
                    "skipped": True,
                    "reason": "rate_limited",
                    "last_maintenance_at": last,
                }

    decayed = store.apply_decay(now=now)
    candidates = store.purge_candidates(now=now)
    deleted = []
    for candidate in candidates:
        if has_current_kg_facts_for_drawer(palace_path, candidate.drawer_id):
            continue
        result = forget_drawer(
            palace_path,
            candidate.drawer_id,
            reason="automatic lifecycle purge",
            origin=_ORIGIN_MAINTENANCE,
            config=config,
            dry_run=dry_run,
        )
        if result.get("success"):
            deleted.append(result)
    if not dry_run:
        store.set_last_maintenance_at(now.isoformat())
    return {
        "success": True,
        "skipped": False,
        "decayed_marked": decayed,
        "purge_candidates": len(candidates),
        "deleted": deleted,
        "deleted_count": len(deleted),
        "dry_run": dry_run,
    }
