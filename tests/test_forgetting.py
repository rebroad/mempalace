import json
from datetime import timedelta
from pathlib import Path

from mempalace.config import MempalaceConfig
from mempalace.forgetting import (
    _content_sha256,
    _iso_now,
    _utcnow,
    forget_drawer,
    get_lifecycle_store,
    has_current_kg_facts_for_drawer,
    run_forgetting_maintenance,
)
from mempalace.knowledge_graph import KnowledgeGraph
from mempalace.searcher import search_memories


class FakeCollection:
    def __init__(self):
        self.rows = {}

    def upsert(self, ids, documents, metadatas):
        for drawer_id, document, metadata in zip(ids, documents, metadatas):
            self.rows[drawer_id] = {"document": document, "metadata": dict(metadata or {})}

    def get(self, ids=None, where=None, include=None, limit=None, offset=None):
        items = list(self.rows.items())
        if ids is not None:
            wanted = set(ids)
            items = [(drawer_id, row) for drawer_id, row in items if drawer_id in wanted]
        if where:
            items = [(drawer_id, row) for drawer_id, row in items if _matches_where(row["metadata"], where)]
        if offset:
            items = items[offset:]
        if limit is not None:
            items = items[:limit]
        return {
            "ids": [drawer_id for drawer_id, _ in items],
            "documents": [row["document"] for _, row in items],
            "metadatas": [row["metadata"] for _, row in items],
        }

    def delete(self, ids=None, where=None):
        if ids is not None:
            for drawer_id in ids:
                self.rows.pop(drawer_id, None)
            return
        if where:
            doomed = [
                drawer_id
                for drawer_id, row in self.rows.items()
                if _matches_where(row["metadata"], where)
            ]
            for drawer_id in doomed:
                self.rows.pop(drawer_id, None)

    def query(self, query_texts, n_results, include, where=None):
        query = (query_texts or [""])[0].lower()
        terms = [term for term in query.split() if term]
        items = list(self.rows.items())
        if where:
            items = [(drawer_id, row) for drawer_id, row in items if _matches_where(row["metadata"], where)]
        scored = []
        for drawer_id, row in items:
            document = row["document"]
            doc_lower = document.lower()
            matches = sum(1 for term in terms if term in doc_lower)
            distance = 1.5 if matches == 0 else max(0.0, 1.0 - 0.2 * matches)
            scored.append((distance, drawer_id, row))
        scored.sort(key=lambda item: item[0])
        top = scored[:n_results]
        return {
            "ids": [[drawer_id for _, drawer_id, _ in top]],
            "documents": [[row["document"] for _, _, row in top]],
            "metadatas": [[row["metadata"] for _, _, row in top]],
            "distances": [[distance for distance, _, _ in top]],
        }


def _matches_where(metadata, where):
    if not where:
        return True
    if "$and" in where:
        return all(_matches_where(metadata, clause) for clause in where["$and"])
    for key, expected in where.items():
        value = metadata.get(key)
        if isinstance(expected, dict):
            if "$in" in expected:
                if value not in expected["$in"]:
                    return False
            else:
                return False
        elif value != expected:
            return False
    return True


def _write_config(tmp_path):
    cfg_dir = tmp_path / ".mempalace"
    cfg_dir.mkdir()
    config = {
        "palace_path": str(tmp_path / "palace"),
        "forgetting": {
            "enabled": True,
            "mode": "hard_delete",
            "profile": "conservative",
            "auto_run_hooks": True,
            "maintenance_min_interval_hours": 24,
            "initial_strength_days": 30,
            "decay_retrievability_threshold": 0.15,
            "decay_after_days": 180,
            "purge_after_days": 365,
            "max_strength_days": 365,
            "include_decayed_default": False,
        },
    }
    (cfg_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")
    return cfg_dir


def test_config_exposes_default_forgetting(tmp_path, monkeypatch):
    cfg_dir = _write_config(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))
    cfg = MempalaceConfig(config_dir=str(cfg_dir))
    assert cfg.forgetting["enabled"] is True
    assert cfg.forgetting["mode"] == "hard_delete"
    assert cfg.forgetting["include_decayed_default"] is False
    store = get_lifecycle_store(str(tmp_path / "palace"), cfg.forgetting)
    assert Path(store.db_path).exists()


def test_tombstoned_identical_content_is_blocked_but_changed_content_allowed(tmp_path, monkeypatch):
    cfg_dir = _write_config(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))
    palace_path = str(tmp_path / "palace")
    store = get_lifecycle_store(palace_path, MempalaceConfig(config_dir=str(cfg_dir)).forgetting)

    drawer_id = "drawer_abc"
    content = "hello world"
    store.record_tombstone(
        drawer_id,
        _content_sha256(content),
        "/var/tmp/source.txt",
        "test",
        "explicit",
    )
    assert store.should_ingest(drawer_id, content, "/var/tmp/source.txt") is False
    assert store.should_ingest(drawer_id, "hello world changed", "/var/tmp/source.txt") is True


def test_search_hides_decayed_by_default_and_can_include_them(tmp_path, monkeypatch):
    cfg_dir = _write_config(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))
    palace_path = str(tmp_path / "palace")
    col = FakeCollection()
    monkeypatch.setattr("mempalace.searcher.get_collection", lambda *args, **kwargs: col)
    monkeypatch.setattr("mempalace.searcher.get_closets_collection", lambda *args, **kwargs: FakeCollection())
    lifecycle = get_lifecycle_store(palace_path, MempalaceConfig(config_dir=str(cfg_dir)).forgetting)

    doc = "The auth migration uses JWT refresh tokens."
    meta = {
        "wing": "project",
        "room": "backend",
        "source_file": "/repo/auth.md",
        "filed_at": _iso_now(),
    }
    col.upsert(ids=["drawer_auth"], documents=[doc], metadatas=[meta])
    lifecycle.register_ingest("drawer_auth", doc, meta)
    lifecycle.apply_decay(now=_utcnow() + timedelta(days=181))

    hidden = search_memories("JWT refresh", palace_path, include_decayed=False)
    assert hidden["results"] == []

    shown = search_memories("JWT refresh", palace_path, include_decayed=True)
    assert len(shown["results"]) == 1
    assert shown["results"][0]["state"] == "decayed"


def test_forget_drawer_invalidates_kg_and_tombstones(tmp_path, monkeypatch):
    cfg_dir = _write_config(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))
    palace_path = str(tmp_path / "palace")
    col = FakeCollection()
    monkeypatch.setattr("mempalace.forgetting.get_collection", lambda *args, **kwargs: col)
    monkeypatch.setattr("mempalace.forgetting.rebuild_closets_for_source_file", lambda *args, **kwargs: 0)
    lifecycle = get_lifecycle_store(palace_path, MempalaceConfig(config_dir=str(cfg_dir)).forgetting)

    doc = "Edmund Broadley prefers direct language."
    meta = {
        "wing": "user",
        "room": "identity",
        "source_file": "/repo/user.txt",
        "filed_at": _iso_now(),
    }
    col.upsert(ids=["drawer_user_1"], documents=[doc], metadatas=[meta])
    lifecycle.register_ingest("drawer_user_1", doc, meta)

    kg = KnowledgeGraph(db_path=str(tmp_path / "palace" / "knowledge_graph.sqlite3"))
    kg.add_triple(
        "Edmund Broadley",
        "prefers",
        "direct language",
        source_drawer_id="drawer_user_1",
        adapter_name="test",
    )
    assert has_current_kg_facts_for_drawer(palace_path, "drawer_user_1") is True

    result = forget_drawer(palace_path, "drawer_user_1", reason="cleanup", config={})
    assert result["success"] is True
    assert has_current_kg_facts_for_drawer(palace_path, "drawer_user_1") is False
    assert col.get(ids=["drawer_user_1"]).get("ids") == []
    assert lifecycle.has_tombstone("drawer_user_1") is True


def test_maintenance_dry_run_and_live_delete_only_stale_low_value_drawers(tmp_path, monkeypatch):
    cfg_dir = _write_config(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))
    palace_path = str(tmp_path / "palace")
    col = FakeCollection()
    monkeypatch.setattr("mempalace.forgetting.get_collection", lambda *args, **kwargs: col)
    monkeypatch.setattr("mempalace.forgetting.rebuild_closets_for_source_file", lambda *args, **kwargs: 0)
    cfg = MempalaceConfig(config_dir=str(cfg_dir)).forgetting
    lifecycle = get_lifecycle_store(palace_path, cfg)

    stale_doc = "Old note about a temporary build issue."
    stale_meta = {
        "wing": "project",
        "room": "technical",
        "source_file": "/repo/old.txt",
        "filed_at": _iso_now(),
    }
    col.upsert(ids=["drawer_old"], documents=[stale_doc], metadatas=[stale_meta])
    lifecycle.register_ingest("drawer_old", stale_doc, stale_meta)
    lifecycle._conn.execute(
        """
        UPDATE drawer_state
           SET state = 'decayed',
               decayed_at = ?,
               created_at = ?,
               last_accessed_at = ?,
               access_count = 1
         WHERE drawer_id = 'drawer_old'
        """,
        (
            (_utcnow() - timedelta(days=400)).isoformat(),
            (_utcnow() - timedelta(days=400)).isoformat(),
            (_utcnow() - timedelta(days=400)).isoformat(),
        ),
    )
    lifecycle._conn.commit()

    preview = run_forgetting_maintenance(palace_path, cfg, dry_run=True, force=True)
    assert preview["deleted_count"] == 1
    assert col.get(ids=["drawer_old"]).get("ids") == ["drawer_old"]

    actual = run_forgetting_maintenance(palace_path, cfg, dry_run=False, force=True)
    assert actual["deleted_count"] == 1
    assert col.get(ids=["drawer_old"]).get("ids") == []


def test_maintenance_rate_limit_skips_without_force(tmp_path, monkeypatch):
    cfg_dir = _write_config(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))
    palace_path = str(tmp_path / "palace")
    cfg = MempalaceConfig(config_dir=str(cfg_dir)).forgetting
    lifecycle = get_lifecycle_store(palace_path, cfg)
    lifecycle.set_last_maintenance_at(_iso_now())

    result = run_forgetting_maintenance(palace_path, cfg, dry_run=False, force=False)
    assert result["success"] is True
    assert result["skipped"] is True
    assert result["reason"] == "rate_limited"


def test_maintenance_handles_naive_stored_timestamps(tmp_path, monkeypatch):
    cfg_dir = _write_config(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))
    palace_path = str(tmp_path / "palace")
    col = FakeCollection()
    monkeypatch.setattr("mempalace.forgetting.get_collection", lambda *args, **kwargs: col)
    monkeypatch.setattr("mempalace.forgetting.rebuild_closets_for_source_file", lambda *args, **kwargs: 0)
    cfg = MempalaceConfig(config_dir=str(cfg_dir)).forgetting
    lifecycle = get_lifecycle_store(palace_path, cfg)

    doc = "Naive timestamp regression."
    meta = {
        "wing": "project",
        "room": "technical",
        "source_file": "/repo/naive.txt",
        "filed_at": _iso_now(),
    }
    col.upsert(ids=["drawer_naive"], documents=[doc], metadatas=[meta])
    lifecycle.register_ingest("drawer_naive", doc, meta)
    lifecycle._conn.execute(
        """
        UPDATE drawer_state
           SET created_at = ?,
               last_accessed_at = ?
         WHERE drawer_id = 'drawer_naive'
        """,
        (
            "2026-01-01T12:00:00",
            "2026-01-01T12:00:00",
        ),
    )
    lifecycle._conn.commit()

    result = run_forgetting_maintenance(
        palace_path,
        cfg,
        dry_run=True,
        force=True,
    )
    assert result["success"] is True
    assert result["skipped"] is False
