"""Unit tests for ProvenanceStore.

Covers the SQLite half of the declarative layer: insert / query / status /
foreign-key enforcement / the append-only invariant / the current_beliefs view.

Owner: Person 1 (P1). See PROJECT_PLAN.md sections F.2 and G.1.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from iagent.declarative.models import KnowledgeError, ProvenanceEntry, Source
from iagent.declarative.provenance import ProvenanceStore

# ------------------------- fixtures -----------------------------------------


@pytest.fixture()
def store(tmp_path):
    """Fresh SQLite DB per test, pre-seeded with two sources."""
    db_path = str(tmp_path / "test.db")
    s = ProvenanceStore(db_path)
    s.insert_source(Source(source_id="lidar_front", kind="sensor", trust_prior=0.9))
    s.insert_source(Source(source_id="default_map", kind="static_map", trust_prior=0.6))
    yield s
    s.close()


def _entry(
    *,
    subject: str = "path_ahead",
    predicate: str = "is_clear",
    object_: str = "true",
    confidence: float = 1.0,
    source_id: str = "lidar_front",
    perspective: str = "self",
    status: str = "active",
    observed_at: datetime | None = None,
    payload: dict | None = None,
) -> ProvenanceEntry:
    return ProvenanceEntry(
        subject=subject,
        predicate=predicate,
        object=object_,
        confidence=confidence,
        source_id=source_id,
        perspective=perspective,
        status=status,
        observed_at=observed_at or datetime(2026, 9, 26, 10, 0, 0, tzinfo=UTC),
        payload=payload,
    )


# ------------------------- tests --------------------------------------------


def test_insert_populates_id_and_recorded_at(store):
    e = store.insert(_entry())
    assert e.provenance_id is not None and e.provenance_id > 0
    assert e.recorded_at is not None
    # recorded_at is server-side (SQLite `datetime('now')` returns UTC);
    # compare in UTC to avoid local-time offset noise.
    delta = datetime.now(UTC) - e.recorded_at.replace(tzinfo=UTC)
    assert abs(delta.total_seconds()) < 60


def test_get_active_returns_most_recent_active(store):
    older = store.insert(
        _entry(observed_at=datetime(2026, 9, 26, 10, 0, 0, tzinfo=UTC))
    )
    newer = store.insert(
        _entry(observed_at=datetime(2026, 9, 26, 10, 5, 0, tzinfo=UTC))
    )
    got = store.get_active("path_ahead", "is_clear")
    assert got is not None
    # Both rows are 'active' — newest by recorded_at wins.
    assert got.provenance_id == newer.provenance_id
    assert got.provenance_id != older.provenance_id


def test_get_active_returns_none_when_all_downgraded(store):
    e = store.insert(_entry())
    store.mark_status(e.provenance_id, "downgraded")
    assert store.get_active("path_ahead", "is_clear") is None


def test_mark_status_only_touches_status(store):
    e = store.insert(_entry(confidence=0.9, payload={"note": "keep me"}))
    store.mark_status(e.provenance_id, "downgraded")
    reread = store.query_by_subject("path_ahead")[0]
    # Every other column preserved — append-only invariant for individual columns.
    assert reread.provenance_id == e.provenance_id
    assert reread.status == "downgraded"
    assert reread.confidence == 0.9
    assert reread.payload == {"note": "keep me"}


def test_mark_status_rejects_unknown_id(store):
    with pytest.raises(KnowledgeError, match="no provenance row"):
        store.mark_status(99999, "downgraded")


def test_mark_status_rejects_invalid_status(store):
    e = store.insert(_entry())
    with pytest.raises(KnowledgeError, match="invalid status"):
        store.mark_status(e.provenance_id, "totally_fake")


def test_query_by_subject_is_newest_first(store):
    a = store.insert(_entry(predicate="is_clear"))
    b = store.insert(_entry(predicate="is_blocked"))
    c = store.insert(_entry(predicate="is_wet"))
    hist = store.query_by_subject("path_ahead")
    assert [h.provenance_id for h in hist] == [c.provenance_id, b.provenance_id, a.provenance_id]


def test_query_by_subject_filtered_by_predicate(store):
    store.insert(_entry(predicate="is_clear"))
    store.insert(_entry(predicate="is_blocked"))
    store.insert(_entry(predicate="is_blocked"))
    only_blocked = store.query_by_subject("path_ahead", predicate="is_blocked")
    assert len(only_blocked) == 2
    assert all(e.predicate == "is_blocked" for e in only_blocked)


def test_query_by_subject_empty_when_unknown(store):
    assert store.query_by_subject("mystery_thing_42") == []


def test_current_beliefs_view_excludes_downgraded(store):
    e1 = store.insert(_entry(predicate="is_clear", confidence=1.0))
    store.insert(_entry(predicate="is_blocked", confidence=0.95))
    # downgrade the is_clear one
    store.mark_status(e1.provenance_id, "downgraded")

    current = list(store.iter_current_beliefs())
    predicates = {c.predicate for c in current}
    assert predicates == {"is_blocked"}  # is_clear filtered out


def test_current_beliefs_view_preserves_perspective(store):
    # Same (subject, predicate) held from three perspectives should coexist in the view.
    store.insert_source(Source(source_id="user", kind="user", trust_prior=0.4))
    store.insert_source(Source(source_id="bot_02", kind="agent", trust_prior=0.5))
    store.insert(_entry(subject="box_01", predicate="has_color", object_="red",
                        source_id="user", perspective="user"))
    store.insert(_entry(subject="box_01", predicate="has_color", object_="brown",
                        source_id="lidar_front", perspective="self"))
    store.insert(_entry(subject="box_01", predicate="has_color", object_="blue",
                        source_id="bot_02", perspective="third_party:bot_02"))

    box_beliefs = [c for c in store.iter_current_beliefs() if c.subject == "box_01"]
    perspectives = {c.perspective for c in box_beliefs}
    assert perspectives == {"user", "self", "third_party:bot_02"}
    # None of them merged into one.
    assert len(box_beliefs) == 3


def test_insert_source_is_idempotent(store):
    # Fixture already inserted lidar_front + default_map.
    store.insert_source(Source(source_id="lidar_front", kind="sensor", trust_prior=0.99))
    ids = [s.source_id for s in store.list_sources()]
    # Still only one row for lidar_front — INSERT OR IGNORE.
    assert ids.count("lidar_front") == 1
    # And trust_prior of the original is unchanged (IGNORE = no overwrite).
    lidar = next(s for s in store.list_sources() if s.source_id == "lidar_front")
    assert lidar.trust_prior == 0.9


def test_insert_with_unknown_source_raises_knowledge_error(store):
    with pytest.raises(KnowledgeError, match="unknown source_id"):
        store.insert(_entry(source_id="UNREGISTERED"))


def test_history_survives_downgrade_cycle(store):
    # Simulate Scenario A: insert active belief, downgrade + insert successor,
    # verify all three provenance rows are queryable.
    e1 = store.insert(_entry(predicate="is_clear", confidence=1.0))
    # Downgrade with a *new* row + status flip on the old.
    e2 = store.insert(
        _entry(
            predicate="is_clear",
            confidence=0.2,
            status="downgraded",
            payload={"reason": "lidar contradicts", "supersedes": e1.provenance_id},
            observed_at=datetime(2026, 9, 26, 10, 15, 0, tzinfo=UTC),
        )
    )
    store.mark_status(e1.provenance_id, "downgraded")
    e3 = store.insert(
        _entry(
            predicate="is_blocked",
            confidence=0.95,
            observed_at=datetime(2026, 9, 26, 10, 15, 1, tzinfo=UTC),
        )
    )
    hist = store.query_by_subject("path_ahead")
    assert [h.provenance_id for h in hist] == [e3.provenance_id, e2.provenance_id, e1.provenance_id]
    assert [h.status for h in hist] == ["active", "downgraded", "downgraded"]


def test_context_manager_closes_connection(tmp_path):
    db_path = str(tmp_path / "cm.db")
    with ProvenanceStore(db_path) as s:
        s.insert_source(Source(source_id="lidar_front", kind="sensor", trust_prior=0.9))
        s.insert(_entry())
    # After exit, further use should raise (sqlite3 raises ProgrammingError on closed conn).
    import sqlite3

    with pytest.raises(sqlite3.ProgrammingError):
        s.list_sources()
