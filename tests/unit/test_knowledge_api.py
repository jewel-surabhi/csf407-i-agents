"""Unit tests for `KnowledgeAPI` — the declarative-layer facade.

The per-store tests (test_provenance.py, test_belief_graph.py) cover the two
underlying stores individually. These tests cover the COMPOSITION — the
SQL-first atomic-write policy, the auto-supersede semantics, and the two
scenarios end-to-end through the public API.

Owner: Person 1 (P1). See PROJECT_PLAN.md sections G.1, I, J.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from iagent.declarative.api import KnowledgeAPI
from iagent.declarative.belief_graph import BeliefGraph
from iagent.declarative.models import Belief, KnowledgeError
from iagent.declarative.provenance import ProvenanceStore

# ------------------------- fixtures -----------------------------------------


@pytest.fixture()
def api(tmp_path) -> KnowledgeAPI:
    """Fresh KnowledgeAPI per test: tmp SQLite + empty graph + standard sources."""
    store = ProvenanceStore(str(tmp_path / "test.db"))
    graph = BeliefGraph()
    api_ = KnowledgeAPI(graph, store)
    # Register the sources Scenarios A + B need. log_source is idempotent.
    api_.log_source("default_map", "static_map", 0.6, "floor plan")
    api_.log_source("lidar_front", "sensor", 0.9, "front lidar")
    api_.log_source("camera_front", "sensor", 0.7, "front camera")
    api_.log_source("bot_02", "agent", 0.5, "maintenance bot")
    api_.log_source("user", "user", 0.4, "human operator")
    yield api_
    store.close()


def _belief(
    *,
    subject: str = "path_ahead",
    predicate: str = "is_clear",
    object_: str = "true",
    confidence: float = 1.0,
    source: str = "default_map",
    perspective: str = "self",
    status: str = "active",
    observed_at: datetime | None = None,
) -> Belief:
    return Belief(
        subject=subject,
        predicate=predicate,
        object=object_,
        confidence=confidence,
        source=source,
        perspective=perspective,
        status=status,
        observed_at=observed_at or datetime(2026, 9, 26, 10, 0, 0, tzinfo=UTC),
    )


# ------------------------- construction --------------------------------------


def test_init_requires_both_graph_and_store():
    with pytest.raises(KnowledgeError, match="requires both"):
        KnowledgeAPI(None, None)


# ------------------------- reads --------------------------------------------


def test_get_belief_returns_none_for_absent_key(api):
    assert api.get_belief("mystery", "located_at") is None


def test_get_belief_returns_active_after_update(api):
    api.update_belief(_belief(), reason="seed")
    b = api.get_belief("path_ahead", "is_clear")
    assert b is not None
    assert b.confidence == 1.0
    assert b.source == "default_map"
    assert b.provenance_id is not None


def test_query_perspective_returns_empty_for_absent_subject(api):
    assert api.query_perspective("mystery") == []


def test_query_perspective_groups_three_perspectives(api):
    """Scenario B guarantee at the API level: three views coexist, grouped."""
    api.update_belief(_belief(
        subject="box_01", predicate="has_color", object_="red_atom",
        source="user", perspective="user", confidence=0.9,
    ), reason="user statement")
    api.update_belief(_belief(
        subject="box_01", predicate="has_color", object_="brown_atom",
        source="camera_front", perspective="self", confidence=0.82,
    ), reason="live camera")
    api.update_belief(_belief(
        subject="box_01", predicate="has_color", object_="blue_atom",
        source="bot_02", perspective="third_party:bot_02", confidence=0.7,
    ), reason="maintenance log")

    views = api.query_perspective("box_01", predicate="has_color")
    by_persp = {v.perspective: v for v in views}
    assert set(by_persp) == {"user", "self", "third_party:bot_02"}
    assert by_persp["user"].beliefs[0].object == "red_atom"
    assert by_persp["self"].beliefs[0].object == "brown_atom"
    assert by_persp["third_party:bot_02"].beliefs[0].object == "blue_atom"


def test_query_provenance_returns_history_newest_first(api):
    a = api.update_belief(_belief(predicate="is_clear"), reason="1st")
    b = api.update_belief(_belief(predicate="is_wet", confidence=0.3), reason="2nd")
    c = api.update_belief(_belief(predicate="is_muddy", confidence=0.5), reason="3rd")
    hist = api.query_provenance("path_ahead")
    assert [h.provenance_id for h in hist] == [c.provenance_id, b.provenance_id, a.provenance_id]


def test_query_provenance_filtered_by_predicate(api):
    api.update_belief(_belief(predicate="is_clear"), reason="a")
    api.update_belief(_belief(predicate="is_wet", confidence=0.3), reason="b")
    only_wet = api.query_provenance("path_ahead", predicate="is_wet")
    assert len(only_wet) == 1
    assert only_wet[0].predicate == "is_wet"


def test_list_sources_reflects_log_source_and_is_idempotent(api):
    ids_before = {s.source_id for s in api.list_sources()}
    api.log_source("new_source", "sensor", 0.5, "unit test")
    api.log_source("new_source", "sensor", 0.99, "re-register")  # idempotent
    ids_after = {s.source_id for s in api.list_sources()}
    assert "new_source" in ids_after - ids_before
    # Idempotent: trust_prior of the ORIGINAL row is preserved, not overwritten.
    new = next(s for s in api.list_sources() if s.source_id == "new_source")
    assert new.trust_prior == 0.5


# ------------------------- update_belief -------------------------------------


def test_update_belief_writes_sql_row_and_graph_edge_with_matching_prov_id(api):
    entry = api.update_belief(_belief(), reason="seed")
    graph_belief = api.get_belief("path_ahead", "is_clear")
    assert graph_belief is not None
    # Graph edge carries the DB-assigned provenance_id.
    assert graph_belief.provenance_id == entry.provenance_id
    # SQL row is the same one.
    hist = api.query_provenance("path_ahead", predicate="is_clear")
    assert hist[0].provenance_id == entry.provenance_id
    assert hist[0].payload == {"reason": "seed"}


def test_update_belief_auto_supersedes_existing_active_belief(api):
    """Re-asserting the same (subject, predicate, perspective) key
    auto-downgrades the old row so the invariant always holds."""
    first = api.update_belief(_belief(confidence=0.6), reason="initial")
    second = api.update_belief(_belief(confidence=0.9, source="lidar_front"),
                               reason="refined")

    # Graph: exactly one active belief for this key, and it's the new one.
    active = api.get_belief("path_ahead", "is_clear")
    assert active is not None
    assert active.provenance_id == second.provenance_id
    assert active.confidence == 0.9
    assert active.source == "lidar_front"

    # SQL: history has TWO rows for this key; the older one is downgraded.
    hist = api.query_provenance("path_ahead", predicate="is_clear")
    by_id = {h.provenance_id: h for h in hist}
    assert by_id[first.provenance_id].status == "downgraded"
    assert by_id[second.provenance_id].status == "active"


def test_update_belief_refuses_non_active_status(api):
    with pytest.raises(KnowledgeError, match="update_belief"):
        api.update_belief(_belief(status="downgraded"), reason="wrong method")


def test_update_belief_across_perspectives_does_not_supersede(api):
    """A write to `self` must NOT touch the user-perspective belief for the same key."""
    api.update_belief(_belief(
        subject="box_01", predicate="has_color", object_="red_atom",
        source="user", perspective="user",
    ), reason="user")
    api.update_belief(_belief(
        subject="box_01", predicate="has_color", object_="brown_atom",
        source="camera_front", perspective="self",
    ), reason="self")

    user_view = api.get_belief("box_01", "has_color", "user")
    self_view = api.get_belief("box_01", "has_color", "self")
    assert user_view.object == "red_atom"
    assert self_view.object == "brown_atom"


# ------------------------- downgrade_belief ----------------------------------


def test_downgrade_belief_writes_postmortem_and_flips_old_and_graph(api):
    orig = api.update_belief(_belief(confidence=1.0), reason="seed")
    dg = api.downgrade_belief("path_ahead", "is_clear",
                              new_confidence=0.2, reason="sensor contradicts")

    # Postmortem row cites the superseded id.
    assert dg.status == "downgraded"
    assert dg.confidence == 0.2
    assert dg.payload == {"reason": "sensor contradicts", "supersedes": orig.provenance_id}

    # Old row also flipped.
    hist = api.query_provenance("path_ahead", predicate="is_clear")
    by_id = {h.provenance_id: h for h in hist}
    assert by_id[orig.provenance_id].status == "downgraded"

    # Graph: no active belief for this key anymore.
    assert api.get_belief("path_ahead", "is_clear") is None


def test_downgrade_belief_raises_when_no_active_belief(api):
    with pytest.raises(KnowledgeError, match="no active belief"):
        api.downgrade_belief("mystery", "located_at", new_confidence=0.1, reason="nope")


def test_downgrade_belief_rejects_out_of_range_confidence(api):
    api.update_belief(_belief(), reason="seed")
    with pytest.raises(KnowledgeError, match="confidence out of range"):
        api.downgrade_belief("path_ahead", "is_clear",
                             new_confidence=1.5, reason="bad")


def test_downgrade_belief_respects_perspective(api):
    """Downgrading `self` must not touch `user`'s belief for the same key."""
    api.update_belief(_belief(
        subject="box_01", predicate="has_color", object_="red_atom",
        source="user", perspective="user",
    ), reason="user")
    api.update_belief(_belief(
        subject="box_01", predicate="has_color", object_="brown_atom",
        source="camera_front", perspective="self",
    ), reason="self")

    api.downgrade_belief("box_01", "has_color", new_confidence=0.1,
                         reason="lighting bad", perspective="self")

    assert api.get_belief("box_01", "has_color", "self") is None
    still_user = api.get_belief("box_01", "has_color", "user")
    assert still_user is not None
    assert still_user.object == "red_atom"


# ------------------------- Scenario A end-to-end -----------------------------


def test_scenario_a_full_shape(api):
    """Full Scenario A pipeline through the public API."""
    # Seed: map says path is clear.
    api.update_belief(_belief(
        confidence=1.0, source="default_map",
        observed_at=datetime(2026, 9, 26, 9, 0, tzinfo=UTC),
    ), reason="baseline map")

    # LLM step 1: read the belief.
    initial = api.get_belief("path_ahead", "is_clear")
    assert initial.confidence == 1.0

    # LLM step 4: downgrade (lidar contradicts).
    api.downgrade_belief("path_ahead", "is_clear",
                         new_confidence=0.2, reason="live LiDAR contradicts")

    # LLM step 5: assert new blocked belief.
    api.update_belief(_belief(
        predicate="is_blocked", object_="obstacle@12cm",
        confidence=0.95, source="lidar_front",
        observed_at=datetime(2026, 9, 26, 10, 15, tzinfo=UTC),
    ), reason="live lidar")

    # Post-conditions asserted by the epistemic tests in Phase 8:
    # 1. `is_clear` no longer active.
    assert api.get_belief("path_ahead", "is_clear") is None
    # 2. `is_blocked` active with lidar source.
    blocked = api.get_belief("path_ahead", "is_blocked")
    assert blocked is not None
    assert blocked.confidence == 0.95
    assert blocked.source == "lidar_front"
    # 3. History: three rows for `path_ahead`, newest first, statuses right.
    hist = api.query_provenance("path_ahead")
    assert len(hist) == 3
    assert hist[0].predicate == "is_blocked" and hist[0].status == "active"
    assert hist[1].status == "downgraded"  # the postmortem row
    assert hist[2].status == "downgraded"  # the original seed row


# ------------------------- Scenario B end-to-end -----------------------------


def test_scenario_b_full_shape(api):
    """Full Scenario B pipeline through the public API."""
    # Seed: user believes red, bot_02 logged blue.
    api.update_belief(_belief(
        subject="box_01", predicate="has_color", object_="red_atom",
        confidence=0.9, source="user", perspective="user",
    ), reason="user statement")
    api.update_belief(_belief(
        subject="box_01", predicate="has_color", object_="blue_atom",
        confidence=0.7, source="bot_02", perspective="third_party:bot_02",
    ), reason="maintenance log")

    # LLM step 1: query all perspectives on box_01.has_color.
    views = api.query_perspective("box_01", predicate="has_color")
    persps = {v.perspective for v in views}
    assert persps == {"user", "third_party:bot_02"}  # no `self` yet

    # LLM step 2 (via sensor tool → update_belief on our side): egocentric read.
    api.update_belief(_belief(
        subject="box_01", predicate="has_color", object_="brown_atom",
        confidence=0.82, source="camera_front", perspective="self",
    ), reason="live camera; ambient=yellow")

    # Post-conditions the perspective_no_merge test asserts:
    # All THREE perspectives now visible, none merged.
    views = api.query_perspective("box_01", predicate="has_color")
    by_persp = {v.perspective: v.beliefs[0].object for v in views}
    assert by_persp == {
        "user": "red_atom",
        "self": "brown_atom",
        "third_party:bot_02": "blue_atom",
    }
    # And each perspective retrievable by name individually.
    assert api.get_belief("box_01", "has_color", "user").object == "red_atom"
    assert api.get_belief("box_01", "has_color", "self").object == "brown_atom"
    assert api.get_belief("box_01", "has_color", "third_party:bot_02").object == "blue_atom"
