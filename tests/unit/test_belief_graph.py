"""Unit tests for BeliefGraph.

Covers the NetworkX half of the declarative layer: node/edge add + auto-node
creation + perspective coexistence (the Scenario B guarantee) + status
transitions + the at-most-one-active invariant.

Owner: Person 1 (P1). See PROJECT_PLAN.md section F.1.

Note on the strict-raise behavior of `set_edge_status`: current design raises
`KnowledgeError` when no active edge matches. We may relax this to a silent
no-op after integrating with the procedural layer; if so, update
`test_set_edge_status_raises_on_no_active_edge` accordingly.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from iagent.declarative.belief_graph import BeliefGraph
from iagent.declarative.models import Belief, KnowledgeError

# ------------------------- fixtures -----------------------------------------


@pytest.fixture()
def graph() -> BeliefGraph:
    return BeliefGraph()


def _belief(
    *,
    subject: str = "path_ahead",
    predicate: str = "is_clear",
    object_: str = "true",
    confidence: float = 1.0,
    source: str = "default_map",
    perspective: str = "self",
    status: str = "active",
    provenance_id: int | None = 1,
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
        provenance_id=provenance_id,
    )


# ------------------------- node tests ----------------------------------------


def test_add_node_is_idempotent(graph):
    graph.add_node("robot", type="agent", labels=["self"])
    graph.add_node("robot", type="something_else", labels=["would_be_overwrite"])
    # Second call must NOT overwrite — earliest wins.
    snap = graph.snapshot()
    robot = next(n for n in snap["nodes"] if n["id"] == "robot")
    assert robot["type"] == "agent"
    assert robot["labels"] == ["self"]


def test_add_edge_auto_creates_missing_nodes(graph):
    key = graph.add_edge(_belief())
    assert isinstance(key, int)
    assert graph.has_node("path_ahead")
    assert graph.has_node("true")  # `object_` also auto-created


# ------------------------- edge read tests -----------------------------------


def test_get_active_edge_returns_single_active(graph):
    graph.add_edge(_belief(confidence=0.95, source="lidar_front"))
    got = graph.get_active_edge("path_ahead", "is_clear")
    assert got is not None
    assert got.confidence == 0.95
    assert got.source == "lidar_front"
    assert got.status == "active"


def test_get_active_edge_returns_none_for_unknown_subject(graph):
    assert graph.get_active_edge("mystery_thing_42", "located_at") is None


def test_get_active_edge_returns_none_when_only_downgraded(graph):
    graph.add_edge(_belief(status="downgraded"))
    assert graph.get_active_edge("path_ahead", "is_clear") is None


def test_get_active_edge_raises_when_invariant_violated(graph):
    # Add two active edges for the same key — should never happen through the
    # API layer, but the graph must SURFACE the bug rather than pick one.
    graph.add_edge(_belief())
    graph.add_edge(_belief(object_="also_true"))
    with pytest.raises(KnowledgeError, match="invariant"):
        graph.get_active_edge("path_ahead", "is_clear")


def test_iter_active_edges_skips_downgraded_and_filters_by_predicate(graph):
    graph.add_edge(_belief(predicate="is_clear", confidence=1.0))
    graph.add_edge(_belief(predicate="is_wet", confidence=0.3))
    downgraded_key = graph.add_edge(_belief(predicate="was_swept", status="downgraded"))
    assert isinstance(downgraded_key, int)

    all_active = list(graph.iter_active_edges_for("path_ahead"))
    preds = {b.predicate for b in all_active}
    assert preds == {"is_clear", "is_wet"}  # downgraded 'was_swept' skipped

    only_wet = list(graph.iter_active_edges_for("path_ahead", predicate="is_wet"))
    assert len(only_wet) == 1
    assert only_wet[0].confidence == 0.3


def test_iter_active_edges_for_unknown_subject_yields_nothing(graph):
    assert list(graph.iter_active_edges_for("mystery")) == []


# ------------------------- the Scenario B guarantee --------------------------


def test_three_perspectives_coexist_without_merging(graph):
    """The whole point of MultiDiGraph: same (subject, predicate) can be held
    from user / self / third-party simultaneously, as three distinct edges."""
    graph.add_edge(_belief(
        subject="box_01", predicate="has_color", object_="red_atom",
        confidence=0.9, source="user", perspective="user",
    ))
    graph.add_edge(_belief(
        subject="box_01", predicate="has_color", object_="brown_atom",
        confidence=0.82, source="camera_front", perspective="self",
    ))
    graph.add_edge(_belief(
        subject="box_01", predicate="has_color", object_="blue_atom",
        confidence=0.7, source="bot_02", perspective="third_party:bot_02",
    ))

    beliefs = list(graph.iter_active_edges_for("box_01", predicate="has_color"))
    assert len(beliefs) == 3
    by_persp = {b.perspective: b for b in beliefs}
    assert by_persp["user"].object == "red_atom"
    assert by_persp["self"].object == "brown_atom"
    assert by_persp["third_party:bot_02"].object == "blue_atom"

    # And each perspective is retrievable individually via get_active_edge.
    assert graph.get_active_edge("box_01", "has_color", "user").object == "red_atom"
    assert graph.get_active_edge("box_01", "has_color", "self").object == "brown_atom"
    assert (
        graph.get_active_edge("box_01", "has_color", "third_party:bot_02").object
        == "blue_atom"
    )


# ------------------------- set_edge_status tests -----------------------------


def test_set_edge_status_flips_status(graph):
    graph.add_edge(_belief(confidence=1.0))
    graph.set_edge_status("path_ahead", "is_clear", "self", "downgraded")
    assert graph.get_active_edge("path_ahead", "is_clear") is None
    # The edge itself still exists — we didn't delete it, just flipped the status.
    all_edges = list(graph.iter_all_edges())
    assert len(all_edges) == 1
    assert all_edges[0].status == "downgraded"
    # Confidence untouched because we didn't pass new_confidence.
    assert all_edges[0].confidence == 1.0


def test_set_edge_status_updates_confidence_when_requested(graph):
    graph.add_edge(_belief(confidence=1.0))
    graph.set_edge_status(
        "path_ahead", "is_clear", "self", "downgraded", new_confidence=0.2
    )
    all_edges = list(graph.iter_all_edges())
    assert all_edges[0].status == "downgraded"
    assert all_edges[0].confidence == 0.2


def test_set_edge_status_respects_perspective(graph):
    graph.add_edge(_belief(
        subject="box_01", predicate="has_color", object_="red_atom",
        source="user", perspective="user",
    ))
    graph.add_edge(_belief(
        subject="box_01", predicate="has_color", object_="brown_atom",
        source="camera_front", perspective="self",
    ))
    # Downgrade only the self perspective; user perspective must remain active.
    graph.set_edge_status("box_01", "has_color", "self", "downgraded")
    assert graph.get_active_edge("box_01", "has_color", "self") is None
    still_user = graph.get_active_edge("box_01", "has_color", "user")
    assert still_user is not None
    assert still_user.object == "red_atom"


def test_set_edge_status_raises_on_no_active_edge(graph):
    # Empty graph, no matching edge to flip.
    with pytest.raises(KnowledgeError, match="no active edge"):
        graph.set_edge_status("path_ahead", "is_clear", "self", "downgraded")


def test_set_edge_status_rejects_invalid_status(graph):
    graph.add_edge(_belief())
    with pytest.raises(KnowledgeError, match="invalid status"):
        graph.set_edge_status("path_ahead", "is_clear", "self", "quite_stale")


def test_set_edge_status_rejects_out_of_range_confidence(graph):
    graph.add_edge(_belief())
    with pytest.raises(KnowledgeError, match="confidence out of range"):
        graph.set_edge_status(
            "path_ahead", "is_clear", "self", "downgraded", new_confidence=1.5
        )


# ------------------------- snapshot / debugging ------------------------------


def test_snapshot_captures_nodes_and_edges(graph):
    graph.add_node("robot", type="agent", labels=["self"])
    graph.add_edge(_belief(confidence=0.95))

    snap = graph.snapshot()
    assert isinstance(snap, dict)
    assert {n["id"] for n in snap["nodes"]} >= {"robot", "path_ahead", "true"}
    # Edge fields survive round-trip.
    e = snap["edges"][0]
    assert e["subject"] == "path_ahead"
    assert e["object"] == "true"
    assert e["relation"] == "is_clear"
    assert e["confidence"] == 0.95
    assert e["status"] == "active"


def test_len_reports_edge_count(graph):
    assert len(graph) == 0
    graph.add_edge(_belief())
    graph.add_edge(_belief(predicate="is_wet"))
    assert len(graph) == 2
