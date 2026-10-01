"""Unit tests for `iagent.declarative.seeding`.

Owner: Person 1 (P1). Thin glue over BeliefGraph + ProvenanceStore, so these
tests focus on file-format handling and idempotency.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from iagent.declarative.belief_graph import BeliefGraph
from iagent.declarative.models import KnowledgeError
from iagent.declarative.provenance import ProvenanceStore
from iagent.declarative.seeding import (
    apply_beliefs_to_graph,
    load_beliefs_from_json,
    seed_from_files,
    seed_graph,
    seed_sources,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SEED_DIR = _REPO_ROOT / "data" / "seed"
_BELIEFS_JSON = _SEED_DIR / "initial_beliefs.json"
_PROVENANCE_SQL = _SEED_DIR / "initial_provenance.sql"


# ------------------------- fixtures -----------------------------------------


@pytest.fixture()
def store(tmp_path) -> ProvenanceStore:
    s = ProvenanceStore(str(tmp_path / "seed.db"))
    yield s
    s.close()


@pytest.fixture()
def graph() -> BeliefGraph:
    return BeliefGraph()


# ------------------------- load_beliefs_from_json ---------------------------


def test_load_beliefs_from_repo_seed():
    """The seed JSON that ships with the repo must be parseable."""
    data = load_beliefs_from_json(_BELIEFS_JSON)
    assert "nodes" in data
    assert isinstance(data["nodes"], list)
    ids = {n["id"] for n in data["nodes"]}
    # The repo seed contains at least the entities Scenarios A + B need.
    assert {"robot", "room_101", "path_ahead", "box_01"} <= ids


def test_load_beliefs_missing_nodes_raises(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"edges": []}), encoding="utf-8")
    with pytest.raises(KnowledgeError, match="missing required 'nodes'"):
        load_beliefs_from_json(bad)


def test_load_beliefs_invalid_json_raises(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{not: valid json", encoding="utf-8")
    with pytest.raises(KnowledgeError, match="cannot parse"):
        load_beliefs_from_json(bad)


def test_load_beliefs_missing_file_raises(tmp_path):
    with pytest.raises(KnowledgeError, match="cannot parse"):
        load_beliefs_from_json(tmp_path / "does_not_exist.json")


# ------------------------- apply_beliefs_to_graph ---------------------------


def test_apply_beliefs_adds_nodes(graph):
    data = {
        "nodes": [
            {"id": "robot", "type": "agent", "labels": ["self"]},
            {"id": "room_101", "type": "location", "labels": ["room"]},
        ],
        "edges": [],
    }
    added_nodes, added_edges = apply_beliefs_to_graph(graph, data)
    assert added_nodes == 2
    assert added_edges == 0
    assert graph.has_node("robot")
    assert graph.has_node("room_101")


def test_apply_beliefs_is_idempotent(graph):
    data = {"nodes": [{"id": "robot", "type": "agent", "labels": ["self"]}]}
    apply_beliefs_to_graph(graph, data)
    added_nodes, _ = apply_beliefs_to_graph(graph, data)
    # Second call adds nothing because the node already exists.
    assert added_nodes == 0


def test_apply_beliefs_adds_edges_when_present(graph):
    data = {
        "nodes": [
            {"id": "path_ahead", "type": "location", "labels": []},
        ],
        "edges": [
            {
                "subject": "path_ahead",
                "predicate": "is_clear",
                "object": "true",
                "confidence": 1.0,
                "source": "default_map",
                "perspective": "self",
                "status": "active",
                "observed_at": "2026-09-06T09:00:00Z",
            }
        ],
    }
    n, e = apply_beliefs_to_graph(graph, data)
    assert (n, e) == (1, 1)
    active = graph.get_active_edge("path_ahead", "is_clear")
    assert active is not None
    assert active.source == "default_map"


# ------------------------- seed_sources -------------------------------------


def test_seed_sources_populates_from_repo_seed(store):
    seed_sources(store, _PROVENANCE_SQL)
    ids = {s.source_id for s in store.list_sources()}
    # The 6 sources our scenarios rely on.
    assert ids == {"default_map", "lidar_front", "camera_front", "bot_02",
                   "user", "llm_infer"}


def test_seed_sources_is_idempotent(store):
    seed_sources(store, _PROVENANCE_SQL)
    seed_sources(store, _PROVENANCE_SQL)  # second run — INSERT OR IGNORE
    ids = [s.source_id for s in store.list_sources()]
    # Still exactly 6, not 12.
    assert len(ids) == 6
    assert len(set(ids)) == 6


# ------------------------- seed_graph ---------------------------------------


def test_seed_graph_from_repo_json(graph):
    added_nodes, added_edges = seed_graph(graph, _BELIEFS_JSON)
    assert added_nodes >= 4  # at least the four scenarios need
    # Second call is a no-op.
    n2, _ = seed_graph(graph, _BELIEFS_JSON)
    assert n2 == 0


# ------------------------- seed_from_files (end-to-end) ---------------------


def test_seed_from_files_full_round_trip(store, graph):
    summary = seed_from_files(
        store,
        graph,
        provenance_sql=_PROVENANCE_SQL,
        beliefs_json=_BELIEFS_JSON,
    )
    assert summary["sources"] == 6
    assert summary["nodes_added"] >= 4
    assert graph.has_node("robot")
    # Newly seeded graph can be used with the API façade for a full scenario.
    from iagent.declarative.api import KnowledgeAPI

    api = KnowledgeAPI(graph, store)
    # sources are all queryable
    src_ids = {s.source_id for s in api.list_sources()}
    assert "lidar_front" in src_ids
