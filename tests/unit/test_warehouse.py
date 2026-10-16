"""Tests for the warehouse demo layer — move scoring + inventory reconciliation.

Uses the real KnowledgeAPI seeded from configs/warehouse.yaml. Confidence and
reconciliation are asserted against the belief state, proving they're grounded.

Owner: Person 4.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from iagent.declarative.api import KnowledgeAPI
from iagent.declarative.belief_graph import BeliefGraph
from iagent.declarative.provenance import ProvenanceStore
from iagent.declarative.seeding import seed_sources
from iagent.warehouse.service import (
    evaluate_moves,
    reconcile_inventory,
    seed_warehouse,
    transfer,
)
from iagent.warehouse.topology import load_topology, segment_id

_REPO = Path(__file__).resolve().parents[2]
_SEED = _REPO / "data" / "seed" / "initial_provenance.sql"
_WAREHOUSE = _REPO / "configs" / "warehouse.yaml"


@pytest.fixture()
def topo():
    return load_topology(_WAREHOUSE)


@pytest.fixture()
def knowledge(tmp_path, topo):
    store = ProvenanceStore(str(tmp_path / "wh.db"))
    seed_sources(store, _SEED)
    kn = KnowledgeAPI(BeliefGraph(), store)
    seed_warehouse(kn, topo)
    yield kn
    store.close()


# ───────────── topology ─────────────


def test_topology_loads(topo):
    assert topo.id == "warehouse_7"
    assert topo.start == "A-S"
    assert topo.goal == "WIDGET-42"
    assert len(topo.junctions) == 9
    assert len(topo.segments) == 12


def test_segment_id_is_order_independent():
    assert segment_id("A-S", "B-S") == segment_id("B-S", "A-S")


# ───────────── navigation / move scoring ─────────────


def test_moves_from_start_are_scored_and_sorted(knowledge, topo):
    opts = evaluate_moves(knowledge, topo, "A-S")
    assert len(opts) == 2  # A-S connects to B-S and A-M
    # sorted best-first
    assert opts[0].confidence >= opts[1].confidence
    tos = {o.to for o in opts}
    assert tos == {"B-S", "A-M"}


def test_blocked_segment_scores_low_from_sensor_conflict(knowledge, topo):
    """The mid-aisle B-S<->B-M is seeded blocked by LiDAR at 0.95 -> passability ~0.05."""
    opts = evaluate_moves(knowledge, topo, "B-S")
    bm = next(o for o in opts if o.to == "B-M")
    assert bm.passable is False
    assert bm.confidence <= 0.15
    assert bm.tier == "low"
    assert "obstruction" in bm.reason.lower()
    assert bm.source == "lidar_front"


def test_sensor_confirmed_segment_scores_high(knowledge, topo):
    opts = evaluate_moves(knowledge, topo, "A-S")
    bs = next(o for o in opts if o.to == "B-S")  # scanner-confirmed 0.93
    assert bs.passable is True
    assert bs.confidence >= 0.8
    assert bs.tier == "high"
    assert "confirmed" in bs.reason.lower()


def test_map_only_segment_scores_medium_and_unverified(knowledge, topo):
    """A-M<->A-N is map-only at 0.58 -> medium, flagged unverified."""
    opts = evaluate_moves(knowledge, topo, "A-M")
    an = next(o for o in opts if o.to == "A-N")
    assert an.tier == "medium"
    assert an.source == "default_map"
    assert "verif" in an.reason.lower()


def test_unknown_junction_yields_no_moves(knowledge, topo):
    assert evaluate_moves(knowledge, topo, "NOWHERE") == []


def test_high_confidence_route_to_goal_exists(knowledge, topo):
    """The honest route A-S -> B-S -> C-S -> C-M -> C-N must be all high-confidence."""
    route = ["A-S", "B-S", "C-S", "C-M", "C-N"]
    for a, b in zip(route[:-1], route[1:], strict=True):
        opts = evaluate_moves(knowledge, topo, a)
        leg = next(o for o in opts if o.to == b)
        assert leg.confidence >= 0.8, f"{a}->{b} should be high confidence, got {leg.confidence}"


# ───────────── inventory reconciliation ─────────────


def test_reconcile_widget42_nets_shipment(knowledge, topo):
    """ERP 50 (0.6) vs scanner 48 (0.9), minus 3 shipped -> 45, trust scanner."""
    view = reconcile_inventory(knowledge, "WIDGET-42")
    assert view.reconciled == 45
    assert view.winning_source == "warehouse_scanner"
    # all three source rows present
    preds = {(r.source, r.predicate) for r in view.sources}
    assert ("erp_system", "count") in preds
    assert ("warehouse_scanner", "count") in preds
    assert ("sales_log", "shipped") in preds


def test_reconcile_single_source_shelf(knowledge, topo):
    view = reconcile_inventory(knowledge, "BOLT-08")
    assert view.reconciled == 118
    assert view.winning_source == "warehouse_scanner"


# ───────────── transfers ─────────────


def test_transfer_decrements_source(knowledge, topo):
    before = reconcile_inventory(knowledge, "WIDGET-42").reconciled  # 45
    res = transfer(knowledge, topo, "WIDGET-42", "PANEL-07", 10)
    assert res.ok is True
    assert res.source_after == before - 10  # 35
    assert res.dest_after == 10
    # re-reconcile reflects the change
    assert reconcile_inventory(knowledge, "WIDGET-42").reconciled == before - 10


def test_transfer_rejects_more_than_available(knowledge, topo):
    before = reconcile_inventory(knowledge, "WIDGET-42").reconciled
    res = transfer(knowledge, topo, "WIDGET-42", "PANEL-07", 9999)
    assert res.ok is False
    assert "available" in res.message.lower()
    # inventory unchanged
    assert reconcile_inventory(knowledge, "WIDGET-42").reconciled == before


def test_transfer_rejects_unknown_destination(knowledge, topo):
    res = transfer(knowledge, topo, "WIDGET-42", "NOPE", 1)
    assert res.ok is False
    assert "destination" in res.message.lower()


def test_transfer_rejects_zero_quantity(knowledge, topo):
    res = transfer(knowledge, topo, "WIDGET-42", "PANEL-07", 0)
    assert res.ok is False


def test_transfer_leaves_provenance(knowledge, topo):
    transfer(knowledge, topo, "WIDGET-42", "PANEL-07", 5)
    hist = knowledge.query_provenance("WIDGET-42", predicate="count")
    # at least the seed + the transfer-out row
    assert any("transfer out" in (p.payload or {}).get("reason", "") for p in hist)
