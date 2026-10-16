"""Tests for the warehouse HTTP routes. No network — belief-graph backed.

Owner: Person 4.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from iagent.api import warehouse_routes
from iagent.api.server import app


@pytest.fixture()
def client(tmp_path, monkeypatch):
    # Fresh warehouse DB per test; rebuild the singleton against it.
    monkeypatch.setenv("IAGENT_WAREHOUSE_DB", str(tmp_path / "wh.db"))
    warehouse_routes.reset_warehouse()
    c = TestClient(app)
    yield c
    warehouse_routes.reset_warehouse()


def test_get_view_returns_topology_and_scored_options(client):
    r = client.get("/api/warehouse")
    assert r.status_code == 200
    body = r.json()
    assert body["position"] == "A-S"
    assert body["goal"] == "WIDGET-42"
    assert len(body["junctions"]) == 9
    assert len(body["segments"]) == 12
    assert len(body["options"]) == 2  # from A-S
    assert all("confidence" in o for o in body["options"])


def test_options_endpoint(client):
    r = client.get("/api/warehouse/options")
    assert r.status_code == 200
    opts = r.json()
    assert opts[0]["confidence"] >= opts[-1]["confidence"]  # sorted best-first


def test_move_updates_position(client):
    r = client.post("/api/warehouse/move", json={"to": "B-S"})
    assert r.status_code == 200
    assert r.json()["position"] == "B-S"


def test_move_rejects_non_adjacent(client):
    r = client.post("/api/warehouse/move", json={"to": "C-N"})
    assert r.status_code == 400
    assert "not adjacent" in r.json()["detail"].lower()


def test_blocked_segment_is_low_confidence_over_http(client):
    client.post("/api/warehouse/move", json={"to": "B-S"})
    opts = client.get("/api/warehouse/options").json()
    bm = next(o for o in opts if o["to"] == "B-M")
    assert bm["passable"] is False
    assert bm["confidence"] <= 0.15


def test_inventory_reconciliation_over_http(client):
    r = client.get("/api/warehouse/inventory/WIDGET-42")
    assert r.status_code == 200
    body = r.json()
    assert body["reconciled"] == 45
    assert body["winning_source"] == "warehouse_scanner"


def test_transfer_over_http(client):
    before = client.get("/api/warehouse/inventory/WIDGET-42").json()["reconciled"]
    r = client.post("/api/warehouse/transfer",
                    json={"sku": "WIDGET-42", "dest": "PANEL-07", "quantity": 10})
    assert r.status_code == 200
    assert r.json()["ok"] is True
    after = client.get("/api/warehouse/inventory/WIDGET-42").json()["reconciled"]
    assert after == before - 10


def test_transfer_insufficient_stock_over_http(client):
    before = client.get("/api/warehouse/inventory/WIDGET-42").json()["reconciled"]
    r = client.post("/api/warehouse/transfer",
                    json={"sku": "WIDGET-42", "dest": "PANEL-07", "quantity": 9999})
    assert r.json()["ok"] is False
    after = client.get("/api/warehouse/inventory/WIDGET-42").json()["reconciled"]
    assert after == before  # unchanged
