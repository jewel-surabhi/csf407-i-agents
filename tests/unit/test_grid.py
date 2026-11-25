"""Grid dashboard: world geometry, LiDAR beams, merge rule, write-back, API."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from iagent.api import grid_routes
from iagent.api.server import app
from iagent.grid.agent import run_grid_question
from iagent.grid.session import GridSession, plan_route, resolve_total
from iagent.grid.world import GridWorld, all_segments, scan, segment_id
from iagent.procedural.llm_client import RecordedLLMClient

HAPPY_SCRIPT = [
    {"tool": "get_belief", "arguments": {}},
    {"tool": "sense_lidar", "arguments": {}},
    {"tool": "resolve_total", "arguments": {}},
    {"tool": "finalize_answer", "arguments": {"answer": "Route found."}},
]


def test_grid_has_20_nodes_and_31_segments():
    segs = all_segments()
    assert len(segs) == 31
    assert len({s["id"] for s in segs}) == 31
    assert segment_id("B1", "A1") == "A1-B1"
    with pytest.raises(ValueError):
        segment_id("A1", "C1")


def test_scan_stops_at_first_box_and_confidence_falls_with_distance():
    world = GridWorld(robot="A1", target="E4", boxes={"C1-D1"})
    r = scan(world)
    # east: A1-B1 clear (100 cm), B1-C1 clear (200 cm), C1-D1 blocked at 250 cm, D1-E1 unseen
    assert r["A1-B1"].blocked is False and r["A1-B1"].distance_cm == 100
    assert r["B1-C1"].distance_cm == 200
    assert r["C1-D1"].blocked is True and r["C1-D1"].distance_cm == 250
    assert "D1-E1" not in r
    assert r["A1-B1"].confidence > r["B1-C1"].confidence > r["C1-D1"].confidence
    # south: whole column clear, three segments
    assert [r[s].blocked for s in ("A1-A2", "A2-A3", "A3-A4")] == [False, False, False]
    # north and west are off-grid from A1
    assert len(r) == 6


def test_box_next_to_robot_reads_full_confidence():
    r = scan(GridWorld(robot="C2", target="E4", boxes={"C2-D2"}))
    assert r["C2-D2"].distance_cm == 50
    assert r["C2-D2"].confidence == pytest.approx(0.95)


def test_first_setup_declares_every_segment_later_setups_do_not():
    s = GridSession()
    assert s.complete_setup(GridWorld(robot="A1", target="E4", boxes={"B2-C2"})) is True
    d = s.declarations(0)
    assert len(d) == 31
    assert d["B2-C2"]["blocked"] is True and d["B2-C2"]["confidence"] == pytest.approx(0.9)
    assert d["A1-B1"]["blocked"] is False and d["A1-B1"]["source"] == "operator_setup"

    assert s.complete_setup(GridWorld(robot="A1", target="E4", boxes={"A1-B1"})) is False
    d = s.declarations(0)
    assert d["A1-B1"]["blocked"] is False  # the new box is real but not declared
    assert d["B2-C2"]["blocked"] is True


def test_total_prefers_higher_confidence_and_flags_conflicts_only():
    s = GridSession()
    s.complete_setup(GridWorld(robot="A1", target="E4", boxes=set()))
    s.complete_setup(GridWorld(robot="A1", target="E4", boxes={"A1-B1", "D1-E1"}))
    readings = s.sense()
    total = resolve_total(s.declarations(1), readings)
    # box 50 cm away: LiDAR 0.95 beats the operator's clear 0.70 (0.9 - 0.20 at hour 1) -> conflict
    assert total["A1-B1"]["winner"] == "lidar" and total["A1-B1"]["write"] == "conflict"
    # south column agrees with the declaration and LiDAR is stronger -> no write
    assert total["A1-A2"]["winner"] == "lidar" and total["A1-A2"]["write"] == "refresh"
    # D1-E1 is behind the box: unseen, declaration stands
    assert total["D1-E1"]["winner"] == "declaration" and total["D1-E1"]["blocked"] is False


def test_weaker_lidar_reading_loses_to_stronger_declaration():
    from iagent.grid.world import LidarReading

    declared = {"A1-B1": {"segment": "A1-B1", "blocked": False, "confidence": 0.9}}
    readings = {"A1-B1": LidarReading("A1-B1", True, 0.56, 350, "east")}
    total = resolve_total(declared, readings)
    assert total["A1-B1"]["winner"] == "declaration"
    assert total["A1-B1"]["lidar_disagrees"] is True
    assert total["A1-B1"]["write"] is None


def test_decay_follows_the_source_table():
    s = GridSession()
    s.complete_setup(GridWorld(robot="A1", target="E4", boxes=set()))
    s.complete_setup(GridWorld(robot="A1", target="E4", boxes={"A1-B1"}))
    run_grid_question(s, RecordedLLMClient(HAPPY_SCRIPT), "q", 1)
    user = s._sources["operator_setup"].decay_rate_per_hour
    assert user == pytest.approx(0.20)
    lidar = s._sources["lidar_declared"].decay_rate_per_hour
    assert lidar == pytest.approx(0.15)
    d5 = s.declarations(5)
    d2 = s.declarations(2)
    # C3-C4 is outside the robot's row and column, so it keeps the operator declaration
    assert d5["C3-C4"]["source"] == "operator_setup"
    assert d2["C3-C4"]["confidence"] == pytest.approx(0.9 - 0.20 * 2)  # 0.50
    assert s.declarations(0)["C3-C4"]["confidence"] == pytest.approx(0.9)
    assert d5["C3-C4"]["confidence"] == pytest.approx(0.0)  # fully faded after 4.5 hours
    # A1-A2 was in the scan and beat it, so it now fades at the LiDAR rate
    assert d5["A1-A2"]["source"] == "lidar_declared"
    assert d5["A1-B1"]["confidence"] == pytest.approx(max(0.0, 0.95 - lidar * 4))


def test_timeline_has_setup_then_questions():
    s = GridSession()
    s.complete_setup(GridWorld(robot="A1", target="E4", boxes={"B1-C1"}))
    run_grid_question(s, RecordedLLMClient(HAPPY_SCRIPT), "first", 2)
    s.complete_setup(GridWorld(robot="A2", target="E4", boxes=set()))
    run_grid_question(s, RecordedLLMClient(HAPPY_SCRIPT), "second", 3)
    t = s.timeline()
    assert [(i["kind"], i["hour"]) for i in t] == [("setup", 0), ("question", 2), ("question", 5)]
    assert t[0]["declarations"]["B1-C1"]["blocked"] is True
    assert t[2]["world"]["robot"] == "A2" and t[2]["question"] == "second"


def test_route_avoids_blocked_segments():
    s = GridSession()
    s.complete_setup(GridWorld(robot="A1", target="C1", boxes={"A1-B1"}))
    total = resolve_total(s.declarations(0), s.sense())
    route = plan_route(total, "A1", "C1")
    assert route["found"] and "A1-B1" not in route["segments"]
    assert route["nodes"][0] == "A1" and route["nodes"][-1] == "C1"


def test_episode_writes_back_every_winning_reading_and_lidar_beliefs_decay():
    s = GridSession()
    s.complete_setup(GridWorld(robot="A1", target="E4", boxes=set()))
    s.complete_setup(GridWorld(robot="A1", target="E4", boxes={"A1-B1"}))
    rec = run_grid_question(s, RecordedLLMClient(HAPPY_SCRIPT), "Is the way east clear?", 2)

    assert rec["hour"] == 2 and s.clock_h == 2
    assert rec["answer"] == "Route found." and rec["auto_steps"] == []
    # the declaration snapshot is from BEFORE the scan
    assert rec["declarations"]["A1-B1"]["blocked"] is False
    written = {w["segment"]: w["kind"] for w in rec["writes"]}
    assert written["A1-B1"] == "conflict"
    # agreeing but stronger LiDAR readings are written too: at hour 2 the operator's clear has
    # faded to 0.50, so the whole south column (0.89, 0.76, 0.63) beats it
    assert written["A1-A2"] == "refresh" and written["A2-A3"] == "refresh" and written["A3-A4"] == "refresh"
    assert set(written) == set(rec["lidar"])

    after = s.declarations(2)["A1-B1"]
    assert after["blocked"] is True and after["source"] == "lidar_declared"
    assert after["confidence"] == pytest.approx(0.95)
    # declared LiDAR beliefs decay 0.15 per hour
    assert s.declarations(3)["A1-B1"]["confidence"] == pytest.approx(0.80)
    assert s.declarations(2 + 7)["A1-B1"]["confidence"] == pytest.approx(0.0)


def test_safety_net_scans_when_model_skips_tools():
    s = GridSession()
    s.complete_setup(GridWorld(robot="A1", target="E4", boxes=set()))
    rec = run_grid_question(
        s, RecordedLLMClient([{"tool": "finalize_answer", "arguments": {"answer": "ok"}}]), "?", 1
    )
    assert rec["auto_steps"] == ["sense_lidar", "resolve_total"]
    assert rec["lidar"]


def test_previous_scan_is_kept():
    s = GridSession()
    s.complete_setup(GridWorld(robot="A1", target="E4", boxes=set()))
    run_grid_question(s, RecordedLLMClient(HAPPY_SCRIPT), "q1", 1)
    rec = run_grid_question(s, RecordedLLMClient(HAPPY_SCRIPT), "q2", 1)
    assert rec["previous_lidar"]["hour"] == 1 and rec["hour"] == 2


@pytest.fixture
def client():
    grid_routes.reset_grid_session()
    app.dependency_overrides[grid_routes.get_grid_llm] = lambda: RecordedLLMClient(HAPPY_SCRIPT)
    yield TestClient(app)
    app.dependency_overrides.pop(grid_routes.get_grid_llm, None)
    grid_routes.reset_grid_session()


def test_api_flow(client):
    assert client.post("/api/grid/ask", json={"question": "x"}).status_code == 409
    r = client.post("/api/grid/setup", json={"robot": "A1", "target": "E4", "boxes": ["B1-C1"]})
    assert r.status_code == 200 and r.json()["declared_now"] is True
    r = client.post("/api/grid/setup", json={"robot": "A2", "target": "E4", "boxes": []})
    assert r.json()["declared_now"] is False
    r = client.post("/api/grid/ask", json={"question": "Can I reach the target?", "hours": 3})
    body = r.json()
    assert r.status_code == 200 and body["clock_h"] == 3 and body["last"]["answer"] == "Route found."
    assert client.post("/api/grid/setup", json={"robot": "Z9", "target": "E4"}).status_code == 422
    assert [i["kind"] for i in body["timeline"]] == ["setup", "question"]
    prev = client.get("/api/grid/declarations", params={"hour": 5}).json()
    assert prev["hour"] == 5 and len(prev["declarations"]) == 31
    assert client.get("/api/grid/declarations", params={"hour": 1}).status_code == 422
    assert client.post("/api/grid/reset").json()["clock_h"] == 0


def test_backend_serves_the_page(client):
    r = client.get("/grid")
    assert r.status_code == 200 and "Scenario A" in r.text
