"""Grid dashboard: world, layer readers, procedural Total map, write-back, API."""

from __future__ import annotations

from datetime import timedelta

import pytest
from fastapi.testclient import TestClient

from iagent.api import grid_routes
from iagent.api.server import app
from iagent.decision.total_map import plan_route
from iagent.grid.agent import run_grid_question
from iagent.grid.session import GridSession
from iagent.grid.world import GridWorld, all_segments, ray, segment_id
from iagent.layer_readers.sensor_map import grid_to_world_state, read_sensor_map
from iagent.procedural.llm_client import RecordedLLMClient
from iagent.sensorimotor.sensors import camera_confidence, lidar_confidence

# ------------------------------------------------------------------ helpers


def ideal_total(decl: dict, sensor: dict) -> list[dict]:
    """The Total map a model following the prompt's rule would submit."""
    out = []
    for seg in all_segments():
        pid = seg["id"]
        d, s = decl.get(pid), sensor.get(pid)
        if d is None and s is None:
            continue
        if s and (d is None or s["state_conf"] >= d["state_conf"]):
            state, conf, frm = s["state"], s["state_conf"], "sensor"
        else:
            state, conf, frm = d["state"], d["state_conf"], "declaration"
        entry = {"path": pid, "state": state, "state_conf": conf, "state_from": frm}
        if state == "blocked":
            dc = (d["color"], d["color_conf"]) if d and d["color"] else None
            sc = (s["color"], s["color_conf"]) if s and s["color"] else None
            if sc and (dc is None or sc[1] >= dc[1]):
                entry.update(color=sc[0], color_conf=sc[1], color_from="sensor")
            elif dc:
                entry.update(color=dc[0], color_conf=dc[1], color_from="declaration")
        out.append(entry)
    return out


def script_for(session: GridSession, hours: int, answer: str = "Done.", total: list | None = None) -> RecordedLLMClient:
    hour = session.clock_h + hours
    decl = session.declarations(hour)
    sensor = read_sensor_map(session.world, now=session.at(hour))
    paths = total if total is not None else ideal_total(decl, sensor)
    return RecordedLLMClient(
        [
            {"tool": "get_belief", "arguments": {}},
            [{"tool": "sense_lidar", "arguments": {}}, {"tool": "sense_camera", "arguments": {}}],
            {"tool": "submit_total_map", "arguments": {"paths": paths}},
            {"tool": "plan_route", "arguments": {}},
            {"tool": "finalize_answer", "arguments": {"answer": answer}},
        ]
    )


# ------------------------------------------------------------------ world


def test_grid_has_31_paths_and_rays():
    assert len(all_segments()) == 31
    assert segment_id("B1", "A1") == "A1-B1"
    assert ray("A1", "east") == ["A1-B1", "B1-C1", "C1-D1", "D1-E1"]
    assert ray("A1", "north") == []
    with pytest.raises(ValueError):
        GridWorld(robot="A1", target="E4", boxes={"A1-B1": "green"})


# ------------------------------------------------------------------ sensor map (sensorimotor layer)


def test_grid_converts_to_world_state_for_the_sensorimotor_layer():
    ws = grid_to_world_state(GridWorld(robot="B2", target="E4", boxes={"D2-E2": "red", "C4-D4": "blue"}))
    assert ws.extras["beam_range_cm"] == {"front": 100, "right": 300, "back": 200, "left": 100}
    (ob,) = ws.obstacles  # C4-D4 is not on the robot's row or column
    assert ob["in_right_of"] == "robot" and ob["distance_cm"] == 250 and ob["color"] == "red"


def test_sensor_map_reads_row_and_column_and_stops_at_boxes():
    world = GridWorld(robot="A1", target="E4", boxes={"C1-D1": "yellow", "A1-A2": "blue"})
    m = read_sensor_map(world, now=GridSession().at(1))
    assert m["A1-B1"]["state"] == "clear" and m["A1-B1"]["state_conf"] == lidar_confidence(100)
    assert m["C1-D1"]["state"] == "blocked" and m["C1-D1"]["distance_cm"] == 250
    assert m["C1-D1"]["color"] == "yellow" and m["C1-D1"]["color_conf"] == camera_confidence(250)
    assert m["A1-A2"]["state_conf"] == 0.95 and m["A1-A2"]["color_conf"] == 0.98
    assert "D1-E1" not in m and "A2-A3" not in m  # behind boxes
    assert m["A1-B1"]["color"] is None


# ------------------------------------------------------------------ declaration map (declarative layer)


def test_first_setup_declares_state_and_colour_later_setups_do_not():
    s = GridSession()
    assert s.complete_setup(GridWorld(robot="A1", target="E4", boxes={"B2-C2": "red"})) is True
    d = s.declarations(0)
    assert len(d) == 31
    assert d["B2-C2"]["state"] == "blocked" and d["B2-C2"]["state_conf"] == pytest.approx(0.9)
    assert d["B2-C2"]["color"] == "red" and d["B2-C2"]["color_conf"] == pytest.approx(0.9)
    assert d["A1-B1"]["state"] == "clear" and d["A1-B1"]["color"] is None
    assert d["A1-B1"]["state_meta"]["source"] == "operator_setup"

    assert s.complete_setup(GridWorld(robot="A1", target="E4", boxes={"A1-B1": "blue"})) is False
    assert s.declarations(0)["A1-B1"]["state"] == "clear"  # real, not declared


def test_declarations_decay_by_source():
    s = GridSession()
    s.complete_setup(GridWorld(robot="A1", target="E4", boxes={"B2-C2": "red"}))
    d2 = s.declarations(2)
    assert d2["B2-C2"]["state_conf"] == pytest.approx(0.5) and d2["B2-C2"]["color_conf"] == pytest.approx(0.5)
    assert s.declarations(5)["B2-C2"]["state_conf"] == 0.0


# ------------------------------------------------------------------ procedural Total map + write-back


def test_total_map_is_what_the_llm_submitted_and_winners_are_written_back():
    s = GridSession()
    s.complete_setup(GridWorld(robot="A1", target="E4", boxes={"A1-B1": "red"}))
    s.complete_setup(GridWorld(robot="A1", target="E4", boxes={"A1-B1": "blue", "A2-A3": "yellow"}))
    rec = run_grid_question(s, script_for(s, 1, "Blue box on A1-B1."), "What is around me?", 1)

    assert rec["hour"] == 1 and s.clock_h == 1 and rec["answer"] == "Blue box on A1-B1."
    assert rec["submitted"] and rec["missing"] == [] and rec["issues"] == []
    # declaration snapshot before the scan: red at 0.70
    assert rec["declarations"]["A1-B1"]["color"] == "red"
    # camera 0.98 beats declared red 0.70 -> colour conflict written
    t = rec["total"]["A1-B1"]
    assert t["color"] == "blue" and t["color_from"] == "sensor"
    writes = {(w["path"], w["what"]): w["kind"] for w in rec["writes"]}
    assert writes[("A1-B1", "color")] == "conflict"
    assert writes[("A1-B1", "state")] == "refresh"  # blocked 0.95 beats declared blocked 0.70
    assert writes[("A2-A3", "state")] == "conflict" and writes[("A2-A3", "color")] == "new"
    after = s.declarations(1)
    assert after["A1-B1"]["color"] == "blue" and after["A1-B1"]["color_meta"]["source"] == "camera_declared"
    assert after["A2-A3"]["state"] == "blocked" and after["A2-A3"]["state_meta"]["source"] == "lidar_declared"
    assert s.declarations(2)["A1-B1"]["state_conf"] == pytest.approx(0.95 - 0.15)


def test_path_seen_clear_loses_its_declared_colour():
    s = GridSession()
    s.complete_setup(GridWorld(robot="A1", target="E4", boxes={"A1-B1": "red"}))
    s.complete_setup(GridWorld(robot="A1", target="E4", boxes={}))
    rec = run_grid_question(s, script_for(s, 1), "q", 1)
    assert rec["total"]["A1-B1"]["state"] == "clear"
    assert ("A1-B1", "color", "removed") in {(w["path"], w["what"], w["kind"]) for w in rec["writes"]}
    assert s.declarations(1)["A1-B1"]["color"] is None


def test_llm_decision_is_not_overridden_and_gaps_are_unknown():
    s = GridSession()
    s.complete_setup(GridWorld(robot="A1", target="E4", boxes={}))
    s.complete_setup(GridWorld(robot="A1", target="E4", boxes={"A1-B1": "red"}))
    # The model wrongly keeps the declaration for A1-B1 and submits only two paths.
    wrong = [
        {"path": "A1-B1", "state": "clear", "state_conf": 0.7, "state_from": "declaration"},
        {"path": "b1-a1", "state": "clear", "state_conf": 0.7, "state_from": "declaration"},
        {"path": "Z9-Z8", "state": "clear", "state_conf": 0.7, "state_from": "declaration"},
    ]
    rec = run_grid_question(s, script_for(s, 1, total=wrong), "q", 1)
    assert rec["total"]["A1-B1"]["state"] == "clear"  # the LLM's call stands
    assert len(rec["missing"]) == 30 and any("Z9-Z8" in i for i in rec["issues"])
    assert rec["writes"] == []  # it never picked the sensor, so nothing is written
    assert s.declarations(1)["A1-B1"]["state"] == "clear"


def test_no_submission_means_every_path_is_unknown():
    s = GridSession()
    s.complete_setup(GridWorld(robot="A1", target="E4", boxes={}))
    rec = run_grid_question(s, RecordedLLMClient([{"tool": "finalize_answer", "arguments": {"answer": "?"}}]), "q", 1)
    assert rec["total"] == {} and not rec["submitted"] and len(rec["missing"]) == 31
    assert rec["sensor"]  # the sensor map is still shown


def test_route_avoids_blocked_paths():
    total = {"A1-B1": {"state": "blocked", "state_conf": 0.9}}
    r = plan_route(total, "A1", "C1")
    assert r["found"] and "A1-B1" not in r["paths"] and r["nodes"][0] == "A1"


def test_timeline():
    s = GridSession()
    s.complete_setup(GridWorld(robot="A1", target="E4", boxes={"B1-C1": "red"}))
    run_grid_question(s, script_for(s, 2), "first", 2)
    s.complete_setup(GridWorld(robot="A2", target="E4", boxes={}))
    run_grid_question(s, script_for(s, 3), "second", 3)
    t = s.timeline()
    assert [(i["kind"], i["hour"]) for i in t] == [("setup", 0), ("question", 2), ("question", 5)]
    assert t[2]["world"]["robot"] == "A2"


# ------------------------------------------------------------------ API


@pytest.fixture
def client():
    grid_routes.reset_grid_session()
    yield TestClient(app)
    app.dependency_overrides.pop(grid_routes.get_grid_llm, None)
    grid_routes.reset_grid_session()


def test_api_flow(client):
    app.dependency_overrides[grid_routes.get_grid_llm] = lambda: RecordedLLMClient([])
    assert client.post("/api/grid/ask", json={"question": "x"}).status_code == 409
    r = client.post("/api/grid/setup", json={"robot": "A1", "target": "E4", "boxes": {"B1-C1": "yellow"}})
    assert r.status_code == 200 and r.json()["declared_now"] is True
    assert r.json()["declarations"]["B1-C1"]["color"] == "yellow"
    session = grid_routes.get_grid_session()
    app.dependency_overrides[grid_routes.get_grid_llm] = lambda: script_for(session, 3, "ok")
    body = client.post("/api/grid/ask", json={"question": "Route?", "hours": 3}).json()
    assert body["clock_h"] == 3 and body["last"]["answer"] == "ok" and body["last"]["sensor"]
    assert [i["kind"] for i in body["timeline"]] == ["setup", "question"]
    assert client.get("/api/grid/declarations", params={"hour": 4}).status_code == 200
    assert client.get("/api/grid/declarations", params={"hour": 1}).status_code == 422
    bad = client.post("/api/grid/setup", json={"robot": "A1", "target": "E4", "boxes": {"A1-B1": "green"}})
    assert bad.status_code == 422
    assert client.post("/api/grid/reset").json()["clock_h"] == 0


def test_backend_serves_the_page(client):
    r = client.get("/grid")
    assert r.status_code == 200 and "Scenario A" in r.text


def test_clock_epoch_math():
    s = GridSession()
    assert s.at(3) - s.at(0) == timedelta(hours=3)
