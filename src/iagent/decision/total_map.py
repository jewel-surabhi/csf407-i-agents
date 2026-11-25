"""Total map: the procedural layer's decision, and its write-back.

The Total map is exactly what the LLM submitted through ``submit_total_map``.
Code does not re-decide any path. Paths the LLM left out are "unknown".

Write-back stores the LLM's choices in the declarative layer when it picked the
sensor and the sensor tells the declaration layer something it did not
already hold at least as confidently:

* state   -> ``is_blocked`` from ``lidar_declared``  (new / conflict / refresh)
* colour  -> ``box_color`` from ``camera_declared`` (new / conflict / refresh)
* a path the LLM decided is clear from the sensor loses its declared colour.

The stored values are the sensor's own readings, not numbers retyped by the model.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

from iagent.declarative.api import KnowledgeAPI
from iagent.declarative.models import Belief
from iagent.grid.sources import CAMERA_DECLARED, LIDAR_DECLARED
from iagent.grid.world import GridWorld, all_segments, segment_ids
from iagent.layer_readers.declaration_map import COLOR, STATE, path_history, read_declaration_map
from iagent.layer_readers.sensor_map import read_sensor_map
from iagent.procedural.grid_task import GridReaders, run_grid_task

__all__ = ["Decision", "decide", "write_back", "plan_route"]


@dataclass
class Decision:
    hour: int
    declarations: dict[str, dict[str, Any]]
    sensor: dict[str, dict[str, Any]]
    total: dict[str, dict[str, Any]]
    answer: str | None
    route: dict[str, Any]
    steps: list[dict[str, Any]] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    submitted: bool = False


def plan_route(total: dict[str, dict[str, Any]], start: str, goal: str) -> dict[str, Any]:
    """Cheapest path over paths the Total map does not call blocked.

    Cost = 1 + (1 - confidence) for a clear path, 2 for an unknown one; blocked
    paths are never used.
    """
    adj: dict[str, list[tuple[str, str, float]]] = {}
    for seg in all_segments():
        cell = total.get(seg["id"])
        if cell and cell["state"] == "blocked":
            continue
        cost = 2.0 if not cell else 1.0 + (1.0 - float(cell["state_conf"]))
        adj.setdefault(seg["a"], []).append((seg["b"], seg["id"], cost))
        adj.setdefault(seg["b"], []).append((seg["a"], seg["id"], cost))

    best, prev, queue = {start: 0.0}, {}, [(0.0, start)]
    while queue:
        cost, node = heapq.heappop(queue)
        if node == goal:
            break
        if cost > best.get(node, float("inf")):
            continue
        for nxt, sid, c in adj.get(node, []):
            if cost + c < best.get(nxt, float("inf")):
                best[nxt], prev[nxt] = cost + c, (node, sid)
                heapq.heappush(queue, (cost + c, nxt))
    if goal not in best:
        return {"found": False, "nodes": [], "paths": [], "cost": None}
    nodes, paths, cur = [goal], [], goal
    while cur != start:
        cur, sid = prev[cur]
        nodes.append(cur)
        paths.append(sid)
    return {"found": True, "nodes": nodes[::-1], "paths": paths[::-1], "cost": round(best[goal], 3)}


def decide(
    *,
    knowledge: KnowledgeAPI,
    world: GridWorld,
    llm: Any,
    question: str,
    hour: int,
    clock_h: int,
    epoch: datetime,
) -> Decision:
    """Read both layers, let the procedural layer decide, return the Total map."""
    now = epoch + timedelta(hours=hour)
    declarations = read_declaration_map(knowledge, hour=hour, epoch=epoch)  # snapshot before any write
    sensor = read_sensor_map(world, now=now)

    readers = GridReaders(
        read_declarations=lambda: declarations,
        read_lidar=lambda: sensor,
        read_camera=lambda: {k: v for k, v in sensor.items() if v["color"]},
        history=lambda path: path_history(knowledge, path, epoch=epoch),
        plan_route=lambda total: plan_route(total, world.robot, world.target),
    )
    context = (
        f"Question time: hour {hour} (clock was hour {clock_h}). "
        f"Robot at {world.robot}, target at {world.target}."
    )
    ep = run_grid_task(llm, readers, question=question, context=context, known_paths=set(segment_ids()))
    total = ep.total or {}
    route = ep.route or plan_route(total, world.robot, world.target)
    return Decision(
        hour=hour,
        declarations=declarations,
        sensor=sensor,
        total=total,
        answer=ep.answer,
        route=route,
        steps=ep.steps,
        issues=ep.issues,
        missing=ep.missing if ep.submissions else sorted(segment_ids()),
        submitted=ep.submissions > 0,
    )


def write_back(knowledge: KnowledgeAPI, decision: Decision, *, epoch: datetime) -> list[dict[str, Any]]:
    """Store the LLM's sensor-backed winners in the declarative layer."""
    when = epoch + timedelta(hours=decision.hour)
    writes: list[dict[str, Any]] = []
    for pid, t in sorted(decision.total.items()):
        d, s = decision.declarations.get(pid), decision.sensor.get(pid)
        if s is None:
            continue  # the sensors never saw it: nothing to store, whatever the model said
        if t["state_from"] == "sensor":
            kind = _kind(d and d["state"], d and d["state_conf"], s["state"], s["state_conf"])
            if kind:
                knowledge.update_belief(
                    Belief(
                        subject=pid,
                        predicate=STATE,
                        object=str(s["state"] == "blocked").lower(),
                        confidence=s["state_conf"],
                        source=LIDAR_DECLARED,
                        observed_at=when,
                    ),
                    reason=f"LiDAR at hour {decision.hour}: {kind}",
                )
                writes.append({"path": pid, "what": "state", "kind": kind, "value": s["state"], "conf": s["state_conf"]})
            if t["state"] == "clear" and d and d["color"]:
                knowledge.downgrade_belief(pid, COLOR, 0.0, reason=f"path seen clear at hour {decision.hour}")
                writes.append({"path": pid, "what": "color", "kind": "removed", "value": None, "conf": None})
        if t["state"] == "blocked" and t.get("color_from") == "sensor" and s.get("color"):
            kind = _kind(d and d["color"], d and d["color_conf"], s["color"], s["color_conf"])
            if kind:
                knowledge.update_belief(
                    Belief(
                        subject=pid,
                        predicate=COLOR,
                        object=s["color"],
                        confidence=s["color_conf"],
                        source=CAMERA_DECLARED,
                        observed_at=when,
                    ),
                    reason=f"camera at hour {decision.hour}: {kind}",
                )
                writes.append({"path": pid, "what": "color", "kind": kind, "value": s["color"], "conf": s["color_conf"]})
    return writes


def _kind(old_value: Any, old_conf: Any, new_value: Any, new_conf: float) -> str | None:
    if not old_value:
        return "new"
    if old_value != new_value:
        return "conflict"
    if new_conf > (old_conf or 0.0):
        return "refresh"
    return None
