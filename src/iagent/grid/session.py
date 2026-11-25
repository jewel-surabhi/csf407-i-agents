"""One dashboard session: the real world, a simulated clock and the declarative layer.

Lifecycle (matches the Scenario A dashboard):

1. ``complete_setup(world)`` the FIRST time declares the whole world into the
   declarative layer at hour 0: every segment gets an ``is_blocked`` belief
   from the operator (source ``operator_setup``: trust 0.9, decay 0.20 / hour),
   with that source's trust prior as its confidence.
2. ``ask`` (see ``grid.agent``) advances the clock by +X hours, lets the ReAct
   loop read the declarations and sense the LiDAR, and then
   ``commit_question`` writes the LiDAR findings that won a conflict back into
   the declarative layer as ``lidar_declared`` beliefs (0.15 / hour).
3. ``complete_setup(world)`` LATER (after "Change setup") only changes the
   real world. Nothing is declared; the next scan is what updates beliefs.

Time is simulated: hour ``h`` is ``epoch + h hours``. Every belief is written
with that ``observed_at`` and every read applies decay at the requested hour.
Decay is a property of the source (the kind of information), read from the
``sources`` table seeded by ``data/seed/initial_provenance.sql``:
``effective = max(0, stored - decay_rate_per_hour * age_hours)``.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from iagent.declarative.api import KnowledgeAPI
from iagent.declarative.belief_graph import BeliefGraph
from iagent.declarative.decay import annotate_with_decay
from iagent.declarative.models import Belief
from iagent.declarative.provenance import ProvenanceStore
from iagent.declarative.seeding import seed_sources
from iagent.grid.world import GridWorld, LidarReading, all_segments, scan

__all__ = ["GridSession", "PREDICATE", "DECLARE_SOURCE", "LIDAR_SOURCE", "resolve_total", "plan_route"]

PREDICATE = "is_blocked"
DECLARE_SOURCE = "operator_setup"  # the human operator's first declaration
# The operator's first declaration starts at 0.90 and fades 0.20 / hour.
OPERATOR_TRUST = 0.9
OPERATOR_DECAY_PER_HOUR = 0.20
LIDAR_SOURCE = "lidar_declared"  # a LiDAR reading once it is stored in the declaration layer
LIDAR_TRUST = 0.9  # same trust as the project's `lidar_front`
LIDAR_DECAY_PER_HOUR = 0.15  # a declared LiDAR reading fades more slowly than the raw sensor source

_REPO = Path(__file__).resolve().parents[3]
_SEED_SQL = _REPO / "data" / "seed" / "initial_provenance.sql"


def _label(blocked: bool) -> str:
    return "blocked" if blocked else "clear"


@dataclass
class GridSession:
    world: GridWorld = field(default_factory=GridWorld)
    clock_h: int = 0
    declared: bool = False
    epoch: datetime = field(default_factory=lambda: datetime(2026, 1, 1, tzinfo=UTC))
    last_scan: dict[str, LidarReading] | None = None
    last_scan_hour: int | None = None
    previous_scan: dict[str, LidarReading] | None = None
    previous_scan_hour: int | None = None
    history: list[dict[str, Any]] = field(default_factory=list)
    setup_record: dict[str, Any] | None = None
    world_changes: list[dict[str, Any]] = field(default_factory=list)
    db_path: str = ":memory:"

    def __post_init__(self) -> None:
        store = ProvenanceStore(self.db_path)
        seed_sources(store, _SEED_SQL)
        self.knowledge = KnowledgeAPI(BeliefGraph(), store)
        self.knowledge.log_source(
            DECLARE_SOURCE,
            "user",
            OPERATOR_TRUST,
            "Operator's first declaration (setup); fades slowly",
            decay_rate_per_hour=OPERATOR_DECAY_PER_HOUR,
        )
        self.knowledge.log_source(
            LIDAR_SOURCE,
            "sensor",
            LIDAR_TRUST,
            "LiDAR reading declared into the declaration layer",
            decay_rate_per_hour=LIDAR_DECAY_PER_HOUR,
        )
        self._sources = {s.source_id: s for s in self.knowledge.list_sources()}

    # ------------------------------------------------------------- time

    def at(self, hour: float) -> datetime:
        return self.epoch + timedelta(hours=hour)

    # ------------------------------------------------------------- setup

    def complete_setup(self, world: GridWorld) -> bool:
        """Install ``world`` as the real world. Returns True if it was declared."""
        world.validate()
        self.world = world
        if self.declared:
            self.world_changes.append({"hour": self.clock_h, "world": world.to_dict()})
            return False
        when = self.at(self.clock_h)
        for seg in all_segments():
            blocked = seg["id"] in world.boxes
            self.knowledge.update_belief(
                Belief(
                    subject=seg["id"],
                    predicate=PREDICATE,
                    object=str(blocked).lower(),
                    confidence=self._sources[DECLARE_SOURCE].trust_prior,
                    source=DECLARE_SOURCE,
                    observed_at=when,
                ),
                reason="operator setup declaration",
            )
        self.declared = True
        self.setup_record = {
            "kind": "setup",
            "hour": self.clock_h,
            "world": world.to_dict(),
            "robot": world.robot,
            "target": world.target,
            "declarations": self.declarations(self.clock_h),
        }
        return True

    def timeline(self) -> list[dict[str, Any]]:
        """Significant hours, oldest first: the setup declaration, then every answered question."""
        items = [self.setup_record] if self.setup_record else []
        return items + [{"kind": "question", **rec} for rec in self.history]

    # ------------------------------------------------------------- reads

    def declarations(self, hour: float) -> dict[str, dict[str, Any]]:
        """Every active ``is_blocked`` belief, decayed to ``hour``."""
        now = self.at(hour)
        out: dict[str, dict[str, Any]] = {}
        for seg in all_segments():
            belief = self.knowledge.get_belief(seg["id"], PREDICATE)
            if belief is None:
                continue
            b = annotate_with_decay(belief, self._sources, now=now)
            src = self._sources.get(b.source)
            out[seg["id"]] = {
                "segment": seg["id"],
                "blocked": b.object == "true",
                "stored": round(b.confidence, 3),
                "confidence": round(b.effective_confidence or 0.0, 3),
                "source": b.source,
                "trust_prior": src.trust_prior if src else None,
                "decay_per_hour": src.decay_rate_per_hour if src else 0.0,
                "declared_at_hour": round((b.observed_at - self.epoch).total_seconds() / 3600, 2),
                "age_hours": round(b.age_hours or 0.0, 2),
                "provenance_id": b.provenance_id,
            }
        return out

    def history_for(self, segment: str, limit: int = 10) -> list[dict[str, Any]]:
        rows = self.knowledge.query_provenance(segment, PREDICATE, limit=limit)
        return [
            {
                "blocked": r.object == "true",
                "confidence": r.confidence,
                "source": r.source_id,
                "status": r.status,
                "hour": round((r.observed_at - self.epoch).total_seconds() / 3600, 2),
                "reason": (r.payload or {}).get("reason"),
            }
            for r in rows
        ]

    def sense(self) -> dict[str, LidarReading]:
        return scan(self.world)

    # ------------------------------------------------------------- writes

    def commit_question(
        self,
        *,
        hour: int,
        readings: dict[str, LidarReading],
        total: dict[str, dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Write the Total map's LiDAR wins that changed the answer, advance the clock."""
        writes: list[dict[str, Any]] = []
        for seg_id, cell in total.items():
            if cell.get("write") is None:
                continue
            reading = readings[seg_id]
            self.knowledge.update_belief(
                Belief(
                    subject=seg_id,
                    predicate=PREDICATE,
                    object=str(reading.blocked).lower(),
                    confidence=reading.confidence,
                    source=LIDAR_SOURCE,
                    observed_at=self.at(hour),
                ),
                reason=f"LiDAR scan at hour {hour}: {cell['write']}",
            )
            writes.append({"segment": seg_id, "kind": cell["write"], **_short(cell)})
        self.previous_scan, self.previous_scan_hour = self.last_scan, self.last_scan_hour
        self.last_scan, self.last_scan_hour = readings, hour
        self.clock_h = hour
        return writes


def _short(cell: dict[str, Any]) -> dict[str, Any]:
    return {"blocked": cell["blocked"], "confidence": cell["confidence"]}


# ----------------------------------------------------------------- procedural code


def resolve_total(
    declared: dict[str, dict[str, Any]],
    readings: dict[str, LidarReading],
) -> dict[str, dict[str, Any]]:
    """Per segment, pick the higher confidence between declaration and LiDAR.

    * both present: higher confidence wins; a tie goes to the LiDAR (fresher)
    * only one present: that one is used
    * neither: the segment is unknown

    ``write`` is set whenever the LiDAR wins, because a winning reading is the
    best current evidence and replaces the weaker declaration:
    ``"conflict"`` (LiDAR disagrees with the declaration), ``"refresh"`` (it
    agrees but is more confident) or ``"new"`` (nothing was declared). A
    declaration that is at least as confident as the scan keeps its place and
    nothing is written.
    """
    total: dict[str, dict[str, Any]] = {}
    for seg in all_segments():
        sid = seg["id"]
        d, r = declared.get(sid), readings.get(sid)
        if d is None and r is None:
            total[sid] = {"segment": sid, "blocked": None, "confidence": 0.0, "winner": None, "write": None}
            continue
        if r is not None and (d is None or r.confidence >= d["confidence"]):
            if d is None:
                write = "new"
            elif d["blocked"] != r.blocked:
                write = "conflict"
            elif r.confidence > d["confidence"]:
                write = "refresh"
            else:
                write = None  # exact tie on the same label: nothing better to record
            total[sid] = {
                "segment": sid,
                "blocked": r.blocked,
                "confidence": r.confidence,
                "winner": "lidar",
                "write": write,
            }
        else:
            assert d is not None
            total[sid] = {
                "segment": sid,
                "blocked": d["blocked"],
                "confidence": d["confidence"],
                "winner": "declaration",
                "write": None,
                "lidar_disagrees": r is not None and r.blocked != d["blocked"],
            }
    return total


def plan_route(total: dict[str, dict[str, Any]], start: str, goal: str) -> dict[str, Any]:
    """Cheapest path over segments the Total map does not call blocked.

    Cost of a segment = 1 + (1 - confidence) when it is believed clear, 2 when
    nothing is known about it. Blocked segments are never used.
    """
    adj: dict[str, list[tuple[str, str, float]]] = {}
    for seg in all_segments():
        cell = total[seg["id"]]
        if cell["blocked"] is True:
            continue
        cost = 2.0 if cell["blocked"] is None else 1.0 + (1.0 - cell["confidence"])
        adj.setdefault(seg["a"], []).append((seg["b"], seg["id"], cost))
        adj.setdefault(seg["b"], []).append((seg["a"], seg["id"], cost))

    best = {start: 0.0}
    prev: dict[str, tuple[str, str]] = {}
    queue = [(0.0, start)]
    while queue:
        cost, node = heapq.heappop(queue)
        if node == goal:
            break
        if cost > best.get(node, float("inf")):
            continue
        for nxt, sid, c in adj.get(node, []):
            nc = cost + c
            if nc < best.get(nxt, float("inf")):
                best[nxt], prev[nxt] = nc, (node, sid)
                heapq.heappush(queue, (nc, nxt))

    if goal not in best:
        return {"found": False, "nodes": [], "segments": [], "cost": None}
    nodes, segs, cur = [goal], [], goal
    while cur != start:
        cur, sid = prev[cur]
        nodes.append(cur)
        segs.append(sid)
    nodes.reverse()
    segs.reverse()
    return {"found": True, "nodes": nodes, "segments": segs, "cost": round(best[goal], 3)}
