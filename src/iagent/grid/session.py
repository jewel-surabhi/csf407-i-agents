"""One dashboard session: the real world, a simulated clock and the declarative layer.

Lifecycle:

1. ``complete_setup(world)`` the FIRST time declares the world into the
   declarative layer at hour 0, as the operator (source ``operator_setup``):
   every path gets ``is_blocked``, and every path with a box also gets
   ``box_color``.
2. ``grid.agent.run_grid_question`` advances the clock by +X hours. The
   ``decision`` folder reads the declaration map and the sensor map, the
   procedural layer (LLM) submits the Total map, and its sensor-backed winners
   are written back to the declarative layer.
3. ``complete_setup(world)`` LATER (after "Change setup") only changes the
   real world. Nothing is declared; the next question's sensors find it.

Time is simulated: hour ``h`` is ``epoch + h hours``. Decay is a property of
each source (see ``grid.sources``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from iagent.declarative.api import KnowledgeAPI
from iagent.declarative.belief_graph import BeliefGraph
from iagent.declarative.models import Belief
from iagent.declarative.provenance import ProvenanceStore
from iagent.declarative.seeding import seed_sources
from iagent.grid.sources import GRID_SOURCES, OPERATOR, register_grid_sources
from iagent.grid.world import GridWorld, all_segments
from iagent.layer_readers.declaration_map import COLOR, STATE, path_history, read_declaration_map

__all__ = ["GridSession"]

_REPO = Path(__file__).resolve().parents[3]
_SEED_SQL = _REPO / "data" / "seed" / "initial_provenance.sql"


@dataclass
class GridSession:
    world: GridWorld = field(default_factory=GridWorld)
    clock_h: int = 0
    declared: bool = False
    epoch: datetime = field(default_factory=lambda: datetime(2026, 1, 1, tzinfo=UTC))
    history: list[dict[str, Any]] = field(default_factory=list)
    setup_record: dict[str, Any] | None = None
    world_changes: list[dict[str, Any]] = field(default_factory=list)
    db_path: str = ":memory:"

    def __post_init__(self) -> None:
        store = ProvenanceStore(self.db_path)
        seed_sources(store, _SEED_SQL)
        self.knowledge = KnowledgeAPI(BeliefGraph(), store)
        register_grid_sources(self.knowledge)
        self._sources = {s.source_id: s for s in self.knowledge.list_sources()}

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
        conf = GRID_SOURCES[OPERATOR][1]
        for seg in all_segments():
            sid = seg["id"]
            self.knowledge.update_belief(
                Belief(
                    subject=sid,
                    predicate=STATE,
                    object=str(sid in world.boxes).lower(),
                    confidence=conf,
                    source=OPERATOR,
                    observed_at=when,
                ),
                reason="operator setup declaration",
            )
            if sid in world.boxes:
                self.knowledge.update_belief(
                    Belief(
                        subject=sid,
                        predicate=COLOR,
                        object=world.boxes[sid],
                        confidence=conf,
                        source=OPERATOR,
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

    # ------------------------------------------------------------- reads

    def declarations(self, hour: float) -> dict[str, dict[str, Any]]:
        return read_declaration_map(self.knowledge, hour=hour, epoch=self.epoch)

    def history_for(self, path: str) -> list[dict[str, Any]]:
        return path_history(self.knowledge, path, epoch=self.epoch)

    def timeline(self) -> list[dict[str, Any]]:
        """Significant hours, oldest first: the setup declaration, then every answered question."""
        items = [self.setup_record] if self.setup_record else []
        return items + [{"kind": "question", **rec} for rec in self.history]

    def sources_view(self) -> dict[str, dict[str, float]]:
        return {
            sid: {"trust_prior": self._sources[sid].trust_prior, "decay_per_hour": self._sources[sid].decay_rate_per_hour}
            for sid in GRID_SOURCES
        }
