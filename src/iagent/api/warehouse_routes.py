"""HTTP routes for the Warehouse 7 demo.

Serves the topology, scores moves with grounded confidence, commits moves,
reconciles inventory, and executes transfers — all through a KnowledgeAPI
seeded from configs/warehouse.yaml.

The confidence pills and reconciliation the frontend shows come straight from
these endpoints (belief-graph derived). The `/api/query` agent endpoint (in
routes.py) remains the LLM-narration path for the "Explain" toggle.

Owner: Person 4.
"""

from __future__ import annotations

import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from iagent.declarative.api import KnowledgeAPI
from iagent.declarative.belief_graph import BeliefGraph
from iagent.declarative.models import Belief, ProvenanceEntry, Source
from iagent.declarative.provenance import ProvenanceStore
from iagent.declarative.seeding import seed_sources, seed_world_facts
from iagent.warehouse.events import EventEngine
from iagent.warehouse.models import (
    InventoryView,
    MoveOption,
    ShelfContents,
    Topology,
    TransferResult,
)
from iagent.warehouse.service import (
    evaluate_moves,
    reconcile_inventory,
    seed_warehouse,
    shelf_contents,
    transfer,
)
from iagent.warehouse.topology import load_topology, neighbors

router = APIRouter(prefix="/api/warehouse", tags=["warehouse"])

_REPO = Path(__file__).resolve().parents[3]
_WAREHOUSE_YAML = _REPO / "configs" / "warehouse.yaml"
_SEED_SQL = _REPO / "data" / "seed" / "initial_provenance.sql"
_WORLD_FACTS_YAML = _REPO / "data" / "seed" / "world_facts.yaml"
_EVENTS_YAML = _REPO / "configs" / "events.yaml"


def _write_position_belief(knowledge: KnowledgeAPI, junction: str, reason: str) -> None:
    """Persist the robot's current junction as a first-class belief so the
    ask-bar agent can answer 'nearby' questions by calling
    get_belief(robot, located_at). Auto-supersedes any prior position belief
    via the at-most-one-active invariant."""
    knowledge.update_belief(
        Belief(
            subject="robot",
            predicate="located_at",
            object=junction,
            confidence=1.0,
            source="warehouse_scanner",
            perspective="self",
            observed_at=datetime.now(UTC),
        ),
        reason=reason,
    )


class _WarehouseState:
    def __init__(self) -> None:
        self.topo: Topology = load_topology(_WAREHOUSE_YAML)
        self.store = ProvenanceStore(os.environ.get("IAGENT_WAREHOUSE_DB",
                                                    str(_REPO / "warehouse.db")))
        seed_sources(self.store, _SEED_SQL)
        self.knowledge = KnowledgeAPI(BeliefGraph(), self.store)
        seed_warehouse(self.knowledge, self.topo)
        # Also seed world-facts (agent identity, facility, policies, aisle
        # metadata, staff roster) so the ask-bar agent can answer general
        # questions, not just physical queries about segments and stock.
        if _WORLD_FACTS_YAML.exists():
            seed_world_facts(self.knowledge, _WORLD_FACTS_YAML)
        self.pos: str = self.topo.start
        # Write the starting position as a belief so the agent can read it.
        # (This OVERWRITES the "room_101" seed from world_facts.yaml with the
        # actual warehouse junction, since that's the ground truth here.)
        _write_position_belief(self.knowledge, self.pos, "warehouse start position")
        # YAML-driven dynamic event engine — belief changes that fire as the
        # session progresses, so the agent handles new information mid-episode.
        self.events = EventEngine(_EVENTS_YAML if _EVENTS_YAML.exists() else None)


_STATE: _WarehouseState | None = None


def get_warehouse() -> _WarehouseState:
    global _STATE
    if _STATE is None:
        _STATE = _WarehouseState()
    return _STATE


def reset_warehouse() -> None:
    global _STATE
    if _STATE is not None:
        _STATE.store.close()
    _STATE = None


WH = Annotated[_WarehouseState, Depends(get_warehouse)]


# ───────────── request/response bodies ─────────────


class MoveRequest(BaseModel):
    to: str


class TransferRequest(BaseModel):
    sku: str
    dest: str
    quantity: int = Field(ge=1, le=100000)


class ReflectionRequest(BaseModel):
    lesson: str = Field(min_length=1, max_length=400)
    topic: str = Field(default="general", min_length=1, max_length=60)
    confidence: float = Field(default=0.6, ge=0.0, le=1.0)


class WarehouseView(BaseModel):
    id: str
    position: str
    goal: str
    junctions: dict
    segments: list[dict]
    shelves: list[dict]
    options: list[MoveOption]
    # Any dynamic events that just fired (populated only by the move endpoint).
    events_fired: list[dict] = Field(default_factory=list)
    moves_made: int = 0


# ───────────── endpoints ─────────────


def _snapshot(wh: _WarehouseState, fired: list | None = None) -> WarehouseView:
    return WarehouseView(
        id=wh.topo.id,
        position=wh.pos,
        goal=wh.topo.goal,
        junctions={k: j.model_dump() for k, j in wh.topo.junctions.items()},
        segments=[s.model_dump() for s in wh.topo.segments],
        shelves=[s.model_dump() for s in wh.topo.shelves],
        options=evaluate_moves(wh.knowledge, wh.topo, wh.pos),
        events_fired=[e.to_dict() for e in (fired or [])],
        moves_made=wh.events.moves_made,
    )


@router.get("", response_model=WarehouseView)
def get_view(wh: WH) -> WarehouseView:
    """Full warehouse snapshot + the agent's scored options from the current spot."""
    return _snapshot(wh)


@router.get("/options", response_model=list[MoveOption])
def get_options(wh: WH) -> list[MoveOption]:
    """Grounded-confidence score for every move reachable from the current spot."""
    return evaluate_moves(wh.knowledge, wh.topo, wh.pos)


@router.post("/move", response_model=WarehouseView)
def do_move(req: MoveRequest, wh: WH) -> WarehouseView:
    """Commit a move to an adjacent junction. 400 if it isn't adjacent.

    Also ticks the EventEngine — any YAML-defined events whose triggers match
    the new move count or destination fire now and their belief-writes are
    applied before the snapshot is returned, so the UI sees the mutation in
    the SAME response as the move.
    """
    reachable = {to for _seg, to in neighbors(wh.topo, wh.pos)}
    if req.to not in reachable:
        raise HTTPException(
            status_code=400,
            detail=f"{req.to!r} is not adjacent to {wh.pos!r}. Reachable: {sorted(reachable)}",
        )
    wh.pos = req.to
    # Persist the move as a belief so the ask-bar agent knows where we are
    # (lets it resolve "nearby" / "adjacent" / "ahead" references correctly).
    _write_position_belief(wh.knowledge, wh.pos, f"moved to {wh.pos}")
    fired = wh.events.on_move(wh.knowledge, wh.pos)
    return _snapshot(wh, fired=fired)


@router.get("/inventory/{sku}", response_model=InventoryView)
def get_inventory(sku: str, wh: WH) -> InventoryView:
    """Reconcile a SKU across its source perspectives."""
    return reconcile_inventory(wh.knowledge, sku)


@router.get("/bin/{shelf_id}", response_model=ShelfContents)
def get_bin(shelf_id: str, wh: WH) -> ShelfContents:
    """List every lot currently stored at a shelf (bin), including transferred-in."""
    return shelf_contents(wh.knowledge, wh.topo, shelf_id)


@router.get("/provenance/{sku}", response_model=list[ProvenanceEntry])
def get_provenance(sku: str, wh: WH, limit: int = 20) -> list[ProvenanceEntry]:
    """Full provenance timeline for a SKU — every claim, by every source, over time.

    This is the SQLite provenance log surfaced for the UI. Each entry shows WHO
    said WHAT, from WHICH PERSPECTIVE, with WHAT CONFIDENCE, and WHEN. Lets the
    user see the historical ledger behind the current reconciled view.
    """
    return wh.knowledge.query_provenance(sku, limit=limit)


@router.post("/transfer", response_model=TransferResult)
def do_transfer(req: TransferRequest, wh: WH) -> TransferResult:
    """Validate and execute a stock transfer (real belief-state update + provenance)."""
    return transfer(wh.knowledge, wh.topo, req.sku, req.dest, req.quantity)


@router.post("/reset", response_model=WarehouseView)
def do_reset(wh: WH) -> WarehouseView:
    """Rebuild the warehouse to its seeded starting state. The event engine
    is reset too so dynamic events can fire again on the fresh session."""
    reset_warehouse()
    fresh = get_warehouse()
    return _snapshot(fresh)


# ───────────── temporal decay + reflections ─────────────


@router.get("/sources", response_model=list[Source])
def list_sources(wh: WH) -> list[Source]:
    """Every registered source with its trust prior AND decay rate.

    The frontend reads this to render the Trust panel with decay-rate badges
    so the user can see WHY a two-hour-old LiDAR reading is less trustworthy
    than a fresh one.
    """
    return wh.knowledge.list_sources()


@router.get("/reflections", response_model=list[Belief])
def list_reflections(wh: WH) -> list[Belief]:
    """Return every active self-reflection belief (newest first).

    These are Reflexion-style meta-beliefs the agent has learned about its
    own reasoning — stored as first-class beliefs under subject='self.policy'
    so they carry provenance, trust prior, and decay like any sensor reading.
    """
    return wh.knowledge.list_reflections()


@router.post("/reflections", response_model=Belief)
def add_reflection(req: ReflectionRequest, wh: WH) -> Belief:
    """Write a new self-reflection into the belief graph.

    The UI exposes this as 'teach the agent something new about itself' —
    the written lesson becomes a queryable belief that influences future
    episodes exactly like a sensor reading or a user statement would.
    """
    wh.knowledge.add_reflection(
        lesson=req.lesson,
        topic=req.topic,
        confidence=req.confidence,
    )
    # Return the belief view so the UI can render it immediately with decay.
    belief = wh.knowledge.get_belief(
        "self.policy", f"learned_rule:{req.topic}", "self"
    )
    if belief is None:  # pragma: no cover - should always exist right after write
        raise HTTPException(status_code=500, detail="reflection vanished after write")
    return belief
