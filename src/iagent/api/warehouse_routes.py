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
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from iagent.declarative.api import KnowledgeAPI
from iagent.declarative.belief_graph import BeliefGraph
from iagent.declarative.provenance import ProvenanceStore
from iagent.declarative.seeding import seed_sources
from iagent.warehouse.models import InventoryView, MoveOption, Topology, TransferResult
from iagent.warehouse.service import (
    evaluate_moves,
    reconcile_inventory,
    seed_warehouse,
    transfer,
)
from iagent.warehouse.topology import load_topology, neighbors

router = APIRouter(prefix="/api/warehouse", tags=["warehouse"])

_REPO = Path(__file__).resolve().parents[3]
_WAREHOUSE_YAML = _REPO / "configs" / "warehouse.yaml"
_SEED_SQL = _REPO / "data" / "seed" / "initial_provenance.sql"


class _WarehouseState:
    def __init__(self) -> None:
        self.topo: Topology = load_topology(_WAREHOUSE_YAML)
        self.store = ProvenanceStore(os.environ.get("IAGENT_WAREHOUSE_DB",
                                                    str(_REPO / "warehouse.db")))
        seed_sources(self.store, _SEED_SQL)
        self.knowledge = KnowledgeAPI(BeliefGraph(), self.store)
        seed_warehouse(self.knowledge, self.topo)
        self.pos: str = self.topo.start


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


class WarehouseView(BaseModel):
    id: str
    position: str
    goal: str
    junctions: dict
    segments: list[dict]
    shelves: list[dict]
    options: list[MoveOption]


# ───────────── endpoints ─────────────


@router.get("", response_model=WarehouseView)
def get_view(wh: WH) -> WarehouseView:
    """Full warehouse snapshot + the agent's scored options from the current spot."""
    return WarehouseView(
        id=wh.topo.id,
        position=wh.pos,
        goal=wh.topo.goal,
        junctions={k: j.model_dump() for k, j in wh.topo.junctions.items()},
        segments=[s.model_dump() for s in wh.topo.segments],
        shelves=[s.model_dump() for s in wh.topo.shelves],
        options=evaluate_moves(wh.knowledge, wh.topo, wh.pos),
    )


@router.get("/options", response_model=list[MoveOption])
def get_options(wh: WH) -> list[MoveOption]:
    """Grounded-confidence score for every move reachable from the current spot."""
    return evaluate_moves(wh.knowledge, wh.topo, wh.pos)


@router.post("/move", response_model=WarehouseView)
def do_move(req: MoveRequest, wh: WH) -> WarehouseView:
    """Commit a move to an adjacent junction. 400 if it isn't adjacent."""
    reachable = {to for _seg, to in neighbors(wh.topo, wh.pos)}
    if req.to not in reachable:
        raise HTTPException(
            status_code=400,
            detail=f"{req.to!r} is not adjacent to {wh.pos!r}. Reachable: {sorted(reachable)}",
        )
    wh.pos = req.to
    return get_view(wh)


@router.get("/inventory/{sku}", response_model=InventoryView)
def get_inventory(sku: str, wh: WH) -> InventoryView:
    """Reconcile a SKU across its source perspectives."""
    return reconcile_inventory(wh.knowledge, sku)


@router.post("/transfer", response_model=TransferResult)
def do_transfer(req: TransferRequest, wh: WH) -> TransferResult:
    """Validate and execute a stock transfer (real belief-state update + provenance)."""
    return transfer(wh.knowledge, wh.topo, req.sku, req.dest, req.quantity)


@router.post("/reset", response_model=WarehouseView)
def do_reset(wh: WH) -> WarehouseView:
    """Rebuild the warehouse to its seeded starting state."""
    reset_warehouse()
    fresh = get_warehouse()
    return get_view(fresh)
