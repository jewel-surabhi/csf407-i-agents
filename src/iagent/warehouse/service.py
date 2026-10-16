"""Warehouse service — grounded move scoring + inventory reconciliation.

Everything here reads/writes through `KnowledgeAPI`. The confidence numbers the
UI shows are belief confidences after conflict resolution — never invented.

- Navigation  = Scenario A: each segment has a passability belief; a sensor
  belief (is_blocked) overrides the floor map, dropping confidence.
- Inventory   = Scenario B: each SKU holds count/shipped beliefs across source
  perspectives; reconciliation trusts the highest-prior source and nets
  shipments.

Owner: Person 4 (demo application layer).
"""

from __future__ import annotations

from datetime import UTC, datetime

from iagent.declarative.api import Belief, KnowledgeAPI
from iagent.warehouse.models import (
    InventoryView,
    MoveOption,
    SourceCount,
    Topology,
    TransferResult,
)
from iagent.warehouse.topology import neighbors, segment_id

_SENSORS = {"lidar_front", "warehouse_scanner", "camera_front"}

# Sources the warehouse introduces beyond the base seed (initial_provenance.sql).
_EXTRA_SOURCES = [
    ("warehouse_scanner", "sensor", 0.90, "Floor inventory scanner"),
    ("erp_system", "agent", 0.60, "ERP stock system"),
    ("sales_log", "agent", 0.80, "Sales / shipping log"),
]


def _now() -> datetime:
    return datetime.now(UTC)


def _src_label(src: str) -> str:
    return {
        "lidar_front": "front LiDAR",
        "warehouse_scanner": "floor scanner",
        "camera_front": "dock camera",
        "default_map": "floor map",
        "erp_system": "ERP",
        "sales_log": "sales log",
    }.get(src, src)


def _tier(c: float) -> str:
    return "high" if c >= 0.8 else "medium" if c >= 0.4 else "low"


# ───────────────────────── seeding ─────────────────────────


def seed_warehouse(knowledge: KnowledgeAPI, topo: Topology) -> None:
    """Load the topology's segment + stock beliefs into the declarative layer."""
    for sid, kind, trust, desc in _EXTRA_SOURCES:
        knowledge.log_source(sid, kind, trust, desc)

    for seg in topo.segments:
        sid = segment_id(seg.a, seg.b)
        if seg.state == "clear":
            knowledge.update_belief(
                Belief(subject=sid, predicate="is_clear", object="true",
                       confidence=seg.confidence, source=seg.source,
                       perspective="self", observed_at=_now()),
                reason="warehouse seed",
            )
        else:  # blocked
            knowledge.update_belief(
                Belief(subject=sid, predicate="is_blocked", object="obstacle",
                       confidence=seg.confidence, source=seg.source,
                       perspective="self", observed_at=_now()),
                reason="warehouse seed",
            )

    for shelf in topo.shelves:
        for st in shelf.stock:
            knowledge.update_belief(
                Belief(subject=shelf.id, predicate=st.predicate, object=str(st.value),
                       confidence=0.9 if st.source in _SENSORS else 0.6,
                       source=st.source, perspective=st.perspective, observed_at=_now()),
                reason=f"warehouse seed ({st.note})" if st.note else "warehouse seed",
            )


# ───────────────────────── navigation ─────────────────────────


def score_segment(knowledge: KnowledgeAPI, seg_id: str) -> tuple[float, str, str, bool, str]:
    """Return (confidence, tier, source, passable, reason) for one segment,
    derived from its active beliefs."""
    blocked = knowledge.get_belief(seg_id, "is_blocked")
    if blocked is not None:
        conf = round(1.0 - blocked.confidence, 2)
        return (conf, _tier(conf), blocked.source, False,
                f"Sensor conflict — the {_src_label(blocked.source)} reports an "
                f"obstruction. Live sensors outrank the floor map, so avoid.")
    clear = knowledge.get_belief(seg_id, "is_clear")
    if clear is not None:
        conf = round(clear.confidence, 2)
        if clear.source in _SENSORS:
            return (conf, _tier(conf), clear.source, True,
                    f"Clear — confirmed by the {_src_label(clear.source)}.")
        return (conf, _tier(conf), clear.source, True,
                "Floor map says clear, but no sensor has verified it — treat as uncertain.")
    return (0.5, "medium", "unknown", True, "No belief on record for this segment.")


def evaluate_moves(knowledge: KnowledgeAPI, topo: Topology, from_junction: str) -> list[MoveOption]:
    """Score every move reachable from `from_junction`, best first."""
    if from_junction not in topo.junctions:
        return []
    opts: list[MoveOption] = []
    for seg, to in neighbors(topo, from_junction):
        sid = segment_id(seg.a, seg.b)
        conf, tier, source, passable, reason = score_segment(knowledge, sid)
        opts.append(MoveOption(
            to=to, to_label=topo.junctions[to].label, segment=sid,
            confidence=conf, tier=tier, source=source, passable=passable, reason=reason,
        ))
    opts.sort(key=lambda o: o.confidence, reverse=True)
    return opts


# ───────────────────────── inventory ─────────────────────────


def reconcile_inventory(knowledge: KnowledgeAPI, sku: str) -> InventoryView:
    """Reconcile a SKU's stock across source perspectives (Scenario B)."""
    trust = {s.source_id: s.trust_prior for s in knowledge.list_sources()}
    views = knowledge.query_perspective(sku)

    rows: list[SourceCount] = []
    counts: list[tuple[int, float, float, str]] = []  # value, trust, conf, source
    shipped_total = 0
    for v in views:
        for b in v.beliefs:
            try:
                val = int(b.object)
            except (TypeError, ValueError):
                continue
            rows.append(SourceCount(source=b.source, perspective=b.perspective,
                                    predicate=b.predicate, value=val,
                                    confidence=b.confidence))
            if b.predicate == "count":
                counts.append((val, trust.get(b.source, 0.5), b.confidence, b.source))
            elif b.predicate == "shipped":
                shipped_total += val

    if not counts:
        return InventoryView(sku=sku, reconciled=0, confidence=0.0,
                             winning_source="none", sources=rows,
                             explanation="No count on record for this SKU.")

    val, tr, conf, src = max(counts, key=lambda c: (c[1], c[2]))
    reconciled = val - shipped_total
    expl = (f"Trust the {_src_label(src)} ({tr:.2f}) over the other sources"
            + (f"; net {shipped_total} shipped → {val} − {shipped_total} = {reconciled}."
               if shipped_total else f" → {reconciled}."))
    return InventoryView(sku=sku, reconciled=reconciled, confidence=round(conf, 2),
                         winning_source=src, sources=rows, explanation=expl)


def transfer(knowledge: KnowledgeAPI, topo: Topology, sku: str,
             dest_shelf: str, quantity: int) -> TransferResult:
    """Move `quantity` units of `sku` to `dest_shelf`. Validates against the
    reconciled source count, then writes real belief updates + provenance.

    Honest scope: this updates the warehouse's *beliefs* about stock (SQLite +
    graph), not a transactional WMS. The change is durable and audited.
    """
    shelf_ids = {s.id for s in topo.shelves}
    if sku not in shelf_ids:
        return TransferResult(ok=False, sku=sku, quantity=quantity,
                              source_shelf=sku, dest_shelf=dest_shelf,
                              message=f"Unknown SKU/shelf '{sku}'. Inventory unchanged.")
    if dest_shelf not in shelf_ids:
        return TransferResult(ok=False, sku=sku, quantity=quantity,
                              source_shelf=sku, dest_shelf=dest_shelf,
                              message=f"Unknown destination '{dest_shelf}'. Inventory unchanged.")
    if quantity <= 0:
        return TransferResult(ok=False, sku=sku, quantity=quantity,
                              source_shelf=sku, dest_shelf=dest_shelf,
                              message="Quantity must be at least 1. Inventory unchanged.")

    src_view = reconcile_inventory(knowledge, sku)
    available = src_view.reconciled
    if quantity > available:
        return TransferResult(ok=False, sku=sku, quantity=quantity,
                              source_shelf=sku, dest_shelf=dest_shelf,
                              message=f"Only {available} available — can't move {quantity}. "
                                      "Inventory unchanged.")

    src_after = available - quantity
    # Source: the floor scanner now reads the lower count (authoritative, self perspective).
    # This recount already reflects reality, so retire any "shipped" belief that
    # reconcile_inventory would otherwise subtract a second time.
    knowledge.update_belief(
        Belief(subject=sku, predicate="count", object=str(src_after),
               confidence=0.95, source="warehouse_scanner", perspective="self",
               observed_at=_now()),
        reason=f"transfer out {quantity} to {dest_shelf}",
    )
    for v in knowledge.query_perspective(sku):
        for b in v.beliefs:
            if b.predicate == "shipped":
                knowledge.downgrade_belief(sku, "shipped", 0.1,
                                           "absorbed into post-transfer recount",
                                           perspective=b.perspective)
    # Destination: record units now located there.
    dest_subject = f"{sku}@{dest_shelf}"
    existing = knowledge.get_belief(dest_subject, "count")
    dest_after = (int(existing.object) if existing else 0) + quantity
    knowledge.update_belief(
        Belief(subject=dest_subject, predicate="count", object=str(dest_after),
               confidence=0.95, source="warehouse_scanner", perspective="self",
               observed_at=_now()),
        reason=f"transfer in {quantity} from {sku}",
    )
    return TransferResult(ok=True, sku=sku, quantity=quantity,
                          source_shelf=sku, dest_shelf=dest_shelf,
                          source_after=src_after, dest_after=dest_after,
                          message=f"Moved {quantity} × {sku} to {dest_shelf}. "
                                  f"Source now {src_after}; {quantity} logged at {dest_shelf}.")
