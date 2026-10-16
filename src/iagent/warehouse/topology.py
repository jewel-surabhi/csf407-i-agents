"""Warehouse topology loader + graph helpers.

The topology is NOT used for pathfinding — only to answer "from where I am,
what are my options?" The human chooses; the agent scores each option.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from iagent.warehouse.models import SegmentSeed, Topology


def load_topology(path: str | Path) -> Topology:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    # Junction id comes from its dict key — inject it before validation.
    for key, j in data.get("junctions", {}).items():
        j.setdefault("id", key)
    return Topology.model_validate(data)


def segment_id(a: str, b: str) -> str:
    """Canonical, order-independent id for the segment between two junctions."""
    lo, hi = sorted((a, b))
    return f"seg:{lo}|{hi}"


def find_segment(topo: Topology, a: str, b: str) -> SegmentSeed | None:
    want = segment_id(a, b)
    for s in topo.segments:
        if segment_id(s.a, s.b) == want:
            return s
    return None


def neighbors(topo: Topology, junction_id: str) -> list[tuple[SegmentSeed, str]]:
    """Return (segment, neighbor_junction_id) pairs reachable from `junction_id`."""
    out: list[tuple[SegmentSeed, str]] = []
    for s in topo.segments:
        if s.a == junction_id:
            out.append((s, s.b))
        elif s.b == junction_id:
            out.append((s, s.a))
    return out
