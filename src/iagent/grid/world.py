"""Grid world for the Scenario A dashboard: a 5 x 4 node graph.

Nodes are named like a spreadsheet: columns A-E, rows 1-4 (``A1`` is the
top-left node). A segment is the straight edge between two neighbouring
nodes and is named ``<node>-<node>`` with the nodes in sorted order, so
``A1-B1`` and ``B1-A1`` are the same segment. 16 horizontal + 15 vertical
segments = 31 segments.

The robot stands on a node, the target is a node, and a box always sits at
the MIDPOINT of a segment. Neighbouring nodes are ``SPACING_CM`` apart, so a
box on the first segment in front of the robot is 50 cm away.

This module is pure (no I/O, no LLM). It is the "real world" the LiDAR
reads during the usage phase.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from iagent.sensorimotor.sensors import lidar_confidence

__all__ = [
    "COLS",
    "ROWS",
    "SPACING_CM",
    "DIRECTIONS",
    "GridWorld",
    "LidarReading",
    "node_id",
    "node_xy",
    "all_nodes",
    "all_segments",
    "segment_id",
    "scan",
]

COLS = 5
ROWS = 4
SPACING_CM = 100

_COL_NAMES = "ABCDE"

# (d_col, d_row) — row 1 is the top of the grid, so "north" decreases the row.
DIRECTIONS: dict[str, tuple[int, int]] = {
    "north": (0, -1),
    "east": (1, 0),
    "south": (0, 1),
    "west": (-1, 0),
}


def node_id(col: int, row: int) -> str:
    return f"{_COL_NAMES[col]}{row + 1}"


def node_xy(node: str) -> tuple[int, int]:
    """``'B3'`` -> ``(1, 2)`` (zero-based column, row). Raises ValueError."""
    if len(node) != 2 or node[0] not in _COL_NAMES[:COLS] or not node[1].isdigit():
        raise ValueError(f"unknown node {node!r}")
    col, row = _COL_NAMES.index(node[0]), int(node[1]) - 1
    if not 0 <= row < ROWS:
        raise ValueError(f"unknown node {node!r}")
    return col, row


def all_nodes() -> list[str]:
    return [node_id(c, r) for r in range(ROWS) for c in range(COLS)]


def segment_id(a: str, b: str) -> str:
    """Canonical segment name for two neighbouring nodes."""
    (ac, ar), (bc, br) = node_xy(a), node_xy(b)
    if abs(ac - bc) + abs(ar - br) != 1:
        raise ValueError(f"{a} and {b} are not neighbours")
    first, second = sorted((a, b))
    return f"{first}-{second}"


def all_segments() -> list[dict[str, str]]:
    """Every segment with its two end nodes and orientation."""
    out: list[dict[str, str]] = []
    for r in range(ROWS):
        for c in range(COLS):
            here = node_id(c, r)
            if c + 1 < COLS:
                there = node_id(c + 1, r)
                out.append({"id": segment_id(here, there), "a": here, "b": there, "axis": "h"})
            if r + 1 < ROWS:
                there = node_id(c, r + 1)
                out.append({"id": segment_id(here, there), "a": here, "b": there, "axis": "v"})
    return out


_SEGMENT_IDS = frozenset(s["id"] for s in all_segments())


@dataclass
class GridWorld:
    """Ground truth: where the robot is, where it must go, which segments hold a box."""

    robot: str = "A1"
    target: str = "E4"
    boxes: set[str] = field(default_factory=set)

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        node_xy(self.robot)
        node_xy(self.target)
        if self.robot == self.target:
            raise ValueError("robot and target must be on different nodes")
        unknown = sorted(set(self.boxes) - _SEGMENT_IDS)
        if unknown:
            raise ValueError(f"unknown segment(s): {', '.join(unknown)}")

    def to_dict(self) -> dict:
        return {"robot": self.robot, "target": self.target, "boxes": sorted(self.boxes)}


@dataclass(frozen=True)
class LidarReading:
    """What one LiDAR beam says about one segment."""

    segment: str
    blocked: bool
    confidence: float
    distance_cm: int
    direction: str

    def to_dict(self) -> dict:
        return {
            "segment": self.segment,
            "blocked": self.blocked,
            "confidence": self.confidence,
            "distance_cm": self.distance_cm,
            "direction": self.direction,
        }


def scan(world: GridWorld) -> dict[str, LidarReading]:
    """Sweep the robot's row and column in all four directions.

    Each beam runs until the grid ends or it hits a box. Segments the beam
    passes through completely are read as clear, with the confidence for the
    distance to the segment's far end. The segment holding the box is read
    as blocked, with the confidence for the distance to the box (the
    midpoint). Segments behind a box are not seen and get no reading.
    """
    readings: dict[str, LidarReading] = {}
    col, row = node_xy(world.robot)
    for direction, (dc, dr) in DIRECTIONS.items():
        c, r, travelled = col, row, 0
        while 0 <= c + dc < COLS and 0 <= r + dr < ROWS:
            here, there = node_id(c, r), node_id(c + dc, r + dr)
            seg = segment_id(here, there)
            if seg in world.boxes:
                distance = travelled + SPACING_CM // 2
                readings[seg] = LidarReading(seg, True, lidar_confidence(distance), distance, direction)
                break
            travelled += SPACING_CM
            readings[seg] = LidarReading(seg, False, lidar_confidence(travelled), travelled, direction)
            c, r = c + dc, r + dr
    return readings
