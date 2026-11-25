"""Sources the grid dashboard registers in the declarative layer.

Decay is a property of the source (the kind of information):
``effective = max(0, stored - decay_per_hour * age_hours)``.
"""

from __future__ import annotations

from iagent.declarative.api import KnowledgeAPI

__all__ = ["OPERATOR", "LIDAR_DECLARED", "CAMERA_DECLARED", "GRID_SOURCES", "register_grid_sources"]

OPERATOR = "operator_setup"  # the operator's first declaration (path state and box colour)
LIDAR_DECLARED = "lidar_declared"  # a LiDAR reading once stored in the declaration layer
CAMERA_DECLARED = "camera_declared"  # a camera colour once stored in the declaration layer

# source_id: (kind, trust prior, decay per hour, description)
GRID_SOURCES: dict[str, tuple[str, float, float, str]] = {
    OPERATOR: ("user", 0.90, 0.20, "Operator's first declaration (setup)"),
    LIDAR_DECLARED: ("sensor", 0.90, 0.15, "LiDAR reading declared into the declaration layer"),
    CAMERA_DECLARED: ("sensor", 0.90, 0.15, "Camera colour declared into the declaration layer"),
}


def register_grid_sources(knowledge: KnowledgeAPI) -> None:
    for sid, (kind, trust, decay, desc) in GRID_SOURCES.items():
        knowledge.log_source(sid, kind, trust, desc, decay_rate_per_hour=decay)
