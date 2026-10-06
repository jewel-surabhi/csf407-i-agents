"""Pure functions turning world state → `SensorPayload`.

Sensors are deterministic: the same `WorldState` and `observed_at` always
produce the same payload. They must never raise; hardware-like failures are
reported by returning a payload with `status="error"`.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from iagent.sensorimotor.models import SensorPayload, WorldState

__all__ = ["read_lidar", "read_camera", "read_clock"]

# LiDAR: obstacles closer than this count as "blocked".
LIDAR_BLOCK_THRESHOLD_CM = 30
# LiDAR nominal max range in this mock — beyond this we report "clear".
LIDAR_MAX_RANGE_CM = 400
LIDAR_CONFIDENCE = 0.95


def read_lidar(
    world_state: WorldState,
    direction: str = "front",
    *,
    observed_at: datetime,
) -> SensorPayload:
    """Scan `world_state.obstacles` for one in `direction`, return distance."""
    sensor_id = f"lidar_{direction}"
    match = _nearest_obstacle_in_direction(world_state.obstacles, direction)

    if match is None:
        # Nothing between us and the wall — treat as clear at max range.
        return SensorPayload(
            sensor=sensor_id,
            kind="distance_cm",
            value=LIDAR_MAX_RANGE_CM,
            status="clear",
            confidence=LIDAR_CONFIDENCE,
            observed_at=observed_at,
            raw={"distance_cm": LIDAR_MAX_RANGE_CM},
        )

    distance_cm = int(match.get("distance_cm", LIDAR_MAX_RANGE_CM))
    status = "blocked" if distance_cm <= LIDAR_BLOCK_THRESHOLD_CM else "clear"
    return SensorPayload(
        sensor=sensor_id,
        kind="distance_cm",
        value=distance_cm,
        status=status,
        confidence=LIDAR_CONFIDENCE,
        observed_at=observed_at,
        raw={
            "distance_cm": distance_cm,
            "material_guess": match.get("material"),
            "obstacle_id": match.get("id"),
        },
    )


def _nearest_obstacle_in_direction(
    obstacles: list[dict[str, Any]], direction: str
) -> dict[str, Any] | None:
    """Pick the closest obstacle recorded as being in `direction` of the robot.

    The scenario YAML uses `in_front_of: robot` today; we treat that as the
    "front" bucket. Additional buckets (`in_left_of`, `in_right_of`, ...) can be
    added later without touching call sites.
    """
    key = _direction_key(direction)
    candidates = [o for o in obstacles if key in o and o[key] == "robot"]
    if not candidates:
        return None
    return min(candidates, key=lambda o: o.get("distance_cm", LIDAR_MAX_RANGE_CM))


def _direction_key(direction: str) -> str:
    return {
        "front": "in_front_of",
        "back": "in_back_of",
        "left": "in_left_of",
        "right": "in_right_of",
    }.get(direction, "in_front_of")


# --- Camera and clock: implementation lands with Scenario B. ---


def read_camera(world_state: WorldState, direction: str = "front", *, observed_at: datetime):
    raise NotImplementedError("Phase 3 — read_camera lands with Scenario B (P2).")


def read_clock(*, observed_at: datetime):
    raise NotImplementedError("Phase 3 — read_clock lands with Scenario B (P2).")
