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


# --- Camera and clock: return preset payloads from the scenario YAML.
#
# Unlike LiDAR (which derives its reading from `world_state.obstacles`), the
# camera's perception is scenario-scripted — the YAML specifies what the
# camera "sees" so the test author controls perception noise (e.g. box_01
# perceived as 'brown' under yellow lighting in scenario_b).


def read_camera(
    world_state: WorldState,
    direction: str = "front",
    *,
    observed_at: datetime,
    presets: dict | None = None,
) -> SensorPayload:
    """Return the scenario's preset camera reading if present, else an empty frame."""
    sensor_id = f"camera_{direction}"
    preset = ((presets or {}).get(sensor_id) or {}).get("reading") or {}
    detections = preset.get("detections", [])
    ambient = preset.get("ambient", {"lighting": world_state.lighting})

    return SensorPayload(
        sensor=sensor_id,
        kind="camera_frame",
        value=len(detections),
        status="ok" if detections else "empty",
        confidence=_avg_detection_confidence(detections),
        observed_at=observed_at,
        raw={"detections": detections, "ambient": ambient},
    )


def _avg_detection_confidence(detections: list[dict]) -> float:
    if not detections:
        return 0.0
    vals = [float(d.get("confidence", 0.0)) for d in detections]
    return round(sum(vals) / len(vals), 3)


def read_clock(*, observed_at: datetime) -> SensorPayload:
    return SensorPayload(
        sensor="clock",
        kind="iso8601",
        value=observed_at.isoformat(),
        status="ok",
        confidence=1.0,
        observed_at=observed_at,
        raw={"now": observed_at.isoformat()},
    )
