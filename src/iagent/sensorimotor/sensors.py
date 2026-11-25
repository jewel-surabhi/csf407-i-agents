"""Pure functions turning world state → `SensorPayload`.

Sensors are deterministic: the same `WorldState` and `observed_at` always
produce the same payload. They must never raise; hardware-like failures are
reported by returning a payload with `status="error"`.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from iagent.sensorimotor.models import SensorPayload, WorldState

__all__ = [
    "read_lidar",
    "read_lidar_sweep",
    "read_camera",
    "read_camera_sweep",
    "read_clock",
    "lidar_confidence",
    "camera_confidence",
    "SWEEP_DIRECTIONS",
]

# LiDAR nominal max range in this mock. Any obstacle detected within range is
# reported as "blocked" (there is no distance cut-off); how far away it is only
# lowers the confidence. Beyond this range nothing is detected -> "clear".
LIDAR_MAX_RANGE_CM = 400
# Kept for backward compatibility only; no longer used to decide the status.
LIDAR_BLOCK_THRESHOLD_CM = 30

# LiDAR confidence falls off with distance: returns from far-away surfaces are
# weaker and more prone to noise / missed thin objects. Readings up to
# LIDAR_FULL_CONFIDENCE_RANGE_CM keep full confidence; beyond that it drops
# linearly to LIDAR_MIN_CONFIDENCE at LIDAR_MAX_RANGE_CM.
LIDAR_MAX_CONFIDENCE = 0.95
LIDAR_MIN_CONFIDENCE = 0.50
LIDAR_FULL_CONFIDENCE_RANGE_CM = 50
# Kept for backward compatibility: confidence of a close-range reading.
LIDAR_CONFIDENCE = LIDAR_MAX_CONFIDENCE


def lidar_confidence(distance_cm: float) -> float:
    """Confidence of a LiDAR return at `distance_cm`.

    - distance <= LIDAR_FULL_CONFIDENCE_RANGE_CM  -> LIDAR_MAX_CONFIDENCE
    - distance >= LIDAR_MAX_RANGE_CM              -> LIDAR_MIN_CONFIDENCE
    - in between: linear interpolation, rounded to 3 decimals.
    """
    if distance_cm <= LIDAR_FULL_CONFIDENCE_RANGE_CM:
        return LIDAR_MAX_CONFIDENCE
    if distance_cm >= LIDAR_MAX_RANGE_CM:
        return LIDAR_MIN_CONFIDENCE
    span = LIDAR_MAX_RANGE_CM - LIDAR_FULL_CONFIDENCE_RANGE_CM
    frac = (distance_cm - LIDAR_FULL_CONFIDENCE_RANGE_CM) / span
    conf = LIDAR_MAX_CONFIDENCE - frac * (LIDAR_MAX_CONFIDENCE - LIDAR_MIN_CONFIDENCE)
    return round(conf, 3)


# Camera colour confidence also falls with distance, from a higher ceiling:
# the camera identifies a box's colour more reliably than LiDAR ranges it.
CAMERA_MAX_CONFIDENCE = 0.98
CAMERA_MIN_CONFIDENCE = 0.70
CAMERA_FULL_CONFIDENCE_RANGE_CM = 50
CAMERA_MAX_RANGE_CM = 400


def camera_confidence(distance_cm: float) -> float:
    """Confidence of a camera colour detection at `distance_cm` (linear fall-off)."""
    return _linear_confidence(
        distance_cm,
        CAMERA_FULL_CONFIDENCE_RANGE_CM,
        CAMERA_MAX_RANGE_CM,
        CAMERA_MAX_CONFIDENCE,
        CAMERA_MIN_CONFIDENCE,
    )


def _linear_confidence(d: float, full: float, far: float, hi: float, lo: float) -> float:
    if d <= full:
        return hi
    if d >= far:
        return lo
    return round(hi - (d - full) / (far - full) * (hi - lo), 3)


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
            # Nothing returned within range: the weakest kind of evidence.
            confidence=lidar_confidence(LIDAR_MAX_RANGE_CM),
            observed_at=observed_at,
            raw={"distance_cm": LIDAR_MAX_RANGE_CM},
        )

    distance_cm = int(match.get("distance_cm", LIDAR_MAX_RANGE_CM))
    # Something was detected in range: that is a blocked direction. Distance only
    # changes how sure the sensor is, not the status.
    status = "blocked" if distance_cm < LIDAR_MAX_RANGE_CM else "clear"
    return SensorPayload(
        sensor=sensor_id,
        kind="distance_cm",
        value=distance_cm,
        status=status,
        confidence=lidar_confidence(distance_cm),
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


# --- Sweeps: read the robot's whole row and column -------------------------
#
# A sweep casts one beam in each direction. The beam is divided into cells of
# `world_state.extras["cell_cm"]` (a cell is one stretch of floor, e.g. one
# path between two grid nodes) and runs for `extras["beam_range_cm"][dir]`
# (the distance to the edge of the floor in that direction). Every cell the
# beam crosses is reported: clear cells with the confidence for the distance
# to their far end, and the first cell holding an obstacle as blocked with the
# confidence for the distance to the obstacle. The beam stops there; cells
# behind an obstacle are not seen. Without those extras the beam is one cell
# of LIDAR_MAX_RANGE_CM.

SWEEP_DIRECTIONS = ("front", "right", "back", "left")


def _beam(world_state: WorldState, direction: str) -> tuple[int, int, dict[str, Any] | None]:
    extras = world_state.extras or {}
    cell = int(extras.get("cell_cm", LIDAR_MAX_RANGE_CM))
    ranges = extras.get("beam_range_cm") or {}
    reach = int(ranges.get(direction, LIDAR_MAX_RANGE_CM))
    return cell, reach, _nearest_obstacle_in_direction(world_state.obstacles, direction)


def read_lidar_sweep(world_state: WorldState, *, observed_at: datetime) -> SensorPayload:
    """LiDAR along all four directions, cell by cell (see module note above)."""
    beams: dict[str, list[dict[str, Any]]] = {}
    for direction in SWEEP_DIRECTIONS:
        cell, reach, hit = _beam(world_state, direction)
        hit_at = int(hit.get("distance_cm", 10**9)) if hit else None
        cells: list[dict[str, Any]] = []
        start = 0
        while start < reach:
            end = min(start + cell, reach)
            if hit_at is not None and start < hit_at <= end:
                cells.append(
                    {
                        "index": len(cells),
                        "from_cm": start,
                        "to_cm": end,
                        "status": "blocked",
                        "distance_cm": hit_at,
                        "confidence": lidar_confidence(hit_at),
                        "obstacle_id": hit.get("id"),
                    }
                )
                break
            cells.append(
                {
                    "index": len(cells),
                    "from_cm": start,
                    "to_cm": end,
                    "status": "clear",
                    "distance_cm": end,
                    "confidence": lidar_confidence(end),
                }
            )
            start = end
        beams[direction] = cells
    seen = sum(len(c) for c in beams.values())
    return SensorPayload(
        sensor="lidar_sweep",
        kind="lidar_sweep",
        value=seen,
        status="ok",
        confidence=None,
        observed_at=observed_at,
        raw={"beams": beams},
    )


def read_camera_sweep(world_state: WorldState, *, observed_at: datetime) -> SensorPayload:
    """Camera along all four directions: the colour of the first obstacle in each."""
    detections: list[dict[str, Any]] = []
    for direction in SWEEP_DIRECTIONS:
        _cell, reach, hit = _beam(world_state, direction)
        if hit is None:
            continue
        d = int(hit.get("distance_cm", 10**9))
        if d > reach or d > CAMERA_MAX_RANGE_CM:
            continue
        detections.append(
            {
                "direction": direction,
                "id": hit.get("id"),
                "color": hit.get("color"),
                "distance_cm": d,
                "confidence": camera_confidence(d),
            }
        )
    return SensorPayload(
        sensor="camera_sweep",
        kind="camera_sweep",
        value=len(detections),
        status="ok" if detections else "empty",
        confidence=_avg_detection_confidence(detections),
        observed_at=observed_at,
        detections=detections,
        ambient={"lighting": world_state.lighting},
        raw={"detections": detections},
    )
