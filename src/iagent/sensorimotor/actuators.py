"""Actuators mutate world state and return ActionResult.

Disabled by default (`SensorimotorAPI(enable_actuators=False)`) so the
epistemic test suite stays deterministic. Individual actuators are added
per-commit (move_forward → rotate → grasp); the uniform `disabled_result`
helper gives the façade a well-shaped response for every flag-off call.
Owner: Person 2 (Phase 3).
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from iagent.sensorimotor.models import ActionResult, WorldState

__all__ = ["disabled_result", "move_forward", "rotate", "grasp"]


def disabled_result(
    action: str,
    params: dict[str, Any],
    *,
    observed_at: datetime,
) -> ActionResult:
    """Uniform ActionResult returned when actuators are feature-flagged off.

    The LLM / caller gets a clean `success=False, reason="actuators_disabled"`
    envelope instead of a raised exception, so a mis-called tool never crashes
    the ReAct loop.
    """
    return ActionResult(
        action=action,
        params=params,
        success=False,
        reason="actuators_disabled",
        world_delta={},
        observed_at=observed_at,
    )


def move_forward(
    world_state: WorldState, distance_cm: int, *, observed_at: datetime
) -> ActionResult:
    """Try to move the robot `distance_cm` forward.

    Fails (no mutation) if any obstacle `in_front_of: robot` sits within
    `distance_cm`. Otherwise shrinks every in-front obstacle's `distance_cm`
    by the move length so the next `sense_lidar` reflects the new geometry,
    and bumps `robot.traveled_cm` so callers can read total displacement.

    `distance_cm <= 0` is treated as a successful no-op — handy for the
    ReAct loop retrying with 0 to probe the current state without moving.
    """
    params: dict[str, Any] = {"distance_cm": distance_cm, "direction": "forward"}

    if distance_cm <= 0:
        return ActionResult(
            action="move",
            params=params,
            success=True,
            reason="no_op",
            world_delta={},
            observed_at=observed_at,
        )

    blockers = [
        o
        for o in world_state.obstacles
        if o.get("in_front_of") == "robot"
        and int(o.get("distance_cm", _FAR)) <= distance_cm
    ]
    if blockers:
        nearest = min(blockers, key=lambda o: int(o.get("distance_cm", _FAR)))
        return ActionResult(
            action="move",
            params=params,
            success=False,
            reason=f"blocked_by:{nearest.get('id')}",
            world_delta={},
            observed_at=observed_at,
        )

    # Clear path — commit the move.
    moved_obstacles: list[dict[str, Any]] = []
    for o in world_state.obstacles:
        if o.get("in_front_of") == "robot":
            o["distance_cm"] = int(o.get("distance_cm", _FAR)) - distance_cm
            moved_obstacles.append({"id": o.get("id"), "distance_cm": o["distance_cm"]})

    world_state.robot["traveled_cm"] = (
        int(world_state.robot.get("traveled_cm", 0)) + distance_cm
    )

    return ActionResult(
        action="move",
        params=params,
        success=True,
        reason=None,
        world_delta={
            "robot": {"traveled_cm": world_state.robot["traveled_cm"]},
            "obstacles": moved_obstacles,
        },
        observed_at=observed_at,
    )


# Treat obstacles without a distance as "effectively infinitely far" for
# blockage checks — the sensor already caps at LIDAR_MAX_RANGE_CM when reporting.
_FAR = 10**9


# Clockwise ordering of compass-style buckets the robot can face.
# rotate(90) advances one step along this list, wrapping at the end.
_DIRS_CLOCKWISE = ["front", "right", "back", "left"]

# Scenario YAMLs use "forward"/"backward"; normalize to the canonical buckets.
_FACING_ALIASES = {"forward": "front", "backward": "back"}


def rotate(
    world_state: WorldState, degrees: int, *, observed_at: datetime
) -> ActionResult:
    """Rotate the robot in-place by `degrees` (positive = clockwise).

    Only exact multiples of 90° are supported in this mock — the robot's
    `facing` is one of {front, right, back, left}, so sub-quarter turns have
    no meaningful bucket. `front`/`back` accept the legacy `forward`/`backward`
    aliases coming from the scenario YAMLs.
    """
    params: dict[str, Any] = {"degrees": degrees}

    normalized = degrees % 360
    if normalized % 90 != 0:
        return ActionResult(
            action="rotate",
            params=params,
            success=False,
            reason="unsupported_angle:only_multiples_of_90",
            world_delta={},
            observed_at=observed_at,
        )

    current = world_state.robot.get("facing", "front")
    current = _FACING_ALIASES.get(current, current)
    if current not in _DIRS_CLOCKWISE:
        return ActionResult(
            action="rotate",
            params=params,
            success=False,
            reason=f"unknown_facing:{current}",
            world_delta={},
            observed_at=observed_at,
        )

    quarter_turns = normalized // 90
    new_idx = (_DIRS_CLOCKWISE.index(current) + quarter_turns) % 4
    new_facing = _DIRS_CLOCKWISE[new_idx]

    world_state.robot["facing"] = new_facing

    return ActionResult(
        action="rotate",
        params=params,
        success=True,
        reason=None,
        world_delta={"robot": {"facing": new_facing}},
        observed_at=observed_at,
    )


def grasp(
    world_state: WorldState, target_id: str, *, observed_at: datetime
) -> ActionResult:
    """Pick up the object `target_id` if it exists and is within reach.

    "Within reach" in this mock means the object has an `in_view_of` entry
    (it is in front of one of the robot's sensors). The robot can hold one
    object at a time; on success `robot.holding` is set and the object is
    marked `held_by: robot`.
    """
    params: dict[str, Any] = {"target_id": target_id}

    def _fail(reason: str) -> ActionResult:
        return ActionResult(
            action="grasp",
            params=params,
            success=False,
            reason=reason,
            world_delta={},
            observed_at=observed_at,
        )

    holding = world_state.robot.get("holding")
    if holding:
        return _fail(f"already_holding:{holding}")

    target = next((o for o in world_state.objects if o.get("id") == target_id), None)
    if target is None:
        return _fail("target_not_found")

    if not target.get("in_view_of"):
        return _fail("target_not_in_reach")

    world_state.robot["holding"] = target_id
    target["held_by"] = "robot"

    return ActionResult(
        action="grasp",
        params=params,
        success=True,
        reason=None,
        world_delta={
            "robot": {"holding": target_id},
            "objects": [{"id": target_id, "held_by": "robot"}],
        },
        observed_at=observed_at,
    )