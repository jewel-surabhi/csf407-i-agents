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


def rotate(
    world_state: WorldState, degrees: int, *, observed_at: datetime
) -> ActionResult:
    raise NotImplementedError("Lands in the rotate commit.")


def grasp(
    world_state: WorldState, target_id: str, *, observed_at: datetime
) -> ActionResult:
    raise NotImplementedError("Lands in the grasp commit.")