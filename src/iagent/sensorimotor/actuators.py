"""Actuators mutate world state and return ActionResult.

Disabled by default (`SensorimotorAPI(enable_actuators=False)`) so the
epistemic test suite stays deterministic. Individual actuators are added
per-commit (move_forward -> rotate -> grasp); the uniform `disabled_result`
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
    """Uniform ActionResult returned when actuators are feature-flagged off."""
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
    raise NotImplementedError("Lands in the move_forward commit.")


def rotate(
    world_state: WorldState, degrees: int, *, observed_at: datetime
) -> ActionResult:
    raise NotImplementedError("Lands in the rotate commit.")


def grasp(
    world_state: WorldState, target_id: str, *, observed_at: datetime
) -> ActionResult:
    raise NotImplementedError("Lands in the grasp commit.")