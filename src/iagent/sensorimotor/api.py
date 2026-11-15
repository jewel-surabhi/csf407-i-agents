"""SensorimotorAPI — the SOLE public entrypoint for the sensorimotor layer.

Signatures were frozen at end of Phase 1 (see PROJECT_PLAN.md section G.2).
Owner: Person 2 (Phase 3).

Sensors (`sense_lidar`, `sense_camera`, `sense_clock`) are always available.
Actuators (`act_move`, `act_rotate`, `act_grasp`) run when called. Whether the
LLM may call them at all is decided by `features.enable_actuators` in the
procedural dispatcher (off by default, so the epistemic tests stay
deterministic). Building the API with `enable_actuators=False` is an extra
hard switch: every act_* call then returns `success=False,
reason="actuators_disabled"` and changes nothing.
"""

from __future__ import annotations

from typing import Literal

from iagent.sensorimotor import actuators, sensors
from iagent.sensorimotor.mock_env import MockEnvironment
from iagent.sensorimotor.models import ActionResult, SensorPayload

__all__ = ["SensorimotorAPI", "SensorPayload", "ActionResult"]


class SensorimotorAPI:
    """Facade wrapping a `MockEnvironment`.

    Pass `enable_actuators=False` to hard-disable the act_* methods; they then
    return a well-shaped "disabled" response instead of touching the world.
    """

    def __init__(
        self,
        env: MockEnvironment | None = None,
        *,
        enable_actuators: bool = True,
    ) -> None:
        self._env: MockEnvironment = env or MockEnvironment()
        self._enable_actuators: bool = enable_actuators

    # ---------- sensors ----------

    def sense_lidar(
        self,
        direction: Literal["front", "left", "right", "back"] = "front",
    ) -> SensorPayload:
        return sensors.read_lidar(self._env.world_state, direction, observed_at=self._env.now)

    def sense_camera(
        self,
        direction: Literal["front", "left", "right", "back"] = "front",
    ) -> SensorPayload:
        return sensors.read_camera(
            self._env.world_state,
            direction,
            observed_at=self._env.now,
            presets=self._env.sensor_presets,
        )

    def sense_clock(self) -> SensorPayload:
        return sensors.read_clock(observed_at=self._env.now)

    # ---------- actuators (feature-flagged off for epistemic tests) ----------

    def act_move(
        self,
        distance_cm: int,
        direction: Literal["forward", "backward"] = "forward",
    ) -> ActionResult:
        params = {"distance_cm": distance_cm, "direction": direction}
        if not self._enable_actuators:
            return actuators.disabled_result("move", params, observed_at=self._env.now)
        if direction != "forward":
            # Only forward motion is modelled; refuse instead of moving the wrong way.
            return ActionResult(
                action="move",
                params=params,
                success=False,
                reason=f"unsupported_direction:{direction}",
                world_delta={},
                observed_at=self._env.now,
            )
        return actuators.move_forward(self._env.world_state, distance_cm, observed_at=self._env.now)

    def act_rotate(self, degrees: int) -> ActionResult:
        params = {"degrees": degrees}
        if not self._enable_actuators:
            return actuators.disabled_result("rotate", params, observed_at=self._env.now)
        return actuators.rotate(self._env.world_state, degrees, observed_at=self._env.now)

    def act_grasp(self, target_id: str) -> ActionResult:
        params = {"target_id": target_id}
        if not self._enable_actuators:
            return actuators.disabled_result("grasp", params, observed_at=self._env.now)
        return actuators.grasp(self._env.world_state, target_id, observed_at=self._env.now)

    # ---------- debug (never exposed to the LLM) ----------

    def describe_world(self) -> dict:
        return self._env.describe()