"""SensorimotorAPI — the SOLE public entrypoint for the sensorimotor layer.

Signatures were frozen at end of Phase 1 (see PROJECT_PLAN.md section G.2).
Owner: Person 2 (Phase 3).

Only `sense_lidar` is implemented in this pass — Scenario A only needs the
LiDAR. Camera, clock and actuators still raise `NotImplementedError` and land
in later commits.
"""

from __future__ import annotations

from typing import Literal

from iagent.sensorimotor import actuators, sensors
from iagent.sensorimotor.mock_env import MockEnvironment
from iagent.sensorimotor.models import ActionResult, SensorPayload

__all__ = ["SensorimotorAPI", "SensorPayload", "ActionResult"]


class SensorimotorAPI:
    """Facade wrapping a `MockEnvironment`.

    `enable_actuators` is OFF by default so the 10 epistemic tests stay
    deterministic — any act_* call returns a well-shaped "disabled" response.
    Tests / scenarios that need real actuators pass `enable_actuators=True`.
    """

    def __init__(
        self,
        env: MockEnvironment | None = None,
        *,
        enable_actuators: bool = False,
    ) -> None:
        self._env: MockEnvironment = env or MockEnvironment()
        self._enable_actuators: bool = enable_actuators

    # ---------- sensors ----------

    def sense_lidar(
        self,
        direction: Literal["front", "left", "right", "back"] = "front",
    ) -> SensorPayload:
        return sensors.read_lidar(
            self._env.world_state, direction, observed_at=self._env.now
        )

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
            return actuators.disabled_result(
                "move", params, observed_at=self._env.now
            )
        # Real implementations land per-actuator in follow-up commits.
        return actuators.move_forward(
            self._env.world_state, distance_cm, observed_at=self._env.now
        )

    def act_rotate(self, degrees: int) -> ActionResult:
        params = {"degrees": degrees}
        if not self._enable_actuators:
            return actuators.disabled_result(
                "rotate", params, observed_at=self._env.now
            )
        return actuators.rotate(
            self._env.world_state, degrees, observed_at=self._env.now
        )

    def act_grasp(self, target_id: str) -> ActionResult:
        params = {"target_id": target_id}
        if not self._enable_actuators:
            return actuators.disabled_result(
                "grasp", params, observed_at=self._env.now
            )
        return actuators.grasp(
            self._env.world_state, target_id, observed_at=self._env.now
        )

    # ---------- debug (never exposed to the LLM) ----------

    def describe_world(self) -> dict:
        return self._env.describe()