"""SensorimotorAPI — the SOLE public entrypoint for the sensorimotor layer.

Signatures were frozen at end of Phase 1 (see PROJECT_PLAN.md section G.2).
Owner: Person 2 (Phase 3).

Only `sense_lidar` is implemented in this pass — Scenario A only needs the
LiDAR. Camera, clock and actuators still raise `NotImplementedError` and land
in later commits.
"""

from __future__ import annotations

from typing import Literal

from iagent.sensorimotor import sensors
from iagent.sensorimotor.mock_env import MockEnvironment
from iagent.sensorimotor.models import ActionResult, SensorPayload

__all__ = ["SensorimotorAPI", "SensorPayload", "ActionResult"]


class SensorimotorAPI:
    """Facade wrapping a `MockEnvironment`."""

    def __init__(self, env: MockEnvironment | None = None) -> None:
        self._env: MockEnvironment = env or MockEnvironment()

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
        raise NotImplementedError("Phase 3 — sense_camera lands with Scenario B.")

    def sense_clock(self) -> SensorPayload:
        raise NotImplementedError("Phase 3 — sense_clock lands with Scenario B.")

    # ---------- actuators (feature-flagged off for epistemic tests) ----------

    def act_move(
        self,
        distance_cm: int,
        direction: Literal["forward", "backward"] = "forward",
    ) -> ActionResult:
        raise NotImplementedError("Phase 3 — behind feature flag.")

    def act_rotate(self, degrees: int) -> ActionResult:
        raise NotImplementedError("Phase 3 — behind feature flag.")

    def act_grasp(self, target_id: str) -> ActionResult:
        raise NotImplementedError("Phase 3 — behind feature flag.")

    # ---------- debug (never exposed to the LLM) ----------

    def describe_world(self) -> dict:
        return self._env.describe()
