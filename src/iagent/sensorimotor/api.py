"""SensorimotorAPI — the SOLE public entrypoint for the sensorimotor layer.

Signatures frozen at end of Phase 1. Placeholder impls raise NotImplementedError.
Owner: Person 2 (Phase 3).
"""

from __future__ import annotations

from typing import Literal

from iagent.sensorimotor.models import ActionResult, SensorPayload

__all__ = ["SensorimotorAPI", "SensorPayload", "ActionResult"]


class SensorimotorAPI:
    """Facade wrapping a `MockEnvironment`."""

    def __init__(self, env=None) -> None:
        # EMPTY PLACEHOLDER — IMPLEMENTATION LATER (Phase 3, Person 2).
        self._env = env

    # ---------- sensors ----------

    def sense_lidar(
        self,
        direction: Literal["front", "left", "right", "back"] = "front",
    ) -> SensorPayload:
        raise NotImplementedError("Phase 3 — see PROJECT_PLAN.md section G.2.")

    def sense_camera(
        self,
        direction: Literal["front", "left", "right", "back"] = "front",
    ) -> SensorPayload:
        raise NotImplementedError("Phase 3 — see PROJECT_PLAN.md section G.2.")

    def sense_clock(self) -> SensorPayload:
        raise NotImplementedError("Phase 3 — see PROJECT_PLAN.md section G.2.")

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
        raise NotImplementedError("Phase 3 — debug helper.")
