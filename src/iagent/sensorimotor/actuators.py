"""Actuators mutate world state and return ActionResult.

Disabled by default (config `features.enable_actuators = false`) so epistemic
tests stay deterministic. Placeholder until Phase 3 (Owner: Person 2).
"""

from __future__ import annotations

# EMPTY PLACEHOLDER — IMPLEMENTATION LATER
# Owner: Person 2 (Phase 3).


def move_forward(world_state, distance_cm: int):
    raise NotImplementedError("Phase 3 — behind feature flag.")


def rotate(world_state, degrees: int):
    raise NotImplementedError("Phase 3 — behind feature flag.")


def grasp(world_state, target_id: str):
    raise NotImplementedError("Phase 3 — behind feature flag.")
