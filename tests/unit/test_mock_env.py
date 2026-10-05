"""Unit tests for MockEnvironment + scenario loader. Owner: Person 2."""

from __future__ import annotations

from pathlib import Path

from iagent.sensorimotor.mock_env import MockEnvironment
from iagent.sensorimotor.models import WorldState
from iagent.sensorimotor.scenarios import load_scenario

SCENARIO_A = Path(__file__).resolve().parents[2] / "configs" / "scenario_a.yaml"


def test_load_scenario_a_builds_world_state():
    world, scenario = load_scenario(SCENARIO_A)

    assert isinstance(world, WorldState)
    assert world.robot == {"position": "room_101", "facing": "forward"}
    assert world.lighting == "normal"
    assert world.objects == []
    assert len(world.obstacles) == 1

    crate = world.obstacles[0]
    assert crate["id"] == "crate_A"
    assert crate["in_front_of"] == "robot"
    assert crate["distance_cm"] == 12

    # Non-world blocks stay in the raw scenario dict.
    assert scenario["id"] == "scenario_a"
    assert "seed_beliefs" in scenario


def test_mock_environment_from_scenario_a():
    env = MockEnvironment.from_scenario(SCENARIO_A)

    assert env.world_state.lighting == "normal"
    assert env.world_state.obstacles[0]["distance_cm"] == 12
    # Deterministic clock so sensor readings are reproducible.
    assert env.now.isoformat().startswith("2026-09-06T10:15:00")


def test_describe_world_is_a_plain_dict():
    env = MockEnvironment.from_scenario(SCENARIO_A)
    snapshot = env.describe()

    assert isinstance(snapshot, dict)
    assert "world" in snapshot and "now" in snapshot
    assert snapshot["world"]["obstacles"][0]["id"] == "crate_A"


def test_empty_environment_has_safe_defaults():
    env = MockEnvironment()
    assert env.world_state.robot == {}
    assert env.world_state.obstacles == []
    assert env.world_state.lighting == "normal"
