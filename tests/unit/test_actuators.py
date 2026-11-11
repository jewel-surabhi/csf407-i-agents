"""Unit tests for actuators + their feature-flag plumbing. Owner: Person 2.

Actuator implementations themselves (move_forward, rotate, grasp) land in
their own commits; this file grows alongside them.
"""

from __future__ import annotations

from pathlib import Path

from iagent.sensorimotor.api import SensorimotorAPI
from iagent.sensorimotor.mock_env import MockEnvironment
from iagent.sensorimotor.models import ActionResult

SCENARIO_A = Path(__file__).resolve().parents[2] / "configs" / "scenario_a.yaml"


# --- feature flag --------------------------------------------------------


def test_actuators_disabled_by_default_act_move():
    api = SensorimotorAPI(MockEnvironment.from_scenario(SCENARIO_A))

    result = api.act_move(10, direction="forward")

    assert isinstance(result, ActionResult)
    assert result.action == "move"
    assert result.success is False
    assert result.reason == "actuators_disabled"
    assert result.params == {"distance_cm": 10, "direction": "forward"}
    assert result.world_delta == {}


def test_actuators_disabled_by_default_act_rotate():
    api = SensorimotorAPI(MockEnvironment.from_scenario(SCENARIO_A))

    result = api.act_rotate(90)

    assert result.action == "rotate"
    assert result.success is False
    assert result.reason == "actuators_disabled"
    assert result.params == {"degrees": 90}


def test_actuators_disabled_by_default_act_grasp():
    api = SensorimotorAPI(MockEnvironment.from_scenario(SCENARIO_A))

    result = api.act_grasp("box_01")

    assert result.action == "grasp"
    assert result.success is False
    assert result.reason == "actuators_disabled"
    assert result.params == {"target_id": "box_01"}


def test_disabled_response_does_not_mutate_world():
    env = MockEnvironment.from_scenario(SCENARIO_A)
    api = SensorimotorAPI(env)
    before = env.describe()

    api.act_move(100)
    api.act_rotate(180)
    api.act_grasp("ghost")

    assert env.describe() == before