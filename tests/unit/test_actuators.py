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


# --- move_forward --------------------------------------------------------


def test_act_move_forward_blocked_by_scenario_a_crate():
    env = MockEnvironment.from_scenario(SCENARIO_A)
    api = SensorimotorAPI(env, enable_actuators=True)

    # Scenario A has crate_A at 12cm in front — any move ≥ 12cm must fail.
    result = api.act_move(20, direction="forward")

    assert result.success is False
    assert result.reason == "blocked_by:crate_A"
    # No state change on failure.
    assert env.world_state.obstacles[0]["distance_cm"] == 12
    assert "traveled_cm" not in env.world_state.robot


def test_act_move_forward_clear_updates_robot_traveled_cm():
    # Empty world — no obstacles. Moving should succeed and record displacement.
    api = SensorimotorAPI(MockEnvironment(), enable_actuators=True)

    result = api.act_move(50, direction="forward")

    assert result.success is True
    assert result.reason is None
    assert result.world_delta["robot"]["traveled_cm"] == 50


def test_act_move_forward_shrinks_obstacle_distance_for_next_sensor_read():
    # Scenario A crate sits at 12cm. If the robot moves 5cm, the next LiDAR
    # read should see the crate at 7cm — i.e. the actuator updates world state
    # that sensors will subsequently observe.
    env = MockEnvironment.from_scenario(SCENARIO_A)
    api = SensorimotorAPI(env, enable_actuators=True)

    result = api.act_move(5, direction="forward")
    assert result.success is True

    lidar = api.sense_lidar("front")
    assert lidar.value == 7
    assert lidar.status == "blocked"  # still within the 30cm block threshold


def test_act_move_zero_distance_is_a_successful_no_op():
    env = MockEnvironment.from_scenario(SCENARIO_A)
    api = SensorimotorAPI(env, enable_actuators=True)

    result = api.act_move(0, direction="forward")

    assert result.success is True
    assert result.reason == "no_op"
    assert result.world_delta == {}
    # Crate distance untouched.
    assert env.world_state.obstacles[0]["distance_cm"] == 12


def test_act_move_forward_respects_feature_flag_still_off_when_unset():
    # Sanity check that Day 2's wiring did not accidentally flip the default.
    api = SensorimotorAPI(MockEnvironment.from_scenario(SCENARIO_A))
    result = api.act_move(5)
    assert result.success is False
    assert result.reason == "actuators_disabled"