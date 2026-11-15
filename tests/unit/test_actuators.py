"""Unit tests for actuators + their feature-flag plumbing. Owner: Person 2.

Actuator implementations themselves (move_forward, rotate, grasp) land in
their own commits; this file grows alongside them.
"""

from __future__ import annotations

from pathlib import Path

from iagent.sensorimotor.api import SensorimotorAPI
from iagent.sensorimotor.mock_env import MockEnvironment
from iagent.sensorimotor.models import ActionResult, WorldState

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


# --- rotate --------------------------------------------------------------


def test_act_rotate_90_clockwise_from_forward():
    # Scenario A seeds robot.facing = "forward" (alias for "front").
    env = MockEnvironment.from_scenario(SCENARIO_A)
    api = SensorimotorAPI(env, enable_actuators=True)

    result = api.act_rotate(90)

    assert result.success is True
    assert result.world_delta == {"robot": {"facing": "right"}}
    assert env.world_state.robot["facing"] == "right"


def test_act_rotate_180_flips_direction():
    env = MockEnvironment.from_scenario(SCENARIO_A)
    api = SensorimotorAPI(env, enable_actuators=True)

    result = api.act_rotate(180)

    assert result.success is True
    assert env.world_state.robot["facing"] == "back"


def test_act_rotate_two_90_turns_compose_to_180():
    env = MockEnvironment.from_scenario(SCENARIO_A)
    api = SensorimotorAPI(env, enable_actuators=True)

    api.act_rotate(90)
    api.act_rotate(90)

    assert env.world_state.robot["facing"] == "back"


def test_act_rotate_360_returns_to_original_facing():
    env = MockEnvironment.from_scenario(SCENARIO_A)
    api = SensorimotorAPI(env, enable_actuators=True)

    api.act_rotate(360)

    # 360° is a full turn → back to front.
    assert env.world_state.robot["facing"] == "front"


def test_act_rotate_negative_90_goes_counterclockwise():
    env = MockEnvironment.from_scenario(SCENARIO_A)
    api = SensorimotorAPI(env, enable_actuators=True)

    result = api.act_rotate(-90)

    assert result.success is True
    # -90° clockwise == 90° counter-clockwise: front → left.
    assert env.world_state.robot["facing"] == "left"


def test_act_rotate_rejects_non_multiples_of_90():
    env = MockEnvironment.from_scenario(SCENARIO_A)
    api = SensorimotorAPI(env, enable_actuators=True)

    result = api.act_rotate(45)

    assert result.success is False
    assert result.reason == "unsupported_angle:only_multiples_of_90"
    # No mutation on failure.
    assert env.world_state.robot["facing"] == "forward"


# --- grasp ---------------------------------------------------------------

SCENARIO_B = SCENARIO_A.parent / "scenario_b.yaml"


def test_act_grasp_picks_up_box_in_view_in_scenario_b():
    # Scenario B puts box_01 in view of camera_front.
    env = MockEnvironment.from_scenario(SCENARIO_B)
    api = SensorimotorAPI(env, enable_actuators=True)

    result = api.act_grasp("box_01")

    assert result.success is True
    assert result.world_delta["robot"] == {"holding": "box_01"}
    assert env.world_state.robot["holding"] == "box_01"
    assert env.world_state.objects[0]["held_by"] == "robot"


def test_act_grasp_fails_for_unknown_target():
    env = MockEnvironment.from_scenario(SCENARIO_B)
    api = SensorimotorAPI(env, enable_actuators=True)

    result = api.act_grasp("ghost_box")

    assert result.success is False
    assert result.reason == "target_not_found"
    assert "holding" not in env.world_state.robot


def test_act_grasp_fails_when_target_not_in_reach():
    world = WorldState(
        robot={"position": "room_101"},
        objects=[{"id": "far_box"}],  # exists, but not in view of any sensor
    )
    api = SensorimotorAPI(MockEnvironment(world), enable_actuators=True)

    result = api.act_grasp("far_box")

    assert result.success is False
    assert result.reason == "target_not_in_reach"


def test_act_grasp_fails_when_already_holding_something():
    env = MockEnvironment.from_scenario(SCENARIO_B)
    api = SensorimotorAPI(env, enable_actuators=True)

    api.act_grasp("box_01")
    result = api.act_grasp("box_01")

    assert result.success is False
    assert result.reason == "already_holding:box_01"