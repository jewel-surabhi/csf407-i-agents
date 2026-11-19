"""Unit tests for sensor primitives. Owner: Person 2.

Scenario A only exercises the LiDAR — camera and clock land in a later commit.
"""

from __future__ import annotations

from pathlib import Path

from iagent.sensorimotor.api import SensorimotorAPI
from iagent.sensorimotor.mock_env import MockEnvironment
from iagent.sensorimotor.models import SensorPayload, WorldState
from iagent.sensorimotor.sensors import (
    LIDAR_MAX_CONFIDENCE,
    LIDAR_MIN_CONFIDENCE,
    lidar_confidence,
    read_lidar,
)

SCENARIO_A = Path(__file__).resolve().parents[2] / "configs" / "scenario_a.yaml"


# --- LiDAR ---------------------------------------------------------------


def test_lidar_scenario_a_reports_blocked_at_12cm():
    env = MockEnvironment.from_scenario(SCENARIO_A)
    api = SensorimotorAPI(env)

    reading = api.sense_lidar("front")

    assert isinstance(reading, SensorPayload)
    assert reading.sensor == "lidar_front"
    assert reading.kind == "distance_cm"
    assert reading.value == 12
    assert reading.status == "blocked"
    assert reading.confidence == 0.95
    # The obstacle_id lets the LLM cite what it saw.
    assert reading.raw and reading.raw["obstacle_id"] == "crate_A"


def test_lidar_is_deterministic():
    env = MockEnvironment.from_scenario(SCENARIO_A)
    api = SensorimotorAPI(env)

    a = api.sense_lidar("front").model_dump()
    b = api.sense_lidar("front").model_dump()
    assert a == b


def test_lidar_clear_when_no_obstacle_in_direction():
    env = MockEnvironment.from_scenario(SCENARIO_A)
    api = SensorimotorAPI(env)

    # Scenario A only puts a crate in front of the robot.
    reading = api.sense_lidar("left")

    assert reading.sensor == "lidar_left"
    assert reading.status == "clear"
    assert reading.value >= 30  # anything past the block threshold reads clear


def test_lidar_status_flips_when_obstacle_is_beyond_threshold():
    world = WorldState(
        robot={"position": "room_101", "facing": "forward"},
        obstacles=[{"id": "far_crate", "in_front_of": "robot", "distance_cm": 200}],
    )
    reading = read_lidar(world, "front", observed_at=MockEnvironment().now)

    assert reading.status == "clear"
    assert reading.value == 200


def test_lidar_picks_nearest_obstacle_when_multiple_in_direction():
    world = WorldState(
        robot={"position": "room_101"},
        obstacles=[
            {"id": "far", "in_front_of": "robot", "distance_cm": 80},
            {"id": "near", "in_front_of": "robot", "distance_cm": 20},
        ],
    )
    reading = read_lidar(world, "front", observed_at=MockEnvironment().now)

    assert reading.value == 20
    assert reading.status == "blocked"
    assert reading.raw["obstacle_id"] == "near"


# --- Camera / clock (preset-backed in Scenario B, empty elsewhere) ------


def test_camera_returns_preset_in_scenario_b():
    from pathlib import Path
    scenario_b = Path(SCENARIO_A).parent / "scenario_b.yaml"
    api = SensorimotorAPI(MockEnvironment.from_scenario(scenario_b))
    reading = api.sense_camera("front")
    assert reading.sensor == "camera_front"
    assert reading.status == "ok"
    # scenario_b seeds box_01 perceived as brown under yellow lighting.
    detections = reading.raw["detections"]
    assert any(d["item_id"] == "box_01" and d["color_perceived"] == "brown" for d in detections)
    assert reading.raw["ambient"]["lighting"] == "yellow"


def test_camera_empty_when_no_preset():
    api = SensorimotorAPI(MockEnvironment.from_scenario(SCENARIO_A))
    reading = api.sense_camera("front")
    assert reading.status == "empty"
    assert reading.raw["detections"] == []


def test_clock_returns_now():
    api = SensorimotorAPI(MockEnvironment.from_scenario(SCENARIO_A))
    reading = api.sense_clock()
    assert reading.sensor == "clock"
    assert reading.status == "ok"
    assert reading.value  # ISO-8601 string


# --- LiDAR distance-dependent confidence ------------------------------------


def test_lidar_confidence_full_at_close_range():
    assert lidar_confidence(0) == LIDAR_MAX_CONFIDENCE
    assert lidar_confidence(12) == LIDAR_MAX_CONFIDENCE
    assert lidar_confidence(50) == LIDAR_MAX_CONFIDENCE


def test_lidar_confidence_floor_at_and_beyond_max_range():
    assert lidar_confidence(400) == LIDAR_MIN_CONFIDENCE
    assert lidar_confidence(1000) == LIDAR_MIN_CONFIDENCE


def test_lidar_confidence_decreases_monotonically_with_distance():
    values = [lidar_confidence(d) for d in range(0, 401, 10)]
    assert all(a >= b for a, b in zip(values, values[1:]))
    assert values[0] > values[-1]


def test_lidar_confidence_linear_midpoint():
    # Halfway between 50 and 400 cm -> halfway between 0.95 and 0.50.
    assert lidar_confidence(225) == 0.725


def test_lidar_far_obstacle_reading_has_lower_confidence():
    world = WorldState(
        robot={"position": "room_101"},
        obstacles=[{"id": "far_crate", "in_front_of": "robot", "distance_cm": 200}],
    )
    reading = read_lidar(world, "front", observed_at=MockEnvironment().now)
    assert reading.confidence == lidar_confidence(200)
    assert reading.confidence < LIDAR_MAX_CONFIDENCE


def test_lidar_no_obstacle_reports_minimum_confidence():
    env = MockEnvironment.from_scenario(SCENARIO_A)
    reading = SensorimotorAPI(env).sense_lidar("left")
    assert reading.confidence == LIDAR_MIN_CONFIDENCE
