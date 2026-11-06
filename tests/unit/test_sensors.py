"""Unit tests for sensor primitives. Owner: Person 2.

Scenario A only exercises the LiDAR — camera and clock land in a later commit.
"""

from __future__ import annotations

from pathlib import Path

from iagent.sensorimotor.api import SensorimotorAPI
from iagent.sensorimotor.mock_env import MockEnvironment
from iagent.sensorimotor.models import SensorPayload, WorldState
from iagent.sensorimotor.sensors import read_lidar

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
