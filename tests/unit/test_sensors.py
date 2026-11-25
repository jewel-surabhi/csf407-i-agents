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


def test_lidar_reports_far_obstacle_as_blocked_with_lower_confidence():
    world = WorldState(
        robot={"position": "room_101", "facing": "forward"},
        obstacles=[{"id": "far_crate", "in_front_of": "robot", "distance_cm": 200}],
    )
    reading = read_lidar(world, "front", observed_at=MockEnvironment().now)

    # No 30 cm cut-off: anything detected in range is blocked; distance lowers confidence.
    assert reading.status == "blocked"
    assert reading.value == 200
    assert reading.confidence == lidar_confidence(200) < lidar_confidence(12)


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



# --- sweeps --------------------------------------------------------------


def _grid_world(**ranges):
    return WorldState(
        obstacles=[
            {"id": "box_e", "in_right_of": "robot", "distance_cm": 250, "color": "red"},
            {"id": "box_s", "in_back_of": "robot", "distance_cm": 50, "color": "blue"},
        ],
        extras={"cell_cm": 100, "beam_range_cm": ranges},
    )


def test_lidar_sweep_reads_every_cell_until_the_first_obstacle():
    from iagent.sensorimotor.sensors import read_lidar_sweep

    w = _grid_world(front=0, right=400, back=300, left=0)
    beams = read_lidar_sweep(w, observed_at=MockEnvironment().now).raw["beams"]
    right = beams["right"]
    assert [c["status"] for c in right] == ["clear", "clear", "blocked"]
    assert [c["confidence"] for c in right] == [lidar_confidence(100), lidar_confidence(200), lidar_confidence(250)]
    assert beams["back"] == [beams["back"][0]] and beams["back"][0]["status"] == "blocked"
    assert beams["front"] == [] and beams["left"] == []


def test_lidar_sweep_clear_beam_runs_to_the_edge():
    from iagent.sensorimotor.sensors import read_lidar_sweep

    w = WorldState(extras={"cell_cm": 100, "beam_range_cm": {"front": 300}})
    front = read_lidar_sweep(w, observed_at=MockEnvironment().now).raw["beams"]["front"]
    assert [c["to_cm"] for c in front] == [100, 200, 300]
    assert all(c["status"] == "clear" for c in front)


def test_camera_sweep_sees_colour_of_first_box_with_distance_confidence():
    from iagent.sensorimotor.sensors import camera_confidence, read_camera_sweep

    w = _grid_world(front=0, right=400, back=300, left=0)
    det = {d["direction"]: d for d in read_camera_sweep(w, observed_at=MockEnvironment().now).detections}
    assert det["right"]["color"] == "red" and det["back"]["color"] == "blue"
    assert det["back"]["confidence"] == camera_confidence(50) == 0.98
    assert det["right"]["confidence"] == camera_confidence(250) < 0.98
    assert camera_confidence(400) == 0.70
