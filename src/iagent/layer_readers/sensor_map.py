"""Sensor map: what the sensorimotor layer reports from the robot's node.

The grid is converted into the sensorimotor layer's own input, a
``WorldState``, and read with the unchanged ``SensorimotorAPI``:

* The robot faces north, so north/east/south/west are the sensor's
  front/right/back/left.
* Every box on the robot's row or column becomes an obstacle in that direction,
  at its distance in cm (boxes sit at the middle of a path), carrying its colour.
* ``extras`` tells the sweep how long one path is (``cell_cm``) and how far each
  beam can run before the floor ends (``beam_range_cm``).

``sense_lidar_sweep`` then reports every path along the four beams up to the
first box, and ``sense_camera_sweep`` the colour of each first box. The cells
are mapped back to path ids here.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from iagent.grid.world import SPACING_CM, GridWorld, ray
from iagent.sensorimotor.api import SensorimotorAPI
from iagent.sensorimotor.mock_env import MockEnvironment
from iagent.sensorimotor.models import WorldState

__all__ = ["COMPASS_TO_SENSOR", "grid_to_world_state", "read_sensor_map"]

COMPASS_TO_SENSOR = {"north": "front", "east": "right", "south": "back", "west": "left"}
_SENSOR_TO_COMPASS = {v: k for k, v in COMPASS_TO_SENSOR.items()}
_KEY = {"front": "in_front_of", "right": "in_right_of", "back": "in_back_of", "left": "in_left_of"}


def grid_to_world_state(world: GridWorld) -> WorldState:
    obstacles: list[dict[str, Any]] = []
    ranges: dict[str, int] = {}
    for compass, sensor_dir in COMPASS_TO_SENSOR.items():
        line = ray(world.robot, compass)
        ranges[sensor_dir] = len(line) * SPACING_CM
        for k, seg in enumerate(line):
            if seg in world.boxes:
                obstacles.append(
                    {
                        "id": f"box@{seg}",
                        _KEY[sensor_dir]: "robot",
                        "distance_cm": k * SPACING_CM + SPACING_CM // 2,
                        "color": world.boxes[seg],
                        "segment": seg,
                    }
                )
    return WorldState(
        robot={"position": world.robot, "facing": "north"},
        obstacles=obstacles,
        extras={"cell_cm": SPACING_CM, "beam_range_cm": ranges},
    )


def read_sensor_map(world: GridWorld, *, now: datetime) -> dict[str, dict[str, Any]]:
    """Per seen path: LiDAR state + confidence + distance, and camera colour where a box is seen."""
    api = SensorimotorAPI(MockEnvironment(grid_to_world_state(world), now=now), enable_actuators=False)
    lidar = api.sense_lidar_sweep()
    camera = api.sense_camera_sweep()

    out: dict[str, dict[str, Any]] = {}
    for sensor_dir, cells in (lidar.raw or {}).get("beams", {}).items():
        compass = _SENSOR_TO_COMPASS[sensor_dir]
        line = ray(world.robot, compass)
        for cell in cells:
            seg = line[cell["index"]]
            out[seg] = {
                "path": seg,
                "direction": compass,
                "state": cell["status"],
                "state_conf": cell["confidence"],
                "distance_cm": cell["distance_cm"],
                "color": None,
                "color_conf": None,
            }
    for det in camera.detections or []:
        compass = _SENSOR_TO_COMPASS[det["direction"]]
        line = ray(world.robot, compass)
        seg = line[(det["distance_cm"] - SPACING_CM // 2) // SPACING_CM]
        if seg in out:
            out[seg]["color"] = det["color"]
            out[seg]["color_conf"] = det["confidence"]
    return out
