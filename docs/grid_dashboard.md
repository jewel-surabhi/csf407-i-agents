# Scenario A grid dashboard

Open <http://localhost:8000/grid> (served by the FastAPI backend). Every question runs the
real Groq ReAct loop, so `GROQ_API_KEY` must be set in `.env`.

## World

* 5 x 4 nodes, columns A-E and rows 1-4, 100 cm apart; 31 paths named like `A1-B1`.
* The robot and the target sit on nodes; nothing is decided about nodes.
* A box sits at the middle of a path and blocks it. Boxes are red, blue or yellow. The
  colour is only a label that identifies the box; it does not change routing.

## How each map is made

| Map | Layer | Code |
|---|---|---|
| Declaration map | declarative (`KnowledgeAPI`) | `layer_readers/declaration_map.py` |
| Sensor map | sensorimotor (`SensorimotorAPI`) | `layer_readers/sensor_map.py` |
| Total map | procedural (LLM) | `procedural/grid_task.py`, read by `decision/total_map.py` |

**Declaration map.** Each path is a subject with `is_blocked` and, while a box is
believed there, `box_color`. Confidence is decayed to the shown hour by the source's rate.

**Sensor map.** The grid is converted into the sensorimotor layer's own input, a
`WorldState`. The robot faces north, so north, east, south and west are front, right, back
and left, and each box on the robot's row or column is an obstacle at its distance with its
colour. Two sensor sweeps then read it:

* `sense_lidar_sweep` reads the robot's whole row and column. Every path up to the first box
  gets a reading: clear paths at the distance to their far end, the box's path at the box's
  distance. The beam stops at the box. There is no distance cut-off: anything detected is
  blocked, and distance only lowers the confidence.
* `sense_camera_sweep` gives the colour of the first box in each direction.

**Total map.** The LLM reads the other two maps through tools (`get_belief`,
`sense_lidar`, `sense_camera`) and submits every path in one `submit_total_map` call. For
each path, the state and the colour each go to whichever side has the higher confidence. The
Total map is exactly what the LLM submitted. Code does not change it, and paths it left out
show as unknown.

## Confidence

Confidence is a 0 to 1 rating of how sure a source is of the value it states (blocked or
clear, or a colour). It is not a blocked-to-open scale.

| Source | Confidence | Fade per hour |
|---|---|---|
| Operator's first declaration (`operator_setup`), path and colour | 0.90 | 0.20 |
| Live LiDAR | 0.95 up to 50 cm, falling to 0.50 at 400 cm | not stored |
| Live camera | 0.98 up to 50 cm, falling to 0.70 at 400 cm | not stored |
| LiDAR reading stored in the declaration layer (`lidar_declared`) | its live value | 0.15 |
| Camera colour stored in the declaration layer (`camera_declared`) | its live value | 0.15 |

`effective = max(0, stored - fade_per_hour * age_hours)`. The three grid sources are
registered in `grid/sources.py`; the project's own sources are unchanged.

## Flow

1. **Setup (hour 0).** Place the robot, the target and the coloured boxes (click a path
   with a colour selected; click again with the same colour to remove). *Complete setup*
   declares every path's state and every box's colour as the operator.
2. **Ask.** Choose +X hours (default 1) and ask a free-text question. The clock moves to
   clock + X. The declaration map is read as it stood before the scan, the sensors read the
   real world, and the LLM submits the Total map and answers.
3. **Write-back.** Where the LLM chose the sensor and the reading is news to the declaration
   layer (it is new, disagrees, or is more confident), the sensor's reading is stored:
   state as `lidar_declared`, colour as `camera_declared`. A path the LLM decided is clear
   from the sensor loses its declared colour.
4. **Change setup** returns to the setup phase at the current hour. *Complete setup* then
   changes only the real world; the next question's sensors find the changes.

## Display

* Grey line: clear. Dark line: blocked. Dashed: unknown, or not in line of sight.
* A blocked path shows the box in the colour that map believes, with the colour confidence
  on the box and the path confidence in the chip beside it. "?" means the colour is unknown.
* On the Total map, the route is highlighted and a teal chip marks a path written back.
* Click any path to compare its value in all three maps.

## Timeline

The strip above the map lists the setup and every answered question. Click one to see its
three maps, question, answer and agent steps as they were. Snapshots are read-only; *Back to
now* returns to the live view.
