# Scenario A grid dashboard

`web/public/grid.html`, served at <http://localhost:5173/grid.html> by the Vite dev server.
It talks to the FastAPI backend (`/api/grid/*`), and every question runs the real Groq
ReAct loop, so the backend and `GROQ_API_KEY` are required.

## World

* 5 x 4 nodes, columns A-E and rows 1-4, 100 cm apart; 31 segments named like `A1-B1`.
* The robot and the target sit on nodes. A box always sits at the middle of a segment.

## Flow

1. **Setup phase (hour 0).** Place the robot, the target and the boxes. *Complete setup*
   declares every segment into the declarative layer (`is_blocked`) as the operator
   (source `operator_setup`): confidence = that source's trust prior.
2. **Usage phase.** Choose *+X hours* (default +1) and ask a free-text question. The clock
   moves to `clock + X` and one ReAct episode runs:
   * `get_belief` reads the declarations as they stood before the scan, decayed to the
     question hour;
   * `sense_lidar` scans from the robot's node along its row and column. A beam stops at
     the first box. Confidence follows the LiDAR distance rule (0.95 up to 50 cm,
     0.50 at 400 cm);
   * `resolve_total` is code: per segment the higher confidence wins (a tie goes to the
     LiDAR), plus the cheapest route that avoids blocked segments;
   * `finalize_answer` is the LLM's answer, built from the three maps.
3. **Write-back.** Every LiDAR reading that wins the Total map is written to the declarative
   layer as a `lidar_declared` belief, replacing the weaker declaration: a *conflict* (it
   disagrees), a *refresh* (it agrees but is more confident) or *new* (nothing declared). A
   declaration that is at least as confident as the scan stays, and segments the scan did not
   see are untouched. A written reading then fades at 0.15 per hour.
4. **Change setup** goes back to the setup phase at the current hour. *Complete setup* then
   changes only the real world. Nothing is declared; the next scan finds the changes.

If the model skips `sense_lidar` or `resolve_total`, the backend runs them so the maps are
always complete, and the step list says so.

## Decay

Decay belongs to the source, the kind of information, and is read from the `sources` table
(`data/seed/initial_provenance.sql`): `effective = max(0, stored - decay_per_hour * age_hours)`.

| Declared by | Source | Trust (= stored confidence) | Decay per hour |
|---|---|---|---|
| Operator, at the first setup | `operator_setup` | 0.9 | 0.20 |
| LiDAR reading written back after a question | `lidar_declared` | distance-based, 0.50-0.95 | 0.15 |

Both sources are registered by the grid session, so the project's own `user` and `lidar_front` sources (and the warehouse demo) are unchanged. `operator_setup` starts at 0.9 and fades 0.20 per hour (it reaches 0 after 4.5 hours). `lidar_declared` has the trust of `lidar_front` (0.9) but fades at 0.15 per hour (a 0.95 reading reaches 0 after about 6.3 hours).

## Maps

| Map | Shows |
|---|---|
| Declaration | Live: the declarative layer decayed to the hour the next question will be asked at (clock + time set). In a timeline snapshot: as it stood just before that scan |
| LiDAR | The scan for the question; grey dashed = not in line of sight |
| Total | Winner per segment, the chosen route, and an outline on segments written back |

Click any segment in the usage phase to compare all three values for it.

## Timeline

The strip above the map lists every significant hour: the setup declaration and each answered
question. Click one to see its three maps, the question, the answer and the agent steps as they
were. Snapshots are read-only; *Back to now* returns to the live view where you can ask.

## Run

```powershell
# terminal 1, project root
$env:PYTHONPATH="src"          # or: pip install -e .
uvicorn iagent.api.server:app --reload --port 8000

# terminal 2
cd web
npm run dev
```

Open <http://localhost:5173/grid.html>.

## Code

* `src/iagent/grid/world.py`: grid geometry and LiDAR beams
* `src/iagent/grid/session.py`: simulated clock, declarations, merge rule, route
* `src/iagent/grid/agent.py`: the grid ReAct loop and tool specs
* `src/iagent/api/grid_routes.py`: `/api/grid/state`, `/setup`, `/ask`, `/reset`
* `tests/unit/test_grid.py`
