# Ghost-block timeline dashboard

`web/public/ghost.html` is a standalone page served by the web app's Vite dev server at **http://localhost:5173/ghost.html** (`cd web && npm run dev`). It needs no backend, and the Ops co-pilot header links to it as **Ghost timeline**. You can also open the file directly in a browser. It simulates Warehouse 7 hour by hour. The junctions, segments and shelves come from `configs/warehouse.yaml`.

There are two kinds of box:

- **Real box:** what is actually on a segment. The robot is never told. It finds out only with a LiDAR scan or by bumping into it. Toggle "Show real world" to see these.
- **Ghost box:** a belief in the declarative layer that the segment is blocked. It's faded according to its decayed confidence, and the number on it is the confidence now. A ghost box may or may not still be real.

## Using it

1. **Build hour 0.**
   - Drag *Real box*, *Ghost box* or *"Clear" report* from the bar above the map onto any segment. Pick the reporting source and its confidence in the bar.
   - Drag boxes along a segment to move them, or onto 🗑 to remove them.
   - Drag the robot **R** to its starting junction.
2. **Start the timeline.** The robot never drives on its own. At every hour you can:
   - **Continue** to the next hour. Every belief decays at its source's rate.
   - **Run a LiDAR scan** from the robot's junction. The scan is *sensed now* (sensorimotor layer) and shown as **pending**; nothing in memory changes yet. When you **continue to the next hour** it is written to the declarative layer. That hour then shows which stored ratings changed (amber, `old→new` on the map), a before/after table, and the new predicted path, with the previous path dotted in grey.
   - **Drag the robot R** to any junction to scan from there. This only repositions it; nothing is driven.
   - **Drag in new boxes.** A real box changes the world without telling the robot; a ghost box is a report from the chosen source.
   - **Preview the route** to a shelf or junction: pick it in the Robot tab or click it on the map. The blue route and its explanation (detour, segments steered around, riskiest segment) update after every scan, change or hour.
   - **Ask the robot** in the ask bar, e.g. "why this path?", "what did the scan change?", "is C-D to D-D blocked?", "which ghost boxes can I trust?" or "what changed since hour 1?". The Ask tab shows the answer and the reasoning trace (tool calls and observations).
3. **Read the map.** **Labels** shows each segment's confidence as stored in the declarative layer: the winning belief, decayed to the current hour. Green means it says clear, red means it says blocked, and — means no belief. **Explain** adds the full reasoning to the segment cards. **Real world** shows the hidden boxes.
4. **Reset** (↺ in the header, click twice) returns to hour 0 as it was when you pressed Start. Your hour-0 boxes and the changes scheduled before Start stay; scans, robot placements and anything added during the timeline are cleared.
5. **Scrub back** to any earlier hour (read-only). Use **Export YAML** to get the hour-0 scenario as a `warehouse.yaml` plus the timeline as an `events.yaml`-style list.

## Who answers the questions

- **Built-in grounded agent (default).** Works offline. It answers only from tool calls on the simulation, using the same tool names as the project's ReAct loop: `get_belief`, `query_perspective`, `query_provenance`, `list_sources`, plus `explain_route`, `last_scan`, `list_ghost_blocks`, `recent_changes` and `get_robot_state`.
- **Live model (optional).** Under Settings, add a Groq API key. The model gets the same tools and must call them before answering. If the call fails, the built-in agent answers instead. The key is stored only in this browser's local storage.

## Rules it uses

| What | Rule | Where it comes from |
|---|---|---|
| Decay | `max(0, confidence − decay_rate × age_hours)` | `declarative/decay.py` |
| Trust and decay per source | Editable under Settings | `initial_provenance.sql` + `warehouse/service.py` |
| LiDAR confidence | 0.95 up to 50 cm, falling linearly to 0.50 at 400 cm | `sensorimotor/sensors.py` |
| Conflicts | Highest *trust × decayed confidence* wins. The project's current rule (trust prior only) can be switched on under Settings | |
| P(blocked) | 50% ± half the winning belief's decayed confidence. A fully decayed belief means "unknown", not "clear" | |
| Route | Shortest path where riskier segments cost more; segments above the avoid threshold (default 75%) are skipped. The explanation gives the detour compared with the shortest possible route, and names the risky segments it steered around | |

The map scale is 2 cm per pixel of the map, so segments are 3.0–3.6 m long and within LiDAR range. `at_hour` triggers and the `truth` / `world_*` / `place_robot` / `lidar_scan` entries in the export are new; the Python backend doesn't read them yet.
