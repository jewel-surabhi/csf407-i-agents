"""Grid task for the procedural layer: the LLM builds the Total map.

Runs through the standard ``react_loop.run_react`` with its own prompt and
tools. The procedural layer never reads the other layers directly; the caller
(``decision.total_map``) hands it reader callables:

    read_declarations() -> {path: {...}}  declarative layer, decayed to the question hour
    read_lidar()        -> {path: {...}}  sensorimotor layer, LiDAR part of the sensor map
    read_camera()       -> {path: {...}}  sensorimotor layer, camera part of the sensor map
    history(path)       -> [...]          provenance log for one path

Tools the LLM sees:

    get_belief          declaration map (or one path + its history)
    sense_lidar         LiDAR along the robot's row and column
    sense_camera        colour of the first box in each direction
    submit_total_map    the LLM's decision for EVERY path (state + colour, with confidence)
    plan_route          cheapest route over the submitted Total map
    finalize_answer     the answer to the user's question

What the LLM submits IS the Total map; nothing in code overrides it. Entries
that are malformed are reported back so the model can resubmit; paths it never
submits are shown as unknown.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

from iagent.procedural.react_loop import MaxStepsExceeded, run_react

__all__ = [
    "GRID_SYSTEM_PROMPT",
    "GRID_TOOL_SPECS",
    "GridReaders",
    "GridEpisode",
    "run_grid_task",
    "MAX_STEPS",
]

MAX_STEPS = 8
COLORS = ("red", "blue", "yellow")
STATES = ("blocked", "clear")
ORIGINS = ("declaration", "sensor")

GRID_SYSTEM_PROMPT = """You are the procedural layer of a grounded warehouse robot.

The floor is a 5 x 4 grid of nodes, columns A-E (west to east) and rows 1-4 (north to
south). A path joins two neighbouring nodes and is named like "A1-B1" (31 paths). Nodes
are 100 cm apart. A box always sits at the middle of a path and blocks it. Every box has a
colour, red, blue or yellow, which is only a label to identify it.

You know the floor only through tools:
- get_belief: the declaration map from the declarative layer. Each path's declared state
  (blocked/clear) and, if a box is declared, its colour, each with a confidence already
  decayed to the question time.
- sense_lidar: the LiDAR sweep along the robot's row and column. Each beam stops at the
  first box; paths behind it are not seen. Confidence falls with distance.
- sense_camera: the colour of the first box in each direction, confidence falling with
  distance.
- submit_total_map: YOUR decision for every path.
- plan_route: cheapest route from the robot to the target over your submitted map.

Confidence always means "how sure the source is of the value it states" (0 to 1).

Build the Total map path by path:
- State: if the LiDAR saw the path, compare the LiDAR confidence with the declared state's
  confidence. The higher one wins (a tie goes to the sensor). If the LiDAR did not see it,
  use the declaration. If neither has it, leave it out.
- Colour: only for paths you decide are blocked. Compare the camera's colour confidence
  with the declared colour's confidence; the higher one wins (a tie goes to the sensor).
  If only one has a colour, use it. A clear path has colour null.
- Copy the winner's confidence exactly, and set state_from / color_from to "declaration"
  or "sensor".

Procedure: get_belief, sense_lidar, sense_camera, then submit_total_map with ALL 31 paths
in ONE call, then plan_route if the question is about moving, then finalize_answer
exactly once. Answer the user's question briefly from your Total map: name paths, colours
and the route as node lists, and say where the declaration and the sensors disagreed.
"""


def _fn(name: str, description: str, properties: dict | None = None, required: list | None = None) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": properties or {}, "required": required or []},
        },
    }


# Kept loose on purpose: the provider rejects a whole tool call when one field
# breaks the schema, which would throw away a 31-path submission. Values are
# checked by `_validate`, which reports problems back so the model can resubmit.
_ENTRY = {
    "type": "object",
    "properties": {
        "path": {"type": "string", "description": "Path id, e.g. 'A1-B1'."},
        "state": {"type": "string", "description": "'blocked' or 'clear'."},
        "state_conf": {"type": "number", "description": "Confidence of the winning state, 0..1."},
        "state_from": {"type": "string", "description": "'declaration' or 'sensor'."},
        "color": {"type": "string", "description": "'red', 'blue' or 'yellow'; empty for a clear path."},
        "color_conf": {"type": "number", "description": "Confidence of the winning colour; 0 if none."},
        "color_from": {"type": "string", "description": "'declaration' or 'sensor'; empty if none."},
    },
    "required": ["path", "state", "state_conf", "state_from"],
}

GRID_TOOL_SPECS: list[dict[str, Any]] = [
    _fn(
        "get_belief",
        "Declaration map at the question time. Without `path`: every declared path. With it: "
        "that path plus its provenance history.",
        {"path": {"type": "string"}},
    ),
    _fn("sense_lidar", "LiDAR sweep along the robot's row and column."),
    _fn("sense_camera", "Camera: colour of the first box in each of the four directions."),
    _fn(
        "submit_total_map",
        "Submit your Total map: one entry for every path (all 31) in a single call.",
        {"paths": {"type": "array", "items": _ENTRY}},
        ["paths"],
    ),
    _fn("plan_route", "Cheapest route from the robot to the target over your submitted Total map."),
    _fn(
        "finalize_answer",
        "Give the final answer to the user's question. Call exactly once.",
        {"answer": {"type": "string"}},
        ["answer"],
    ),
]


@dataclass
class GridReaders:
    read_declarations: Callable[[], dict[str, dict[str, Any]]]
    read_lidar: Callable[[], dict[str, dict[str, Any]]]
    read_camera: Callable[[], dict[str, dict[str, Any]]]
    history: Callable[[str], list[dict[str, Any]]]
    plan_route: Callable[[dict[str, dict[str, Any]]], dict[str, Any]]


@dataclass
class GridEpisode:
    answer: str | None = None
    total: dict[str, dict[str, Any]] | None = None
    submissions: int = 0
    issues: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    route: dict[str, Any] | None = None
    steps: list[dict[str, Any]] = field(default_factory=list)
    used: list[str] = field(default_factory=list)


# ------------------------------------------------------------------ formatting


def _decl_line(d: dict[str, Any]) -> str:
    s = d["state_meta"]
    line = (
        f"{d['path']}: {d['state']} {d['state_conf']:.2f} "
        f"({s['source']}, stored {s['stored']:.2f} at h{s['declared_at_hour']:g}, -{s['decay_per_hour']}/h)"
    )
    if d["color"]:
        c = d["color_meta"]
        line += (
            f"; box {d['color']} {d['color_conf']:.2f} "
            f"({c['source']}, stored {c['stored']:.2f} at h{c['declared_at_hour']:g}, -{c['decay_per_hour']}/h)"
        )
    return line


def _lidar_line(r: dict[str, Any]) -> str:
    return f"{r['path']}: {r['state']} {r['state_conf']:.2f} ({r['direction']}, {r['distance_cm']} cm)"


def _camera_line(r: dict[str, Any]) -> str:
    return f"{r['path']}: box {r['color']} {r['color_conf']:.2f} ({r['direction']}, {r['distance_cm']} cm)"


def _num(x: Any) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if 0.0 <= v <= 1.0 else None


def _validate(entries: Any, known: set[str]) -> tuple[dict[str, dict[str, Any]], list[str]]:
    total: dict[str, dict[str, Any]] = {}
    issues: list[str] = []
    if not isinstance(entries, list):
        return total, ["`paths` must be a list of path entries"]
    for i, e in enumerate(entries):
        if not isinstance(e, dict):
            issues.append(f"entry {i}: not an object")
            continue
        pid = str(e.get("path", "")).strip().upper()
        if pid not in known:
            parts = pid.split("-")
            pid = "-".join(sorted(parts)) if len(parts) == 2 else pid
        if pid not in known:
            issues.append(f"entry {i}: unknown path {e.get('path')!r}")
            continue
        state, conf, origin = e.get("state"), _num(e.get("state_conf")), e.get("state_from")
        if state not in STATES or conf is None or origin not in ORIGINS:
            issues.append(f"{pid}: needs state blocked|clear, state_conf 0..1, state_from declaration|sensor")
            continue
        color, cconf, corigin = e.get("color"), e.get("color_conf"), e.get("color_from")
        if state == "clear" or color in (None, "", "null", "none"):
            color, cconf, corigin = None, None, None
        else:
            cconf = _num(cconf)
            if color not in COLORS or cconf is None or corigin not in ORIGINS:
                issues.append(f"{pid}: colour needs red|blue|yellow, color_conf 0..1, color_from")
                continue
        total[pid] = {
            "path": pid,
            "state": state,
            "state_conf": round(conf, 3),
            "state_from": origin,
            "color": color,
            "color_conf": None if cconf is None else round(cconf, 3),
            "color_from": corigin,
        }
    return total, issues


# ------------------------------------------------------------------ episode


def run_grid_task(
    llm: Any,
    readers: GridReaders,
    *,
    question: str,
    context: str,
    known_paths: set[str],
    max_steps: int = MAX_STEPS,
) -> GridEpisode:
    """One ReAct episode on the grid. Returns what the LLM decided and said."""
    ep = GridEpisode()
    cache: dict[str, Any] = {}

    def once(key: str, fn: Callable[[], Any]) -> Any:
        if key not in cache:
            cache[key] = fn()
        return cache[key]

    def dispatch(name: str, args: dict[str, Any]) -> dict[str, Any]:
        ep.used.append(name)
        args = args or {}
        if name == "get_belief":
            decl = once("decl", readers.read_declarations)
            path = args.get("path")
            if path:
                pid = str(path).strip().upper()
                if pid not in decl:
                    parts = pid.split("-")
                    pid = "-".join(sorted(parts)) if len(parts) == 2 else pid
                if pid not in known_paths:
                    return {"ok": False, "error": {"kind": "unknown_path", "message": str(path)}}
                return {
                    "ok": True,
                    "data": {
                        "belief": _decl_line(decl[pid]) if pid in decl else f"{pid}: nothing declared",
                        "history": readers.history(pid),
                    },
                }
            return {"ok": True, "data": {"paths": [_decl_line(d) for d in decl.values()]}}
        if name == "sense_lidar":
            lid = once("lidar", readers.read_lidar)
            return {
                "ok": True,
                "data": {"readings": [_lidar_line(r) for r in lid.values()], "note": "Paths not listed were not seen."},
            }
        if name == "sense_camera":
            cam = once("camera", readers.read_camera)
            return {
                "ok": True,
                "data": {"detections": [_camera_line(r) for r in cam.values()] or ["no box in sight"]},
            }
        if name == "submit_total_map":
            total, issues = _validate(args.get("paths"), known_paths)
            ep.submissions += 1
            ep.total, ep.issues = total, issues
            ep.missing = sorted(known_paths - set(total))
            return {
                "ok": not issues,
                "data": {"accepted": len(total), "missing": ep.missing, "issues": issues},
                **({"error": {"kind": "invalid_entries", "message": "; ".join(issues)}} if issues else {}),
            }
        if name == "plan_route":
            if not ep.total:
                return {"ok": False, "error": {"kind": "no_total_map", "message": "Call submit_total_map first."}}
            ep.route = readers.plan_route(ep.total)
            return {"ok": True, "data": ep.route}
        if name == "finalize_answer":
            answer = args.get("answer")
            if not isinstance(answer, str) or not answer.strip():
                return {"ok": False, "error": {"kind": "missing_argument", "message": "answer"}}
            ep.answer = answer.strip()
            return {"ok": True, "data": {"answer": ep.answer}}
        return {"ok": False, "error": {"kind": "unknown_tool", "message": name}}

    agent = SimpleNamespace(knowledge=None, sensors=None, llm=llm)
    try:
        trace = run_react(
            agent,
            f"{context}\nQuestion: {question}",
            max_steps=max_steps,
            system_prompt=GRID_SYSTEM_PROMPT,
            tool_specs=GRID_TOOL_SPECS,
            dispatch_fn=dispatch,
            token_budget=20000,
            max_context_messages=24,
        )
    except MaxStepsExceeded as exc:
        trace = exc.trace

    for st in trace.steps:
        if st.tool_call is None:
            ep.steps.append({"idx": st.idx, "thought": st.thought})
            continue
        obs = st.observation
        ep.steps.append(
            {
                "idx": st.idx,
                "tool": st.tool_call.name,
                "ok": bool(obs and obs.ok),
                "summary": _summarize(st.tool_call.name, st.tool_call.arguments, obs.content if obs else None),
            }
        )
    return ep


def _summarize(name: str, args: dict[str, Any], content: Any) -> str:
    if name == "submit_total_map":
        n = len(args.get("paths") or []) if isinstance(args.get("paths"), list) else 0
        extra = ""
        if isinstance(content, dict) and (content.get("missing") or content.get("issues")):
            extra = f", {len(content.get('missing') or [])} missing, {len(content.get('issues') or [])} invalid"
        return f"{n} paths submitted{extra}"
    if name == "finalize_answer":
        return "answer given"
    if isinstance(content, dict):
        for key in ("paths", "readings", "detections"):
            if isinstance(content.get(key), list):
                return f"{len(content[key])} {key}"
        if "belief" in content:
            return str(content["belief"])
        if "found" in content:
            return " -> ".join(content.get("nodes") or []) or "no route"
        if "message" in content:
            return str(content["message"])
    return json.dumps(content)[:120] if content is not None else ""
