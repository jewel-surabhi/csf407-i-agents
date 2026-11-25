"""ReAct loop for the grid dashboard: the procedural layer between the other two.

One question = one episode at question hour ``clock + X``:

    get_belief          -> declarative layer, decayed to the question hour
                           (the declarations as they stood just before the scan)
    sense_lidar         -> sensorimotor layer, beams from the robot's node
    get_previous_scan   -> the scan from the previous question, if any
    resolve_total       -> deterministic merge (higher confidence wins) + route
    finalize_answer     -> the LLM's answer to the free-text question

The LLM chooses the calls and writes the answer; the merge rule and the
write-back are code, so all 31 segments are always resolved the same way.
After the episode the session writes the LiDAR conflict wins into the
declarative layer and moves the clock to the question hour.

Uses the same ``LLMClient`` (Groq) and ``chat_with_tools`` protocol as
``procedural.react_loop``; tests pass a ``RecordedLLMClient`` instead.
"""

from __future__ import annotations

import json
from typing import Any

from iagent.grid.session import GridSession, plan_route, resolve_total
from iagent.grid.world import SPACING_CM, LidarReading

__all__ = ["GRID_SYSTEM_PROMPT", "GRID_TOOL_SPECS", "run_grid_question", "MAX_STEPS"]

MAX_STEPS = 8

GRID_SYSTEM_PROMPT = f"""You are the procedural layer of a grounded robot agent in a warehouse.

The floor is a 5 x 4 grid of nodes: columns A-E (west to east), rows 1-4 (north to south).
A segment joins two neighbouring nodes and is named like "A1-B1". Nodes are {SPACING_CM} cm
apart; a box always sits at the middle of a segment and blocks it.

You can only know the floor through tools:
- get_belief: the declarative layer. Each segment's declared state with its confidence
  already decayed to the question time (stored - decay_per_hour * age). The decay rate
  depends on the source: the operator's setup declaration (source "operator_setup") and LiDAR
  findings written back after earlier questions (source "lidar_declared") fade differently.
- sense_lidar: a fresh LiDAR scan from the robot's node along its row and column. A beam
  stops at the first box; segments behind it are not seen. Confidence falls with distance
  (0.95 up to 50 cm, 0.50 at 400 cm).
- get_previous_scan: the scan made for the previous question, if any.
- resolve_total: the code-computed Total map. Per segment the higher confidence wins
  (a tie goes to the LiDAR). It also lists which LiDAR findings will be written back to
  the declarative layer, and the cheapest route from the robot to the target that avoids
  blocked segments.

Procedure, in order: call get_belief, then sense_lidar, then resolve_total
(get_previous_scan when the question is about change over time). Then call
finalize_answer exactly once.

Rules:
- Never state a segment's status without a tool result behind it.
- Base the answer on the Total map. When the declaration and the LiDAR disagree, say which
  one won and why (confidence after decay vs. confidence from distance).
- Mention segments the LiDAR could not see when they matter to the question.
- Keep the answer short and concrete: name segments and the route as node lists.
- At most {MAX_STEPS} tool calls in total.
"""


def _fn(name: str, description: str, properties: dict | None = None) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties or {},
                "required": [],
            },
        },
    }


GRID_TOOL_SPECS: list[dict[str, Any]] = [
    _fn(
        "get_belief",
        "Read the declarative layer at the question time. Without `segment` returns every "
        "segment; with it returns one segment plus its belief history.",
        {"segment": {"type": "string", "description": "Optional segment id, e.g. 'B2-C2'."}},
    ),
    _fn("sense_lidar", "Scan with the LiDAR from the robot's current node (4 directions)."),
    _fn("get_previous_scan", "The LiDAR scan made for the previous question, if any."),
    _fn(
        "resolve_total",
        "Merge declarations and the latest scan into the Total map, list write-backs (every LiDAR reading that beats its declaration is written back) and "
        "plan the route to the target. Call after sense_lidar.",
    ),
    {
        "type": "function",
        "function": {
            "name": "finalize_answer",
            "description": "Give the final answer to the user's question. Call exactly once.",
            "parameters": {
                "type": "object",
                "properties": {"answer": {"type": "string"}},
                "required": ["answer"],
            },
        },
    },
]


def _fmt_decl(d: dict[str, Any]) -> str:
    state = "blocked" if d["blocked"] else "clear"
    return (
        f"{d['segment']}: {state} {d['confidence']:.2f} "
        f"(stored {d['stored']:.2f}, {d['source']}, declared h{d['declared_at_hour']:g}, "
        f"age {d['age_hours']:g}h)"
    )


def _fmt_scan(r: LidarReading) -> str:
    state = "blocked" if r.blocked else "clear"
    return f"{r.segment}: {state} {r.confidence:.2f} ({r.direction}, {r.distance_cm} cm)"


class _Episode:
    """Holds the per-question state the tools read and fill."""

    def __init__(self, session: GridSession, hour: int) -> None:
        self.session = session
        self.hour = hour
        # Snapshot BEFORE any scan: the declarations the question is judged against.
        self.declared = session.declarations(hour)
        self.readings: dict[str, LidarReading] | None = None
        self.total: dict[str, dict[str, Any]] | None = None
        self.route: dict[str, Any] | None = None
        self.called: list[str] = []
        self.auto: list[str] = []

    # tools ---------------------------------------------------------------

    def get_belief(self, segment: str | None = None) -> dict[str, Any]:
        if segment:
            seg = segment.strip().upper()
            if seg not in self.declared:
                parts = seg.split("-")
                seg = "-".join(sorted(parts)) if len(parts) == 2 else seg
            if seg not in self.declared:
                return {"ok": False, "error": {"kind": "unknown_segment", "message": segment}}
            return {
                "ok": True,
                "data": {
                    "question_hour": self.hour,
                    "belief": self.declared[seg],
                    "history": self.session.history_for(seg),
                },
            }
        return {
            "ok": True,
            "data": {
                "question_hour": self.hour,
                "segments": [_fmt_decl(d) for d in self.declared.values()],
            },
        }

    def sense_lidar(self) -> dict[str, Any]:
        if self.readings is None:
            self.readings = self.session.sense()
        return {
            "ok": True,
            "data": {
                "robot": self.session.world.robot,
                "hour": self.hour,
                "readings": [_fmt_scan(r) for r in self.readings.values()],
                "note": "Segments not listed were not in line of sight.",
            },
        }

    def get_previous_scan(self) -> dict[str, Any]:
        prev = self.session.last_scan
        if prev is None:
            return {"ok": True, "data": {"previous_scan": None, "note": "No earlier scan."}}
        return {
            "ok": True,
            "data": {
                "hour": self.session.last_scan_hour,
                "readings": [_fmt_scan(r) for r in prev.values()],
            },
        }

    def resolve_total(self) -> dict[str, Any]:
        if self.readings is None:
            return {
                "ok": False,
                "error": {"kind": "no_scan", "message": "Call sense_lidar before resolve_total."},
            }
        self._compute_total()
        assert self.total is not None and self.route is not None
        lines = []
        for c in self.total.values():
            if c["blocked"] is None:
                lines.append(f"{c['segment']}: unknown")
                continue
            state = "blocked" if c["blocked"] else "clear"
            extra = f", write-back: {c['write']}" if c["write"] else ""
            if c.get("lidar_disagrees"):
                extra += ", LiDAR disagreed but was weaker"
            lines.append(f"{c['segment']}: {state} {c['confidence']:.2f} (winner {c['winner']}{extra})")
        return {
            "ok": True,
            "data": {
                "robot": self.session.world.robot,
                "target": self.session.world.target,
                "total": lines,
                "route": self.route,
            },
        }

    # helpers -------------------------------------------------------------

    def _compute_total(self) -> None:
        assert self.readings is not None
        self.total = resolve_total(self.declared, self.readings)
        self.route = plan_route(self.total, self.session.world.robot, self.session.world.target)

    def dispatch(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        self.called.append(name)
        if name == "get_belief":
            return self.get_belief(args.get("segment"))
        if name == "sense_lidar":
            return self.sense_lidar()
        if name == "get_previous_scan":
            return self.get_previous_scan()
        if name == "resolve_total":
            return self.resolve_total()
        return {"ok": False, "error": {"kind": "unknown_tool", "message": name}}

    def ensure_complete(self) -> None:
        """Safety net: the maps need a scan and a Total even if the model skipped them."""
        if self.readings is None:
            self.readings = self.session.sense()
            self.auto.append("sense_lidar")
        if self.total is None:
            self._compute_total()
            self.auto.append("resolve_total")


def _summary(result: dict[str, Any]) -> str:
    if not result.get("ok"):
        return f"error: {result.get('error', {}).get('message', '')}"
    data = result.get("data") or {}
    for key in ("readings", "segments", "total"):
        if isinstance(data.get(key), list):
            return f"{len(data[key])} {key}"
    if "belief" in data:
        return _fmt_decl(data["belief"])
    if "answer" in data:
        return "answer given"
    return json.dumps(data)[:120]


def run_grid_question(session: GridSession, llm: Any, question: str, hours: int) -> dict[str, Any]:
    """Run one episode and commit its results. Raises whatever the LLM client raises."""
    if not session.declared:
        raise ValueError("complete the setup before asking a question")
    if hours < 1:
        raise ValueError("time set must be at least +1 hour")

    hour = session.clock_h + hours
    ep = _Episode(session, hour)
    world = session.world
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": GRID_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Question time: hour {hour} (current clock was hour {session.clock_h}).\n"
                f"Robot at {world.robot}, target at {world.target}.\n"
                f"Question: {question}"
            ),
        },
    ]

    steps: list[dict[str, Any]] = []
    answer: str | None = None
    for step_idx in range(1, MAX_STEPS + 1):
        if step_idx == MAX_STEPS and answer is None:
            messages.append(
                {"role": "user", "content": "LAST TURN: call finalize_answer now with your best answer."}
            )
        resp = llm.chat_with_tools(messages, GRID_TOOL_SPECS)
        if not resp.tool_calls:
            text = (resp.content or "").strip()
            steps.append({"idx": step_idx, "thought": text or "(empty)"})
            if text:
                messages.append({"role": "assistant", "content": text})
            messages.append({"role": "user", "content": "Call a tool next. Use finalize_answer to end."})
            continue

        messages.append(
            {
                "role": "assistant",
                "content": resp.content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)},
                    }
                    for tc in resp.tool_calls
                ],
            }
        )
        # Side effects first, finalize last (same order as procedural.react_loop).
        ordered = sorted(resp.tool_calls, key=lambda tc: tc.name == "finalize_answer")
        for tc in ordered:
            if tc.name == "finalize_answer":
                if answer is None and isinstance(tc.arguments.get("answer"), str):
                    answer = tc.arguments["answer"].strip()
                    result = {"ok": True, "data": {"answer": answer}}
                else:
                    result = {"ok": False, "error": {"kind": "duplicate_or_empty_finalize", "message": ""}}
            else:
                result = ep.dispatch(tc.name, tc.arguments)
            steps.append(
                {
                    "idx": step_idx,
                    "thought": resp.content,
                    "tool": tc.name,
                    "arguments": tc.arguments,
                    "ok": bool(result.get("ok")),
                    "summary": _summary(result),
                }
            )
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": json.dumps(result)})
        if answer is not None:
            break

    ep.ensure_complete()
    assert ep.readings is not None and ep.total is not None and ep.route is not None

    previous = session.last_scan
    previous_hour = session.last_scan_hour
    writes = session.commit_question(hour=hour, readings=ep.readings, total=ep.total)

    record = {
        "hour": hour,
        "hours_added": hours,
        "question": question,
        "answer": answer or "The model did not give a final answer within the step limit.",
        "finalized": answer is not None,
        "steps": steps,
        "auto_steps": ep.auto,
        "robot": world.robot,
        "target": world.target,
        "world": world.to_dict(),
        "declarations": ep.declared,
        "declarations_after": session.declarations(hour),
        "lidar": {k: v.to_dict() for k, v in ep.readings.items()},
        "previous_lidar": (
            {"hour": previous_hour, "readings": {k: v.to_dict() for k, v in previous.items()}}
            if previous
            else None
        ),
        "total": ep.total,
        "route": ep.route,
        "writes": writes,
    }
    session.history.append(record)
    return record
