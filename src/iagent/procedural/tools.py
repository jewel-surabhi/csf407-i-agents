"""Tool registry and dispatcher.

`TOOL_SPECS` is the list of JSON-Schema tool definitions handed to the LLM
(OpenAI/Groq function-tool format). `dispatch` routes a name+args pair to
the right layer façade and returns a uniform `{ok, data|error}` envelope so
the LLM sees the same shape whether the call succeeded or blew up.

Owner: Person 3 (Phase 4). See PROJECT_PLAN.md section G.4.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, ValidationError

from iagent.declarative.api import Belief, KnowledgeAPI, KnowledgeError
from iagent.sensorimotor.api import SensorimotorAPI

__all__ = [
    "TOOL_SPECS",
    "TOOL_NAMES",
    "FINALIZE_TOOL",
    "dispatch",
    "DispatchError",
]


FINALIZE_TOOL = "finalize_answer"


class DispatchError(RuntimeError):
    """Raised only when the caller misuses dispatch (unknown APIs dict)."""


def _fn(name: str, description: str, parameters: dict[str, Any]) -> dict[str, Any]:
    """Shorthand for one OpenAI/Groq function-tool spec."""
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": parameters,
        },
    }


_PERSPECTIVE = {
    "type": "string",
    "description": "'self' (default), 'user', or 'third_party:<agent_id>'.",
}


TOOL_SPECS: list[dict[str, Any]] = [
    # ---------- declarative reads ----------
    _fn(
        "get_belief",
        "Read the agent's current belief about (subject, predicate) from the "
        "declarative memory. Returns null if no active belief is held.",
        {
            "type": "object",
            "properties": {
                "subject": {"type": "string"},
                "predicate": {"type": "string"},
                "perspective": _PERSPECTIVE,
            },
            "required": ["subject", "predicate"],
        },
    ),
    _fn(
        "query_perspective",
        "Return every perspective (user / self / third_party:*) held about a "
        "subject, grouped. Use this before answering perspective questions so "
        "you don't merge distinct views.",
        {
            "type": "object",
            "properties": {
                "subject": {"type": "string"},
                "predicate": {"type": "string"},
            },
            "required": ["subject"],
        },
    ),
    _fn(
        "query_provenance",
        "Historical audit trail for a belief (newest first). Use to see how a "
        "belief evolved or which source recorded it originally.",
        {
            "type": "object",
            "properties": {
                "subject": {"type": "string"},
                "predicate": {"type": "string"},
                "limit": {"type": "integer", "minimum": 1, "maximum": 100},
            },
            "required": ["subject"],
        },
    ),
    _fn(
        "list_sources",
        "List every registered source with its trust_prior. Call this when "
        "declarative memory and a sensor disagree so you know which to trust.",
        {"type": "object", "properties": {}},
    ),
    # ---------- declarative writes ----------
    _fn(
        "update_belief",
        "Write a new active belief AND its provenance row (atomic). Use after "
        "a sensor reading contradicts memory, to record the fresh state.",
        {
            "type": "object",
            "properties": {
                "subject": {"type": "string"},
                "predicate": {"type": "string"},
                "object": {"type": "string"},
                "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                "source": {"type": "string"},
                "perspective": _PERSPECTIVE,
                "reason": {"type": "string"},
            },
            "required": [
                "subject",
                "predicate",
                "object",
                "confidence",
                "source",
                "reason",
            ],
        },
    ),
    _fn(
        "downgrade_belief",
        "Lower confidence on a stale belief without deleting it (history is "
        "preserved). Use on the LOSING side of a conflict.",
        {
            "type": "object",
            "properties": {
                "subject": {"type": "string"},
                "predicate": {"type": "string"},
                "new_confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                "reason": {"type": "string"},
                "perspective": _PERSPECTIVE,
            },
            "required": ["subject", "predicate", "new_confidence", "reason"],
        },
    ),
    # ---------- sensorimotor reads ----------
    _fn(
        "sense_lidar",
        "Read the front/left/right/back LiDAR beam. Returns distance and a "
        "status flag ('ok' | 'blocked' | ...).",
        {
            "type": "object",
            "properties": {
                "direction": {
                    "type": "string",
                    "enum": ["front", "left", "right", "back"],
                },
            },
        },
    ),
    _fn(
        "sense_camera",
        "Capture the current camera frame in one direction. Returns detected "
        "objects and ambient lighting.",
        {
            "type": "object",
            "properties": {
                "direction": {
                    "type": "string",
                    "enum": ["front", "left", "right", "back"],
                },
            },
        },
    ),
    _fn(
        "sense_clock",
        "Return the current simulated time.",
        {"type": "object", "properties": {}},
    ),
    # ---------- terminator ----------
    _fn(
        FINALIZE_TOOL,
        "End the episode. Emit your final natural-language answer plus the "
        "belief IDs and sensor readings you cited. Call this exactly once.",
        {
            "type": "object",
            "properties": {
                "answer": {"type": "string"},
                "cited_beliefs": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "cited_sensors": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
            "required": ["answer"],
        },
    ),
]


TOOL_NAMES: frozenset[str] = frozenset(spec["function"]["name"] for spec in TOOL_SPECS)


def _ok(data: Any) -> dict[str, Any]:
    return {"ok": True, "data": _jsonable(data)}


def _err(kind: str, message: str, **extra: Any) -> dict[str, Any]:
    payload = {"ok": False, "error": {"kind": kind, "message": message}}
    if extra:
        payload["error"].update(extra)
    return payload


def _jsonable(value: Any) -> Any:
    """Turn Pydantic models / lists / datetimes into JSON-safe primitives."""
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    if isinstance(value, tuple):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, datetime):
        return value.astimezone(UTC).isoformat()
    return value


def dispatch(
    name: str,
    arguments: dict[str, Any],
    apis: dict[str, Any],
) -> dict[str, Any]:
    """Route a tool call to the right layer façade.

    `apis` is `{"knowledge": KnowledgeAPI, "sensors": SensorimotorAPI}`.

    Returns a uniform envelope so the LLM always sees the same shape:
        success:  {"ok": True,  "data": <json>}
        failure:  {"ok": False, "error": {"kind": ..., "message": ...}}

    `finalize_answer` is passed through as `{"ok": True, "data": <arguments>}`;
    the ReAct loop (P4) detects the name separately and terminates.
    """
    knowledge: KnowledgeAPI | None = apis.get("knowledge")
    sensors: SensorimotorAPI | None = apis.get("sensors")

    if name not in TOOL_NAMES:
        return _err("unknown_tool", f"No tool named {name!r}.")

    args = arguments or {}

    try:
        if name == "get_belief":
            _require(knowledge, "knowledge")
            result = knowledge.get_belief(
                subject=args["subject"],
                predicate=args["predicate"],
                perspective=args.get("perspective", "self"),
            )
            return _ok(result)

        if name == "query_perspective":
            _require(knowledge, "knowledge")
            result = knowledge.query_perspective(
                subject=args["subject"],
                predicate=args.get("predicate"),
            )
            return _ok(result)

        if name == "query_provenance":
            _require(knowledge, "knowledge")
            result = knowledge.query_provenance(
                subject=args["subject"],
                predicate=args.get("predicate"),
                limit=args.get("limit", 20),
            )
            return _ok(result)

        if name == "list_sources":
            _require(knowledge, "knowledge")
            return _ok(knowledge.list_sources())

        if name == "update_belief":
            _require(knowledge, "knowledge")
            belief = Belief(
                subject=args["subject"],
                predicate=args["predicate"],
                object=args["object"],
                confidence=args["confidence"],
                source=args["source"],
                perspective=args.get("perspective", "self"),
                observed_at=datetime.now(UTC),
            )
            return _ok(knowledge.update_belief(belief, reason=args["reason"]))

        if name == "downgrade_belief":
            _require(knowledge, "knowledge")
            return _ok(
                knowledge.downgrade_belief(
                    subject=args["subject"],
                    predicate=args["predicate"],
                    new_confidence=args["new_confidence"],
                    reason=args["reason"],
                    perspective=args.get("perspective", "self"),
                )
            )

        if name == "sense_lidar":
            _require(sensors, "sensors")
            return _ok(sensors.sense_lidar(direction=args.get("direction", "front")))

        if name == "sense_camera":
            _require(sensors, "sensors")
            return _ok(sensors.sense_camera(direction=args.get("direction", "front")))

        if name == "sense_clock":
            _require(sensors, "sensors")
            return _ok(sensors.sense_clock())

        if name == FINALIZE_TOOL:
            return _ok(
                {
                    "answer": args["answer"],
                    "cited_beliefs": args.get("cited_beliefs", []),
                    "cited_sensors": args.get("cited_sensors", []),
                }
            )

    except DispatchError:
        # Caller misuse (missing api in `apis`) — bubble up so P4 sees the bug.
        raise
    except KeyError as exc:
        return _err("missing_argument", f"Missing required argument: {exc.args[0]!r}.")
    except ValidationError as exc:
        return _err("invalid_argument", str(exc))
    except KnowledgeError as exc:
        return _err("knowledge_error", str(exc))
    except NotImplementedError as exc:
        return _err("not_implemented", str(exc))
    except Exception as exc:  # noqa: BLE001 — LLM must never see an unhandled crash
        return _err("tool_error", f"{type(exc).__name__}: {exc}")

    return _err("unknown_tool", f"No dispatch branch for {name!r}.")


def _require(api: Any, label: str) -> None:
    if api is None:
        raise DispatchError(f"apis[{label!r}] is required for this tool call.")
