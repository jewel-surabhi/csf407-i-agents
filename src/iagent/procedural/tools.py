"""Tool registry and dispatcher.

TOOL_SPECS is the list of JSON-Schema definitions handed to the LLM.
`dispatch` routes a name+args pair to the correct layer facade.

Placeholder — implementation lands in Phase 4 (Owner: Person 3).
"""

from __future__ import annotations

from typing import Any

# EMPTY PLACEHOLDER — IMPLEMENTATION LATER
# Owner: Person 3 (Phase 4). See PROJECT_PLAN.md section G.4.

TOOL_SPECS: list[dict[str, Any]] = [
    # Each entry: {"type": "function", "function": {"name": ..., "description": ...,
    #                                                "parameters": {json-schema}}}
    # Names required: get_belief, query_perspective, query_provenance, list_sources,
    #                 update_belief, downgrade_belief,
    #                 sense_lidar, sense_camera, sense_clock, finalize_answer
]


def dispatch(name: str, arguments: dict[str, Any], apis: dict[str, Any]) -> dict[str, Any]:
    """Route a tool call to the right layer facade.

    `apis` is `{"knowledge": KnowledgeAPI, "sensors": SensorimotorAPI}`.
    Returns `{"ok": bool, "data": ...}` — wrapped errors so the LLM can recover.
    """
    raise NotImplementedError("Phase 4 — see PROJECT_PLAN.md section G.4.")
