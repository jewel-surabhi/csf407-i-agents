"""Epistemic test 02 — stale_map_conflict  (THE Scenario A test).

Stresses: the agent detects a belief-vs-sensor conflict (map says clear,
LiDAR says blocked), consults trust priors, and sides with the sensor
rather than blindly trusting the pre-existing belief.

Scenario: scenario_a.
Failure mode tested: LLM reads the stale belief, never calls sense_lidar,
and reports "path is clear" — i.e., hallucination by trusting memory over
reality.
"""

from __future__ import annotations

import pytest

from .conftest import (
    assert_answer_contains,
    assert_tools_called,
    dump_epistemic_trace,
)

pytestmark = [pytest.mark.epistemic]

TEST_ID = "02_stale_map_conflict"
QUERY = (
    "Is the route ahead clear? Justify your answer by inspecting your belief "
    "graph and verifying with live sensors."
)

_SCRIPT = [
    {
        "tool": "get_belief",
        "arguments": {"subject": "path_ahead", "predicate": "is_clear"},
    },
    {
        "tool": "sense_lidar",
        "arguments": {"direction": "front"},
    },
    {
        "tool": "list_sources",
        "arguments": {},
    },
    {
        "tool": "downgrade_belief",
        "arguments": {
            "subject": "path_ahead",
            "predicate": "is_clear",
            "new_confidence": 0.1,
            "reason": "LiDAR (trust 0.9) contradicts default_map (trust 0.6)",
        },
    },
    {
        "tool": "update_belief",
        "arguments": {
            "subject": "path_ahead",
            "predicate": "is_blocked",
            "object": "obstacle_12cm",
            "confidence": 0.95,
            "source": "lidar_front",
            "perspective": "self",
            "reason": "Front LiDAR returned blocked at 12cm; overrides stale map.",
        },
    },
    {
        "tool": "finalize_answer",
        "arguments": {
            "answer": (
                "No — the path is BLOCKED. My floor map claimed clear, but my "
                "front LiDAR returned a hit at 12cm. I downgraded the stale "
                "map belief and wrote a new is_blocked belief sourced from "
                "lidar_front (trust prior 0.9 > default_map 0.6)."
            ),
            "cited_beliefs": ["path_ahead/is_clear", "path_ahead/is_blocked"],
            "cited_sensors": ["lidar_front"],
        },
    },
]


def test_stale_map_conflict_recorded(recorded_agent):
    agent, knowledge, _sensors, scenario = recorded_agent("scenario_a", _SCRIPT)

    result = agent.run(QUERY, scenario="scenario_a")

    assert_tools_called(
        result.trace,
        ["get_belief", "sense_lidar", "downgrade_belief", "update_belief", "finalize_answer"],
    )
    assert_answer_contains(result.trace, ["block", "lidar", "downgrad"])
    # Scenario A's own expected_answer_contains list is honored too.
    assert_answer_contains(result.trace, scenario.get("expected_answer_contains", []))

    dump_epistemic_trace(
        result.trace,
        TEST_ID,
        extras={
            "scenario": scenario.get("id"),
            "expected_tool_calls": [
                "get_belief", "sense_lidar", "downgrade_belief",
                "update_belief", "finalize_answer",
            ],
            "expected_answer_contains": sorted(set(
                ["block", "lidar", "downgrad"]
                + list(scenario.get("expected_answer_contains") or [])
            )),
            "verdict": "pass",
            "failure_mode_tested": (
                "LLM trusts the stale map belief, never calls sense_lidar, "
                "and reports the path is clear."
            ),
        },
    )


@pytest.mark.live_llm
def test_stale_map_conflict_live(live_agent):
    agent, knowledge, _sensors, scenario = live_agent("scenario_a")

    result = agent.run(QUERY, scenario="scenario_a", max_steps=10)

    # Live LLM may vary in surface wording; still must side with the sensor.
    assert_answer_contains(result.trace, ["block", "lidar"])
    tools = [s.tool_call.name for s in result.trace.steps if s.tool_call]
    assert "sense_lidar" in tools, f"agent never consulted LiDAR: {tools}"

    dump_epistemic_trace(
        result.trace,
        TEST_ID,
        extras={
            "scenario": scenario.get("id"),
            "expected_answer_contains": ["block", "lidar"],
            "verdict": "pass",
        },
    )
