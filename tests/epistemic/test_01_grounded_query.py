"""Epistemic test 01 — grounded_query.

Stresses: BASELINE. The agent answers a question whose answer is already in
the belief graph by reading the belief, NOT by hallucinating.

Scenario: scenario_a (robot located_at room_101, confidence 1.0, default_map).
Query:    "Where are you?"
Expected: `get_belief("robot","located_at")` -> `finalize_answer` citing room_101.
Failure mode: LLM invents a location without calling get_belief first.
"""

from __future__ import annotations

import pytest

from .conftest import (
    assert_answer_contains,
    assert_tools_called,
    dump_epistemic_trace,
)

pytestmark = [pytest.mark.epistemic]

TEST_ID = "01_grounded_query"
QUERY = "Where are you?"

_SCRIPT = [
    # Turn 1: read the belief.
    {
        "tool": "get_belief",
        "arguments": {
            "subject": "robot",
            "predicate": "located_at",
            "perspective": "self",
        },
    },
    # Turn 2: cite the belief in the final answer.
    {
        "tool": "finalize_answer",
        "arguments": {
            "answer": (
                "I am located at room_101. This is a self-perspective belief "
                "sourced from the default_map at confidence 1.0."
            ),
            "cited_beliefs": ["robot/located_at@self"],
        },
    },
]


def test_grounded_query_recorded(recorded_agent):
    """Deterministic replay — pins the expected tool sequence + answer shape."""
    agent, knowledge, _sensors, scenario = recorded_agent("scenario_a", _SCRIPT)

    result = agent.run(QUERY)

    # 1. The agent actually read the belief graph.
    assert_tools_called(result.trace, ["get_belief", "finalize_answer"])
    # 2. The answer cites the ground-truth room.
    assert_answer_contains(result.trace, ["room_101"])
    # 3. The belief itself was never modified (baseline test — no writes).
    assert knowledge.get_belief("robot", "located_at").object == "room_101"

    log = dump_epistemic_trace(
        result.trace,
        TEST_ID,
        extras={
            "scenario": scenario.get("id"),
            "expected_tool_calls": ["get_belief", "finalize_answer"],
            "expected_answer_contains": ["room_101"],
            "verdict": "pass",
        },
    )
    assert log.exists(), f"trace log not written: {log}"


@pytest.mark.live_llm
def test_grounded_query_live(live_agent):
    """Same test, driven by real Groq. Produces the committed epistemic log."""
    agent, knowledge, _sensors, scenario = live_agent("scenario_a")

    result = agent.run(QUERY)

    assert_tools_called(result.trace, ["get_belief", "finalize_answer"])
    assert_answer_contains(result.trace, ["room_101"])
    assert knowledge.get_belief("robot", "located_at").object == "room_101"

    dump_epistemic_trace(
        result.trace,
        TEST_ID,
        extras={
            "scenario": scenario.get("id"),
            "expected_tool_calls": ["get_belief", "finalize_answer"],
            "expected_answer_contains": ["room_101"],
            "verdict": "pass",
        },
    )
