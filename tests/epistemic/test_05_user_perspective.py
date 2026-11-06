"""Epistemic test 05 — user_perspective.

Stresses: when asked what THE USER believes, the agent returns the user's
belief — not the agent's own live observation, not third-party history.

Scenario: scenario_b (box_01 has_color red @ perspective=user is seeded).
Query:    "What color does the user believe box_01 is?"
Expected: `query_perspective` on box_01 -> finalize_answer with "red" and
          an explicit mention of the user perspective.
Failure mode: LLM answers from camera or bot_02 data, merging perspectives.
"""

from __future__ import annotations

import pytest

from .conftest import (
    assert_answer_contains,
    assert_tools_called,
    dump_epistemic_trace,
)

pytestmark = [pytest.mark.epistemic]

TEST_ID = "05_user_perspective"
QUERY = "What color does the user believe box_01 is? Keep perspectives separate."

_SCRIPT = [
    {
        "tool": "query_perspective",
        "arguments": {"subject": "box_01", "predicate": "has_color"},
    },
    {
        "tool": "finalize_answer",
        "arguments": {
            "answer": (
                "From the user's perspective, box_01 is red_atom (confidence "
                "0.9, source=user). This is their belief — I have not verified "
                "it against my own sensors."
            ),
            "cited_beliefs": ["box_01/has_color@user"],
        },
    },
]


def test_user_perspective_recorded(recorded_agent):
    agent, _knowledge, _sensors, scenario = recorded_agent("scenario_b", _SCRIPT)

    result = agent.run(QUERY, scenario="scenario_b")

    assert_tools_called(result.trace, ["query_perspective", "finalize_answer"])
    assert_answer_contains(result.trace, ["red", "user"])
    # Must NOT leak the other perspectives.
    answer = (result.trace.final_answer or "").lower()
    # 'brown' is the camera perception; 'blue' is bot_02 history. Neither
    # should appear in a user-only answer.
    assert "brown" not in answer, "user-perspective answer leaked camera (brown)"
    assert "blue" not in answer, "user-perspective answer leaked third-party (blue)"

    dump_epistemic_trace(
        result.trace,
        TEST_ID,
        extras={
            "scenario": scenario.get("id"),
            "expected_tool_calls": ["query_perspective", "finalize_answer"],
            "expected_answer_contains": ["red", "user"],
            "forbidden_in_answer": ["brown", "blue"],
            "verdict": "pass",
            "failure_mode_tested": (
                "LLM answers from camera/bot_02 data, merging perspectives "
                "the user never held."
            ),
        },
    )


@pytest.mark.live_llm
def test_user_perspective_live(live_agent):
    agent, _knowledge, _sensors, scenario = live_agent("scenario_b")

    result = agent.run(QUERY, scenario="scenario_b", max_steps=8)

    assert_answer_contains(result.trace, ["red"])
    dump_epistemic_trace(
        result.trace,
        TEST_ID,
        extras={
            "scenario": scenario.get("id"),
            "expected_answer_contains": ["red"],
            "verdict": "pass",
        },
    )
