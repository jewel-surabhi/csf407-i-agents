"""Epistemic test 06 — egocentric_perspective.

Stresses: when asked what IT (the agent) currently sees, the agent reports
its own LIVE sensor perception, not a stored belief, not a user belief,
not a historical third-party log.

Scenario: scenario_b. Camera returns 'brown' for box_01 (ambient yellow
lighting is noisy — the point is that this is the agent's own current
perception, honest about the ambient warning).
Query:    "What color do you currently see box_01 as?"
Expected: `sense_camera` -> `finalize_answer` with 'brown' and an
          egocentric framing word.
Failure mode: LLM reports the user belief (red) or the historical bot_02
log (blue) instead of what the camera is reporting right now.
"""

from __future__ import annotations

import pytest

from .conftest import (
    assert_answer_contains,
    assert_tools_called,
    dump_epistemic_trace,
)

pytestmark = [pytest.mark.epistemic]

TEST_ID = "06_egocentric_perspective"
QUERY = (
    "Call sense_camera(direction='front') and tell me what color you currently "
    "see box_01 as from your own camera right now. Report the perceived color "
    "plus the ambient lighting context."
)

_SCRIPT = [
    {
        "tool": "sense_camera",
        "arguments": {"direction": "front"},
    },
    {
        "tool": "finalize_answer",
        "arguments": {
            "answer": (
                "Right now my front camera perceives box_01 as brown "
                "(confidence 0.82). Note: the ambient lighting is yellow, "
                "so this is my CURRENT egocentric perception — not a claim "
                "about the box's true color."
            ),
            "cited_sensors": ["camera_front"],
        },
    },
]

# Any of these in the answer is enough to establish the egocentric framing.
_EGOCENTRIC_WORDS = ["currently", "now", "live", "egocentric", "right now", "my camera"]


def _assert_has_any(trace, candidates: list[str]) -> None:
    ans = (trace.final_answer or "").lower()
    assert any(c in ans for c in candidates), (
        f"final answer never asserts egocentric framing (any of {candidates}): "
        f"{trace.final_answer!r}"
    )


def test_egocentric_perspective_recorded(recorded_agent):
    agent, _knowledge, _sensors, scenario = recorded_agent("scenario_b", _SCRIPT)

    result = agent.run(QUERY, scenario="scenario_b")

    assert_tools_called(result.trace, ["sense_camera", "finalize_answer"])
    assert_answer_contains(result.trace, ["brown"])
    _assert_has_any(result.trace, _EGOCENTRIC_WORDS)

    dump_epistemic_trace(
        result.trace,
        TEST_ID,
        extras={
            "scenario": scenario.get("id"),
            "expected_tool_calls": ["sense_camera", "finalize_answer"],
            "expected_answer_contains": ["brown"],
            "expected_egocentric_markers": _EGOCENTRIC_WORDS,
            "verdict": "pass",
            "failure_mode_tested": (
                "LLM reports the user's belief (red) or the historical "
                "bot_02 log (blue) instead of its own current perception."
            ),
        },
    )


@pytest.mark.live_llm
def test_egocentric_perspective_live(live_agent):
    agent, _knowledge, _sensors, scenario = live_agent("scenario_b")

    result = agent.run(QUERY, scenario="scenario_b", max_steps=8)

    assert_answer_contains(result.trace, ["brown"])
    dump_epistemic_trace(
        result.trace,
        TEST_ID,
        extras={
            "scenario": scenario.get("id"),
            "expected_answer_contains": ["brown"],
            "verdict": "pass",
        },
    )
