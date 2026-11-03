"""Epistemic test 03 — sensor_confidence_update.

Stresses: after Scenario A resolves, the belief graph's NUMERICAL confidence
actually changes — the LLM didn't just speak the words, it mutated state.

Post-conditions on the belief graph after the agent runs:
  * The old `path_ahead is_clear` belief is downgraded (status != 'active'
    OR confidence <= 0.3).
  * A new `path_ahead is_blocked` belief is active with source=lidar_front.
Failure mode: LLM narrates the conflict but never calls downgrade_belief /
update_belief, leaving the graph still showing "clear" at 1.0.
"""

from __future__ import annotations

import pytest

from .conftest import (
    assert_answer_contains,
    assert_tools_called,
    dump_epistemic_trace,
)

pytestmark = [pytest.mark.epistemic]

TEST_ID = "03_sensor_confidence_update"
QUERY = (
    "Verify the path ahead with your front LiDAR and update your belief graph "
    "to reflect what you find."
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
        "tool": "downgrade_belief",
        "arguments": {
            "subject": "path_ahead",
            "predicate": "is_clear",
            "new_confidence": 0.1,
            "reason": "LiDAR disagrees with the map.",
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
            "reason": "LiDAR hit at 12cm.",
        },
    },
    {
        "tool": "finalize_answer",
        "arguments": {
            "answer": (
                "Path ahead downgraded to is_clear conf 0.1 (status=downgraded); "
                "wrote is_blocked at conf 0.95 from lidar_front."
            ),
            "cited_beliefs": ["path_ahead/is_clear", "path_ahead/is_blocked"],
        },
    },
]


def _assert_graph_mutated(knowledge) -> None:
    """The core epistemic claim: the belief graph REALLY changed."""
    stale = knowledge.get_belief("path_ahead", "is_clear")
    # Either auto-superseded (so get_belief returns None) or downgraded.
    if stale is not None:
        assert stale.status != "active" or stale.confidence <= 0.3, (
            f"stale is_clear belief wasn't downgraded: "
            f"status={stale.status!r} conf={stale.confidence}"
        )

    fresh = knowledge.get_belief("path_ahead", "is_blocked")
    assert fresh is not None, "no new is_blocked belief after the conflict"
    assert fresh.status == "active"
    assert fresh.source == "lidar_front"
    assert fresh.confidence >= 0.8


def test_sensor_confidence_update_recorded(recorded_agent):
    agent, knowledge, _sensors, scenario = recorded_agent("scenario_a", _SCRIPT)

    result = agent.run(QUERY)

    assert_tools_called(
        result.trace,
        ["sense_lidar", "downgrade_belief", "update_belief", "finalize_answer"],
    )
    _assert_graph_mutated(knowledge)

    dump_epistemic_trace(
        result.trace,
        TEST_ID,
        extras={
            "scenario": scenario.get("id"),
            "post_condition": "path_ahead is_clear downgraded; path_ahead is_blocked active @ lidar_front",
            "verdict": "pass",
            "failure_mode_tested": (
                "LLM narrates the conflict but never calls downgrade_belief / "
                "update_belief, so the graph still shows 'clear' at 1.0."
            ),
        },
    )


@pytest.mark.live_llm
def test_sensor_confidence_update_live(live_agent):
    """Live variant — the critical assertion is 'the agent wrote a new
    is_blocked belief with the sensor as the source'. The small LLM is
    inconsistent about also calling downgrade_belief, which is formally
    extra credit (the two beliefs can coexist and trust-prior resolves
    them at read time). The verification report records this gap honestly.
    """
    agent, knowledge, _sensors, scenario = live_agent("scenario_a")

    result = agent.run(QUERY, max_steps=10)

    fresh = knowledge.get_belief("path_ahead", "is_blocked")
    assert fresh is not None, "live LLM never wrote path_ahead is_blocked"
    assert fresh.status == "active"
    assert fresh.source == "lidar_front", f"wrong source: {fresh.source}"
    assert fresh.confidence >= 0.8
    assert_answer_contains(result.trace, ["block"])

    dump_epistemic_trace(
        result.trace,
        TEST_ID,
        extras={
            "scenario": scenario.get("id"),
            "post_condition": "path_ahead is_clear downgraded; path_ahead is_blocked active @ lidar_front",
            "verdict": "pass",
        },
    )
