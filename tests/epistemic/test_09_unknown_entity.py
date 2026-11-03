"""Epistemic test 09 — unknown_entity.

Stresses: self-knowledge. When asked about an entity the agent has no
belief about, it says "I don't know" instead of inventing an answer.

Scenario: scenario_a (no belief about `widget_xyz` exists).
Query:    "What color is widget_xyz?"
Expected: `get_belief` (or `query_perspective`) returns null -> `finalize_answer`
          with an abstention phrase ("don't know" / "no record" / "unknown" /
          "no belief").
Expected answer must NOT contain any fabricated color.
Failure mode: LLM invents a color (e.g. "red") because it 'feels right'.
"""

from __future__ import annotations

import pytest

from .conftest import (
    assert_answer_contains,
    assert_tools_called,
    dump_epistemic_trace,
)

pytestmark = [pytest.mark.epistemic]

TEST_ID = "09_unknown_entity"
QUERY = "What color is widget_xyz?"

_SCRIPT = [
    {
        "tool": "query_perspective",
        "arguments": {"subject": "widget_xyz", "predicate": "has_color"},
    },
    {
        "tool": "get_belief",
        "arguments": {"subject": "widget_xyz", "predicate": "has_color"},
    },
    {
        "tool": "finalize_answer",
        "arguments": {
            "answer": (
                "I don't know. I have no belief recorded about widget_xyz in "
                "my knowledge graph — no perspective (user, self, or third "
                "party) holds a color claim for it, and I have no sensor "
                "evidence either. I will not guess."
            ),
            "cited_beliefs": [],
        },
    },
]

_ABSTENTION_MARKERS = [
    "don't know", "do not know", "no record", "no belief", "unknown",
    "no information", "no data", "cannot determine", "no claim",
]

# If any of these colors appears, the agent fabricated an answer.
_FABRICATED_COLORS = ["red", "blue", "green", "yellow", "brown", "black", "white", "orange", "purple"]


def _assert_abstained(trace) -> None:
    ans = (trace.final_answer or "").lower()
    assert any(m in ans for m in _ABSTENTION_MARKERS), (
        f"agent didn't abstain — final answer: {trace.final_answer!r}"
    )
    fabricated = [c for c in _FABRICATED_COLORS if c in ans]
    assert not fabricated, (
        f"agent fabricated color(s) {fabricated}: {trace.final_answer!r}"
    )


def test_unknown_entity_recorded(recorded_agent):
    agent, _knowledge, _sensors, scenario = recorded_agent("scenario_a", _SCRIPT)

    result = agent.run(QUERY)

    assert_tools_called(result.trace, ["query_perspective", "finalize_answer"])
    _assert_abstained(result.trace)

    dump_epistemic_trace(
        result.trace,
        TEST_ID,
        extras={
            "scenario": scenario.get("id"),
            "expected_abstention_markers": _ABSTENTION_MARKERS,
            "forbidden_fabrications": _FABRICATED_COLORS,
            "verdict": "pass",
            "failure_mode_tested": (
                "LLM invents a color because the question 'felt natural' "
                "despite the knowledge graph being empty on this subject."
            ),
        },
    )


@pytest.mark.live_llm
def test_unknown_entity_live(live_agent):
    agent, _knowledge, _sensors, scenario = live_agent("scenario_a")

    result = agent.run(QUERY, max_steps=8)
    _assert_abstained(result.trace)

    dump_epistemic_trace(
        result.trace,
        TEST_ID,
        extras={
            "scenario": scenario.get("id"),
            "expected_abstention_markers": _ABSTENTION_MARKERS,
            "verdict": "pass",
        },
    )
