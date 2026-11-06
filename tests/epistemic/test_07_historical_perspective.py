"""Epistemic test 07 — historical_perspective.

Stresses: the agent surfaces a THIRD-PARTY historical log entry (another
agent / another system wrote this) without accidentally re-attributing it
to itself or merging it into the user's view.

Scenario: scenario_b seeds `box_01 has_color blue_atom @ perspective=
third_party:bot_02` from source bot_02.
Query:    "What does your historical record say about box_01's color, and
          who recorded it?"
Expected: `query_perspective` or `query_provenance` -> `finalize_answer`
          with 'blue' AND a mention of bot_02 / historical / third-party.
Failure mode: LLM reports blue but attributes it to itself (as if its own
camera saw blue) or to the user.
"""

from __future__ import annotations

import pytest

from .conftest import (
    assert_answer_contains,
    assert_tools_called,
    dump_epistemic_trace,
)

pytestmark = [pytest.mark.epistemic]

TEST_ID = "07_historical_perspective"
QUERY = (
    "What does your historical record say about box_01's color, and which "
    "source recorded it? Do not confuse this with your own current view."
)

_SCRIPT = [
    {
        "tool": "query_perspective",
        "arguments": {"subject": "box_01", "predicate": "has_color"},
    },
    {
        "tool": "query_provenance",
        "arguments": {"subject": "box_01", "predicate": "has_color", "limit": 10},
    },
    {
        "tool": "finalize_answer",
        "arguments": {
            "answer": (
                "The historical record (third_party:bot_02 perspective) shows "
                "box_01 as blue_atom at confidence 0.7, logged by bot_02. This "
                "is a third-party log entry — not my own observation."
            ),
            "cited_beliefs": ["box_01/has_color@third_party:bot_02"],
        },
    },
]

_HISTORICAL_MARKERS = [
    "bot_02", "third", "historical", "history", "record", "third-party", "third_party",
]


def _assert_has_any(trace, candidates: list[str]) -> None:
    ans = (trace.final_answer or "").lower()
    assert any(c in ans for c in candidates), (
        f"final answer never asserts historical/third-party framing "
        f"(any of {candidates}): {trace.final_answer!r}"
    )


def test_historical_perspective_recorded(recorded_agent):
    agent, _knowledge, _sensors, scenario = recorded_agent("scenario_b", _SCRIPT)

    result = agent.run(QUERY, scenario="scenario_b")

    assert_tools_called(result.trace, ["query_perspective", "finalize_answer"])
    assert_answer_contains(result.trace, ["blue"])
    _assert_has_any(result.trace, _HISTORICAL_MARKERS)

    dump_epistemic_trace(
        result.trace,
        TEST_ID,
        extras={
            "scenario": scenario.get("id"),
            "expected_answer_contains": ["blue"],
            "expected_historical_markers": _HISTORICAL_MARKERS,
            "verdict": "pass",
            "failure_mode_tested": (
                "LLM reports blue but attributes it to itself or to the user "
                "instead of to bot_02's historical log."
            ),
        },
    )


@pytest.mark.live_llm
def test_historical_perspective_live(live_agent):
    agent, _knowledge, _sensors, scenario = live_agent("scenario_b")

    result = agent.run(QUERY, scenario="scenario_b", max_steps=8)

    assert_answer_contains(result.trace, ["blue"])
    dump_epistemic_trace(
        result.trace,
        TEST_ID,
        extras={
            "scenario": scenario.get("id"),
            "expected_answer_contains": ["blue"],
            "verdict": "pass",
        },
    )
