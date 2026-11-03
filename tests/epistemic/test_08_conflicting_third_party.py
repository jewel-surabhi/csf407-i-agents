"""Epistemic test 08 — perspective_no_merge  (THE Scenario B test).

Stresses: all THREE perspectives (user, self/egocentric, third-party/
historical) kept SYNTACTICALLY DISTINCT in a single final answer. No
"the box is blue-brown to me and the user" merging.

Scenario: scenario_b.
Query:    scenario's own query — "What color does the user think the object
          is, what color do you register it as, and what does your data
          history say its true state is?"
Expected: `sense_camera` + `query_perspective` -> `finalize_answer` that
          separately mentions red (user), brown (self/live), blue (history/
          third-party), each attached to its own perspective label.
Failure mode: LLM averages the perspectives, or picks one and drops the
others, or re-attributes (e.g. says the user believes brown).
"""

from __future__ import annotations

import pytest

from .conftest import (
    assert_answer_contains,
    assert_tools_called,
    dump_epistemic_trace,
)

pytestmark = [pytest.mark.epistemic]

TEST_ID = "08_perspective_no_merge"

_SCRIPT = [
    {
        "tool": "query_perspective",
        "arguments": {"subject": "box_01", "predicate": "has_color"},
    },
    {
        "tool": "sense_camera",
        "arguments": {"direction": "front"},
    },
    {
        "tool": "finalize_answer",
        "arguments": {
            "answer": (
                "Three distinct perspectives on box_01's color:\n"
                "- USER perspective: red_atom (confidence 0.9, source=user).\n"
                "- SELF / egocentric: brown as perceived by my front camera "
                "right now (confidence 0.82; ambient lighting yellow, perception "
                "unreliable).\n"
                "- THIRD-PARTY / historical: blue_atom per bot_02's log "
                "(confidence 0.7).\n"
                "I keep these separate — I do not know the true color."
            ),
            "cited_beliefs": [
                "box_01/has_color@user",
                "box_01/has_color@third_party:bot_02",
            ],
            "cited_sensors": ["camera_front"],
        },
    },
]


def _assert_perspectives_distinct(trace) -> dict[str, bool]:
    """Each (color, perspective-marker) pair must appear in the same answer."""
    ans = (trace.final_answer or "").lower()
    checks = {
        "user_red": "red" in ans and "user" in ans,
        "self_brown": "brown" in ans and any(
            m in ans for m in ("self", "egocentric", "camera", "current", "now")
        ),
        "third_party_blue": "blue" in ans and any(
            m in ans for m in ("bot_02", "third", "historical", "history", "record")
        ),
    }
    failing = [k for k, ok in checks.items() if not ok]
    assert not failing, (
        f"perspectives merged or dropped: missing {failing}. "
        f"Final answer: {trace.final_answer!r}"
    )
    return checks


_LIVE_QUERY = (
    "For subject='box_01' predicate='has_color', first call "
    "query_perspective(subject='box_01', predicate='has_color') to retrieve "
    "every stored perspective, then call sense_camera(direction='front') to "
    "get your own current reading. Finally report ALL THREE perspectives "
    "separately in finalize_answer: what the user believes, what you "
    "currently see, and what the historical third-party log says."
)


def test_perspective_no_merge_recorded(recorded_agent):
    scenario_query = (
        "What color does the user think the object is, what color do you "
        "register it as, and what does your data history say its true state is?"
    )
    agent, _knowledge, _sensors, scenario = recorded_agent("scenario_b", _SCRIPT)

    result = agent.run(scenario_query, scenario="scenario_b")

    assert_tools_called(result.trace, ["query_perspective", "sense_camera", "finalize_answer"])
    # All three colors must appear.
    assert_answer_contains(result.trace, ["red", "brown", "blue"])
    # And must be attached to distinct perspective markers.
    checks = _assert_perspectives_distinct(result.trace)

    dump_epistemic_trace(
        result.trace,
        TEST_ID,
        extras={
            "scenario": scenario.get("id"),
            "expected_answer_contains": ["red", "brown", "blue", "user"],
            "perspective_checks": checks,
            "verdict": "pass",
            "failure_mode_tested": (
                "LLM averages the three perspectives, re-attributes them, or "
                "drops one — producing 'the box is blue-brown' or 'the user "
                "says brown'."
            ),
        },
    )


@pytest.mark.live_llm
def test_perspective_no_merge_live(live_agent):
    agent, _knowledge, _sensors, scenario = live_agent("scenario_b")

    result = agent.run(_LIVE_QUERY, scenario="scenario_b", max_steps=10)

    assert_answer_contains(result.trace, ["red", "brown", "blue"])
    checks = _assert_perspectives_distinct(result.trace)

    dump_epistemic_trace(
        result.trace,
        TEST_ID,
        extras={
            "scenario": scenario.get("id"),
            "expected_answer_contains": ["red", "brown", "blue"],
            "perspective_checks": checks,
            "verdict": "pass",
        },
    )
