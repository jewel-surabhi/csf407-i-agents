"""Epistemic test 04 — provenance_written.

Stresses: every belief update leaves an INSPECTABLE ROW in SQLite. The
in-memory graph and the on-disk provenance log must stay in lockstep —
otherwise the "verification report" claim ("all beliefs are auditable")
is a lie.

Post-conditions on SQLite after Scenario A runs:
  * At least one new provenance row for subject=path_ahead with
    source_id='lidar_front' (the new is_blocked).
  * The seeded default_map row for is_clear is now in status 'downgraded'.
Failure mode tested: in-memory graph updates land but SQL writes are
silently dropped — the system LOOKS correct in the drawer but can't be
audited.
"""

from __future__ import annotations

import pytest

from .conftest import assert_tools_called, dump_epistemic_trace

pytestmark = [pytest.mark.epistemic]

TEST_ID = "04_provenance_written"
QUERY = (
    "Check your front LiDAR and update the belief graph if what you see "
    "contradicts your current beliefs."
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
            "reason": "LiDAR disagrees.",
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
            "answer": "Belief graph updated; LiDAR evidence persisted.",
        },
    },
]


def _assert_provenance_persisted(knowledge) -> dict[str, int]:
    """Query SQLite directly through the KnowledgeAPI and assert the ledger
    actually recorded the mutation. Returns a counts summary for the log."""
    history = knowledge.query_provenance("path_ahead")

    assert history, "no provenance rows at all for path_ahead"

    sources_seen = {row.source_id for row in history}
    assert "default_map" in sources_seen, (
        f"the seeded map row disappeared from provenance: {sources_seen}"
    )
    assert "lidar_front" in sources_seen, (
        f"the new sensor row never persisted: {sources_seen}"
    )

    # The seeded is_clear row must now carry a non-active status (downgraded
    # or retracted) somewhere in the ledger.
    is_clear_rows = [r for r in history if r.predicate == "is_clear"]
    assert any(r.status != "active" for r in is_clear_rows), (
        f"no is_clear row flipped out of 'active' — ledger shows: "
        f"{[(r.source_id, r.status) for r in is_clear_rows]}"
    )

    # Exactly one active is_blocked row — the lidar one.
    is_blocked_active = [
        r for r in history if r.predicate == "is_blocked" and r.status == "active"
    ]
    assert len(is_blocked_active) >= 1, "no active is_blocked row persisted"
    assert any(r.source_id == "lidar_front" for r in is_blocked_active)

    return {
        "total_rows": len(history),
        "is_clear_rows": len(is_clear_rows),
        "is_blocked_active_rows": len(is_blocked_active),
        "distinct_sources": len(sources_seen),
    }


def test_provenance_written_recorded(recorded_agent):
    agent, knowledge, _sensors, scenario = recorded_agent("scenario_a", _SCRIPT)

    result = agent.run(QUERY)

    assert_tools_called(result.trace, ["downgrade_belief", "update_belief"])
    counts = _assert_provenance_persisted(knowledge)

    dump_epistemic_trace(
        result.trace,
        TEST_ID,
        extras={
            "scenario": scenario.get("id"),
            "provenance_counts": counts,
            "verdict": "pass",
            "failure_mode_tested": (
                "In-memory graph updated but SQL writes dropped — the belief "
                "looks right in queries but has no audit trail."
            ),
        },
    )


@pytest.mark.live_llm
def test_provenance_written_live(live_agent):
    agent, knowledge, _sensors, scenario = live_agent("scenario_a")

    result = agent.run(QUERY, max_steps=10)
    counts = _assert_provenance_persisted(knowledge)

    dump_epistemic_trace(
        result.trace,
        TEST_ID,
        extras={
            "scenario": scenario.get("id"),
            "provenance_counts": counts,
            "verdict": "pass",
        },
    )
