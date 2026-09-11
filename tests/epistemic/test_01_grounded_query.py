"""Epistemic test 01 — grounded_query.

Stresses: baseline. Agent answers a fact already in memory without hallucinating.
Query: "Where are you?"
Expected: get_belief("robot","located_at") -> finalize_answer containing "room_101".

Owner: Person 4. Implementation in Phase 8. See PROJECT_PLAN.md section K.
"""

import pytest

pytestmark = [
    pytest.mark.epistemic,
    pytest.mark.skip(reason="EMPTY PLACEHOLDER — implement in Phase 8."),
]


def test_grounded_query():
    raise NotImplementedError("EMPTY PLACEHOLDER — implement in the phase named in the skip reason.")
