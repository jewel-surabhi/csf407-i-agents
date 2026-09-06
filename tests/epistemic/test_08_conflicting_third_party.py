"""Epistemic test 08 — conflicting_third_party (Scenario B, full).

Stresses: three perspectives kept syntactically distinct in the final answer.
Failure mode: merging perspectives (e.g. "the box is blue-brown to me and the user").

Assertion helper (to write in Phase 7):
    assert_perspectives_distinct(answer, {"user":"red","self":"brown","third_party":"blue"})

Owner: Person 4. Implementation in Phase 7. See PROJECT_PLAN.md section J.
"""

import pytest

pytestmark = [
    pytest.mark.epistemic,
    pytest.mark.skip(reason="EMPTY PLACEHOLDER — implement in Phase 7."),
]


def test_conflicting_third_party():
    assert False
