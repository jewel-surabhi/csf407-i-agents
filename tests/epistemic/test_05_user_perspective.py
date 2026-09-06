"""Epistemic test 05 — user_perspective.

Stresses: agent isolates the user's belief on request.
Query: "What color does the user believe box_01 is?"
Expected: query_perspective -> answer containing "red" and "user".

Owner: Person 4. Implementation in Phase 7.
"""

import pytest

pytestmark = [
    pytest.mark.epistemic,
    pytest.mark.skip(reason="EMPTY PLACEHOLDER — implement in Phase 7."),
]


def test_user_perspective():
    assert False
