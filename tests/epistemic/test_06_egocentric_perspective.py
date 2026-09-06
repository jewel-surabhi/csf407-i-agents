"""Epistemic test 06 — egocentric_perspective.

Stresses: agent reports its own live sensor perception.
Query: "What do you currently see box_01 as?"
Expected: sense_camera -> answer containing "brown" and a "currently"/"live"/"now" token.

Owner: Person 4. Implementation in Phase 7.
"""

import pytest

pytestmark = [
    pytest.mark.epistemic,
    pytest.mark.skip(reason="EMPTY PLACEHOLDER — implement in Phase 7."),
]


def test_egocentric_perspective():
    assert False
