"""Epistemic test 07 — historical_perspective.

Stresses: agent surfaces a third-party log entry without merging it into other views.
Query: "What does the maintenance history say box_01's color is?"
Expected: query_provenance -> answer containing "blue" and one of "bot_02"/"maintenance".

Owner: Person 4. Implementation in Phase 7.
"""

import pytest

pytestmark = [
    pytest.mark.epistemic,
    pytest.mark.skip(reason="EMPTY PLACEHOLDER — implement in Phase 7."),
]


def test_historical_perspective():
    raise NotImplementedError("EMPTY PLACEHOLDER — implement in the phase named in the skip reason.")
