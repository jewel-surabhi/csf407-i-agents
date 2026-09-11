"""Epistemic test 03 — sensor_confidence_update.

Stresses: after Scenario A, the numerical confidence in the graph actually changes.
Post-condition: get_belief("path_ahead","is_clear").confidence <= 0.3
                and .status == "downgraded".

Owner: Person 4. Implementation in Phase 6.
"""

import pytest

pytestmark = [
    pytest.mark.epistemic,
    pytest.mark.skip(reason="EMPTY PLACEHOLDER — implement in Phase 6."),
]


def test_sensor_confidence_update():
    raise NotImplementedError("EMPTY PLACEHOLDER — implement in the phase named in the skip reason.")
