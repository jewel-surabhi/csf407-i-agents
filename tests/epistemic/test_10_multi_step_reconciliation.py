"""Epistemic test 10 — multi_step_reconciliation.

Stresses: multiple sources disagree; agent picks by trust prior and logs both.

Initial state:
    path_ahead is_clear (default_map, 1.0) AND path_ahead is_clear (camera_front, 0.7)
Live LiDAR: blocked.

Expected: agent calls both sensors, notices lidar disagrees, prefers lidar over camera
          (both are sensors but lidar's trust_prior=0.9 > camera's 0.7), downgrades map
          AND camera beliefs. Exactly one active is_blocked belief with source lidar_front;
          two downgraded rows.

Owner: Person 4. Implementation in Phase 8.
"""

import pytest

pytestmark = [
    pytest.mark.epistemic,
    pytest.mark.skip(reason="EMPTY PLACEHOLDER — implement in Phase 8."),
]


def test_multi_step_reconciliation():
    assert False
