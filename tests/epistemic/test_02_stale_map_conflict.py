"""Epistemic test 02 — stale_map_conflict (Scenario A).

Stresses: detect belief-vs-sensor conflict; prioritize sensor by trust prior.
Failure mode: LLM trusts the stale map and never downgrades.

Owner: Person 4. Implementation in Phase 6. See PROJECT_PLAN.md sections I + K.
"""

import pytest

pytestmark = [
    pytest.mark.epistemic,
    pytest.mark.skip(reason="EMPTY PLACEHOLDER — implement in Phase 6."),
]


def test_stale_map_conflict():
    raise NotImplementedError("EMPTY PLACEHOLDER — implement in the phase named in the skip reason.")
