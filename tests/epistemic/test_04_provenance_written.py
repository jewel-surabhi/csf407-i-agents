"""Epistemic test 04 — provenance_written.

Stresses: every belief update leaves an inspectable SQLite row.
Failure mode: in-memory graph updated but nothing persisted.

Owner: Person 4. Implementation in Phase 6.
"""

import pytest

pytestmark = [
    pytest.mark.epistemic,
    pytest.mark.skip(reason="EMPTY PLACEHOLDER — implement in Phase 6."),
]


def test_provenance_written():
    assert False
