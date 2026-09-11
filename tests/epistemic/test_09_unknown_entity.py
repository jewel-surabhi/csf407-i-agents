"""Epistemic test 09 — unknown_entity.

Stresses: agent replies "I don't know" instead of hallucinating.
Query: "Where is object `mystery_thing_42`?"
Nothing in graph or provenance.
Expected: answer contains "unknown"/"no information"/"cannot determine"; MUST NOT
          contain a fabricated location.

Owner: Person 4. Implementation in Phase 8.
"""

import pytest

pytestmark = [
    pytest.mark.epistemic,
    pytest.mark.skip(reason="EMPTY PLACEHOLDER — implement in Phase 8."),
]


def test_unknown_entity():
    raise NotImplementedError("EMPTY PLACEHOLDER — implement in the phase named in the skip reason.")
