"""Fails CI if any layer imports another's internals.

Runs the same check as `scripts/verify_isolation.py`. Owner: Person 4.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from verify_isolation import check  # noqa: E402


def test_no_layer_isolation_violations() -> None:
    violations = check()
    assert not violations, "Layer isolation violated:\n  " + "\n  ".join(violations)
