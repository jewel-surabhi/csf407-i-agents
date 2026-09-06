"""Shared pytest fixtures.

Placeholder fixtures until each layer lands. Owner: Person 4.
"""

from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture()
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture()
def tmp_db(tmp_path: Path) -> Path:
    """Fresh SQLite DB path in a tmp dir. Actual creation lives in Phase 2 tests."""
    return tmp_path / "iagent.db"


# TODO(P4, Phase 5): add fixtures that:
#   - build a seeded KnowledgeAPI against tmp_db
#   - build a MockEnvironment from a scenario YAML
#   - build an Agent wired with a RecordedLLMClient
