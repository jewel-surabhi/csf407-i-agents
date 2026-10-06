"""MockEnvironment — owns the mutable world state the sensors read from.

The environment is intentionally dumb: it stores a `WorldState` and hands it to
pure sensor/actuator functions. All I/O and derivation live in `sensors.py` and
`actuators.py`.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from iagent.sensorimotor.models import WorldState
from iagent.sensorimotor.scenarios import load_scenario

__all__ = ["MockEnvironment"]


# Deterministic clock for sensor timestamps. Sensors must be reproducible —
# same world → same reading — so we do NOT use datetime.utcnow(). Tests can
# override via `MockEnvironment(world_state, now=...)`.
DEFAULT_NOW = datetime(2026, 9, 6, 10, 15, 0, 234000, tzinfo=UTC)


class MockEnvironment:
    """Container for the current world state."""

    def __init__(
        self,
        world_state: WorldState | None = None,
        now: datetime | str = DEFAULT_NOW,
    ) -> None:
        self.world_state: WorldState = world_state or WorldState()
        self.now: datetime = _coerce_now(now)

    @classmethod
    def from_scenario(
        cls, path: str | Path, now: datetime | str = DEFAULT_NOW
    ) -> MockEnvironment:
        world_state, _scenario = load_scenario(path)
        return cls(world_state=world_state, now=now)

    def describe(self) -> dict:
        """Debug snapshot of the world. Never exposed to the LLM."""
        return {
            "now": self.now.isoformat(),
            "world": self.world_state.model_dump(),
        }


def _coerce_now(now: datetime | str) -> datetime:
    if isinstance(now, datetime):
        return now
    # ISO-8601; accept trailing Z as UTC.
    return datetime.fromisoformat(now.replace("Z", "+00:00"))
