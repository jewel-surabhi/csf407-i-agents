"""YAML-driven dynamic events for the warehouse demo.

The warehouse state is otherwise static: beliefs are seeded at startup and only
mutate through explicit UI actions (move, transfer). This module adds a lightweight
event layer so scenarios can EVOLVE as the session progresses — new beliefs enter
the graph mid-session, sensors change, perspectives shift. The agent then has to
reason over information it did not have at startup.

Trigger types (first pass):
  * `after_moves`: fires once when the agent has made N position moves.
  * `on_enter`:    fires once when the agent moves INTO a named junction.

Action types (first pass):
  * `update_belief`: write a new belief through `KnowledgeAPI.update_belief` — same
                     code path as every other belief, with full provenance.
  * `toast`:         UI-only notification, no state change.

Each event has a stable `id`; once fired, the engine remembers that id and skips
it on subsequent ticks. That means re-entering a junction, or crossing the same
step threshold again after a reset, won't re-fire the event unless the engine is
reset explicitly.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from iagent.declarative.api import KnowledgeAPI
from iagent.declarative.models import Belief, KnowledgeError

__all__ = ["EventEngine", "FiredEvent"]


class FiredEvent:
    """What the UI gets back after an event fires."""

    def __init__(self, event_id: str, action: str, message: str) -> None:
        self.event_id = event_id
        self.action = action
        self.message = message

    def to_dict(self) -> dict[str, Any]:
        return {"event_id": self.event_id, "action": self.action, "message": self.message}


class EventEngine:
    """Owns the loaded event list, a move counter, and a 'fired' id set."""

    def __init__(self, events_yaml: str | Path | None = None) -> None:
        self._events: list[dict[str, Any]] = []
        self._fired: set[str] = set()
        self._moves: int = 0
        if events_yaml is not None and Path(events_yaml).exists():
            self.load(events_yaml)

    def load(self, yaml_path: str | Path) -> None:
        """Parse `events:` from a YAML file. Idempotent — re-loads replace the list."""
        try:
            data = yaml.safe_load(Path(yaml_path).read_text(encoding="utf-8")) or {}
        except (OSError, yaml.YAMLError) as e:
            raise KnowledgeError(f"cannot parse events YAML {yaml_path}: {e}") from e

        events = data.get("events") or []
        self._events = []
        for i, ev in enumerate(events):
            ev.setdefault("id", f"ev_{i:03d}")
            self._events.append(ev)

    def reset(self) -> None:
        """Clear the fired-set and move counter so events can fire again."""
        self._fired.clear()
        self._moves = 0

    @property
    def moves_made(self) -> int:
        return self._moves

    @property
    def fired_ids(self) -> set[str]:
        return set(self._fired)

    # ---- the tick: called after every move ----

    def on_move(self, knowledge: KnowledgeAPI, new_position: str) -> list[FiredEvent]:
        """Increment the move counter, fire any newly-triggered events.

        Returns the list of events that fired on THIS move (possibly empty).
        Each fired event's action has already been applied to the belief graph.
        """
        self._moves += 1
        fired: list[FiredEvent] = []
        for ev in self._events:
            if ev["id"] in self._fired:
                continue
            if not self._matches(ev.get("trigger") or {}, new_position):
                continue
            # Trigger matched — apply the action and mark fired.
            msg = self._apply(ev, knowledge)
            self._fired.add(ev["id"])
            fired.append(FiredEvent(ev["id"], self._action_name(ev), msg))
        return fired

    # ---- internals ----

    def _matches(self, trigger: dict[str, Any], new_position: str) -> bool:
        if "after_moves" in trigger:
            return self._moves >= int(trigger["after_moves"])
        if "on_enter" in trigger:
            return str(trigger["on_enter"]) == new_position
        return False

    def _action_name(self, ev: dict[str, Any]) -> str:
        fires = ev.get("fires") or {}
        for k in ("update_belief", "toast"):
            if k in fires:
                return k
        return "unknown"

    def _apply(self, ev: dict[str, Any], knowledge: KnowledgeAPI) -> str:
        """Execute the event's action. Returns a short human-readable summary
        the UI can toast."""
        fires = ev.get("fires") or {}

        if "update_belief" in fires:
            spec = fires["update_belief"]
            belief = Belief(
                subject=str(spec["subject"]),
                predicate=str(spec["predicate"]),
                object=str(spec["object"]),
                confidence=float(spec.get("confidence", 0.7)),
                source=str(spec.get("source", "default_map")),
                perspective=str(spec.get("perspective", "self")),
                observed_at=datetime.now(UTC),
            )
            reason = str(spec.get("reason", f"event:{ev['id']}"))
            knowledge.update_belief(belief, reason=reason)
            return (
                ev.get("summary")
                or f"{belief.source} wrote {belief.subject} {belief.predicate}={belief.object}"
            )

        if "toast" in fires:
            return str(fires["toast"])

        return "event fired"
