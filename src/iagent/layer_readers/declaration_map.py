"""Declaration map: the declarative layer's view of every path.

Each path is a subject in the belief graph with up to two predicates:

* ``is_blocked`` -> ``"true"`` / ``"false"``
* ``box_color``  -> ``"red"`` / ``"blue"`` / ``"yellow"`` (only while a box is believed there)

Confidence is read through ``KnowledgeAPI`` and decayed to the requested hour
with each belief's source decay rate (``effective = stored - rate * age``).
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

from iagent.declarative.api import KnowledgeAPI
from iagent.declarative.decay import annotate_with_decay
from iagent.declarative.models import Source
from iagent.grid.world import all_segments

__all__ = ["STATE", "COLOR", "read_declaration_map", "path_history"]

STATE = "is_blocked"
COLOR = "box_color"


def _hours(epoch: datetime, when: datetime) -> float:
    return round((when - epoch).total_seconds() / 3600, 2)


def _read(
    knowledge: KnowledgeAPI,
    sources: dict[str, Source],
    path: str,
    predicate: str,
    now: datetime,
    epoch: datetime,
) -> dict[str, Any] | None:
    belief = knowledge.get_belief(path, predicate)
    if belief is None:
        return None
    b = annotate_with_decay(belief, sources, now=now)
    src = sources.get(b.source)
    return {
        "value": b.object,
        "confidence": round(b.effective_confidence or 0.0, 3),
        "stored": round(b.confidence, 3),
        "source": b.source,
        "decay_per_hour": src.decay_rate_per_hour if src else 0.0,
        "declared_at_hour": _hours(epoch, b.observed_at),
        "age_hours": round(b.age_hours or 0.0, 2),
        "provenance_id": b.provenance_id,
    }


def read_declaration_map(
    knowledge: KnowledgeAPI,
    *,
    hour: float,
    epoch: datetime,
) -> dict[str, dict[str, Any]]:
    """Every declared path at ``hour``: state and (if any) box colour, each with decayed confidence."""
    sources = {s.source_id: s for s in knowledge.list_sources()}
    now = epoch + timedelta(hours=hour)
    out: dict[str, dict[str, Any]] = {}
    for seg in all_segments():
        pid = seg["id"]
        state = _read(knowledge, sources, pid, STATE, now, epoch)
        if state is None:
            continue
        color = _read(knowledge, sources, pid, COLOR, now, epoch)
        out[pid] = {
            "path": pid,
            "state": "blocked" if state["value"] == "true" else "clear",
            "state_conf": state["confidence"],
            "state_meta": state,
            "color": color["value"] if color else None,
            "color_conf": color["confidence"] if color else None,
            "color_meta": color,
        }
    return out


def path_history(knowledge: KnowledgeAPI, path: str, *, epoch: datetime, limit: int = 10) -> list[dict[str, Any]]:
    """Provenance log for one path (both predicates), newest first."""
    rows = knowledge.query_provenance(path, limit=limit)
    return [
        {
            "predicate": r.predicate,
            "value": r.object,
            "confidence": r.confidence,
            "source": r.source_id,
            "status": r.status,
            "hour": _hours(epoch, r.observed_at),
            "reason": (r.payload or {}).get("reason"),
        }
        for r in rows
    ]
