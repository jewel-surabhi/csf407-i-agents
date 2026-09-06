"""Structured JSON trace logger.

Every ReAct step is appended; on completion the full Trace is written to
`data/runs/<timestamp>.json`.

Placeholder — implementation lands in Phase 4 (Owner: Person 3).
"""

from __future__ import annotations

from pathlib import Path

from iagent.procedural.models import Step, Trace

__all__ = ["Trace", "Step", "dump_trace"]


def dump_trace(trace: Trace, runs_dir: str | Path) -> Path:
    """Serialize trace to `<runs_dir>/<started_at>.json` and return the path."""
    raise NotImplementedError("Phase 4 — see PROJECT_PLAN.md section K (test log format).")
