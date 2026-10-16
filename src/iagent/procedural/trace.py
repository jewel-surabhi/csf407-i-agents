"""Structured JSON trace logger.

Every ReAct episode produces a `Trace`. `dump_trace` serializes it to a
timestamped file under the configured runs directory. These files are the
required artifact for the 10 epistemic tests (section K of PROJECT_PLAN.md).

Owner: Person 4 (Phase 4).
"""

from __future__ import annotations

from pathlib import Path

from iagent.procedural.models import Step, Trace

__all__ = ["Trace", "Step", "dump_trace"]


def dump_trace(trace: Trace, runs_dir: str | Path) -> Path:
    """Write `trace` to `<runs_dir>/trace_<started_at>.json` and return the path.

    Creates `runs_dir` if it does not exist. Uses the Trace's `started_at`
    timestamp (microsecond precision) in the filename so concurrent runs
    never collide.
    """
    runs_dir = Path(runs_dir)
    runs_dir.mkdir(parents=True, exist_ok=True)

    stamp = trace.started_at.strftime("%Y%m%dT%H%M%S%f")
    filename = runs_dir / f"trace_{stamp}.json"

    filename.write_text(
        trace.model_dump_json(indent=2),
        encoding="utf-8",
    )
    return filename
