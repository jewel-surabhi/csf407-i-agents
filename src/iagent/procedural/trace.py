"""Structured JSON trace logger.

Every ReAct episode produces a `Trace`; `dump_trace` writes it to
`<runs_dir>/<timestamp>.json` in the exact shape PROJECT_PLAN.md section K
specifies for the test logs. Those 10 files are the graded deliverable, so the
key set here is a contract:

    test_id, query, scenario, started_at, ended_at, steps,
    final_answer, assertions, verdict

`assertions` and `verdict` are present but empty on an ordinary run — the
epistemic tests (P4) fill them in, either by passing them to `dump_trace` or by
re-writing the file after asserting. They are never omitted, so a consumer can
read the same keys from any log.

Owner: Person 3 (Phase 4). See PROJECT_PLAN.md sections H, K.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from iagent.procedural.models import Step, Trace

__all__ = ["Trace", "Step", "trace_to_log", "dump_trace"]


def trace_to_log(
    trace: Trace,
    *,
    test_id: str | None = None,
    assertions: list[dict[str, Any]] | None = None,
    verdict: str | None = None,
) -> dict[str, Any]:
    """Render `trace` as the section K test-log dict.

    `test_id` falls back to `trace.test_id`. `verdict` is derived from
    `assertions` when it isn't given explicitly: "pass" if every assertion
    passed, else "fail". With no assertions at all the verdict stays None,
    because an un-asserted run has not been judged.
    """
    rows = list(assertions or [])
    if verdict is None and rows:
        verdict = "pass" if all(row.get("passed") for row in rows) else "fail"

    body = trace.model_dump(mode="json")

    return {
        "test_id": test_id or trace.test_id,
        "query": body["query"],
        "scenario": body["scenario"],
        "started_at": body["started_at"],
        "ended_at": body["ended_at"],
        "steps": body["steps"],
        "final_answer": body["final_answer"],
        "assertions": rows,
        "verdict": verdict,
    }


def dump_trace(
    trace: Trace,
    runs_dir: str | Path,
    *,
    test_id: str | None = None,
    assertions: list[dict[str, Any]] | None = None,
    verdict: str | None = None,
    filename: str | None = None,
) -> Path:
    """Write the section K log for `trace` and return the path it landed at.

    Creates `runs_dir` if needed. The filename is, in order of precedence:
      1. `filename`, when given;
      2. `epistemic_<test_id>.json`, when a test id is in play (section K);
      3. `<started_at>.json` — microsecond precision, so concurrent runs and
         retries of the same test never collide.
    """
    runs_dir = Path(runs_dir)
    runs_dir.mkdir(parents=True, exist_ok=True)

    resolved_id = test_id or trace.test_id
    if filename is None:
        if resolved_id:
            filename = f"epistemic_{resolved_id}.json"
        else:
            filename = f"{trace.started_at.strftime('%Y%m%dT%H%M%S%f')}.json"

    log = trace_to_log(trace, test_id=resolved_id, assertions=assertions, verdict=verdict)

    path = runs_dir / filename
    path.write_text(json.dumps(log, indent=2) + "\n", encoding="utf-8")
    return path
