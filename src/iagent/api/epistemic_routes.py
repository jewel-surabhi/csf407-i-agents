"""HTTP routes serving the 10 committed epistemic test trace logs.

These logs (`data/runs/epistemic/*.json`) ARE the proposal's main graded
deliverable -- they're produced once by the pytest suite and committed, not
generated at request time. This module just reads them back out so the
frontend can render a visual pass/fail dashboard instead of a grader having
to open 20 JSON files by hand.

Read-only, no side effects, no LLM calls -- safe to hit from the UI on every
page load.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(prefix="/api/epistemic", tags=["epistemic"])

_REPO = Path(__file__).resolve().parents[3]
_RUNS_DIR = _REPO / "data" / "runs" / "epistemic"

# epistemic_<NN>_<slug>.json or epistemic_<NN>_<slug>_live.json
_FILENAME_RE = re.compile(r"^epistemic_(\d+)_(.+?)(_live)?\.json$")


class StepSummary(BaseModel):
    idx: int
    tool_name: str | None = None
    ok: bool | None = None


class TraceSummary(BaseModel):
    test_id: str
    number: int
    slug: str
    title: str
    mode: str  # "recorded" | "live"
    verdict: str  # "pass" | "fail" | "unknown"
    scenario: str | None = None
    query: str
    final_answer: str
    step_count: int
    steps: list[StepSummary]
    failure_mode_tested: str | None = None


def _title_from_slug(slug: str) -> str:
    return slug.replace("_", " ").title()


def _load_one(path: Path) -> TraceSummary | None:
    m = _FILENAME_RE.match(path.name)
    if not m:
        return None
    number = int(m.group(1))
    slug = m.group(2)
    is_live = m.group(3) is not None

    try:
        data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None

    extras = data.get("extras") or {}
    steps = [
        StepSummary(
            idx=s.get("idx", i),
            tool_name=(s.get("tool_call") or {}).get("name"),
            ok=(s.get("observation") or {}).get("ok"),
        )
        for i, s in enumerate(data.get("steps") or [])
    ]

    return TraceSummary(
        test_id=data.get("test_id") or f"{number:02d}_{slug}",
        number=number,
        slug=slug,
        title=_title_from_slug(slug),
        mode="live" if is_live else data.get("mode", "recorded"),
        verdict=extras.get("verdict", "unknown"),
        scenario=extras.get("scenario"),
        query=data.get("query", ""),
        final_answer=data.get("final_answer", ""),
        step_count=len(steps),
        steps=steps,
        failure_mode_tested=extras.get("failure_mode_tested"),
    )


@router.get("/tests", response_model=list[TraceSummary])
def list_epistemic_tests() -> list[TraceSummary]:
    """Every committed trace log (recorded + live), sorted by test number
    then recorded-before-live, so the UI can pair them side by side."""
    if not _RUNS_DIR.exists():
        return []
    summaries = [s for s in (_load_one(p) for p in _RUNS_DIR.glob("*.json")) if s is not None]
    summaries.sort(key=lambda s: (s.number, s.mode))
    return summaries
