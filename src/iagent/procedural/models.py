"""Pydantic models for ReAct step logging.

`Trace` is the project deliverable: PROJECT_PLAN.md section K requires one JSON
log per epistemic test, and these models define its shape. Field names are part
of that contract — the epistemic tests (P4) extend the dumped file with
`assertions` and `verdict`, so renaming anything here breaks the test logs.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from iagent.declarative.api import Belief

__all__ = ["ToolCall", "Observation", "Step", "Trace", "AgentResult"]


class ToolCall(BaseModel):
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class Observation(BaseModel):
    tool: str
    ok: bool
    content: dict[str, Any] | str


class Step(BaseModel):
    idx: int
    thought: str | None = None
    tool_call: ToolCall | None = None
    observation: Observation | None = None
    final_answer: str | None = None


class Trace(BaseModel):
    query: str
    scenario: str | None = None
    steps: list[Step] = Field(default_factory=list)
    final_answer: str = ""
    started_at: datetime
    ended_at: datetime | None = None
    test_id: str | None = Field(
        default=None,
        description="Set by the epistemic tests so the log names its own test.",
    )

    # ---- convenience API (PROJECT_PLAN.md section E: Trace.append / Trace.dump)

    def append(self, step: Step) -> Step:
        """Record one step. Returns it so callers can keep a handle."""
        self.steps.append(step)
        return step

    def dump(self, runs_dir: str | Path, **kwargs: Any) -> Path:
        """Write this trace to `runs_dir` in section K's log format."""
        # Imported here, not at module scope: trace.py imports this module.
        from iagent.procedural.trace import dump_trace

        return dump_trace(self, runs_dir, **kwargs)


class AgentResult(BaseModel):
    answer: str
    trace: Trace
    beliefs_written: list[Belief] = Field(
        default_factory=list,
        description=(
            "Every belief this episode persisted, in write order — the "
            "provenance rows returned by update_belief / downgrade_belief. "
            "Check `.status` to tell an assertion from a demotion."
        ),
    )
    tools_called: list[str] = Field(default_factory=list)
