"""The ReAct loop: reason → tool call → observation → repeat, capped at max_steps.

Placeholder — implementation lands in Phase 4 (Owner: Person 3).
See PROJECT_PLAN.md section H.2 for the loop pseudocode.
"""

from __future__ import annotations

from iagent.procedural.models import Trace


class MaxStepsExceeded(RuntimeError):
    def __init__(self, trace: Trace) -> None:
        super().__init__(f"ReAct loop exceeded max_steps ({len(trace.steps)})")
        self.trace = trace


def run_react(agent, query: str, max_steps: int = 8) -> Trace:
    """Execute the loop and return the completed Trace."""
    raise NotImplementedError("Phase 4 — see PROJECT_PLAN.md section H.2.")
