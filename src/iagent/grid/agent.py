"""One question on the grid dashboard.

    1. move the clock to ``clock + hours``
    2. ``decision.total_map.decide``: declaration map (declarative layer) +
       sensor map (sensorimotor layer) -> the procedural layer's LLM submits
       the Total map and answers the question
    3. ``decision.total_map.write_back``: store the LLM's sensor-backed winners
    4. record everything for the timeline
"""

from __future__ import annotations

from typing import Any

from iagent.decision.total_map import decide, write_back
from iagent.grid.session import GridSession

__all__ = ["run_grid_question"]


def run_grid_question(session: GridSession, llm: Any, question: str, hours: int) -> dict[str, Any]:
    if not session.declared:
        raise ValueError("complete the setup before asking a question")
    if hours < 1:
        raise ValueError("time set must be at least +1 hour")

    hour = session.clock_h + hours
    world = session.world
    decision = decide(
        knowledge=session.knowledge,
        world=world,
        llm=llm,
        question=question,
        hour=hour,
        clock_h=session.clock_h,
        epoch=session.epoch,
    )
    writes = write_back(session.knowledge, decision, epoch=session.epoch)
    session.clock_h = hour

    record = {
        "hour": hour,
        "hours_added": hours,
        "question": question,
        "answer": decision.answer or "The model did not give a final answer within the step limit.",
        "finalized": decision.answer is not None,
        "submitted": decision.submitted,
        "issues": decision.issues,
        "missing": decision.missing,
        "steps": decision.steps,
        "robot": world.robot,
        "target": world.target,
        "world": world.to_dict(),
        "declarations": decision.declarations,
        "sensor": decision.sensor,
        "total": decision.total,
        "route": decision.route,
        "writes": writes,
    }
    session.history.append(record)
    return record
