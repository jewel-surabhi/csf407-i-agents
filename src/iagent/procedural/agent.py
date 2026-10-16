"""Agent -- top-level facade. Composes KnowledgeAPI + SensorimotorAPI + LLM
and exposes `Agent.run(query)` as the single entry point.

The real work happens inside `react_loop.run_react` -- this class is a thin
wrapper that holds the dependencies and summarizes the completed Trace into
an `AgentResult`.

Owner: Person 4 (Phase 4). See PROJECT_PLAN.md section G.3.
"""

from __future__ import annotations

from typing import Any

from iagent.declarative.api import KnowledgeAPI
from iagent.procedural.models import AgentResult
from iagent.procedural.react_loop import run_react
from iagent.sensorimotor.api import SensorimotorAPI

__all__ = ["Agent", "AgentResult"]


class Agent:
    """Compose the three layers behind one `run(query)` call.

    `llm` is anything exposing `chat_with_tools(messages, tools)` -- the real
    `LLMClient` for live runs, `RecordedLLMClient` for tests. `config` is
    optional and reserved for future runtime knobs (default max_steps, trace
    output directory, etc.); the loop reads nothing from it today.
    """

    def __init__(
        self,
        knowledge: KnowledgeAPI,
        sensors: SensorimotorAPI,
        llm: Any,
        config: Any = None,
    ) -> None:
        self.knowledge = knowledge
        self.sensors = sensors
        self.llm = llm
        self.config = config

    def run(
        self,
        query: str,
        *,
        scenario: str | None = None,
        max_steps: int = 8,
    ) -> AgentResult:
        """Execute one ReAct episode and return the final answer + Trace.

        Raises `react_loop.MaxStepsExceeded` if the LLM never calls
        `finalize_answer` within `max_steps` iterations.
        """
        trace = run_react(self, query, scenario=scenario, max_steps=max_steps)
        tools_called = [
            step.tool_call.name
            for step in trace.steps
            if step.tool_call is not None
        ]
        return AgentResult(
            answer=trace.final_answer,
            trace=trace,
            tools_called=tools_called,
        )
