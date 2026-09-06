"""Agent — top-level facade. Composes KnowledgeAPI + SensorimotorAPI + LLMClient.

Placeholder — implementation lands in Phase 4 (Owner: Person 3).
"""

from __future__ import annotations

from iagent.declarative.api import KnowledgeAPI
from iagent.procedural.llm_client import LLMClient
from iagent.procedural.models import AgentResult
from iagent.sensorimotor.api import SensorimotorAPI

__all__ = ["Agent", "AgentResult"]


class Agent:
    def __init__(
        self,
        knowledge: KnowledgeAPI,
        sensors: SensorimotorAPI,
        llm: LLMClient,
        config,
    ) -> None:
        # EMPTY PLACEHOLDER — IMPLEMENTATION LATER (Phase 4, Person 3).
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
        """Execute one ReAct episode. Returns final answer + full trace."""
        raise NotImplementedError("Phase 4 — see PROJECT_PLAN.md section G.3.")
