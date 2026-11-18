"""Agent singleton + FastAPI dependency.

One `Agent` is built lazily on the first request and reused thereafter. The
singleton pattern matches the agent's real lifecycle: the DB connection,
belief graph, and LLM client are all expensive to construct and should
survive across requests.

The agent is backed by the SAME belief graph the warehouse endpoints use,
so questions typed into the ask bar can reach every warehouse belief (31
segments, 10 stock rows across 3 perspectives for WIDGET-42, the full
provenance log, world-facts seed, etc.) — not just the two tiny
scenario_a beliefs the agent used to carry.

Tests override `get_agent` via `app.dependency_overrides` to inject a
`RecordedLLMClient`-backed agent so no test ever hits Groq.

Owner: Person 4.
"""

from __future__ import annotations

import os
from pathlib import Path

from iagent.procedural.agent import Agent
from iagent.procedural.llm_client import LLMClient
from iagent.sensorimotor.api import SensorimotorAPI
from iagent.sensorimotor.mock_env import MockEnvironment

_REPO = Path(__file__).resolve().parents[3]
_DEFAULT_SCENARIO = _REPO / "configs" / "scenario_a.yaml"

_agent: Agent | None = None


def get_agent() -> Agent:
    """FastAPI dependency: return the singleton Agent, building on first call.

    Required env vars (only on first call):
      GROQ_API_KEY  — Groq token (unless caller injects a different LLM via
                      app.dependency_overrides)

    Optional env vars:
      IAGENT_SCENARIO — defaults to configs/scenario_a.yaml (drives sensors)
    """
    global _agent
    if _agent is None:
        _agent = _build_agent()
    return _agent


def reset_agent() -> None:
    """Force rebuild on next `get_agent()`. Used by scenario-reset endpoints
    and by tests that want a clean singleton."""
    global _agent
    _agent = None


def _build_agent() -> Agent:
    # Import locally to avoid a circular import — warehouse_routes imports
    # several declarative modules that would pull deps.py in at import time.
    from iagent.api.warehouse_routes import get_warehouse

    # Share the warehouse's KnowledgeAPI so the ask-bar agent can answer
    # questions about shelves, stock, segments and perspectives — not just
    # the two tiny scenario beliefs it used to carry.
    warehouse = get_warehouse()
    knowledge = warehouse.knowledge

    scenario_yaml = os.environ.get("IAGENT_SCENARIO", str(_DEFAULT_SCENARIO))
    env = MockEnvironment.from_scenario(scenario_yaml)
    sensors = SensorimotorAPI(env)

    llm = LLMClient()  # reads GROQ_API_KEY from env, raises LLMClientError if absent

    return Agent(knowledge, sensors, llm)
