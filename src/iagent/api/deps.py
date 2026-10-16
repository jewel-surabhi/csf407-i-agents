"""Agent singleton + FastAPI dependency.

One `Agent` is built lazily on the first request and reused thereafter. The
singleton pattern matches the agent's real lifecycle: the DB connection,
belief graph, and LLM client are all expensive to construct and should
survive across requests.

Tests override `get_agent` via `app.dependency_overrides` to inject a
`RecordedLLMClient`-backed agent so no test ever hits Groq.

Owner: Person 4.
"""

from __future__ import annotations

import os
from pathlib import Path

from iagent.declarative.api import KnowledgeAPI
from iagent.declarative.belief_graph import BeliefGraph
from iagent.declarative.provenance import ProvenanceStore
from iagent.declarative.seeding import seed_sources
from iagent.procedural.agent import Agent
from iagent.procedural.llm_client import LLMClient
from iagent.sensorimotor.api import SensorimotorAPI
from iagent.sensorimotor.mock_env import MockEnvironment

_REPO = Path(__file__).resolve().parents[3]
_DEFAULT_SEED_SQL = _REPO / "data" / "seed" / "initial_provenance.sql"
_DEFAULT_SCENARIO = _REPO / "configs" / "scenario_a.yaml"
_DEFAULT_DB_PATH = _REPO / "iagent.db"

_agent: Agent | None = None


def get_agent() -> Agent:
    """FastAPI dependency: return the singleton Agent, building on first call.

    Required env vars (only on first call):
      GROQ_API_KEY  — Groq token (unless caller injects a different LLM via
                      app.dependency_overrides)

    Optional env vars:
      IAGENT_DB_PATH  — defaults to ./iagent.db at the repo root
      IAGENT_SCENARIO — defaults to configs/scenario_a.yaml
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
    db_path = os.environ.get("IAGENT_DB_PATH", str(_DEFAULT_DB_PATH))
    scenario_yaml = os.environ.get("IAGENT_SCENARIO", str(_DEFAULT_SCENARIO))

    store = ProvenanceStore(db_path)
    seed_sources(store, _DEFAULT_SEED_SQL)  # idempotent

    graph = BeliefGraph()
    knowledge = KnowledgeAPI(graph, store)

    env = MockEnvironment.from_scenario(scenario_yaml)
    sensors = SensorimotorAPI(env)

    llm = LLMClient()  # reads GROQ_API_KEY from env, raises LLMClientError if absent

    return Agent(knowledge, sensors, llm)
