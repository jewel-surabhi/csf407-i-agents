"""Tests for the HTTP API.

Uses FastAPI's `TestClient`. The Agent is replaced via `dependency_overrides`
with one backed by a `RecordedLLMClient`, so no test ever hits Groq.

Owner: Person 4.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from iagent.api.deps import get_agent
from iagent.api.server import app
from iagent.declarative.api import Belief, KnowledgeAPI
from iagent.declarative.belief_graph import BeliefGraph
from iagent.declarative.provenance import ProvenanceStore
from iagent.declarative.seeding import seed_sources
from iagent.procedural.agent import Agent
from iagent.procedural.llm_client import RecordedLLMClient
from iagent.sensorimotor.api import SensorimotorAPI
from iagent.sensorimotor.mock_env import MockEnvironment

_REPO = Path(__file__).resolve().parents[2]
_SEED = _REPO / "data" / "seed" / "initial_provenance.sql"
_SCENARIO_A = _REPO / "configs" / "scenario_a.yaml"


# ------------------------- fixtures -----------------------------------------


def _make_agent(tmp_path: Path, script: list) -> tuple[Agent, ProvenanceStore]:
    store = ProvenanceStore(str(tmp_path / "iagent.db"))
    seed_sources(store, _SEED)
    knowledge = KnowledgeAPI(BeliefGraph(), store)
    sensors = SensorimotorAPI(MockEnvironment.from_scenario(_SCENARIO_A))
    llm = RecordedLLMClient(script)
    return Agent(knowledge, sensors, llm), store


@pytest.fixture()
def client_with_recorded_agent(tmp_path):
    """TestClient wired to a RecordedLLMClient-backed Agent.

    Callers set the script via `set_script(...)` on the returned object.
    """

    class _Harness:
        def __init__(self) -> None:
            self._store = None
            self.client = None

        def set_script(self, script: list) -> None:
            agent, self._store = _make_agent(tmp_path, script)
            app.dependency_overrides[get_agent] = lambda: agent
            self.client = TestClient(app)

        def close(self) -> None:
            app.dependency_overrides.clear()
            if self._store is not None:
                self._store.close()

    h = _Harness()
    yield h
    h.close()


# ------------------------- /api/health --------------------------------------


def test_health_endpoint_does_not_require_agent():
    """/api/health must work without GROQ_API_KEY or any Agent setup."""
    with TestClient(app) as c:
        resp = c.get("/api/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert "version" in body


# ------------------------- /api/query ---------------------------------------


def test_query_returns_answer_trace_and_tools_called(client_with_recorded_agent):
    client_with_recorded_agent.set_script([
        {"tool": "finalize_answer",
         "arguments": {"answer": "I am I, Agent."}},
    ])
    resp = client_with_recorded_agent.client.post(
        "/api/query", json={"query": "Who are you?"}
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["answer"] == "I am I, Agent."
    assert body["tools_called"] == ["finalize_answer"]
    assert body["trace"]["query"] == "Who are you?"
    assert body["trace"]["steps"][0]["tool_call"]["name"] == "finalize_answer"


def test_query_runs_full_scenario_a_pipeline(client_with_recorded_agent):
    """End-to-end: the API composes KnowledgeAPI + Sensors + Loop + Dispatch
    through a single HTTP call."""
    # Seed the stale map belief that scenario A relies on.
    client_with_recorded_agent.set_script([
        {"tool": "get_belief",
         "arguments": {"subject": "path_ahead", "predicate": "is_clear"}},
        {"tool": "sense_lidar", "arguments": {"direction": "front"}},
        {"tool": "list_sources", "arguments": {}},
        {"tool": "downgrade_belief",
         "arguments": {"subject": "path_ahead", "predicate": "is_clear",
                       "new_confidence": 0.2,
                       "reason": "lidar contradicts the map"}},
        {"tool": "update_belief",
         "arguments": {"subject": "path_ahead", "predicate": "is_blocked",
                       "object": "obstacle@12cm", "confidence": 0.95,
                       "source": "lidar_front",
                       "reason": "live lidar reading"}},
        {"tool": "finalize_answer",
         "arguments": {"answer": "Blocked -- sensor beats stale map."}},
    ])
    # Seed the baseline belief manually via a direct knowledge-side call --
    # in production this would come from an /api/scenarios/A/reset call.
    agent = app.dependency_overrides[get_agent]()
    from datetime import UTC, datetime

    agent.knowledge.update_belief(Belief(
        subject="path_ahead", predicate="is_clear", object="true",
        confidence=1.0, source="default_map", perspective="self",
        observed_at=datetime(2026, 10, 16, 9, 0, tzinfo=UTC),
    ), reason="baseline floor plan")

    resp = client_with_recorded_agent.client.post(
        "/api/query",
        json={"query": "Is your route clear?", "scenario": "A"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert "blocked" in body["answer"].lower()
    assert "downgrade_belief" in body["tools_called"]
    assert "update_belief" in body["tools_called"]
    assert "sense_lidar" in body["tools_called"]
    # Real sensor payload came through
    sensor_step = next(
        s for s in body["trace"]["steps"] if s["tool_call"]["name"] == "sense_lidar"
    )
    assert sensor_step["observation"]["content"]["value"] == 12
    assert sensor_step["observation"]["content"]["status"] == "blocked"


# ------------------------- validation ---------------------------------------
# Use the harness so get_agent is overridden even for 422-returning requests;
# FastAPI resolves dependencies in parallel with body validation, so without
# the override the real LLMClient() would fire (and crash without GROQ_API_KEY).


def test_query_rejects_empty_body(client_with_recorded_agent):
    client_with_recorded_agent.set_script([
        {"tool": "finalize_answer", "arguments": {"answer": "unused"}},
    ])
    resp = client_with_recorded_agent.client.post("/api/query", json={"query": ""})
    assert resp.status_code == 422


def test_query_rejects_oversized_max_steps(client_with_recorded_agent):
    client_with_recorded_agent.set_script([
        {"tool": "finalize_answer", "arguments": {"answer": "unused"}},
    ])
    resp = client_with_recorded_agent.client.post(
        "/api/query", json={"query": "hi", "max_steps": 100}
    )
    assert resp.status_code == 422


# ------------------------- max_steps handling -------------------------------


def test_query_returns_422_with_partial_trace_when_max_steps_exceeded(
    client_with_recorded_agent,
):
    client_with_recorded_agent.set_script([
        {"tool": "list_sources", "arguments": {}},
        {"tool": "list_sources", "arguments": {}},
        {"tool": "list_sources", "arguments": {}},
    ])
    resp = client_with_recorded_agent.client.post(
        "/api/query", json={"query": "x", "max_steps": 3}
    )
    assert resp.status_code == 422
    detail = resp.json()["detail"]
    assert detail["error"] == "max_steps_exceeded"
    assert len(detail["partial_trace"]["steps"]) == 3
