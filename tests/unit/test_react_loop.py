"""Unit tests for the ReAct loop + Agent facade.

Uses `RecordedLLMClient` so no test ever hits Groq. The real `KnowledgeAPI`,
`BeliefGraph`, `ProvenanceStore`, and `MockEnvironment` are composed through
the Agent -- the tests exercise the loop against the actual layer facades,
not fakes.

Owner: Person 4 (Phase 4). See PROJECT_PLAN.md sections G.3, H.2.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from iagent.declarative.api import Belief, KnowledgeAPI
from iagent.declarative.belief_graph import BeliefGraph
from iagent.declarative.provenance import ProvenanceStore
from iagent.declarative.seeding import seed_sources
from iagent.procedural.agent import Agent, AgentResult
from iagent.procedural.llm_client import RecordedLLMClient, ScriptExhausted
from iagent.procedural.models import Trace
from iagent.procedural.react_loop import MaxStepsExceeded, run_react
from iagent.sensorimotor.api import SensorimotorAPI
from iagent.sensorimotor.mock_env import MockEnvironment

_REPO = Path(__file__).resolve().parents[2]
_SEED = _REPO / "data" / "seed" / "initial_provenance.sql"
_SCENARIO_A = _REPO / "configs" / "scenario_a.yaml"


# -------------------- fixtures --------------------


@pytest.fixture()
def store(tmp_path):
    s = ProvenanceStore(str(tmp_path / "iagent.db"))
    seed_sources(s, _SEED)
    yield s
    s.close()


@pytest.fixture()
def knowledge(store):
    return KnowledgeAPI(BeliefGraph(), store)


@pytest.fixture()
def sensors():
    """Real Scenario A sensorimotor stack from P2's branch."""
    return SensorimotorAPI(MockEnvironment.from_scenario(_SCENARIO_A))


def _agent(knowledge, sensors, script) -> Agent:
    return Agent(knowledge, sensors, RecordedLLMClient(script))


# -------------------- happy paths --------------------


def test_single_step_finalize_only(knowledge, sensors):
    """The simplest possible episode: LLM calls finalize_answer immediately."""
    script = [
        {"tool": "finalize_answer", "arguments": {"answer": "hello world"}},
    ]
    result = _agent(knowledge, sensors, script).run("hi")

    assert isinstance(result, AgentResult)
    assert result.answer == "hello world"
    assert result.tools_called == ["finalize_answer"]
    assert len(result.trace.steps) == 1
    assert result.trace.steps[0].tool_call.name == "finalize_answer"
    assert result.trace.steps[0].final_answer == "hello world"


def test_two_tool_episode_populates_full_trace(knowledge, sensors):
    """Memory read then finalize. Trace captures both steps in order."""
    # Seed a belief so get_belief returns something non-null.
    knowledge.update_belief(Belief(
        subject="robot", predicate="located_at", object="room_101",
        confidence=1.0, source="default_map", perspective="self",
        observed_at=datetime(2026, 10, 16, 9, 0, tzinfo=UTC),
    ), reason="baseline")

    script = [
        {"tool": "get_belief",
         "arguments": {"subject": "robot", "predicate": "located_at"}},
        {"tool": "finalize_answer",
         "arguments": {"answer": "You are in room_101."}},
    ]
    result = _agent(knowledge, sensors, script).run("Where am I?")

    assert result.answer == "You are in room_101."
    assert result.tools_called == ["get_belief", "finalize_answer"]
    # The get_belief step recorded the actual belief as its observation.
    first = result.trace.steps[0]
    assert first.observation.tool == "get_belief"
    assert first.observation.ok is True
    assert first.observation.content["object"] == "room_101"


def test_trace_has_start_and_end_timestamps(knowledge, sensors):
    script = [{"tool": "finalize_answer", "arguments": {"answer": "done"}}]
    before = datetime.now(UTC)
    result = _agent(knowledge, sensors, script).run("test")
    after = datetime.now(UTC)

    assert result.trace.started_at is not None
    assert result.trace.ended_at is not None
    # Replace tzinfo on comparison because Trace may store with/without tz consistently.
    assert before <= result.trace.started_at.replace(tzinfo=UTC) <= after
    assert result.trace.started_at <= result.trace.ended_at


# -------------------- Scenario A end-to-end --------------------


def test_scenario_a_full_pipeline_through_react_loop(knowledge, sensors):
    """The whole Scenario A sequence: memory -> sensor -> sources -> downgrade
    -> update -> finalize, driven through the ReAct loop with the real
    KnowledgeAPI and real P2 LiDAR."""

    # Seed the stale map belief that the sensor will contradict.
    knowledge.update_belief(Belief(
        subject="path_ahead", predicate="is_clear", object="true",
        confidence=1.0, source="default_map", perspective="self",
        observed_at=datetime(2026, 10, 16, 9, 0, tzinfo=UTC),
    ), reason="baseline floor plan")

    script = [
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
         "arguments": {"answer": "Path is blocked -- stale map downgraded.",
                       "cited_beliefs": ["path_ahead:is_blocked"],
                       "cited_sensors": ["lidar_front"]}},
    ]
    result = _agent(knowledge, sensors, script).run(
        "Is your route clear?", scenario="A"
    )

    assert result.answer.startswith("Path is blocked")
    assert result.tools_called == [
        "get_belief", "sense_lidar", "list_sources",
        "downgrade_belief", "update_belief", "finalize_answer",
    ]
    # Post-conditions on the real stores:
    assert knowledge.get_belief("path_ahead", "is_clear") is None
    now_active = knowledge.get_belief("path_ahead", "is_blocked")
    assert now_active is not None
    assert now_active.confidence == 0.95
    assert now_active.source == "lidar_front"

    # The sensor reading really came from P2's MockEnvironment.
    lidar_step = result.trace.steps[1]
    assert lidar_step.observation.ok is True
    assert lidar_step.observation.content["value"] == 12
    assert lidar_step.observation.content["status"] == "blocked"


# -------------------- control-flow edge cases --------------------


def test_max_steps_exceeded_raises_with_partial_trace(knowledge, sensors):
    """If the LLM keeps calling tools without ever finalizing, the loop
    raises MaxStepsExceeded and attaches the partial trace."""
    script = [
        {"tool": "list_sources", "arguments": {}},
        {"tool": "list_sources", "arguments": {}},
        {"tool": "list_sources", "arguments": {}},
    ]
    agent = _agent(knowledge, sensors, script)

    with pytest.raises(MaxStepsExceeded) as exc:
        agent.run("hi", max_steps=3)

    assert len(exc.value.trace.steps) == 3
    assert all(s.tool_call.name == "list_sources" for s in exc.value.trace.steps)
    assert exc.value.trace.ended_at is not None


def test_free_text_thought_is_logged_and_loop_nudges(knowledge, sensors):
    """If the LLM returns only text (no tool_call), the loop records it as a
    Step.thought, nudges the LLM, then proceeds normally."""
    script = [
        {"content": "I am thinking about what to do."},
        {"tool": "finalize_answer", "arguments": {"answer": "ok"}},
    ]
    result = _agent(knowledge, sensors, script).run("x")

    # Two steps: one thought-only, one tool call.
    assert len(result.trace.steps) == 2
    thought_step = result.trace.steps[0]
    assert thought_step.tool_call is None
    assert thought_step.thought == "I am thinking about what to do."
    assert result.answer == "ok"


def test_dispatch_error_is_captured_but_loop_continues(knowledge, sensors):
    """A tool-level error (e.g. required argument missing) is surfaced in
    the Observation with ok=False, and the loop keeps going."""
    script = [
        # Missing `subject` -- dispatch returns {ok: False, error: ...}
        {"tool": "get_belief", "arguments": {"predicate": "is_clear"}},
        {"tool": "finalize_answer", "arguments": {"answer": "recovered"}},
    ]
    result = _agent(knowledge, sensors, script).run("x")

    first = result.trace.steps[0]
    assert first.observation.tool == "get_belief"
    assert first.observation.ok is False
    assert "subject" in str(first.observation.content).lower()  # missing-arg message
    assert result.answer == "recovered"


def test_multiple_tool_calls_in_one_turn_all_recorded(knowledge, sensors):
    """The provider's API allows the LLM to batch tool_calls in a single turn.
    Each becomes its own Step."""
    script = [
        [
            {"tool": "list_sources", "arguments": {}},
            {"tool": "sense_lidar", "arguments": {"direction": "front"}},
        ],
        {"tool": "finalize_answer", "arguments": {"answer": "done"}},
    ]
    result = _agent(knowledge, sensors, script).run("x")

    tool_steps = [s for s in result.trace.steps if s.tool_call]
    assert [s.tool_call.name for s in tool_steps] == [
        "list_sources", "sense_lidar", "finalize_answer",
    ]


def test_script_exhausted_surfaces_cleanly(knowledge, sensors):
    """If the test script runs out before finalize_answer, the recorded
    client raises ScriptExhausted -- a loud test-authoring failure, not a
    silent hang."""
    script = [
        {"tool": "list_sources", "arguments": {}},
        # No finalize_answer here on purpose.
    ]
    agent = _agent(knowledge, sensors, script)

    with pytest.raises(ScriptExhausted):
        agent.run("x", max_steps=5)


def test_trace_can_be_dumped_as_json(knowledge, sensors, tmp_path):
    """The Trace Pydantic model round-trips through dump_trace cleanly."""
    from iagent.procedural.trace import dump_trace

    script = [{"tool": "finalize_answer", "arguments": {"answer": "done"}}]
    result = _agent(knowledge, sensors, script).run("x")

    path = dump_trace(result.trace, tmp_path)
    assert path.exists()
    assert path.suffix == ".json"
    body = path.read_text(encoding="utf-8")
    assert '"query": "x"' in body
    assert '"final_answer": "done"' in body
    assert '"tool_call"' in body


# -------------------- direct `run_react` without Agent facade --------------------


def test_run_react_directly_against_duck_typed_agent(knowledge, sensors):
    """`run_react` only needs .knowledge/.sensors/.llm attributes -- useful
    when writing scenario-specific test harnesses that don't need the
    `Agent` summary logic."""

    class DuckAgent:
        def __init__(self, k, s, llm):
            self.knowledge = k
            self.sensors = s
            self.llm = llm

    llm = RecordedLLMClient([
        {"tool": "finalize_answer", "arguments": {"answer": "direct"}},
    ])
    trace = run_react(DuckAgent(knowledge, sensors, llm), "q")

    assert isinstance(trace, Trace)
    assert trace.final_answer == "direct"
    assert trace.ended_at is not None
