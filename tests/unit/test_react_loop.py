"""Unit tests for the ReAct loop + Agent facade.

Uses `RecordedLLMClient` so no test ever hits Groq. The real `KnowledgeAPI`,
`BeliefGraph`, `ProvenanceStore`, and `MockEnvironment` are composed through
the Agent -- the tests exercise the loop against the actual layer facades,
not fakes.

Owner: Person 4 (Phase 4). See PROJECT_PLAN.md sections G.3, H.2.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from iagent.declarative.api import Belief, KnowledgeAPI
from iagent.declarative.belief_graph import BeliefGraph
from iagent.declarative.provenance import ProvenanceStore
from iagent.declarative.seeding import seed_sources
from iagent.procedural.agent import Agent, AgentResult
from iagent.procedural.llm_client import (
    DEFAULT_MODEL,
    MALFORMED_ARGS_KEY,
    LLMClient,
    LLMClientError,
    RecordedLLMClient,
    ScriptExhausted,
)
from iagent.procedural.models import Trace
from iagent.procedural.react_loop import (
    MaxStepsExceeded,
    run_react,
    truncate_context,
)
from iagent.procedural.trace import dump_trace
from iagent.sensorimotor.api import SensorimotorAPI, SensorPayload
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
    knowledge.update_belief(
        Belief(
            subject="robot",
            predicate="located_at",
            object="room_101",
            confidence=1.0,
            source="default_map",
            perspective="self",
            observed_at=datetime(2026, 10, 16, 9, 0, tzinfo=UTC),
        ),
        reason="baseline",
    )

    script = [
        {"tool": "get_belief", "arguments": {"subject": "robot", "predicate": "located_at"}},
        {"tool": "finalize_answer", "arguments": {"answer": "You are in room_101."}},
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
    knowledge.update_belief(
        Belief(
            subject="path_ahead",
            predicate="is_clear",
            object="true",
            confidence=1.0,
            source="default_map",
            perspective="self",
            observed_at=datetime(2026, 10, 16, 9, 0, tzinfo=UTC),
        ),
        reason="baseline floor plan",
    )

    script = [
        {"tool": "get_belief", "arguments": {"subject": "path_ahead", "predicate": "is_clear"}},
        {"tool": "sense_lidar", "arguments": {"direction": "front"}},
        {"tool": "list_sources", "arguments": {}},
        {
            "tool": "downgrade_belief",
            "arguments": {
                "subject": "path_ahead",
                "predicate": "is_clear",
                "new_confidence": 0.2,
                "reason": "lidar contradicts the map",
            },
        },
        {
            "tool": "update_belief",
            "arguments": {
                "subject": "path_ahead",
                "predicate": "is_blocked",
                "object": "obstacle@12cm",
                "confidence": 0.95,
                "source": "lidar_front",
                "perspective": "self",
                "reason": "live lidar reading",
            },
        },
        {
            "tool": "finalize_answer",
            "arguments": {
                "answer": "Path is blocked -- stale map downgraded.",
                "cited_beliefs": ["path_ahead:is_blocked"],
                "cited_sensors": ["lidar_front"],
            },
        },
    ]
    result = _agent(knowledge, sensors, script).run("Is your route clear?", scenario="A")

    assert result.answer.startswith("Path is blocked")
    assert result.tools_called == [
        "get_belief",
        "sense_lidar",
        "list_sources",
        "downgrade_belief",
        "update_belief",
        "finalize_answer",
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
        "list_sources",
        "sense_lidar",
        "finalize_answer",
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

    llm = RecordedLLMClient(
        [
            {"tool": "finalize_answer", "arguments": {"answer": "direct"}},
        ]
    )
    trace = run_react(DuckAgent(knowledge, sensors, llm), "q")

    assert isinstance(trace, Trace)
    assert trace.final_answer == "direct"
    assert trace.ended_at is not None


# -------------------- H.2(c): finalize runs last --------------------


def test_finalize_batched_with_another_call_runs_the_other_call_first(knowledge, sensors):
    """A model that emits [finalize_answer, update_belief] in ONE turn must
    still get the belief written. Emission order is the model's; execution
    order is ours (PROJECT_PLAN.md H.2)."""
    script = [
        [
            {"tool": "finalize_answer", "arguments": {"answer": "blocked"}},
            {
                "tool": "update_belief",
                "arguments": {
                    "subject": "path_ahead",
                    "predicate": "is_blocked",
                    "object": "obstacle@12cm",
                    "confidence": 0.95,
                    "source": "lidar_front",
                    "perspective": "self",
                    "reason": "live lidar reading",
                },
            },
        ],
    ]
    result = _agent(knowledge, sensors, script).run("x")

    # The write happened BEFORE the episode closed...
    assert result.tools_called == ["update_belief", "finalize_answer"]
    # ...and it really landed in the declarative layer.
    written = knowledge.get_belief("path_ahead", "is_blocked")
    assert written is not None
    assert written.confidence == 0.95
    assert result.answer == "blocked"


def test_duplicate_finalize_in_one_turn_keeps_the_first_answer(knowledge, sensors):
    """The prompt says call it once. If the model calls it twice anyway, the
    first answer wins and the second gets a refusal -- but every tool_call
    still receives a result, or the next request would be invalid."""
    script = [
        [
            {"tool": "finalize_answer", "arguments": {"answer": "first"}},
            {"tool": "finalize_answer", "arguments": {"answer": "second"}},
        ],
    ]
    result = _agent(knowledge, sensors, script).run("x")

    assert result.answer == "first"
    finals = [s for s in result.trace.steps if s.tool_call.name == "finalize_answer"]
    assert len(finals) == 2
    assert finals[0].observation.ok is True
    assert finals[1].observation.ok is False
    assert finals[1].observation.content["kind"] == "duplicate_finalize"


def test_a_failed_finalize_does_not_end_the_episode(knowledge, sensors):
    """`finalize_answer` without an answer is a dispatch error, so the loop
    keeps going and lets the model try again."""
    script = [
        {"tool": "finalize_answer", "arguments": {}},  # missing `answer`
        {"tool": "finalize_answer", "arguments": {"answer": "done properly"}},
    ]
    result = _agent(knowledge, sensors, script).run("x")

    assert result.trace.steps[0].observation.ok is False
    assert result.answer == "done properly"


# -------------------- H.2(d): the partial trace is dumped --------------------


def test_max_steps_dumps_the_partial_trace_when_given_a_runs_dir(knowledge, sensors, tmp_path):
    script = [{"tool": "list_sources", "arguments": {}} for _ in range(3)]
    agent = _agent(knowledge, sensors, script)

    with pytest.raises(MaxStepsExceeded) as exc:
        agent.run("hi", max_steps=3, runs_dir=tmp_path)

    assert exc.value.dump_path is not None
    assert exc.value.dump_path.exists()
    log = json.loads(exc.value.dump_path.read_text(encoding="utf-8"))
    assert len(log["steps"]) == 3
    assert log["final_answer"] == ""


def test_max_steps_without_a_runs_dir_still_carries_the_trace(knowledge, sensors):
    script = [{"tool": "list_sources", "arguments": {}} for _ in range(2)]
    agent = _agent(knowledge, sensors, script)

    with pytest.raises(MaxStepsExceeded) as exc:
        agent.run("hi", max_steps=2)

    assert exc.value.dump_path is None
    assert len(exc.value.trace.steps) == 2


# -------------------- H.5(e): context truncation --------------------


def _tool_turn(call_id: str, n_results: int = 1) -> list[dict]:
    assistant = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": f"{call_id}_{i}",
                "type": "function",
                "function": {"name": "list_sources", "arguments": "{}"},
            }
            for i in range(n_results)
        ],
    }
    results = [
        {"role": "tool", "tool_call_id": f"{call_id}_{i}", "content": "{}"}
        for i in range(n_results)
    ]
    return [assistant, *results]


def test_truncate_context_is_a_no_op_within_budget():
    messages = [{"role": "system", "content": "rules"}, {"role": "user", "content": "q"}]
    assert truncate_context(messages, token_budget=10_000) is messages


def test_truncate_context_always_keeps_the_system_prompt():
    messages = [
        {"role": "system", "content": "the epistemic rules"},
        *[{"role": "user", "content": f"turn {i}"} for i in range(20)],
    ]
    kept = truncate_context(messages, max_messages=4, token_budget=0)

    assert kept[0]["content"] == "the epistemic rules"
    assert len(kept) == 5  # system + 4
    assert kept[-1]["content"] == "turn 19"


def test_truncate_context_never_orphans_a_tool_result():
    """Dropping an assistant tool_calls turn but keeping its results -- or the
    reverse -- produces a message log the provider rejects."""
    messages = [
        {"role": "system", "content": "rules"},
        {"role": "user", "content": "q"},
        *_tool_turn("a", n_results=2),
        {"role": "user", "content": "nudge"},
        *_tool_turn("b", n_results=2),
    ]
    for max_messages in range(1, 10):
        kept = truncate_context(messages, max_messages=max_messages, token_budget=0)
        for i, msg in enumerate(kept):
            if msg.get("role") != "tool":
                continue
            # Walk back to the owning assistant turn; it must be present.
            owner = next((m for m in reversed(kept[:i]) if m.get("role") == "assistant"), None)
            assert owner is not None, f"orphaned tool result at max={max_messages}"
            assert owner.get("tool_calls"), f"orphaned tool result at max={max_messages}"


def test_truncate_context_keeps_the_newest_group_whole_even_if_oversized():
    """A single turn bigger than the message budget is kept intact -- half a
    tool group is worse than an over-budget one."""
    messages = [
        {"role": "system", "content": "rules"},
        {"role": "user", "content": "old"},
        *_tool_turn("a", n_results=3),  # 4 messages
    ]
    kept = truncate_context(messages, max_messages=2, token_budget=0)

    assert kept[0]["role"] == "system"
    assert len(kept) == 5  # system + the whole 4-message group
    assert sum(1 for m in kept if m.get("role") == "tool") == 3


# -------------------- section K log format --------------------


def test_dumped_log_has_the_section_k_shape(knowledge, sensors, tmp_path):
    script = [{"tool": "finalize_answer", "arguments": {"answer": "done"}}]
    result = _agent(knowledge, sensors, script).run("x", scenario="A")

    path = dump_trace(result.trace, tmp_path, test_id="test_99_demo")
    log = json.loads(path.read_text(encoding="utf-8"))

    assert path.name == "epistemic_test_99_demo.json"
    assert set(log) == {
        "test_id",
        "query",
        "scenario",
        "started_at",
        "ended_at",
        "steps",
        "final_answer",
        "assertions",
        "verdict",
    }
    assert log["test_id"] == "test_99_demo"
    assert log["scenario"] == "A"
    assert log["assertions"] == []
    assert log["verdict"] is None


def test_dumped_log_carries_assertions_and_derives_a_verdict(knowledge, sensors, tmp_path):
    """P4's epistemic tests extend the log; the verdict follows the rows."""
    script = [{"tool": "finalize_answer", "arguments": {"answer": "done"}}]
    result = _agent(knowledge, sensors, script).run("x")

    passing = json.loads(
        dump_trace(
            result.trace,
            tmp_path,
            test_id="ok",
            assertions=[{"name": "answered", "passed": True, "detail": "-"}],
        ).read_text(encoding="utf-8")
    )
    failing = json.loads(
        dump_trace(
            result.trace,
            tmp_path,
            test_id="bad",
            assertions=[{"name": "answered", "passed": False, "detail": "-"}],
        ).read_text(encoding="utf-8")
    )

    assert passing["verdict"] == "pass"
    assert failing["verdict"] == "fail"


def test_trace_dump_method_matches_dump_trace(knowledge, sensors, tmp_path):
    script = [{"tool": "finalize_answer", "arguments": {"answer": "done"}}]
    result = _agent(knowledge, sensors, script).run("x")

    path = result.trace.dump(tmp_path / "via_method")
    assert path.exists()
    assert json.loads(path.read_text(encoding="utf-8"))["final_answer"] == "done"


# -------------------- AgentResult.beliefs_written --------------------


def test_beliefs_written_records_what_was_actually_committed(knowledge, sensors):
    knowledge.update_belief(
        Belief(
            subject="path_ahead",
            predicate="is_clear",
            object="true",
            confidence=1.0,
            source="default_map",
            perspective="self",
            observed_at=datetime(2026, 10, 16, 9, 0, tzinfo=UTC),
        ),
        reason="baseline floor plan",
    )

    script = [
        {
            "tool": "downgrade_belief",
            "arguments": {
                "subject": "path_ahead",
                "predicate": "is_clear",
                "new_confidence": 0.2,
                "reason": "lidar contradicts",
            },
        },
        {
            "tool": "update_belief",
            "arguments": {
                "subject": "path_ahead",
                "predicate": "is_blocked",
                "object": "obstacle@12cm",
                "confidence": 0.95,
                "source": "lidar_front",
                "perspective": "self",
                "reason": "live lidar reading",
            },
        },
        {"tool": "finalize_answer", "arguments": {"answer": "blocked"}},
    ]
    result = _agent(knowledge, sensors, script).run("x")

    assert [b.predicate for b in result.beliefs_written] == ["is_clear", "is_blocked"]
    downgraded, asserted = result.beliefs_written
    assert downgraded.status == "downgraded"
    assert downgraded.confidence == 0.2
    assert asserted.status == "active"
    assert asserted.source == "lidar_front"
    # provenance_id comes from the declarative layer, not from the LLM's args.
    assert all(b.provenance_id is not None for b in result.beliefs_written)


def test_beliefs_written_is_empty_for_a_read_only_episode(knowledge, sensors):
    script = [
        {"tool": "sense_lidar", "arguments": {"direction": "front"}},
        {"tool": "finalize_answer", "arguments": {"answer": "blocked"}},
    ]
    result = _agent(knowledge, sensors, script).run("x")
    assert result.beliefs_written == []


def test_failed_write_is_not_reported_as_written(knowledge, sensors):
    """downgrade_belief with nothing to downgrade is a KnowledgeError."""
    script = [
        {
            "tool": "downgrade_belief",
            "arguments": {
                "subject": "ghost",
                "predicate": "exists",
                "new_confidence": 0.1,
                "reason": "nothing there",
            },
        },
        {"tool": "finalize_answer", "arguments": {"answer": "no such belief"}},
    ]
    result = _agent(knowledge, sensors, script).run("x")

    assert result.trace.steps[0].observation.ok is False
    assert result.beliefs_written == []


# -------------------- Agent reads its config --------------------


def test_agent_takes_max_steps_from_config(knowledge, sensors):
    script = [{"tool": "list_sources", "arguments": {}} for _ in range(5)]
    agent = Agent(knowledge, sensors, RecordedLLMClient(script), {"react": {"max_steps": 2}})

    with pytest.raises(MaxStepsExceeded) as exc:
        agent.run("x")

    assert len(exc.value.trace.steps) == 2


def test_explicit_max_steps_beats_config(knowledge, sensors):
    script = [{"tool": "list_sources", "arguments": {}} for _ in range(5)]
    agent = Agent(knowledge, sensors, RecordedLLMClient(script), {"react": {"max_steps": 2}})

    with pytest.raises(MaxStepsExceeded) as exc:
        agent.run("x", max_steps=4)

    assert len(exc.value.trace.steps) == 4


def test_agent_writes_a_trace_to_the_configured_runs_dir(knowledge, sensors, tmp_path):
    script = [{"tool": "finalize_answer", "arguments": {"answer": "done"}}]
    config = {
        "paths": {"runs_dir": str(tmp_path)},
        "logging": {"json_traces": True},
    }
    Agent(knowledge, sensors, RecordedLLMClient(script), config).run("x")

    dumped = list(Path(tmp_path).glob("*.json"))
    assert len(dumped) == 1
    assert json.loads(dumped[0].read_text(encoding="utf-8"))["final_answer"] == "done"


def test_json_traces_false_writes_nothing(knowledge, sensors, tmp_path):
    script = [{"tool": "finalize_answer", "arguments": {"answer": "done"}}]
    config = {
        "paths": {"runs_dir": str(tmp_path)},
        "logging": {"json_traces": False},
    }
    Agent(knowledge, sensors, RecordedLLMClient(script), config).run("x")

    assert list(Path(tmp_path).glob("*.json")) == []


def test_no_config_means_no_trace_file(knowledge, sensors, tmp_path):
    """The HTTP layer builds `Agent(knowledge, sensors, llm)` with no config;
    it must not litter a file per request."""
    script = [{"tool": "finalize_answer", "arguments": {"answer": "done"}}]
    Agent(knowledge, sensors, RecordedLLMClient(script)).run("x")
    assert list(Path(tmp_path).glob("*.json")) == []


# -------------------- actuators stay off unless asked --------------------


def test_actuator_call_is_refused_in_the_loop_by_default(knowledge, sensors):
    script = [
        {"tool": "act_move", "arguments": {"distance_cm": 30}},
        {"tool": "finalize_answer", "arguments": {"answer": "did not move"}},
    ]
    result = _agent(knowledge, sensors, script).run("x")

    refusal = result.trace.steps[0].observation
    assert refusal.ok is False
    assert refusal.content["kind"] == "tool_disabled"


def test_enabling_actuators_in_config_opens_the_gate(knowledge, sensors):
    """With the flag on, the call reaches P2's facade -- which currently
    returns `not_implemented` rather than being refused outright."""
    script = [
        {"tool": "act_move", "arguments": {"distance_cm": 30}},
        {"tool": "finalize_answer", "arguments": {"answer": "tried to move"}},
    ]
    agent = Agent(
        knowledge,
        sensors,
        RecordedLLMClient(script),
        {"features": {"enable_actuators": True}},
    )
    result = agent.run("x")

    observation = result.trace.steps[0].observation
    assert observation.ok is False
    assert observation.content["kind"] == "not_implemented"


def test_actuator_specs_reach_the_llm_only_when_enabled(knowledge, sensors):
    script = [{"tool": "finalize_answer", "arguments": {"answer": "done"}}]
    llm = RecordedLLMClient(script)
    Agent(knowledge, sensors, llm, {"features": {"enable_actuators": True}}).run("x")

    offered = {spec["function"]["name"] for spec in llm.calls[0]["tools"]}
    assert "act_move" in offered

    llm2 = RecordedLLMClient([{"tool": "finalize_answer", "arguments": {"answer": "d"}}])
    Agent(knowledge, sensors, llm2).run("x")
    assert "act_move" not in {s["function"]["name"] for s in llm2.calls[0]["tools"]}


# -------------------- LLMClient transport --------------------
#
# The real Groq client, driven against a stub provider so no test touches the
# network. What is under test is the retry policy, not Groq.


def _tool_call(name="list_sources", arguments="{}", call_id="call_1"):
    return SimpleNamespace(id=call_id, function=SimpleNamespace(name=name, arguments=arguments))


def _completion(tool_calls=None, content=None, finish_reason="tool_calls"):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=content, tool_calls=tool_calls or []),
                finish_reason=finish_reason,
            )
        ]
    )


class _StubProvider:
    """Stands in for `groq.Groq`. Yields queued responses; raises queued errors."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = 0
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **_kwargs):
        self.calls += 1
        item = self._responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def _client(provider, **kwargs):
    slept: list[float] = []
    client = LLMClient(client=provider, sleep=slept.append, **kwargs)
    return client, slept


def test_malformed_tool_call_arguments_are_retried():
    """Small instruct models sometimes emit non-JSON arguments. One resample
    is cheap; a dead episode is not."""
    provider = _StubProvider(
        [
            _completion([_tool_call(arguments="{not json")]),
            _completion([_tool_call(arguments='{"subject": "a"}')]),
        ]
    )
    client, slept = _client(provider, max_retries=3)

    response = client.chat_with_tools([], [])

    assert provider.calls == 2
    assert response.malformed is False
    assert response.tool_calls[0].arguments == {"subject": "a"}
    assert slept == [0.5]


def test_persistently_malformed_generation_is_returned_not_raised():
    """After the retries are spent the loop still gets something to dispatch,
    which becomes a recoverable `missing_argument` observation."""
    provider = _StubProvider([_completion([_tool_call(arguments="{not json")]) for _ in range(3)])
    client, _slept = _client(provider, max_retries=3)

    response = client.chat_with_tools([], [])

    assert provider.calls == 3
    assert response.malformed is True
    assert response.tool_calls[0].arguments == {MALFORMED_ARGS_KEY: "{not json"}


def test_unnamed_tool_call_counts_as_malformed():
    provider = _StubProvider(
        [
            _completion([_tool_call(name="")]),
            _completion([_tool_call(name="list_sources")]),
        ]
    )
    client, _ = _client(provider, max_retries=2)

    assert client.chat_with_tools([], []).tool_calls[0].name == "list_sources"
    assert provider.calls == 2


def test_free_text_response_is_not_malformed():
    provider = _StubProvider([_completion(content="thinking out loud")])
    client, _ = _client(provider)

    response = client.chat_with_tools([], [])

    assert provider.calls == 1
    assert response.malformed is False
    assert response.content == "thinking out loud"


def test_transient_provider_error_is_retried_then_succeeds():
    provider = _StubProvider(
        [
            RuntimeError("502 bad gateway"),
            _completion([_tool_call()]),
        ]
    )
    client, slept = _client(provider, max_retries=3)

    assert client.chat_with_tools([], []).tool_calls[0].name == "list_sources"
    assert provider.calls == 2
    assert slept == [0.5]


def test_exhausted_retries_raise_llm_client_error():
    provider = _StubProvider([RuntimeError("boom") for _ in range(3)])
    client, slept = _client(provider, max_retries=3)

    with pytest.raises(LLMClientError, match="after 3 attempts"):
        client.chat_with_tools([], [])

    assert provider.calls == 3
    assert slept == [0.5, 1.0]  # no sleep after the final failure


# -------------------- LLMClient config/env wiring --------------------


def test_from_config_reads_the_llm_block(monkeypatch):
    monkeypatch.delenv("LLM_MODEL", raising=False)
    monkeypatch.delenv("LLM_TEMPERATURE", raising=False)
    config = {"llm": {"model": "yaml-model", "temperature": 0.3, "max_retries": 7}}

    client = LLMClient.from_config(config, client=object())

    assert client.model == "yaml-model"
    assert client.temperature == 0.3
    assert client.max_retries == 7


def test_env_overrides_the_config_file(monkeypatch):
    """configs/default.yaml says so at the top of the file."""
    monkeypatch.setenv("LLM_MODEL", "env-model")
    monkeypatch.setenv("LLM_TEMPERATURE", "0.9")
    config = {"llm": {"model": "yaml-model", "temperature": 0.0}}

    client = LLMClient.from_config(config, client=object())

    assert client.model == "env-model"
    assert client.temperature == 0.9


def test_explicit_override_beats_env_and_config(monkeypatch):
    monkeypatch.setenv("LLM_MODEL", "env-model")
    config = {"llm": {"model": "yaml-model"}}

    client = LLMClient.from_config(config, model="explicit", client=object())

    assert client.model == "explicit"


def test_bare_constructor_falls_back_to_the_documented_default(monkeypatch):
    monkeypatch.delenv("LLM_MODEL", raising=False)
    client = LLMClient(client=object())

    assert client.model == DEFAULT_MODEL
    assert client.temperature == 0.0
    assert client.max_retries == 3


def test_unknown_provider_is_rejected():
    with pytest.raises(LLMClientError, match="Unsupported llm.provider"):
        LLMClient.from_config({"llm": {"provider": "openai"}}, client=object())


def test_from_config_tolerates_no_config_at_all(monkeypatch):
    monkeypatch.delenv("LLM_MODEL", raising=False)
    assert LLMClient.from_config(None, client=object()).model == DEFAULT_MODEL


def test_real_default_yaml_loads_into_a_client(monkeypatch, repo_root):
    """configs/default.yaml is mine to keep in sync with this client."""
    monkeypatch.delenv("LLM_MODEL", raising=False)
    monkeypatch.delenv("LLM_TEMPERATURE", raising=False)
    config = yaml.safe_load((repo_root / "configs" / "default.yaml").read_text())

    client = LLMClient.from_config(config, client=object())

    assert client.model == config["llm"]["model"]
    assert client.temperature == 0.0
    assert client.max_retries == config["llm"]["max_retries"]


def test_context_guard_from_config_keeps_every_request_valid(knowledge, sensors):
    """With an absurdly tight budget the loop still produces well-formed
    requests: a system prompt first, and no orphaned tool results."""
    script = [
        {"tool": "list_sources", "arguments": {}},
        {"tool": "sense_lidar", "arguments": {"direction": "front"}},
        {"tool": "finalize_answer", "arguments": {"answer": "done"}},
    ]
    llm = RecordedLLMClient(script)
    config = {"react": {"token_budget": 0, "max_context_messages": 2}}

    result = Agent(knowledge, sensors, llm, config).run("x")

    assert result.answer == "done"
    assert len(llm.calls) == 3
    for turn in llm.calls:
        messages = turn["messages"]
        assert messages[0]["role"] == "system"
        for i, msg in enumerate(messages):
            if msg.get("role") != "tool":
                continue
            owner = next(
                (m for m in reversed(messages[:i]) if m.get("role") == "assistant"),
                None,
            )
            assert owner is not None and owner.get("tool_calls")


# -------------------- Scenario B end-to-end (stubbed camera) --------------------


class _ScenarioBSensors(SensorimotorAPI):
    """P2's `sense_camera` is still NotImplementedError (Phase 3, Scenario B).

    This subclass supplies exactly the payload section J specifies, leaving
    the frozen G.2 signature untouched, so the loop and the declarative layer
    are the real thing and only the camera is stubbed.
    """

    def sense_camera(self, direction="front"):
        return SensorPayload(
            sensor=f"camera_{direction}",
            kind="vision",
            status="ok",
            observed_at=datetime(2026, 10, 16, 10, 15, tzinfo=UTC),
            detections=[
                {
                    "item_id": "box_01",
                    "class": "box",
                    "color_perceived": "brown",
                    "bbox": [120, 88, 240, 210],
                    "confidence": 0.82,
                }
            ],
            ambient={
                "lighting": "yellow",
                "warning": "color perception unreliable",
            },
        )


@pytest.fixture()
def scenario_b_knowledge(knowledge):
    """Seed the user's belief and bot_02's maintenance log (plan section J)."""
    knowledge.update_belief(
        Belief(
            subject="box_01",
            predicate="has_color",
            object="red_atom",
            confidence=0.9,
            source="user",
            perspective="user",
            observed_at=datetime(2026, 10, 16, 8, 0, tzinfo=UTC),
        ),
        reason="user statement",
    )
    knowledge.update_belief(
        Belief(
            subject="box_01",
            predicate="has_color",
            object="blue_atom",
            confidence=0.7,
            source="bot_02",
            perspective="third_party:bot_02",
            observed_at=datetime(2026, 10, 16, 9, 30, tzinfo=UTC),
        ),
        reason="maintenance log: painted_blue",
    )
    return knowledge


def test_scenario_b_replay_keeps_three_perspectives_separate(scenario_b_knowledge):
    knowledge = scenario_b_knowledge
    sensors = _ScenarioBSensors(None)

    script = [
        {"tool": "query_perspective", "arguments": {"subject": "box_01", "predicate": "has_color"}},
        {"tool": "sense_camera", "arguments": {"direction": "front"}},
        {
            "tool": "update_belief",
            "arguments": {
                "subject": "box_01",
                "predicate": "has_color",
                "object": "brown_atom",
                "confidence": 0.82,
                "source": "camera_front",
                "perspective": "self",
                "reason": "live camera reading; ambient=yellow",
            },
        },
        {"tool": "query_provenance", "arguments": {"subject": "box_01", "predicate": "has_color"}},
        {
            "tool": "finalize_answer",
            "arguments": {
                "answer": (
                    "Three perspectives, kept separate: user says red; my camera "
                    "registers brown (0.82, yellow ambient, unreliable); the "
                    "bot_02 maintenance history says blue."
                ),
                "cited_beliefs": [
                    "box_01:has_color@user",
                    "box_01:has_color@self",
                    "box_01:has_color@third_party:bot_02",
                ],
                "cited_sensors": ["camera_front"],
            },
        },
    ]

    result = _agent(knowledge, sensors, script).run(
        "What color does the user think the object is, what color do you "
        "register it as, and what does your data history say?",
        scenario="B",
    )

    assert result.tools_called == [
        "query_perspective",
        "sense_camera",
        "update_belief",
        "query_provenance",
        "finalize_answer",
    ]

    # Step 1 — the grouped views arrive unmerged.
    views = result.trace.steps[0].observation.content["items"]
    by_perspective = {v["perspective"]: v for v in views}
    assert set(by_perspective) == {"user", "third_party:bot_02"}
    assert by_perspective["user"]["beliefs"][0]["object"] == "red_atom"
    assert by_perspective["third_party:bot_02"]["beliefs"][0]["object"] == "blue_atom"

    # Step 2 — the egocentric reading, with its unreliability warning intact.
    camera = result.trace.steps[1].observation.content
    assert camera["detections"][0]["color_perceived"] == "brown"
    assert camera["ambient"]["lighting"] == "yellow"
    assert "unreliable" in camera["ambient"]["warning"]

    # Step 3 — the write landed as 'self' and nothing else moved. This is the
    # no-merge proof: three (subject, predicate) beliefs coexist.
    assert [b.perspective for b in result.beliefs_written] == ["self"]
    assert knowledge.get_belief("box_01", "has_color", "user").object == "red_atom"
    assert knowledge.get_belief("box_01", "has_color", "third_party:bot_02").object == "blue_atom"
    assert knowledge.get_belief("box_01", "has_color", "self").object == "brown_atom"

    # Step 4 — the audit trail holds all three, still distinguishable.
    history = result.trace.steps[3].observation.content["items"]
    assert {row["perspective"] for row in history} == {
        "user",
        "self",
        "third_party:bot_02",
    }


def test_scenario_b_against_the_real_camera_returns_empty_when_no_preset(knowledge, sensors):
    """sense_camera is now implemented (returns the scenario YAML's preset
    reading, or an empty frame when no preset exists). The `sensors` fixture
    here loads scenario_a.yaml, which has no camera preset, so we expect an
    empty-but-successful observation rather than `not_implemented`."""
    script = [
        {"tool": "sense_camera", "arguments": {"direction": "front"}},
        {"tool": "finalize_answer", "arguments": {"answer": "I see no objects right now."}},
    ]
    result = _agent(knowledge, sensors, script).run("what do you see?", scenario="B")

    observation = result.trace.steps[0].observation
    assert observation.ok is True
    assert observation.content["status"] == "empty"
    assert observation.content["raw"]["detections"] == []
    assert result.answer == "I see no objects right now."
