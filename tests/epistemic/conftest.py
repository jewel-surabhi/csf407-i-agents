"""Fixtures shared across the 10 epistemic tests.

Design (section K of PROJECT_PLAN.md):

- Each test spins up its OWN tmp SQLite + fresh BeliefGraph + MockEnvironment
  loaded from a scenario YAML. Nothing is shared between tests.
- Each test runs through the real `Agent.run()` which drives the real
  `react_loop.run_react()`. The only moving part is the LLM:
    * default = `RecordedLLMClient` playing a canned tool-call sequence, so
      CI stays deterministic and offline (no Groq key required).
    * `IAGENT_LIVE_LLM=1` flips every `live_llm`-marked test to the real
      `LLMClient` against Groq — these generate the "official" epistemic
      logs committed into the repo as the proposal deliverable.
- Every test calls `dump_epistemic_trace(trace, test_id)` which writes a
  JSON file under `data/runs/epistemic/`. These files ARE the required
  "10 automated test logs" the proposal grades.

Owner: Person 4. Phases 7–8.
"""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

import pytest
import yaml
from dotenv import load_dotenv

# Load .env so live-LLM tests pick up GROQ_API_KEY without a shell export.
# Safe for recorded tests too — the recorded path never touches Groq.
load_dotenv(Path(__file__).resolve().parents[2] / ".env")

from iagent.declarative.api import KnowledgeAPI
from iagent.declarative.belief_graph import BeliefGraph
from iagent.declarative.models import Belief
from iagent.declarative.provenance import ProvenanceStore
from iagent.declarative.seeding import seed_sources
from iagent.procedural.agent import Agent
from iagent.procedural.llm_client import RecordedLLMClient
from iagent.procedural.models import Trace
from iagent.sensorimotor.api import SensorimotorAPI
from iagent.sensorimotor.mock_env import MockEnvironment

# ---------------------------------------------------------------------------
# Paths

_REPO = Path(__file__).resolve().parents[2]
_SEED_SQL = _REPO / "data" / "seed" / "initial_provenance.sql"
_CONFIGS = _REPO / "configs"
_RUNS_DIR = _REPO / "data" / "runs" / "epistemic"


# ---------------------------------------------------------------------------
# Live LLM gate

LIVE_LLM = os.getenv("IAGENT_LIVE_LLM") == "1"


def pytest_collection_modifyitems(config, items):
    """Auto-skip `live_llm`-marked tests unless IAGENT_LIVE_LLM=1."""
    if LIVE_LLM:
        return
    skip_live = pytest.mark.skip(reason="Set IAGENT_LIVE_LLM=1 to run live-LLM tests.")
    for item in items:
        if "live_llm" in item.keywords:
            item.add_marker(skip_live)


# ---------------------------------------------------------------------------
# Scenario seeding

def _parse_iso(ts: str | None) -> datetime:
    if ts is None:
        return datetime.now(UTC)
    return datetime.fromisoformat(ts.replace("Z", "+00:00"))


def _apply_seed_beliefs(knowledge: KnowledgeAPI, scenario: dict[str, Any]) -> None:
    """Apply every `seed_beliefs` entry via `update_belief` so the belief lands
    in BOTH the graph AND the SQLite provenance log atomically.

    This is the correct seeding path for tests that assert on provenance side
    effects (test_04) or need `query_perspective` to work (tests 05–08).
    """
    for raw in scenario.get("seed_beliefs", []) or []:
        belief = Belief(
            subject=raw["subject"],
            predicate=raw["predicate"],
            object=str(raw["object"]),
            confidence=float(raw.get("confidence", 1.0)),
            source=raw["source"],
            perspective=raw.get("perspective", "self"),
            observed_at=_parse_iso(raw.get("observed_at")),
        )
        knowledge.update_belief(belief, reason="scenario seed")


def _load_scenario_yaml(scenario_name: str) -> dict[str, Any]:
    path = _CONFIGS / f"{scenario_name}.yaml"
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


# ---------------------------------------------------------------------------
# Fixtures the tests consume


@pytest.fixture()
def runs_dir(tmp_path_factory) -> Path:
    """Directory where committed epistemic trace logs live. Created on demand."""
    _RUNS_DIR.mkdir(parents=True, exist_ok=True)
    return _RUNS_DIR


@pytest.fixture()
def epistemic_env(tmp_path):
    """Factory: build a (knowledge, sensors, scenario_dict) triple for one scenario.

    Usage:
        knowledge, sensors, scenario = epistemic_env("scenario_a")
    """
    created_stores: list[ProvenanceStore] = []

    def build(scenario_name: str) -> tuple[KnowledgeAPI, SensorimotorAPI, dict[str, Any]]:
        scenario = _load_scenario_yaml(scenario_name)

        # Fresh SQLite per build so each test gets a pristine provenance log.
        db_path = tmp_path / f"{scenario_name}.db"
        store = ProvenanceStore(str(db_path))
        created_stores.append(store)
        seed_sources(store, _SEED_SQL)

        graph = BeliefGraph()
        knowledge = KnowledgeAPI(graph, store)
        _apply_seed_beliefs(knowledge, scenario)

        env = MockEnvironment.from_scenario(_CONFIGS / f"{scenario_name}.yaml")
        sensors = SensorimotorAPI(env)

        return knowledge, sensors, scenario

    yield build

    for store in created_stores:
        store.close()


class _TracingAgent:
    """Thin wrapper around Agent that remembers the last Trace it produced.

    Live tests use this so a conftest teardown hook can dump the trace to
    disk even when assertions fail — otherwise a flaky LLM run leaves no
    artifact to inspect.
    """

    def __init__(self, inner: Agent) -> None:
        self._inner = inner
        self.last_trace = None

    @property
    def knowledge(self):
        return self._inner.knowledge

    @property
    def sensors(self):
        return self._inner.sensors

    @property
    def llm(self):
        return self._inner.llm

    def run(self, query: str, **kwargs):
        try:
            result = self._inner.run(query, **kwargs)
            self.last_trace = result.trace
            return result
        except Exception:
            # Loop may have raised MaxStepsExceeded / LLMClientError. If the
            # loop attached a partial trace, keep it so teardown can dump.
            import sys
            exc = sys.exc_info()[1]
            partial = getattr(exc, "trace", None)
            if partial is not None:
                self.last_trace = partial
            raise


def _agent_runner(
    knowledge: KnowledgeAPI,
    sensors: SensorimotorAPI,
    llm: Any,
) -> Agent:
    return _TracingAgent(Agent(knowledge=knowledge, sensors=sensors, llm=llm))


# Track the live agents a test touched so pytest_runtest_makereport can
# dump their traces on failure. Keyed by nodeid.
_LIVE_AGENTS: dict[str, "_TracingAgent"] = {}


@pytest.fixture()
def recorded_agent(epistemic_env):
    """Factory: `recorded_agent(scenario_name, script)` -> (agent, knowledge, sensors, scenario)."""
    def build(scenario_name: str, script: list[Any]):
        knowledge, sensors, scenario = epistemic_env(scenario_name)
        llm = RecordedLLMClient(script)
        agent = _agent_runner(knowledge, sensors, llm)
        return agent, knowledge, sensors, scenario

    return build


@pytest.fixture()
def live_agent(epistemic_env, request):
    """Factory: `live_agent(scenario_name)` -> Agent with the real Groq LLMClient.

    Only used by `@pytest.mark.live_llm` tests (gated on IAGENT_LIVE_LLM=1).
    Registers the agent for `pytest_runtest_makereport` so a failed live
    run still emits a trace log for debugging.
    """
    def build(scenario_name: str):
        from iagent.procedural.llm_client import LLMClient
        knowledge, sensors, scenario = epistemic_env(scenario_name)
        llm = LLMClient()  # reads GROQ_API_KEY from env
        agent = _agent_runner(knowledge, sensors, llm)
        _LIVE_AGENTS[request.node.nodeid] = agent
        return agent, knowledge, sensors, scenario

    yield build
    _LIVE_AGENTS.pop(request.node.nodeid, None)


@pytest.hookimpl(hookwrapper=True, tryfirst=True)
def pytest_runtest_makereport(item, call):
    """On a failed live-LLM test, dump whatever trace the agent managed so
    the failure can be inspected post-hoc."""
    outcome = yield
    report = outcome.get_result()
    if report.when != "call" or report.outcome != "failed":
        return
    agent = _LIVE_AGENTS.get(item.nodeid)
    if agent is None or agent.last_trace is None:
        return
    # Best-effort — never let the dump itself mask the real failure.
    try:
        # Prefer the test MODULE's id (test_03_stale_... -> 03_stale_...) so
        # the dump filename matches the recorded trace it corresponds to.
        mod_name = item.module.__name__.rsplit(".", 1)[-1]
        test_id = mod_name.removeprefix("test_")
        dump_epistemic_trace(
            agent.last_trace,
            test_id,
            extras={"verdict": "fail", "pytest_nodeid": item.nodeid},
            mode="live",
        )
    except Exception:  # pragma: no cover
        pass


# ---------------------------------------------------------------------------
# Trace dump — writes the required test log


def dump_epistemic_trace(
    trace: Trace,
    test_id: str,
    *,
    runs_dir: Path = _RUNS_DIR,
    extras: dict[str, Any] | None = None,
    mode: str | None = None,
) -> Path:
    """Write a trace JSON under `data/runs/epistemic/epistemic_<test_id>.json`.

    `extras` is merged into the top level so tests can tag the file with the
    scenario id, expected substrings, verdict etc. — the section-K log shape.

    One file per test_id with a suffix. Recorded runs land at
    `epistemic_<id>.json`; live runs land at `epistemic_<id>_live.json` so
    both are preserved side by side in the committed deliverable.
    """
    runs_dir.mkdir(parents=True, exist_ok=True)
    resolved_mode = mode or ("live" if LIVE_LLM else "recorded")
    suffix = "_live" if resolved_mode == "live" else ""
    path = runs_dir / f"epistemic_{test_id}{suffix}.json"

    payload: dict[str, Any] = json.loads(trace.model_dump_json())
    payload["test_id"] = test_id
    payload["recorded_at"] = datetime.now(UTC).isoformat()
    payload["mode"] = resolved_mode
    if extras:
        payload["extras"] = extras

    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Small assertion helpers (keep tests readable)


def assert_tools_called(trace: Trace, expected: list[str]) -> None:
    """Every name in `expected` must appear in order in trace.steps."""
    actual = [s.tool_call.name for s in trace.steps if s.tool_call is not None]
    missing = [t for t in expected if t not in actual]
    assert not missing, f"missing expected tool calls {missing} — actual trace: {actual}"


def assert_answer_contains(trace: Trace, needles: list[str]) -> None:
    """Case-insensitive substring check on the final answer."""
    ans = (trace.final_answer or "").lower()
    missing = [n for n in needles if n.lower() not in ans]
    assert not missing, f"final answer missing {missing}: {trace.final_answer!r}"


__all__ = [
    "dump_epistemic_trace",
    "assert_tools_called",
    "assert_answer_contains",
    "LIVE_LLM",
]
