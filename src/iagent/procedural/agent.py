"""Agent -- top-level facade. Composes KnowledgeAPI + SensorimotorAPI + LLM
and exposes `Agent.run(query)` as the single entry point.

The real work happens inside `react_loop.run_react` -- this class holds the
dependencies, resolves runtime knobs from config, and summarizes the completed
Trace into an `AgentResult`.

Owner: Person 3 (Phase 4). See PROJECT_PLAN.md section G.3.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import ValidationError

from iagent.declarative.api import Belief, KnowledgeAPI
from iagent.procedural.models import AgentResult, Trace
from iagent.procedural.react_loop import (
    DEFAULT_CONTEXT_MESSAGES,
    DEFAULT_TOKEN_BUDGET,
    run_react,
)
from iagent.sensorimotor.api import SensorimotorAPI

__all__ = ["Agent", "AgentResult"]

_WRITE_TOOLS = frozenset({"update_belief", "downgrade_belief"})

DEFAULT_MAX_STEPS = 8


class Agent:
    """Compose the three layers behind one `run(query)` call.

    `llm` is anything exposing `chat_with_tools(messages, tools)` -- the real
    `LLMClient` for live runs, `RecordedLLMClient` for tests.

    `config` is the parsed `configs/default.yaml` (a dict, or any object with
    matching attributes). It supplies `react.max_steps`,
    `features.enable_actuators`, `paths.runs_dir` and `logging.json_traces`.
    All four have safe defaults, so `Agent(knowledge, sensors, llm)` still
    works -- with no config there is no runs directory, so an episode writes
    no trace file unless `run(runs_dir=...)` says where to put it.
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
        max_steps: int | None = None,
        enable_actuators: bool | None = None,
        runs_dir: str | Path | None = None,
        test_id: str | None = None,
    ) -> AgentResult:
        """Execute one ReAct episode and return the final answer + Trace.

        Every keyword overrides the corresponding config value; anything left
        as None falls back to config, then to the documented default.

        Raises `react_loop.MaxStepsExceeded` if the LLM never calls
        `finalize_answer` within the step budget. The exception carries the
        partial trace (and its dump path, when a runs directory was in play).
        """
        trace = run_react(
            self,
            query,
            scenario=scenario,
            max_steps=self._max_steps(max_steps),
            enable_actuators=self._enable_actuators(enable_actuators),
            runs_dir=self._runs_dir(runs_dir),
            test_id=test_id,
            token_budget=self._int("react", "token_budget", DEFAULT_TOKEN_BUDGET),
            max_context_messages=self._int(
                "react", "max_context_messages", DEFAULT_CONTEXT_MESSAGES
            ),
        )
        return AgentResult(
            answer=trace.final_answer,
            trace=trace,
            beliefs_written=_beliefs_written(trace),
            tools_called=[
                step.tool_call.name for step in trace.steps if step.tool_call is not None
            ],
        )

    # ---------- config resolution ----------

    def _max_steps(self, override: int | None) -> int:
        if override is not None:
            return override
        return self._int("react", "max_steps", DEFAULT_MAX_STEPS)

    def _int(self, section: str, key: str, default: int) -> int:
        value = _cfg(self.config, section, key)
        return int(value) if value is not None else default

    def _enable_actuators(self, override: bool | None) -> bool:
        if override is not None:
            return override
        return bool(_cfg(self.config, "features", "enable_actuators") or False)

    def _runs_dir(self, override: str | Path | None) -> str | Path | None:
        if override is not None:
            return override
        json_traces = _cfg(self.config, "logging", "json_traces")
        if json_traces is False:
            return None
        return _cfg(self.config, "paths", "runs_dir")


def _cfg(config: Any, section: str, key: str) -> Any:
    """Read `config[section][key]` from a dict or an attribute-style object.

    Returns None when anything along the path is missing, so callers can fall
    back to their own default. Tolerant by design: P4 owns `config.py` and may
    land either a plain dict or a Pydantic `AppConfig`, and this layer should
    not care which.
    """
    if config is None:
        return None
    block = config.get(section) if isinstance(config, dict) else getattr(config, section, None)
    if block is None:
        return None
    if isinstance(block, dict):
        return block.get(key)
    return getattr(block, key, None)


def _beliefs_written(trace: Trace) -> list[Belief]:
    """Reconstruct the beliefs this episode persisted, in write order.

    Built from the provenance rows the declarative layer returned, not from
    the arguments the LLM supplied -- so what lands here is what was actually
    committed, including the server-assigned `provenance_id`. A downgrade is
    included too and carries `status="downgraded"`.

    Telemetry must never break an otherwise-good episode, so an observation
    that doesn't parse as a provenance row is skipped rather than raised.
    """
    written: list[Belief] = []
    for step in trace.steps:
        if step.tool_call is None or step.tool_call.name not in _WRITE_TOOLS:
            continue
        obs = step.observation
        if obs is None or not obs.ok or not isinstance(obs.content, dict):
            continue
        row = obs.content
        try:
            written.append(
                Belief(
                    subject=row["subject"],
                    predicate=row["predicate"],
                    object=row["object"],
                    confidence=row["confidence"],
                    source=row["source_id"],
                    perspective=row["perspective"],
                    status=row.get("status", "active"),
                    observed_at=row["observed_at"],
                    provenance_id=row.get("provenance_id"),
                )
            )
        except (KeyError, ValidationError, TypeError):
            continue
    return written
