"""The ReAct loop: reason -> act -> observe -> repeat, capped at `max_steps`.

This is the one place that drives the LLM. Everything upstream (prompts,
tool specs, LLM client) and downstream (dispatcher, layer facades) is
assembled here into one episode.

Protocol per iteration:
  1. Truncate the message log if it has outgrown the token budget, then call
     `agent.llm.chat_with_tools(messages, specs)`.
  2. If the response includes tool_calls:
       - append ONE assistant turn carrying every tool_call (OpenAI-compat),
         then one role=tool turn per call with its result
       - execute every non-terminal call FIRST, then `finalize_answer`, so a
         model that batches `[finalize_answer, update_belief]` still gets its
         belief written before the episode closes
       - a successful `finalize_answer` returns the completed Trace
  3. If the response is only free text (no tool_call), log it as a
     Step(thought=...), add it as an assistant turn, and nudge the LLM
     with a user message telling it to call a tool next.
  4. If `max_steps` iterations elapse without `finalize_answer`, dump the
     partial trace and raise `MaxStepsExceeded(trace)` with it attached.

Both `LLMClient` (real Groq) and `RecordedLLMClient` (test replay) satisfy
the same `chat_with_tools(messages, tools) -> AssistantResponse` shape, so
the loop never knows which one is driving.

Owner: Person 3 (Phase 4). See PROJECT_PLAN.md sections H.2 and H.5.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from iagent.procedural.llm_client import AssistantResponse, ToolCallRequest
from iagent.procedural.models import Observation, Step, ToolCall, Trace
from iagent.procedural.prompts import SYSTEM_PROMPT, format_query
from iagent.procedural.tools import FINALIZE_TOOL, build_tool_specs, dispatch
from iagent.procedural.trace import dump_trace

__all__ = [
    "MaxStepsExceeded",
    "run_react",
    "DEFAULT_TOKEN_BUDGET",
    "DEFAULT_CONTEXT_MESSAGES",
]


# PROJECT_PLAN.md H.5: "all messages truncated to N=8 most-recent (except
# system prompt) if context grows > 6k tokens".
DEFAULT_TOKEN_BUDGET = 6000
DEFAULT_CONTEXT_MESSAGES = 8


class MaxStepsExceeded(RuntimeError):
    """Raised when the loop hits `max_steps` without `finalize_answer`.

    The partial Trace (every step up to the budget cap) is attached so
    callers can still inspect what happened. `dump_path` points at the
    partial trace on disk when the episode was given a runs directory.
    """

    def __init__(self, trace: Trace, dump_path: Path | None = None) -> None:
        super().__init__(f"ReAct loop exceeded max_steps ({len(trace.steps)})")
        self.trace = trace
        self.dump_path = dump_path


_NUDGE = "You must call a tool to continue the episode. " "Use `finalize_answer` to end."

_DUPLICATE_FINALIZE = {
    "ok": False,
    "error": {
        "kind": "duplicate_finalize",
        "message": (
            "finalize_answer was already called in this turn. The episode is "
            "closing with the first answer; this call was ignored."
        ),
    },
}


def run_react(
    agent,
    query: str,
    *,
    scenario: str | None = None,
    max_steps: int = 8,
    enable_actuators: bool = False,
    runs_dir: str | Path | None = None,
    test_id: str | None = None,
    token_budget: int = DEFAULT_TOKEN_BUDGET,
    max_context_messages: int = DEFAULT_CONTEXT_MESSAGES,
) -> Trace:
    """Drive one ReAct episode. Returns the completed `Trace`.

    `agent` is duck-typed: must expose `.knowledge`, `.sensors`, `.llm`.
    This lets tests pass a lightweight fake instead of constructing a full
    `Agent` with Groq credentials.

    When `runs_dir` is given the trace is written there before returning —
    and also before raising `MaxStepsExceeded`, so a capped-out episode still
    leaves a log on disk.
    """
    apis: dict[str, Any] = {
        "knowledge": agent.knowledge,
        "sensors": agent.sensors,
    }
    specs = build_tool_specs(enable_actuators=enable_actuators)

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": format_query(query, scenario)},
    ]
    trace = Trace(
        query=query,
        scenario=scenario,
        steps=[],
        started_at=_now(),
        test_id=test_id,
    )

    for step_idx in range(1, max_steps + 1):
        messages = truncate_context(
            messages,
            max_messages=max_context_messages,
            token_budget=token_budget,
        )
        response: AssistantResponse = agent.llm.chat_with_tools(messages, specs)

        if not response.tool_calls:
            # Free-text response with no tool call -- log it, nudge the LLM.
            thought_text = (response.content or "").strip()
            trace.append(Step(idx=step_idx, thought=thought_text or "(empty)"))
            if thought_text:
                messages.append({"role": "assistant", "content": thought_text})
            messages.append({"role": "user", "content": _NUDGE})
            continue

        # One assistant turn carrying every tool_call the model emitted, in
        # the order it emitted them -- the provider requires the ids to match.
        messages.append(_assistant_tool_call_message(response))

        # Execution order is OURS, though: side effects before termination.
        pending = [tc for tc in response.tool_calls if tc.name != FINALIZE_TOOL]
        finals = [tc for tc in response.tool_calls if tc.name == FINALIZE_TOOL]

        for tc in pending:
            result = dispatch(tc.name, tc.arguments, apis, enable_actuators=enable_actuators)
            trace.append(
                _build_step(
                    idx=step_idx,
                    thought=response.content,
                    tool_call=tc,
                    result=result,
                )
            )
            messages.append(_tool_result_message(tc.id, result))

        terminal: dict[str, Any] | None = None
        for position, tc in enumerate(finals):
            # The prompt says call it exactly once. If the model calls it
            # twice anyway, the first answer wins and the rest are refused --
            # but each still gets a tool result, or the message log would be
            # invalid on the next request.
            result = (
                dispatch(tc.name, tc.arguments, apis, enable_actuators=enable_actuators)
                if position == 0
                else dict(_DUPLICATE_FINALIZE)
            )
            trace.append(
                _build_step(
                    idx=step_idx,
                    thought=response.content,
                    tool_call=tc,
                    result=result,
                )
            )
            messages.append(_tool_result_message(tc.id, result))
            if position == 0 and result.get("ok"):
                terminal = result

        if terminal is not None:
            trace.final_answer = _extract_final_answer(terminal)
            trace.ended_at = _now()
            if runs_dir is not None:
                trace.dump(runs_dir)
            return trace

    # Loop budget exhausted without a final answer.
    trace.ended_at = _now()
    dumped = dump_trace(trace, runs_dir) if runs_dir is not None else None
    raise MaxStepsExceeded(trace, dump_path=dumped)


# ---------------------------------------------------------------------------
# context window
# ---------------------------------------------------------------------------


def truncate_context(
    messages: list[dict[str, Any]],
    *,
    max_messages: int = DEFAULT_CONTEXT_MESSAGES,
    token_budget: int = DEFAULT_TOKEN_BUDGET,
) -> list[dict[str, Any]]:
    """Trim the message log to the most recent turns, safely.

    Two invariants (PROJECT_PLAN.md H.5):
      * the system prompt always survives — it carries every epistemic rule;
      * an assistant turn carrying `tool_calls` is never separated from the
        role=tool messages that answer it. The provider rejects an orphaned
        tool result, and it rejects a tool_calls turn whose results are gone,
        so those messages are kept or dropped as one atomic group.

    Returns `messages` unchanged while the estimated size is within budget.
    """
    if _estimate_tokens(messages) <= token_budget:
        return messages

    head = messages[:1]  # the system prompt
    groups = _atomic_groups(messages[1:])

    kept: list[list[dict[str, Any]]] = []
    size = 0
    for group in reversed(groups):
        # Always keep the newest group even if it alone busts the budget:
        # splitting it would produce a message log the provider refuses.
        if kept and size + len(group) > max_messages:
            break
        kept.append(group)
        size += len(group)

    kept.reverse()
    return [*head, *(msg for group in kept for msg in group)]


def _atomic_groups(messages: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """Group each assistant-with-tool_calls turn together with its results."""
    groups: list[list[dict[str, Any]]] = []
    for msg in messages:
        attaches = msg.get("role") == "tool" and groups and _carries_tool_calls(groups[-1][0])
        if attaches:
            groups[-1].append(msg)
        else:
            groups.append([msg])
    return groups


def _carries_tool_calls(msg: dict[str, Any]) -> bool:
    return msg.get("role") == "assistant" and bool(msg.get("tool_calls"))


def _estimate_tokens(messages: list[dict[str, Any]]) -> int:
    """Rough token count: ~4 characters per token of serialized JSON.

    Deliberately crude. The budget is a safety rail against context overflow,
    not an accounting system, and a real tokenizer would make the loop depend
    on the model id.
    """
    return len(json.dumps(messages, default=str)) // 4


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _now() -> datetime:
    return datetime.now(UTC)


def _assistant_tool_call_message(response: AssistantResponse) -> dict[str, Any]:
    """Build the OpenAI/Groq-compat assistant message carrying tool_calls."""
    return {
        "role": "assistant",
        "content": response.content,  # provider accepts None here
        "tool_calls": [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.name,
                    "arguments": json.dumps(tc.arguments),
                },
            }
            for tc in response.tool_calls
        ],
    }


def _tool_result_message(tool_call_id: str, result: dict[str, Any]) -> dict[str, Any]:
    return {
        "role": "tool",
        "tool_call_id": tool_call_id,
        "content": json.dumps(result),
    }


def _build_step(
    *,
    idx: int,
    thought: str | None,
    tool_call: ToolCallRequest,
    result: dict[str, Any],
) -> Step:
    ok = bool(result.get("ok"))
    raw_content = result.get("data") if ok else result.get("error")
    final_answer = None
    if tool_call.name == FINALIZE_TOOL and ok and isinstance(result.get("data"), dict):
        final_answer = result["data"].get("answer")

    return Step(
        idx=idx,
        thought=thought,
        tool_call=ToolCall(name=tool_call.name, arguments=tool_call.arguments),
        observation=Observation(
            tool=tool_call.name,
            ok=ok,
            content=_coerce_observation_content(raw_content),
        ),
        final_answer=final_answer,
    )


def _coerce_observation_content(x: Any) -> dict[str, Any] | str:
    """`Observation.content` is `dict | str`. Everything else gets wrapped."""
    if isinstance(x, dict):
        return x
    if isinstance(x, str):
        return x
    if x is None:
        return {}
    if isinstance(x, list):
        return {"items": x}
    return {"value": x}


def _extract_final_answer(result: dict[str, Any]) -> str:
    data = result.get("data")
    if isinstance(data, dict):
        answer = data.get("answer")
        if isinstance(answer, str):
            return answer
    return ""
