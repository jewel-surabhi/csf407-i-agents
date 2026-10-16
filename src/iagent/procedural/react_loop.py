"""The ReAct loop: reason -> act -> observe -> repeat, capped at `max_steps`.

This is the one place that drives the LLM. Everything upstream (prompts,
tool specs, LLM client) and downstream (dispatcher, layer facades) is
assembled here into one episode.

Protocol per iteration:
  1. Call `agent.llm.chat_with_tools(messages, TOOL_SPECS)`.
  2. If the response includes tool_calls:
       - append the assistant turn (with tool_calls) to `messages`
       - for each tool_call: dispatch, log a Step, feed the result back
         to `messages` as a role=tool turn
       - if any tool_call is `finalize_answer`, capture the answer and
         return the completed Trace
  3. If the response is only free text (no tool_call), log it as a
     Step(thought=...), add it as an assistant turn, and nudge the LLM
     with a user message telling it to call a tool next.
  4. If `max_steps` iterations elapse without `finalize_answer`, raise
     `MaxStepsExceeded(trace)` with the partial trace attached.

Both `LLMClient` (real Groq) and `RecordedLLMClient` (test replay) satisfy
the same `chat_with_tools(messages, tools) -> AssistantResponse` shape, so
the loop never knows which one is driving.

Owner: Person 4 (Phase 4). See PROJECT_PLAN.md section H.2.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

from iagent.procedural.llm_client import AssistantResponse, ToolCallRequest
from iagent.procedural.models import Observation, Step, ToolCall, Trace
from iagent.procedural.prompts import SYSTEM_PROMPT, format_query
from iagent.procedural.tools import FINALIZE_TOOL, TOOL_SPECS, dispatch

__all__ = ["MaxStepsExceeded", "run_react"]


class MaxStepsExceeded(RuntimeError):
    """Raised when the loop hits `max_steps` without `finalize_answer`.

    The partial Trace (every step up to the budget cap) is attached so
    callers can still inspect what happened.
    """

    def __init__(self, trace: Trace) -> None:
        super().__init__(f"ReAct loop exceeded max_steps ({len(trace.steps)})")
        self.trace = trace


_NUDGE = (
    "You must call a tool to continue the episode. "
    "Use `finalize_answer` to end."
)


def run_react(
    agent,
    query: str,
    *,
    scenario: str | None = None,
    max_steps: int = 8,
) -> Trace:
    """Drive one ReAct episode. Returns the completed `Trace`.

    `agent` is duck-typed: must expose `.knowledge`, `.sensors`, `.llm`.
    This lets tests pass a lightweight fake instead of constructing a full
    `Agent` with Groq credentials.
    """
    apis: dict[str, Any] = {
        "knowledge": agent.knowledge,
        "sensors": agent.sensors,
    }

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": format_query(query, scenario)},
    ]
    trace = Trace(
        query=query,
        scenario=scenario,
        steps=[],
        started_at=_now(),
    )

    for step_idx in range(1, max_steps + 1):
        response: AssistantResponse = agent.llm.chat_with_tools(messages, TOOL_SPECS)

        if response.tool_calls:
            # 1. Record the assistant's tool_call turn in the message log.
            messages.append(_assistant_tool_call_message(response))

            # 2. Execute every tool call in this turn (usually one, sometimes a batch).
            for tc in response.tool_calls:
                result = dispatch(tc.name, tc.arguments, apis)

                trace.steps.append(_build_step(
                    idx=step_idx,
                    thought=response.content,
                    tool_call=tc,
                    result=result,
                ))

                # 3. Feed the result back as a role=tool turn so the LLM sees it.
                messages.append(_tool_result_message(tc.id, result))

                # 4. finalize_answer terminates the episode.
                if tc.name == FINALIZE_TOOL and result.get("ok"):
                    trace.final_answer = _extract_final_answer(result)
                    trace.ended_at = _now()
                    return trace
        else:
            # Free-text response with no tool call -- log it, nudge the LLM.
            thought_text = (response.content or "").strip()
            trace.steps.append(Step(idx=step_idx, thought=thought_text or "(empty)"))
            if thought_text:
                messages.append({"role": "assistant", "content": thought_text})
            messages.append({"role": "user", "content": _NUDGE})

    # Loop budget exhausted without a final answer.
    trace.ended_at = _now()
    raise MaxStepsExceeded(trace)


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
