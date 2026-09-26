"""LLM clients used by the ReAct loop.

Two implementations, one shape:

    * `LLMClient`         — real Groq/OpenAI-compatible tool-calling wrapper.
    * `RecordedLLMClient` — deterministic replay of a scripted sequence, used
                            by every unit + epistemic test so the suite never
                            hits the network.

Both expose `chat_with_tools(messages, tools) -> AssistantResponse` so the
ReAct loop (P4) is oblivious to which one it's driving.

Owner: Person 3 (Phase 4). See PROJECT_PLAN.md sections H, K.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "ToolCallRequest",
    "AssistantResponse",
    "LLMClient",
    "LLMClientError",
    "RecordedLLMClient",
    "ScriptExhausted",
]


class LLMClientError(RuntimeError):
    """Raised when the provider call fails after all retries."""


class ScriptExhausted(RuntimeError):
    """Raised by RecordedLLMClient when the loop asks for more steps than scripted."""


@dataclass(frozen=True)
class ToolCallRequest:
    """One tool call the LLM wants us to execute."""

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class AssistantResponse:
    """Normalized shape of one assistant turn."""

    content: str | None = None
    tool_calls: list[ToolCallRequest] = field(default_factory=list)
    finish_reason: str | None = None
    raw: dict[str, Any] | None = None


# ---------------------------------------------------------------------------
# Real Groq client
# ---------------------------------------------------------------------------


class LLMClient:
    """Thin wrapper around `groq.Groq` with retry + normalization.

    Kept minimal on purpose: transport concerns only. Prompt shaping lives in
    `prompts.py`, tool routing in `tools.py`, the loop in `react_loop.py`.
    """

    def __init__(
        self,
        model: str = "llama-3.1-8b-instant",
        temperature: float = 0.0,
        max_tokens: int = 1024,
        timeout_s: float = 30.0,
        max_retries: int = 3,
        api_key: str | None = None,
        client: Any | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self._sleep = sleep

        if client is not None:
            self._client = client
            return

        try:
            from groq import Groq
        except ImportError as exc:  # pragma: no cover — surfaces only when groq missing
            raise LLMClientError(
                "The 'groq' package is required. `pip install groq` or pass a "
                "pre-built client via the `client=` kwarg."
            ) from exc

        key = api_key or os.environ.get("GROQ_API_KEY")
        if not key:
            raise LLMClientError(
                "GROQ_API_KEY is not set. Export it or pass api_key= explicitly."
            )
        self._client = Groq(api_key=key, timeout=timeout_s)

    def chat_with_tools(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> AssistantResponse:
        """Call the provider once; retry transient failures with backoff."""
        last_exc: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                raw = self._client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=tools,
                    tool_choice="auto",
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                return self._normalize(raw)
            except Exception as exc:  # noqa: BLE001 — provider errors vary widely
                last_exc = exc
                if attempt == self.max_retries:
                    break
                self._sleep(0.5 * (2 ** (attempt - 1)))
        raise LLMClientError(
            f"Groq call failed after {self.max_retries} attempts: {last_exc}"
        ) from last_exc

    @staticmethod
    def _normalize(raw: Any) -> AssistantResponse:
        choice = raw.choices[0]
        message = choice.message
        tool_calls: list[ToolCallRequest] = []
        for tc in getattr(message, "tool_calls", None) or []:
            fn = tc.function
            args_str = fn.arguments or "{}"
            try:
                args = json.loads(args_str) if isinstance(args_str, str) else dict(args_str)
            except json.JSONDecodeError:
                args = {"__raw__": args_str}
            tool_calls.append(
                ToolCallRequest(id=tc.id, name=fn.name, arguments=args)
            )
        return AssistantResponse(
            content=getattr(message, "content", None),
            tool_calls=tool_calls,
            finish_reason=getattr(choice, "finish_reason", None),
            raw=None,
        )


# ---------------------------------------------------------------------------
# Deterministic recorded client (for tests)
# ---------------------------------------------------------------------------


class RecordedLLMClient:
    """Replays a scripted list of turns. Never hits the network.

    Each script entry is either:
        * a dict `{"tool": "name", "arguments": {...}}`
        * a list of such dicts (multiple tool calls in one turn)
        * a dict `{"content": "free text"}` (assistant thought, no tool call)

    Example (the Scenario-A happy path):

        script = [
            {"tool": "get_belief",
             "arguments": {"subject": "path_ahead", "predicate": "is_clear"}},
            {"tool": "sense_lidar",
             "arguments": {"direction": "front"}},
            {"tool": "finalize_answer",
             "arguments": {"answer": "Blocked — sensor beats stale map."}},
        ]
        llm = RecordedLLMClient(script)
    """

    def __init__(self, script: Iterable[dict[str, Any] | list[dict[str, Any]]]) -> None:
        self._script = list(script)
        self._cursor = 0
        self.calls: list[dict[str, Any]] = []  # captured (messages, tools) per turn

    def chat_with_tools(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> AssistantResponse:
        self.calls.append({"messages": messages, "tools": tools})
        if self._cursor >= len(self._script):
            raise ScriptExhausted(
                f"RecordedLLMClient script exhausted after {self._cursor} turns "
                "— the loop asked for one more."
            )
        turn = self._script[self._cursor]
        self._cursor += 1
        return self._turn_to_response(turn, index=self._cursor)

    def remaining(self) -> int:
        return len(self._script) - self._cursor

    @staticmethod
    def _turn_to_response(
        turn: dict[str, Any] | list[dict[str, Any]],
        index: int,
    ) -> AssistantResponse:
        if isinstance(turn, dict) and "content" in turn and "tool" not in turn:
            return AssistantResponse(content=str(turn["content"]))

        raw_calls = turn if isinstance(turn, list) else [turn]
        tool_calls: list[ToolCallRequest] = []
        for i, entry in enumerate(raw_calls):
            if "tool" not in entry:
                raise ValueError(
                    f"RecordedLLMClient script turn {index} entry {i} needs a 'tool' key."
                )
            tool_calls.append(
                ToolCallRequest(
                    id=entry.get("id", f"call_{index}_{i}"),
                    name=entry["tool"],
                    arguments=dict(entry.get("arguments", {})),
                )
            )
        return AssistantResponse(tool_calls=tool_calls, finish_reason="tool_calls")
