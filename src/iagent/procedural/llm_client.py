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
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "ToolCallRequest",
    "AssistantResponse",
    "LLMClient",
    "LLMClientError",
    "RecordedLLMClient",
    "ScriptExhausted",
    "DEFAULT_MODEL",
    "MALFORMED_ARGS_KEY",
]

# Open-weight, tool-calling, available on the Groq free tier. Overridable via
# `LLM_MODEL` in .env or the `llm.model` key in configs/default.yaml.
DEFAULT_MODEL = "openai/gpt-oss-20b"

# Where unparseable tool-call arguments are stashed so the dispatcher reports a
# recoverable `missing_argument` instead of the loop crashing.
MALFORMED_ARGS_KEY = "__raw__"


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
    """Normalized shape of one assistant turn.

    `malformed` flags a generation whose tool calls did not parse -- unnamed
    functions, or an `arguments` payload that is not valid JSON. Small
    instruct-tuned models do this occasionally; `LLMClient` retries such a
    turn before handing it on.
    """

    content: str | None = None
    tool_calls: list[ToolCallRequest] = field(default_factory=list)
    finish_reason: str | None = None
    raw: dict[str, Any] | None = None
    malformed: bool = False


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
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        timeout_s: float | None = None,
        max_retries: int | None = None,
        api_key: str | None = None,
        client: Any | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        # Anything left as None resolves from the environment (.env), then
        # from the documented default -- so `LLMClient()` honours LLM_MODEL.
        self.model = model or _env_str("LLM_MODEL") or DEFAULT_MODEL
        self.temperature = _first_not_none(temperature, _env_float("LLM_TEMPERATURE"), 0.0)
        self.max_tokens = int(_first_not_none(max_tokens, _env_int("LLM_MAX_TOKENS"), 1024))
        self.timeout_s = float(_first_not_none(timeout_s, _env_float("LLM_TIMEOUT_S"), 30.0))
        self.max_retries = int(_first_not_none(max_retries, _env_int("LLM_MAX_RETRIES"), 3))
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
            raise LLMClientError("GROQ_API_KEY is not set. Export it or pass api_key= explicitly.")
        self._client = Groq(api_key=key, timeout=self.timeout_s)

    @classmethod
    def from_config(cls, config: Any = None, **overrides: Any) -> LLMClient:
        """Build a client from the parsed `configs/default.yaml`.

        Precedence, highest first: explicit keyword overrides, environment
        variables (`.env`), the config file's `llm:` block, then this module's
        defaults -- matching the note at the top of configs/default.yaml.

        `config` may be a dict or any object exposing an `llm` attribute, so
        this works before P4's `AppConfig` lands. The `llm.provider` key is
        read for validation only: Groq is the one supported transport.
        """
        block = _llm_block(config)
        provider = str(block.get("provider", "groq")).lower()
        if provider not in {"groq", ""}:
            raise LLMClientError(
                f"Unsupported llm.provider {provider!r}; this client speaks Groq "
                "(OpenAI-compatible) only."
            )

        settings: dict[str, Any] = {
            "model": _env_str("LLM_MODEL") or block.get("model"),
            "temperature": _first_not_none(_env_float("LLM_TEMPERATURE"), block.get("temperature")),
            "max_tokens": _first_not_none(_env_int("LLM_MAX_TOKENS"), block.get("max_tokens")),
            "timeout_s": _first_not_none(_env_float("LLM_TIMEOUT_S"), block.get("timeout_s")),
            "max_retries": _first_not_none(_env_int("LLM_MAX_RETRIES"), block.get("max_retries")),
        }
        settings.update({k: v for k, v in overrides.items() if v is not None})
        return cls(**settings)

    def chat_with_tools(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> AssistantResponse:
        """Call the provider, retrying transient failures with backoff.

        A *malformed* generation is retried too: small models sometimes emit a
        tool call whose `arguments` are not valid JSON, and at temperature 0 a
        resample is still worth one try. If every attempt comes back malformed
        the last response is returned anyway rather than raised -- the
        dispatcher turns it into a `missing_argument` observation the LLM can
        recover from, which beats aborting the episode.
        """
        last_exc: Exception | None = None
        last_malformed: AssistantResponse | None = None

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
            except Exception as exc:  # noqa: BLE001 — provider errors vary widely
                last_exc = exc
                if attempt == self.max_retries:
                    break
                self._sleep(_backoff(attempt))
                continue

            response = self._normalize(raw)
            if not response.malformed:
                return response

            last_malformed = response
            if attempt == self.max_retries:
                break
            self._sleep(_backoff(attempt))

        if last_malformed is not None:
            return last_malformed
        raise LLMClientError(
            f"Groq call failed after {self.max_retries} attempts: {last_exc}"
        ) from last_exc

    @staticmethod
    def _normalize(raw: Any) -> AssistantResponse:
        choice = raw.choices[0]
        message = choice.message
        tool_calls: list[ToolCallRequest] = []
        malformed = False

        for tc in getattr(message, "tool_calls", None) or []:
            fn = getattr(tc, "function", None)
            name = getattr(fn, "name", None) or ""
            if not name:
                malformed = True

            args_str = getattr(fn, "arguments", None) or "{}"
            if isinstance(args_str, str):
                try:
                    args = json.loads(args_str)
                except json.JSONDecodeError:
                    args = {MALFORMED_ARGS_KEY: args_str}
                    malformed = True
            else:
                args = args_str

            if not isinstance(args, dict):
                # e.g. a bare list or string where an object was required.
                args = {MALFORMED_ARGS_KEY: args}
                malformed = True

            tool_calls.append(
                ToolCallRequest(id=getattr(tc, "id", "") or "", name=name, arguments=args)
            )

        return AssistantResponse(
            content=getattr(message, "content", None),
            tool_calls=tool_calls,
            finish_reason=getattr(choice, "finish_reason", None),
            raw=None,
            malformed=malformed,
        )


def _backoff(attempt: int) -> float:
    """Exponential backoff: 0.5s, 1s, 2s, ..."""
    return 0.5 * (2 ** (attempt - 1))


def _first_not_none(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _llm_block(config: Any) -> Mapping[str, Any]:
    """Pull the `llm:` mapping out of a dict- or attribute-style config."""
    if config is None:
        return {}
    block = config.get("llm") if isinstance(config, Mapping) else getattr(config, "llm", None)
    if block is None:
        return {}
    if isinstance(block, Mapping):
        return block
    # A Pydantic sub-model or similar: expose its fields as a mapping.
    return dict(getattr(block, "__dict__", {}) or {})


def _env_str(name: str) -> str | None:
    value = os.environ.get(name, "").strip()
    return value or None


def _env_float(name: str) -> float | None:
    value = _env_str(name)
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _env_int(name: str) -> int | None:
    value = _env_str(name)
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None


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
