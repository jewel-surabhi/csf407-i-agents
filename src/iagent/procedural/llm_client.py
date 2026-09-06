"""LLMClient — Groq/OpenAI-compatible tool-calling wrapper.

Also defines `RecordedLLMClient` for deterministic tests.

Placeholder — implementation lands in Phase 4 (Owner: Person 3).
"""

from __future__ import annotations

from typing import Any

# EMPTY PLACEHOLDER — IMPLEMENTATION LATER
# Owner: Person 3 (Phase 4).


class LLMClient:
    """Wraps `groq.Groq` (or an OpenAI-compat client) with retries + logging."""

    def __init__(self, config: Any) -> None:
        raise NotImplementedError("Phase 4 — see PROJECT_PLAN.md section H.")

    def chat_with_tools(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Return the raw provider response including any tool_calls."""
        raise NotImplementedError("Phase 4.")


class RecordedLLMClient:
    """Replays a canned sequence of tool_calls. Used by unit + epistemic tests."""

    def __init__(self, script: list[dict[str, Any]]) -> None:
        raise NotImplementedError("Phase 4 — see PROJECT_PLAN.md section K.")
