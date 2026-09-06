"""Pydantic models for ReAct step logging."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class ToolCall(BaseModel):
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class Observation(BaseModel):
    tool: str
    ok: bool
    content: dict[str, Any] | str


class Step(BaseModel):
    idx: int
    thought: str | None = None
    tool_call: ToolCall | None = None
    observation: Observation | None = None
    final_answer: str | None = None


class Trace(BaseModel):
    query: str
    scenario: str | None = None
    steps: list[Step] = Field(default_factory=list)
    final_answer: str = ""
    started_at: datetime
    ended_at: datetime | None = None


class AgentResult(BaseModel):
    answer: str
    trace: Trace
    tools_called: list[str] = Field(default_factory=list)
