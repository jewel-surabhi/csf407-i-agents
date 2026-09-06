"""Pydantic models crossing the sensorimotor-layer boundary."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class SensorPayload(BaseModel):
    """Uniform envelope for every sensor reading.

    `kind` names the reading shape; consumers key on it.
    """

    sensor: str
    kind: str
    value: Any = None
    status: str = "ok"
    observed_at: datetime
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    raw: dict[str, Any] | None = None

    # Optional camera-only fields (kept optional so all sensors share one envelope):
    detections: list[dict[str, Any]] | None = None
    ambient: dict[str, Any] | None = None


class ActionResult(BaseModel):
    """Uniform envelope for every actuator response."""

    action: str
    params: dict[str, Any] = Field(default_factory=dict)
    success: bool
    reason: str | None = None
    world_delta: dict[str, Any] = Field(default_factory=dict)
    observed_at: datetime


class WorldState(BaseModel):
    """The mutable world the MockEnvironment owns. Not exposed to the LLM."""

    robot: dict[str, Any] = Field(default_factory=dict)
    objects: list[dict[str, Any]] = Field(default_factory=list)
    obstacles: list[dict[str, Any]] = Field(default_factory=list)
    lighting: str = "normal"
    extras: dict[str, Any] = Field(default_factory=dict)
