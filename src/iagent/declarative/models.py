"""Pydantic models that cross the declarative-layer boundary.

These are the ONLY types the procedural layer sees from us. Field names and
types are frozen at the end of Phase 1 — changes require a coordinated PR
touching `docs/data_contracts.md`.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


class Belief(BaseModel):
    """A single belief edge, addressable by (subject, predicate, perspective)."""

    subject: str
    predicate: str
    object: str
    confidence: float = Field(ge=0.0, le=1.0)
    source: str
    perspective: str = Field(
        default="self",
        description="'self' | 'user' | 'third_party:<agent_id>'",
    )
    status: Literal["active", "downgraded", "retracted"] = "active"
    observed_at: datetime
    provenance_id: int | None = None


class ProvenanceEntry(BaseModel):
    """A single row in the provenance table."""

    provenance_id: int | None = None
    subject: str
    predicate: str
    object: str
    confidence: float = Field(ge=0.0, le=1.0)
    source_id: str
    perspective: str
    observed_at: datetime
    recorded_at: datetime | None = None
    status: Literal["active", "downgraded", "retracted"] = "active"
    payload: dict[str, Any] | None = None


class PerspectiveView(BaseModel):
    """All beliefs held from a single perspective on a subject."""

    perspective: str
    beliefs: list[Belief]


class Source(BaseModel):
    """Registered source with a trust prior."""

    source_id: str
    kind: Literal["static_map", "sensor", "agent", "user", "llm"]
    trust_prior: float = Field(ge=0.0, le=1.0)
    description: str | None = None


class KnowledgeError(ValueError):
    """Any invalid input to KnowledgeAPI raises this instead of DB exceptions."""
