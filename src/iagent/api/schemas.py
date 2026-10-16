"""Request/response Pydantic schemas for the HTTP API.

Response bodies lean heavily on the existing cross-layer models (`Trace`,
`PerspectiveView`, `ProvenanceEntry`, `Source`, ...). This file holds only
the shapes unique to the HTTP surface — request bodies and the thin wrappers
the frontend expects.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from iagent.procedural.models import Trace


class QueryRequest(BaseModel):
    """POST /api/query body."""

    query: str = Field(..., min_length=1, max_length=2000)
    scenario: str | None = Field(default=None, max_length=16)
    max_steps: int | None = Field(default=None, ge=1, le=16)


class QueryResponse(BaseModel):
    """POST /api/query response.

    `trace` is the full ReAct trace (every tool call + observation).
    `tools_called` is a convenience projection the frontend uses for its
    tool-badge row without having to walk the trace.
    """

    answer: str
    trace: Trace
    tools_called: list[str]


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "0.1.0"
