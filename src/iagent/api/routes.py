"""HTTP routes for the Agent.

Keep the surface small and typed. Every endpoint's response is a Pydantic
model so FastAPI's auto-generated OpenAPI (/docs) stays accurate.

Owner: Person 4.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status

from iagent.api.deps import get_agent
from iagent.api.schemas import HealthResponse, QueryRequest, QueryResponse
from iagent.procedural.agent import Agent
from iagent.procedural.react_loop import MaxStepsExceeded

router = APIRouter(prefix="/api", tags=["agent"])

# Modern FastAPI pattern: Annotated[...] carries Depends without triggering
# ruff's B008 "function call in default" false positive.
AgentDep = Annotated[Agent, Depends(get_agent)]


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Trivial liveness probe — does NOT touch the Agent, so it works even
    when GROQ_API_KEY is missing."""
    return HealthResponse()


@router.post("/query", response_model=QueryResponse)
def query_endpoint(req: QueryRequest, agent: AgentDep) -> QueryResponse:
    """Run one ReAct episode against the Agent and return the full trace.

    On `MaxStepsExceeded` we return 422 with the partial trace in the error
    detail, so the frontend can still render what happened before the cap.
    """
    try:
        result = agent.run(
            req.query,
            scenario=req.scenario,
            max_steps=req.max_steps or 8,
        )
    except MaxStepsExceeded as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "error": "max_steps_exceeded",
                "message": str(exc),
                "partial_trace": exc.trace.model_dump(mode="json"),
            },
        ) from exc

    return QueryResponse(
        answer=result.answer,
        trace=result.trace,
        tools_called=result.tools_called,
    )
