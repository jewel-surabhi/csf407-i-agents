"""HTTP routes for the Agent.

Keep the surface small and typed. Every endpoint's response is a Pydantic
model so FastAPI's auto-generated OpenAPI (/docs) stays accurate.

Owner: Person 4.
"""

from __future__ import annotations

import asyncio
import json
import queue
import threading
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse

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


@router.post("/query/stream")
def query_stream(req: QueryRequest, agent: AgentDep) -> StreamingResponse:
    """Run one ReAct episode AND stream each step to the client as it happens.

    Server-sent events, one `data:` chunk per message. Message kinds:
      - "step"  — a Step was just appended to the trace (live reasoning)
      - "done"  — final AgentResult (answer + full trace + tools_called)
      - "error" — the episode raised (message + partial trace if available)

    The frontend uses this to show the agent's tool calls in real time
    instead of a blank "Thinking…" spinner for 25 seconds.
    """
    step_q: queue.Queue = queue.Queue()
    done = threading.Event()
    container: dict[str, Any] = {}
    max_steps = req.max_steps or 8

    def worker() -> None:
        try:
            def on_step(step: Any) -> None:
                step_q.put(("step", step.model_dump(mode="json")))

            result = agent.run(
                req.query,
                scenario=req.scenario,
                max_steps=max_steps,
                on_step=on_step,
            )
            container["result"] = result
        except MaxStepsExceeded as exc:
            container["error"] = {
                "error": "max_steps_exceeded",
                "message": str(exc),
                "partial_trace": exc.trace.model_dump(mode="json"),
            }
        except Exception as exc:  # noqa: BLE001
            container["error"] = {"error": "agent_error", "message": str(exc)}
        finally:
            done.set()

    threading.Thread(target=worker, daemon=True).start()

    async def generate():
        yield 'data: {"kind":"opened"}\n\n'
        while not done.is_set() or not step_q.empty():
            try:
                kind, data = step_q.get(timeout=0.08)
                yield f"data: {json.dumps({'kind': kind, 'data': data})}\n\n"
            except queue.Empty:
                await asyncio.sleep(0.03)
        # Final envelope: either the full AgentResult or the error.
        if "result" in container:
            result = container["result"]
            payload = {
                "kind": "done",
                "data": {
                    "answer": result.answer,
                    "trace": result.trace.model_dump(mode="json"),
                    "tools_called": result.tools_called,
                },
            }
            yield f"data: {json.dumps(payload)}\n\n"
        else:
            payload = {"kind": "error", "data": container.get("error", {"message": "unknown"})}
            yield f"data: {json.dumps(payload)}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")
