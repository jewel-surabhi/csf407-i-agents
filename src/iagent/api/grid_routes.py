"""HTTP routes for the Scenario A grid dashboard (web/public/grid.html).

    GET  /api/grid/state   layout, phase, clock, world, current declarations, last result, timeline
    GET  /api/grid/declarations?hour=H   declarative layer decayed to hour H (preview)
    POST /api/grid/setup   Complete setup (declares only the first time)
    POST /api/grid/ask     one ReAct episode at clock + hours (real Groq loop)
    POST /api/grid/reset   start a fresh session at hour 0
    GET  /grid             the dashboard page itself (no Vite needed)

One in-memory session per server process. Tests override `get_grid_llm`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from iagent.grid.agent import run_grid_question
from iagent.grid.session import GridSession
from iagent.grid.world import COLS, ROWS, SPACING_CM, GridWorld, all_nodes, all_segments

router = APIRouter(prefix="/api/grid", tags=["grid"])
page_router = APIRouter(tags=["grid"])

_PAGE = Path(__file__).resolve().parents[3] / "web" / "public" / "grid.html"


@page_router.get("/grid", include_in_schema=False)
@page_router.get("/grid.html", include_in_schema=False)
def grid_page() -> FileResponse:
    """Serve the dashboard from the backend so it works without the Vite dev server."""
    if not _PAGE.exists():
        raise HTTPException(status_code=404, detail=f"{_PAGE} not found")
    return FileResponse(_PAGE, media_type="text/html")

_session: GridSession | None = None


def get_grid_session() -> GridSession:
    global _session
    if _session is None:
        _session = GridSession()
    return _session


def reset_grid_session() -> GridSession:
    global _session
    _session = GridSession()
    return _session


def get_grid_llm() -> Any:
    """Real Groq client. Raises HTTP 503 with a readable message when no key is set."""
    from iagent.procedural.llm_client import LLMClient, LLMClientError

    try:
        return LLMClient()
    except LLMClientError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


class SetupBody(BaseModel):
    robot: str
    target: str
    boxes: list[str] = Field(default_factory=list)


class AskBody(BaseModel):
    question: str = Field(min_length=1, max_length=1000)
    hours: int = Field(default=1, ge=1, le=1000)


def _state(session: GridSession) -> dict[str, Any]:
    return {
        "layout": {
            "cols": COLS,
            "rows": ROWS,
            "spacing_cm": SPACING_CM,
            "nodes": all_nodes(),
            "segments": all_segments(),
        },
        "clock_h": session.clock_h,
        "declared": session.declared,
        "world": session.world.to_dict(),
        "declarations": session.declarations(session.clock_h) if session.declared else {},
        "last": session.history[-1] if session.history else None,
        "questions": len(session.history),
        "timeline": session.timeline(),
        "sources": {
            k: {"trust_prior": v.trust_prior, "decay_per_hour": v.decay_rate_per_hour}
            for k, v in session._sources.items()
            if k in ("operator_setup", "lidar_declared")
        },
    }


Session = Annotated[GridSession, Depends(get_grid_session)]


@router.get("/state")
def state(session: Session) -> dict[str, Any]:
    return _state(session)


@router.get("/declarations")
def declarations(session: Session, hour: float) -> dict[str, Any]:
    """Preview: the declarative layer as it would read at ``hour`` (now or later)."""
    if hour < session.clock_h:
        raise HTTPException(status_code=422, detail="hour must not be before the current clock")
    return {"hour": hour, "declarations": session.declarations(hour) if session.declared else {}}


@router.post("/setup")
def setup(body: SetupBody, session: Session) -> dict[str, Any]:
    try:
        world = GridWorld(robot=body.robot, target=body.target, boxes=set(body.boxes))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    declared_now = session.complete_setup(world)
    return {"declared_now": declared_now, **_state(session)}


@router.post("/ask")
def ask(
    body: AskBody,
    session: Session,
    llm: Annotated[Any, Depends(get_grid_llm)],
) -> dict[str, Any]:
    if not session.declared:
        raise HTTPException(status_code=409, detail="Complete the setup first.")
    try:
        record = run_grid_question(session, llm, body.question.strip(), body.hours)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001 - provider failures vary
        raise HTTPException(status_code=502, detail=f"LLM call failed: {exc}") from exc
    return _state(session) | {"last": record}


@router.post("/reset")
def reset() -> dict[str, Any]:
    return _state(reset_grid_session())
