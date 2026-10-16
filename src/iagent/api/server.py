"""FastAPI entry point.

Run locally:
    uvicorn iagent.api.server:app --reload --port 8000

The React dev server (Vite, default port 5173) is CORS-allowed. Tighten the
origins list before deploying anywhere public.
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from iagent.api.routes import router
from iagent.api.warehouse_routes import router as warehouse_router

app = FastAPI(
    title="I, Agent API",
    description=(
        "HTTP interface to the three-layer epistemic agent. "
        "Visit /docs for the OpenAPI UI."
    ),
    version="0.1.0",
)

# Dev-only CORS. Replace with the actual frontend origin before deploy.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",   # Vite default
        "http://127.0.0.1:5173",
        "http://localhost:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)
app.include_router(warehouse_router)
