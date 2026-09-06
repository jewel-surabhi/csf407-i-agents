"""Configuration loader (YAML + .env). Placeholder until Phase 5."""

from __future__ import annotations

from pathlib import Path
from typing import Any

# EMPTY PLACEHOLDER — IMPLEMENTATION LATER
# Owner: Person 4 (Phase 5).
# Should:
#   1. Load .env via python-dotenv.
#   2. Load a YAML file (defaulting to configs/default.yaml).
#   3. Environment variables override matching YAML keys.
#   4. Return a validated Pydantic AppConfig model.


def load_config(path: str | Path = "configs/default.yaml") -> dict[str, Any]:
    """TODO(P4): parse YAML + .env, validate, return an AppConfig."""
    raise NotImplementedError("Phase 5 — see PROJECT_PLAN.md section E and G.4.")
