"""SQLite CRUD for the provenance table.

Placeholder — implementation lands in Phase 2 (Owner: Person 1).
"""

from __future__ import annotations

# EMPTY PLACEHOLDER — IMPLEMENTATION LATER
# Owner: Person 1 (Phase 2).
# Must load DDL from `schema.sql` verbatim rather than duplicating in code.
# See PROJECT_PLAN.md section F.2 and G.1.


class ProvenanceStore:
    """Thin sqlite3 wrapper. Only accessed via `KnowledgeAPI`."""

    def __init__(self, db_path: str) -> None:
        raise NotImplementedError("Phase 2 — see PROJECT_PLAN.md section F.2.")
