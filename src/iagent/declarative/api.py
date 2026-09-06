"""KnowledgeAPI — the SOLE public entrypoint for the declarative layer.

Every method signature listed here is frozen at end of Phase 1. Any change
requires a coordinated PR touching `docs/data_contracts.md` and approved by
all four owners.

Placeholder implementations raise NotImplementedError. Owner: Person 1 (Phase 2).
"""

from __future__ import annotations

from iagent.declarative.models import (
    Belief,
    KnowledgeError,  # re-export so callers only import from api
    PerspectiveView,
    ProvenanceEntry,
    Source,
)

__all__ = [
    "KnowledgeAPI",
    "Belief",
    "PerspectiveView",
    "ProvenanceEntry",
    "Source",
    "KnowledgeError",
]


class KnowledgeAPI:
    """Facade combining `BeliefGraph` and `ProvenanceStore`."""

    def __init__(self, graph=None, store=None) -> None:
        # EMPTY PLACEHOLDER — IMPLEMENTATION LATER (Phase 2, Person 1).
        self._graph = graph
        self._store = store

    # ---------- reads ----------

    def get_belief(
        self,
        subject: str,
        predicate: str,
        perspective: str = "self",
    ) -> Belief | None:
        """Return current best belief for (subject, predicate, perspective) or None."""
        raise NotImplementedError("Phase 2 — see PROJECT_PLAN.md section G.1.")

    def query_perspective(
        self,
        subject: str,
        predicate: str | None = None,
    ) -> list[PerspectiveView]:
        """Return all perspectives held about `subject`, grouped."""
        raise NotImplementedError("Phase 2 — see PROJECT_PLAN.md section G.1.")

    def query_provenance(
        self,
        subject: str,
        predicate: str | None = None,
        limit: int = 20,
    ) -> list[ProvenanceEntry]:
        """Full history, newest first."""
        raise NotImplementedError("Phase 2 — see PROJECT_PLAN.md section G.1.")

    def list_sources(self) -> list[Source]:
        """Registered sources with trust priors."""
        raise NotImplementedError("Phase 2 — see PROJECT_PLAN.md section G.1.")

    # ---------- writes ----------

    def update_belief(self, belief: Belief, reason: str) -> ProvenanceEntry:
        """Insert a new provenance row AND update graph edge. Atomic."""
        raise NotImplementedError("Phase 2 — see PROJECT_PLAN.md section G.1.")

    def downgrade_belief(
        self,
        subject: str,
        predicate: str,
        new_confidence: float,
        reason: str,
        perspective: str = "self",
    ) -> ProvenanceEntry:
        raise NotImplementedError("Phase 2 — see PROJECT_PLAN.md section G.1.")

    def log_source(
        self,
        source_id: str,
        kind: str,
        trust_prior: float,
        description: str,
    ) -> None:
        """Register a new source (idempotent)."""
        raise NotImplementedError("Phase 2 — see PROJECT_PLAN.md section G.1.")
