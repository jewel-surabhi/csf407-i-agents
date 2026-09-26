"""KnowledgeAPI — the SOLE public entrypoint for the declarative layer.

The procedural layer talks to this facade and nothing else in `declarative/`.
NetworkX + SQLite live behind these methods.

Every method signature listed here is frozen at end of Phase 1. Any change
requires a coordinated PR touching `docs/data_contracts.md` and approved by
all four owners.

Write semantics (settled at Phase-1 review):
- **SQL-first ordering.** Every write commits to SQLite BEFORE mirroring into
  the graph. If SQL fails, the graph is untouched — no divergence. If the
  graph mutation somehow fails after SQL commit, the seeding routine can
  rebuild the graph by scanning the `current_beliefs` SQL view on next
  startup (idempotent replay). No rollback machinery required.
- **at-most-one-active per (subject, predicate, perspective).** Enforced by
  `BeliefGraph.get_active_edge` — writes that would violate this are
  automatically resolved: `update_belief` auto-downgrades any existing
  active belief for the same key before writing the new one.

Owner: Person 1 (P1).
"""

from __future__ import annotations

from collections import defaultdict

from iagent.declarative.belief_graph import BeliefGraph
from iagent.declarative.models import (
    Belief,
    KnowledgeError,  # re-export so callers only import from api
    PerspectiveView,
    ProvenanceEntry,
    Source,
)
from iagent.declarative.provenance import ProvenanceStore

__all__ = [
    "KnowledgeAPI",
    "Belief",
    "PerspectiveView",
    "ProvenanceEntry",
    "Source",
    "KnowledgeError",
]


class KnowledgeAPI:
    """Facade combining `BeliefGraph` (current view) and `ProvenanceStore` (history)."""

    def __init__(self, graph: BeliefGraph, store: ProvenanceStore) -> None:
        if graph is None or store is None:
            raise KnowledgeError("KnowledgeAPI requires both a BeliefGraph and a ProvenanceStore")
        self._graph = graph
        self._store = store

    # ---------- reads ----------

    def get_belief(
        self,
        subject: str,
        predicate: str,
        perspective: str = "self",
    ) -> Belief | None:
        """Return the currently-active belief for (subject, predicate, perspective).

        Purely a graph read — SQLite is not touched.
        """
        return self._graph.get_active_edge(subject, predicate, perspective)

    def query_perspective(
        self,
        subject: str,
        predicate: str | None = None,
    ) -> list[PerspectiveView]:
        """Return all perspectives held about `subject`, grouped.

        The grouping is what prevents the LLM from accidentally merging user /
        self / third-party views. Each returned `PerspectiveView` has a single
        perspective label plus every active belief that carries that label.
        """
        grouped: dict[str, list[Belief]] = defaultdict(list)
        for belief in self._graph.iter_active_edges_for(subject, predicate=predicate):
            grouped[belief.perspective].append(belief)

        return [
            PerspectiveView(perspective=persp, beliefs=beliefs)
            for persp, beliefs in grouped.items()
        ]

    def query_provenance(
        self,
        subject: str,
        predicate: str | None = None,
        limit: int = 20,
    ) -> list[ProvenanceEntry]:
        """Full history for `subject`, newest first. Reads through SQLite."""
        return self._store.query_by_subject(subject, predicate=predicate, limit=limit)

    def list_sources(self) -> list[Source]:
        """Registered sources with trust priors. Used by the LLM for conflict resolution."""
        return self._store.list_sources()

    # ---------- writes ----------

    def update_belief(self, belief: Belief, reason: str) -> ProvenanceEntry:
        """Assert a new active belief.

        If an active belief already exists for the same (subject, predicate,
        perspective), it is auto-superseded: the old SQL row is marked
        'downgraded' and the graph edge is flipped, then the new belief is
        written as a fresh active row + graph edge. The new provenance row IS
        the audit-log entry for the transition — no separate "downgrade
        postmortem" row is written (that's what `downgrade_belief` is for).

        Returns the new `ProvenanceEntry` (with `provenance_id` populated).
        """
        if belief.status != "active":
            raise KnowledgeError(
                f"update_belief writes active beliefs; got status={belief.status!r}"
            )

        # Supersede any existing active belief for the same key.
        existing = self._graph.get_active_edge(
            belief.subject, belief.predicate, belief.perspective
        )
        if existing is not None:
            if existing.provenance_id is not None:
                self._store.mark_status(existing.provenance_id, "downgraded")
            self._graph.set_edge_status(
                belief.subject, belief.predicate, belief.perspective, "downgraded"
            )

        # SQL insert FIRST — atomicity anchor. If this raises, graph is untouched.
        new_entry = self._store.insert(
            ProvenanceEntry(
                subject=belief.subject,
                predicate=belief.predicate,
                object=belief.object,
                confidence=belief.confidence,
                source_id=belief.source,
                perspective=belief.perspective,
                observed_at=belief.observed_at,
                status="active",
                payload={"reason": reason},
            )
        )

        # Mirror into the graph with the DB-assigned provenance_id.
        self._graph.add_edge(
            belief.model_copy(
                update={"provenance_id": new_entry.provenance_id, "status": "active"}
            )
        )

        return new_entry

    def downgrade_belief(
        self,
        subject: str,
        predicate: str,
        new_confidence: float,
        reason: str,
        perspective: str = "self",
    ) -> ProvenanceEntry:
        """Lower an active belief's confidence and mark it 'downgraded'.

        Unlike `update_belief`, this does NOT write a successor active
        belief — it just demotes the current one and records the reason.
        Use when the agent wants to say "I no longer trust this" without
        asserting a replacement.

        Writes a "downgrade postmortem" row into provenance that cites the
        superseded `provenance_id` in its payload. Returns that new row.

        Raises `KnowledgeError` if there is no active belief for the key
        (per current design; may be relaxed to a silent no-op after
        procedural-layer integration).
        """
        if not 0.0 <= new_confidence <= 1.0:
            raise KnowledgeError(f"confidence out of range: {new_confidence}")

        existing = self._graph.get_active_edge(subject, predicate, perspective)
        if existing is None:
            raise KnowledgeError(
                f"no active belief to downgrade: "
                f"({subject!r}, {predicate!r}, {perspective!r})"
            )

        # SQL: postmortem row explaining the demotion.
        dg_entry = self._store.insert(
            ProvenanceEntry(
                subject=existing.subject,
                predicate=existing.predicate,
                object=existing.object,
                confidence=new_confidence,
                source_id=existing.source,
                perspective=existing.perspective,
                observed_at=existing.observed_at,
                status="downgraded",
                payload={
                    "reason": reason,
                    "supersedes": existing.provenance_id,
                },
            )
        )
        # SQL: flip the old row's status (bookkeeping so `current_beliefs` VIEW
        # correctly excludes it).
        if existing.provenance_id is not None:
            self._store.mark_status(existing.provenance_id, "downgraded")
        # Graph: flip the edge status + drop confidence.
        self._graph.set_edge_status(
            subject,
            predicate,
            perspective,
            "downgraded",
            new_confidence=new_confidence,
        )
        return dg_entry

    def log_source(
        self,
        source_id: str,
        kind: str,
        trust_prior: float,
        description: str,
    ) -> None:
        """Register a new source (idempotent — re-registering is a no-op).

        The `kind` must be one of 'static_map', 'sensor', 'agent', 'user', 'llm'
        per `Source`'s Literal type. Anything else raises via Pydantic validation.
        """
        self._store.insert_source(
            Source(
                source_id=source_id,
                kind=kind,   # type: ignore[arg-type]  # Pydantic validates the literal
                trust_prior=trust_prior,
                description=description,
            )
        )
