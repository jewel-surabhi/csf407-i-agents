"""NetworkX MultiDiGraph wrapper for the current-view store.

The BeliefGraph is the authoritative source for what the agent CURRENTLY
believes. Every belief the API writes lands here as a directed edge whose
attributes carry the belief's metadata (relation, confidence, source,
perspective, status, observed_at, provenance_id).

Multi-edges are essential — the same (subject, object) pair can carry many
beliefs, and the same (subject, predicate) can hold from multiple
perspectives simultaneously. That is what makes Scenario B representable.

History lives in `ProvenanceStore`, not here — when a belief is downgraded,
we flip its `status` attribute (never delete) so a graph traversal can still
see the downgraded edge for debugging.

Owner: Person 1 (P1). See PROJECT_PLAN.md sections F.1 and G.1.
"""

from __future__ import annotations

from collections.abc import Iterator

import networkx as nx

from iagent.declarative.models import Belief, KnowledgeError


class BeliefGraph:
    """Thin `networkx.MultiDiGraph` wrapper. Only accessed via `KnowledgeAPI`."""

    def __init__(self) -> None:
        self._g: nx.MultiDiGraph = nx.MultiDiGraph()

    # ---- node management -----------------------------------------------------

    def add_node(
        self,
        node_id: str,
        *,
        type: str = "entity",
        labels: list[str] | None = None,
    ) -> None:
        """Register a cognitive entity. No-op if the node already exists.

        Never overwrites the attributes of an existing node — earliest wins.
        """
        if node_id not in self._g:
            self._g.add_node(node_id, type=type, labels=list(labels or []))

    def has_node(self, node_id: str) -> bool:
        return node_id in self._g

    # ---- edge writes ---------------------------------------------------------

    def add_edge(self, belief: Belief) -> int:
        """Add a directed edge for this belief. Auto-creates missing nodes.

        Returns the NetworkX multi-edge key so callers can address this exact
        edge later.

        NOTE: `add_edge` always inserts a new edge — it does NOT deduplicate.
        Callers wanting "at most one active belief per (subject, predicate,
        perspective)" should call `set_edge_status(..., 'downgraded')` on the
        existing active edge first. `KnowledgeAPI.update_belief` enforces that
        pattern.
        """
        self.add_node(belief.subject)
        self.add_node(belief.object)
        key = self._g.add_edge(
            belief.subject,
            belief.object,
            relation=belief.predicate,
            confidence=float(belief.confidence),
            source=belief.source,
            perspective=belief.perspective,
            status=belief.status,
            observed_at=belief.observed_at.isoformat(),
            provenance_id=belief.provenance_id,
        )
        return int(key)

    def set_edge_status(
        self,
        subject: str,
        predicate: str,
        perspective: str,
        status: str,
        *,
        new_confidence: float | None = None,
    ) -> None:
        """Flip the active edge for (subject, predicate, perspective)'s status.

        If `new_confidence` is supplied, updates it too (useful for
        `downgrade_belief` which lowers confidence AND flips status in the
        same operation).

        Raises `KnowledgeError` if no active edge matches — we do NOT
        silently no-op, because that would mask a coordination bug between
        the API and the store.
        """
        if status not in ("active", "downgraded", "retracted"):
            raise KnowledgeError(f"invalid status: {status!r}")

        found = self._find_active_edge_key(subject, predicate, perspective)
        if found is None:
            raise KnowledgeError(
                f"no active edge for ({subject!r}, {predicate!r}, {perspective!r})"
            )
        v, k = found
        self._g[subject][v][k]["status"] = status
        if new_confidence is not None:
            if not 0.0 <= new_confidence <= 1.0:
                raise KnowledgeError(f"confidence out of range: {new_confidence}")
            self._g[subject][v][k]["confidence"] = float(new_confidence)

    # ---- edge reads ----------------------------------------------------------

    def get_active_edge(
        self,
        subject: str,
        predicate: str,
        perspective: str = "self",
    ) -> Belief | None:
        """Return the single active belief for this key, or None.

        Raises `KnowledgeError` if MORE than one active edge is found — that
        would be a coordination bug (the invariant is: at most one active per
        (subject, predicate, perspective)).
        """
        if subject not in self._g:
            return None

        matches: list[tuple[str, int, dict]] = [
            (v, k, d)
            for u, v, k, d in self._g.out_edges(subject, keys=True, data=True)
            if d.get("relation") == predicate
            and d.get("perspective") == perspective
            and d.get("status") == "active"
        ]
        if not matches:
            return None
        if len(matches) > 1:
            raise KnowledgeError(
                f"multiple active edges for ({subject!r}, {predicate!r}, "
                f"{perspective!r}); the graph invariant has been violated"
            )
        v, _k, d = matches[0]
        return _edge_to_belief(subject, v, d)

    def iter_active_edges_for(
        self,
        subject: str,
        predicate: str | None = None,
    ) -> Iterator[Belief]:
        """Yield every active belief about `subject`, optionally filtered by
        predicate. Used by `KnowledgeAPI.query_perspective`.
        """
        if subject not in self._g:
            return
        for _u, v, _k, d in self._g.out_edges(subject, keys=True, data=True):
            if d.get("status") != "active":
                continue
            if predicate is not None and d.get("relation") != predicate:
                continue
            yield _edge_to_belief(subject, v, d)

    def iter_all_edges(self) -> Iterator[Belief]:
        """Yield every edge in the graph regardless of status (debug/inspection)."""
        for u, v, _k, d in self._g.edges(keys=True, data=True):
            yield _edge_to_belief(u, v, d)

    # ---- debugging / seeding -------------------------------------------------

    def snapshot(self) -> dict:
        """Return a JSON-serializable dump of the graph (nodes + edges).

        Not exposed to the LLM. Useful for tests and the verification report.
        """
        return {
            "nodes": [
                {"id": n, **{k: v for k, v in self._g.nodes[n].items()}}
                for n in self._g.nodes
            ],
            "edges": [
                {"subject": u, "object": v, "key": k, **d}
                for u, v, k, d in self._g.edges(keys=True, data=True)
            ],
        }

    def __len__(self) -> int:
        """Number of edges (beliefs), not nodes."""
        return self._g.number_of_edges()

    # ---- internals -----------------------------------------------------------

    def _find_active_edge_key(
        self,
        subject: str,
        predicate: str,
        perspective: str,
    ) -> tuple[str, int] | None:
        if subject not in self._g:
            return None
        for _u, v, k, d in self._g.out_edges(subject, keys=True, data=True):
            if (
                d.get("relation") == predicate
                and d.get("perspective") == perspective
                and d.get("status") == "active"
            ):
                return (v, int(k))
        return None


# ------------------------- module-private helpers ---------------------------


def _edge_to_belief(subject: str, obj: str, data: dict) -> Belief:
    """Convert a NetworkX edge attribute dict back into a typed Belief.

    Pydantic parses the ISO datetime string automatically.
    """
    return Belief(
        subject=subject,
        predicate=data["relation"],
        object=obj,
        confidence=data["confidence"],
        source=data["source"],
        perspective=data.get("perspective", "self"),
        status=data.get("status", "active"),
        observed_at=data["observed_at"],  # Pydantic parses ISO strings
        provenance_id=data.get("provenance_id"),
    )
