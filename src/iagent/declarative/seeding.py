"""Load `data/seed/*` into a live declarative layer.

Two concerns handled separately, so they can be used independently:

- **Sources** live in SQLite and are bootstrapped by executing
  `data/seed/initial_provenance.sql` against `ProvenanceStore`. Idempotent
  (the SQL uses `INSERT OR IGNORE`).
- **Belief graph nodes/edges** live in NetworkX (in-memory) and are
  bootstrapped from `data/seed/initial_beliefs.json` into a `BeliefGraph`.

`init_db.py` glues them together for the one-shot CLI bootstrap.
Runtime code (the Agent) can call `seed_from_files` at startup to rebuild
the graph state from disk before serving queries.

Owner: Person 1 (P1). See PROJECT_PLAN.md section E (data/seed/*).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from iagent.declarative.belief_graph import BeliefGraph
from iagent.declarative.models import Belief, KnowledgeError
from iagent.declarative.provenance import ProvenanceStore

__all__ = [
    "load_beliefs_from_json",
    "apply_beliefs_to_graph",
    "seed_sources",
    "seed_graph",
    "seed_from_files",
]


def load_beliefs_from_json(path: str | Path) -> dict[str, Any]:
    """Parse a beliefs seed JSON file. Expected shape:
        {"nodes": [{"id": str, "type": str?, "labels": [str]?}, ...],
         "edges": [{"subject": str, "predicate": str, "object": str,
                    "confidence": float?, "source": str,
                    "perspective": str?, "status": str?,
                    "observed_at": str}, ...]}

    The `edges` list may be empty (or omitted); nodes are the minimum viable
    seed. Missing 'nodes' key raises `KnowledgeError`.
    """
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        raise KnowledgeError(f"cannot parse beliefs seed {path}: {e}") from e

    if not isinstance(data, dict) or "nodes" not in data:
        raise KnowledgeError(
            f"beliefs seed {path} is missing required 'nodes' key"
        )
    return data


def apply_beliefs_to_graph(
    graph: BeliefGraph,
    data: dict[str, Any],
) -> tuple[int, int]:
    """Register nodes and (optional) edges from the parsed JSON.

    Idempotent — `add_node` skips existing nodes; `add_edge` inserts each
    edge in the seed. Returns `(nodes_added, edges_added)`.
    """
    nodes_added = 0
    for node in data.get("nodes", []):
        if not graph.has_node(node["id"]):
            graph.add_node(
                node["id"],
                type=node.get("type", "entity"),
                labels=list(node.get("labels", [])),
            )
            nodes_added += 1

    edges_added = 0
    for edge in data.get("edges", []):
        belief = Belief(
            subject=edge["subject"],
            predicate=edge["predicate"],
            object=edge["object"],
            confidence=float(edge.get("confidence", 1.0)),
            source=edge["source"],
            perspective=edge.get("perspective", "self"),
            status=edge.get("status", "active"),
            observed_at=edge["observed_at"],
            provenance_id=edge.get("provenance_id"),
        )
        graph.add_edge(belief)
        edges_added += 1

    return nodes_added, edges_added


def seed_sources(
    store: ProvenanceStore,
    provenance_sql: str | Path = "data/seed/initial_provenance.sql",
) -> None:
    """Bootstrap the `sources` table from a SQL seed file. Idempotent."""
    store.apply_seed_script(provenance_sql)


def seed_graph(
    graph: BeliefGraph,
    beliefs_json: str | Path = "data/seed/initial_beliefs.json",
) -> tuple[int, int]:
    """Bootstrap graph nodes/edges from a JSON seed file. Returns
    (nodes_added, edges_added)."""
    data = load_beliefs_from_json(beliefs_json)
    return apply_beliefs_to_graph(graph, data)


def seed_from_files(
    store: ProvenanceStore,
    graph: BeliefGraph,
    *,
    provenance_sql: str | Path = "data/seed/initial_provenance.sql",
    beliefs_json: str | Path = "data/seed/initial_beliefs.json",
) -> dict[str, Any]:
    """Convenience: seed both stores in one call. Returns a summary dict
    with keys `sources`, `nodes_added`, `edges_added`.
    """
    seed_sources(store, provenance_sql)
    nodes_added, edges_added = seed_graph(graph, beliefs_json)
    return {
        "sources": len(store.list_sources()),
        "nodes_added": nodes_added,
        "edges_added": edges_added,
    }
