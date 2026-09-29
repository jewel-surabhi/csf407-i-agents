"""Bootstrap `iagent.db` from `data/seed/*`.

Usage:
    python scripts/init_db.py                              # default paths
    python scripts/init_db.py --db /tmp/mydev.db
    python scripts/init_db.py --seed-dir data/seed --force

The SQLite schema is applied automatically the moment `ProvenanceStore` opens
the file, so this script's real job is to also run `initial_provenance.sql`
against the fresh DB and preview the graph node bootstrap from
`initial_beliefs.json`.

Owner: Person 1 (P1).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make `iagent` importable when the script runs directly (not via `python -m`).
_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

from iagent.declarative.belief_graph import BeliefGraph  # noqa: E402
from iagent.declarative.models import KnowledgeError  # noqa: E402
from iagent.declarative.provenance import ProvenanceStore  # noqa: E402
from iagent.declarative.seeding import seed_from_files  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Bootstrap iagent.db from data/seed/*")
    parser.add_argument(
        "--db",
        default=str(_REPO / "iagent.db"),
        help="SQLite file to create (default: ./iagent.db in the repo root)",
    )
    parser.add_argument(
        "--seed-dir",
        default=str(_REPO / "data" / "seed"),
        help="Directory containing initial_provenance.sql + initial_beliefs.json",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing DB file. Without this, an existing file is left alone.",
    )
    args = parser.parse_args(argv)

    db_path = Path(args.db)
    seed_dir = Path(args.seed_dir)
    prov_sql = seed_dir / "initial_provenance.sql"
    beliefs_json = seed_dir / "initial_beliefs.json"

    if db_path.exists():
        if not args.force:
            print(
                f"[init_db] {db_path} already exists. Re-run with --force to overwrite.",
                file=sys.stderr,
            )
            return 2
        db_path.unlink()
        print(f"[init_db] removed existing {db_path}")

    for path in (prov_sql, beliefs_json):
        if not path.exists():
            print(f"[init_db] missing seed file: {path}", file=sys.stderr)
            return 2

    try:
        with ProvenanceStore(str(db_path)) as store:
            graph = BeliefGraph()
            summary = seed_from_files(
                store,
                graph,
                provenance_sql=prov_sql,
                beliefs_json=beliefs_json,
            )
            print()
            print(f"  DB file       : {db_path}")
            print("  Schema        : applied (src/iagent/declarative/schema.sql)")
            print(f"  Sources seeded: {summary['sources']}")
            for src in store.list_sources():
                print(f"    - {src.source_id:<14} kind={src.kind:<10} "
                      f"trust_prior={src.trust_prior}")
            print(f"  Graph nodes   : {summary['nodes_added']} added (in-memory preview)")
            print(f"  Graph edges   : {summary['edges_added']} added (in-memory preview)")
            print()
            print("[init_db] ready.")
    except KnowledgeError as e:
        print(f"[init_db] {e}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
