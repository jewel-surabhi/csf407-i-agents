"""Create the SQLite provenance DB from schema.sql + seed rows.

Placeholder — implementation lands in Phase 2 (Owner: Person 1).

Intended usage:
    python scripts/init_db.py [--db iagent.db]
"""

from __future__ import annotations

# EMPTY PLACEHOLDER — IMPLEMENTATION LATER
# Owner: Person 1 (Phase 2).
# 1. Open sqlite3 connection to args.db.
# 2. executescript(read src/iagent/declarative/schema.sql).
# 3. executescript(read data/seed/initial_provenance.sql).
# 4. Report row counts.


def main() -> int:
    raise NotImplementedError("Phase 2 — see PROJECT_PLAN.md section E.")


if __name__ == "__main__":
    raise SystemExit(main())
