"""SQLite CRUD for the provenance table.

Append-only audit log. The `provenance` table is never mutated in a way that
loses history — a belief "change" always inserts a NEW row and the old row's
`status` is flipped from 'active' to 'downgraded' (also via an UPDATE that
preserves the row itself).

Every method returns Pydantic models (`ProvenanceEntry`, `Source`) so the
layer boundary stays typed. Internal exceptions are translated to
`KnowledgeError` at every public entry point.

Loads DDL from the neighbouring `schema.sql` verbatim — never duplicate DDL
in Python.

Owner: Person 1 (P1). See PROJECT_PLAN.md sections F.2 and G.1.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path
from typing import Any

from iagent.declarative.models import KnowledgeError, ProvenanceEntry, Source

_SCHEMA_PATH = Path(__file__).parent / "schema.sql"


class ProvenanceStore:
    """Thin sqlite3 wrapper. Only accessed via `KnowledgeAPI`."""

    def __init__(self, db_path: str) -> None:
        self._db_path = db_path
        # `check_same_thread=False` keeps the store usable from pytest fixtures
        # that share a connection across setup + test. Access is still serial.
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        # Foreign-key enforcement is OFF by default in SQLite — must PRAGMA on.
        self._conn.execute("PRAGMA foreign_keys = ON;")
        self._apply_schema()

    # ---- context-manager support so `with ProvenanceStore(...) as s:` works ----

    def __enter__(self) -> ProvenanceStore:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def close(self) -> None:
        self._conn.close()

    # ---- schema management ---------------------------------------------------

    def _apply_schema(self) -> None:
        ddl = _SCHEMA_PATH.read_text(encoding="utf-8")
        try:
            with self._conn:
                self._conn.executescript(ddl)
        except sqlite3.Error as e:
            raise KnowledgeError(f"failed to apply schema.sql: {e}") from e

    # ---- writes --------------------------------------------------------------

    def insert(self, entry: ProvenanceEntry) -> ProvenanceEntry:
        """Insert a provenance row. Returns the entry with `provenance_id`
        and `recorded_at` populated from the DB."""
        sql = """
            INSERT INTO provenance
              (subject, predicate, object, confidence, source_id, perspective,
               observed_at, status, payload)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        params = (
            entry.subject,
            entry.predicate,
            entry.object,
            float(entry.confidence),
            entry.source_id,
            entry.perspective,
            _iso(entry.observed_at),
            entry.status,
            json.dumps(entry.payload) if entry.payload is not None else None,
        )
        try:
            with self._conn:
                cur = self._conn.execute(sql, params)
                new_id = cur.lastrowid
        except sqlite3.IntegrityError as e:
            raise KnowledgeError(
                f"provenance insert failed (likely unknown source_id={entry.source_id!r}): {e}"
            ) from e
        except sqlite3.Error as e:
            raise KnowledgeError(f"provenance insert failed: {e}") from e

        # Re-read to pick up the DB-assigned recorded_at.
        return self._get_by_id(int(new_id))

    def mark_status(self, provenance_id: int, status: str) -> None:
        """Flip the status of an existing row. Does NOT modify any other column."""
        if status not in ("active", "downgraded", "retracted"):
            raise KnowledgeError(f"invalid status: {status!r}")
        try:
            with self._conn:
                cur = self._conn.execute(
                    "UPDATE provenance SET status = ? WHERE provenance_id = ?",
                    (status, int(provenance_id)),
                )
                if cur.rowcount == 0:
                    raise KnowledgeError(f"no provenance row with id={provenance_id}")
        except sqlite3.Error as e:
            raise KnowledgeError(f"provenance status update failed: {e}") from e

    def insert_source(self, source: Source) -> None:
        """Register a source. Idempotent — re-inserting the same source_id is a no-op."""
        try:
            with self._conn:
                self._conn.execute(
                    """
                    INSERT OR IGNORE INTO sources (source_id, kind, trust_prior, description)
                    VALUES (?, ?, ?, ?)
                    """,
                    (source.source_id, source.kind, float(source.trust_prior), source.description),
                )
        except sqlite3.Error as e:
            raise KnowledgeError(f"source insert failed: {e}") from e

    def apply_seed_script(self, sql_path: str | Path) -> None:
        """Execute a trusted seed SQL file against the store's connection.

        Meant for one-shot bootstrap of the `sources` table from
        `data/seed/initial_provenance.sql` and equivalent files. Do NOT use
        this for user-supplied SQL — no parameterization.
        """
        path = Path(sql_path)
        try:
            script = path.read_text(encoding="utf-8")
        except OSError as e:
            raise KnowledgeError(f"cannot read seed script {sql_path}: {e}") from e
        try:
            with self._conn:
                self._conn.executescript(script)
        except sqlite3.Error as e:
            raise KnowledgeError(f"seed script {sql_path} failed: {e}") from e

    # ---- reads ---------------------------------------------------------------

    def _get_by_id(self, provenance_id: int) -> ProvenanceEntry:
        row = self._conn.execute(
            "SELECT * FROM provenance WHERE provenance_id = ?", (provenance_id,)
        ).fetchone()
        if row is None:
            raise KnowledgeError(f"no provenance row with id={provenance_id}")
        return _row_to_entry(row)

    def get_active(
        self, subject: str, predicate: str, perspective: str = "self"
    ) -> ProvenanceEntry | None:
        """Return the most-recently-recorded ACTIVE row for this key, or None."""
        row = self._conn.execute(
            """
            SELECT * FROM provenance
             WHERE subject = ? AND predicate = ? AND perspective = ?
               AND status = 'active'
             ORDER BY recorded_at DESC, provenance_id DESC
             LIMIT 1
            """,
            (subject, predicate, perspective),
        ).fetchone()
        return _row_to_entry(row) if row else None

    def query_by_subject(
        self,
        subject: str,
        predicate: str | None = None,
        limit: int = 20,
    ) -> list[ProvenanceEntry]:
        """Full history for a subject, newest first."""
        if predicate is None:
            sql = """
                SELECT * FROM provenance
                 WHERE subject = ?
                 ORDER BY recorded_at DESC, provenance_id DESC
                 LIMIT ?
            """
            params: tuple[Any, ...] = (subject, int(limit))
        else:
            sql = """
                SELECT * FROM provenance
                 WHERE subject = ? AND predicate = ?
                 ORDER BY recorded_at DESC, provenance_id DESC
                 LIMIT ?
            """
            params = (subject, predicate, int(limit))
        return [_row_to_entry(r) for r in self._conn.execute(sql, params).fetchall()]

    def iter_current_beliefs(self) -> Iterator[ProvenanceEntry]:
        """Yield the latest ACTIVE row per (subject, predicate, perspective).

        Reads through the `current_beliefs` view defined in schema.sql.
        """
        for row in self._conn.execute(
            """
            SELECT subject, predicate, object, confidence, source_id, perspective,
                   observed_at
              FROM current_beliefs
            """
        ):
            yield ProvenanceEntry(
                provenance_id=None,          # view is a projection, no row id
                subject=row["subject"],
                predicate=row["predicate"],
                object=row["object"],
                confidence=row["confidence"],
                source_id=row["source_id"],
                perspective=row["perspective"],
                observed_at=_parse_iso(row["observed_at"]),
                status="active",
                payload=None,
            )

    def list_sources(self) -> list[Source]:
        return [
            Source(
                source_id=row["source_id"],
                kind=row["kind"],
                trust_prior=row["trust_prior"],
                description=row["description"],
            )
            for row in self._conn.execute(
                "SELECT source_id, kind, trust_prior, description FROM sources"
            ).fetchall()
        ]


# ------------------------- module-private helpers ---------------------------


def _iso(dt: datetime) -> str:
    """Uniform ISO-8601 UTC-ish string for SQLite storage."""
    return dt.isoformat()


def _parse_iso(s: str) -> datetime:
    """Parse an ISO string coming out of SQLite. Tolerates trailing Z."""
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    return datetime.fromisoformat(s)


def _row_to_entry(row: sqlite3.Row) -> ProvenanceEntry:
    payload = json.loads(row["payload"]) if row["payload"] else None
    return ProvenanceEntry(
        provenance_id=row["provenance_id"],
        subject=row["subject"],
        predicate=row["predicate"],
        object=row["object"],
        confidence=row["confidence"],
        source_id=row["source_id"],
        perspective=row["perspective"],
        observed_at=_parse_iso(row["observed_at"]),
        recorded_at=_parse_iso(row["recorded_at"]),
        status=row["status"],
        payload=payload,
    )
