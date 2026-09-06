-- Canonical SQLite DDL for the provenance log.
-- Source of truth. `src/iagent/declarative/provenance.py` MUST load this file
-- unchanged rather than duplicating the DDL in Python.

-- Sources = origin registry (map file, sensor, external agent, user statement, LLM inference)
CREATE TABLE IF NOT EXISTS sources (
    source_id     TEXT PRIMARY KEY,           -- e.g. 'default_map', 'lidar_front', 'bot_02'
    kind          TEXT NOT NULL,              -- 'static_map'|'sensor'|'agent'|'user'|'llm'
    trust_prior   REAL NOT NULL DEFAULT 0.5,  -- baseline trust weight 0..1
    description   TEXT
);

-- Every belief write leaves a provenance row.
-- History is APPEND-ONLY: to "change" a belief, insert a new row with a newer recorded_at
-- and either flip the old row's `status` OR insert a downgraded successor.
CREATE TABLE IF NOT EXISTS provenance (
    provenance_id INTEGER PRIMARY KEY AUTOINCREMENT,
    subject       TEXT NOT NULL,              -- graph node id
    predicate     TEXT NOT NULL,              -- relation, e.g. 'located_at'
    object        TEXT NOT NULL,              -- graph node id (or JSON literal for atoms)
    confidence    REAL NOT NULL,              -- 0..1
    source_id     TEXT NOT NULL REFERENCES sources(source_id),
    perspective   TEXT NOT NULL,              -- 'self'|'user'|'third_party:<agent_id>'
    observed_at   TEXT NOT NULL,              -- ISO-8601 UTC
    recorded_at   TEXT NOT NULL DEFAULT (datetime('now')),
    status        TEXT NOT NULL DEFAULT 'active',  -- active|downgraded|retracted
    payload       TEXT                        -- optional raw JSON blob (e.g. sensor reading)
);

CREATE INDEX IF NOT EXISTS idx_prov_subject  ON provenance(subject);
CREATE INDEX IF NOT EXISTS idx_prov_source   ON provenance(source_id);
CREATE INDEX IF NOT EXISTS idx_prov_perspect ON provenance(perspective);

-- Convenience view: current "best" belief per (subject, predicate, perspective)
CREATE VIEW IF NOT EXISTS current_beliefs AS
  SELECT subject, predicate, object, confidence, source_id, perspective, observed_at
  FROM provenance
  WHERE status = 'active'
  GROUP BY subject, predicate, perspective
  HAVING MAX(recorded_at);

-- History of belief updates for a subject (used by tests + verification report)
CREATE VIEW IF NOT EXISTS belief_history AS
  SELECT subject, predicate, object, confidence, source_id, perspective,
         observed_at, recorded_at, status
  FROM provenance
  ORDER BY subject, predicate, recorded_at;
