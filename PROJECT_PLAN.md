# I, Agent — Implementation Plan

> Three-Layer Epistemic Architecture for Grounded Agents
> CSF407 · 4-student team · repo: `jewel-surabhi/csf407-i-agents`

This plan turns the *"I, Agent"* proposal into a concrete, buildable software project. It is opinionated where the proposal is silent, and conservative where the proposal is explicit. Every recommendation is labelled either **[PROPOSAL]** (mandatory, from the spec) or **[REC]** (implementation choice, may be swapped).

---

## A. Executive Summary

**What we're building.** A Python agent that answers natural-language queries by reasoning across three strictly-isolated layers — a NetworkX belief graph + SQLite provenance log (**Declarative**), a tool-calling LLM ReAct loop (**Procedural**), and a mock sensor/actuator environment (**Sensorimotor**). The agent must *detect and resolve conflicts* between historical knowledge and live sensor data, and must *keep three perspectives distinct* (user belief, egocentric perception, historical/third-party record).

**Why the architecture matters.** The layers are API-bounded. The LLM never touches the belief graph directly; it only calls tools. The sensor layer never reads history. This isolation is what makes the epistemic tests meaningful — a hallucination cannot be smuggled in by shared state.

**Primary deliverables (from proposal).**
1. A cleanly-structured Python package with independent `declarative`, `procedural`, `sensorimotor` modules.
2. 10 automated epistemic test logs (pytest).
3. Architecture Verification Report (desirable).

**Recommended tech stack.**
- LLM: **Llama-3.1-8B-Instruct via the Groq API** (free tier, OpenAI-compatible, native tool-calling). Fallback: Ollama + `llama3.1:8b-instruct` for local. [REC]
- Orchestration: **hand-written ReAct loop** using the provider's native tool-calling. No LangGraph. Rationale: transparency, easier to log every step for the 10 required test logs, no framework upgrade churn. [REC]
- Belief graph: **NetworkX** MultiDiGraph. [PROPOSAL]
- Provenance: **SQLite** via stdlib `sqlite3`. [PROPOSAL]
- Mock env: hand-rolled `mock_env.py` class. [PROPOSAL]
- Tests: **pytest** + JSON log dumps. [REC]
- UI: **CLI only** for grading. Optional FastAPI + minimal HTML for the demo — **not Streamlit** per team preference. [REC]

**Team of 4.** Divided by *layer ownership* (Declarative, Procedural, Sensorimotor) + one **Integration/Test Lead**. Every layer has a single owner; integration lead owns cross-layer glue, the scenario runners, and the 10-test suite.

**Timeline.** 6 weeks in 9 phases, each with a clear owner and acceptance criteria. See section N.

---

## B. Current Repository Audit

**Live state of `github.com/jewel-surabhi/csf407-i-agents` (branch: `main`):**

| Item | Status |
|---|---|
| Files at root | `README.md` only |
| README contents | Literally `# csf407-i-agents` — 1 line |
| Source code | None |
| Tests | None |
| CI | None |
| `.gitignore` | Missing |
| License | Missing |
| Branches | `main` only |
| Open issues / PRs | 0 / 0 |

**Verdict.** The repository is a blank slate. Nothing to keep, nothing to modify, nothing to remove. There are no architectural problems because there is no architecture yet. This is actually the best possible starting point — we can lay down the layered structure cleanly on day 1.

**Important files missing** (all of these are created in Phase 1):
`requirements.txt`, `.gitignore`, `.env.example`, `pyproject.toml`, package layout under `src/`, `tests/` scaffolding, `configs/`, `scripts/`, `docs/`, GitHub Actions workflow, `CODEOWNERS`, PR template.

---

## C. Target Architecture

### C.1 The layered contract

```
                    ┌──────────────────────────────┐
                    │        USER (CLI / API)      │
                    └──────────────┬───────────────┘
                                   │ natural-language query
                                   ▼
   ┌───────────────────────────────────────────────────────────┐
   │           PROCEDURAL LAYER  (src/procedural)              │
   │  • ReAct loop        • Tool dispatch                      │
   │  • LLM client        • Trace logger                       │
   │                                                           │
   │  Can call ONLY: declarative_tools.*, sensorimotor_tools.* │
   │  Cannot touch:  NetworkX graph, SQLite, mock_env directly │
   └───────┬────────────────────────────────────────┬──────────┘
           │ tool call: read_belief / write_belief  │ tool call: sense / act
           │           / query_provenance           │
           ▼                                        ▼
 ┌─────────────────────────┐             ┌─────────────────────────┐
 │  DECLARATIVE LAYER      │             │  SENSORIMOTOR LAYER     │
 │  (src/declarative)      │             │  (src/sensorimotor)     │
 │                         │             │                         │
 │  • BeliefGraph          │             │  • MockEnvironment      │
 │    (NetworkX)           │             │  • Sensor primitives    │
 │  • ProvenanceStore      │             │  • Action primitives    │
 │    (SQLite)             │             │  • Scenario loaders     │
 │  • KnowledgeAPI         │             │  • SensorPayload dcls   │
 │    (public interface)   │             │                         │
 └─────────────────────────┘             └─────────────────────────┘
```

### C.2 Isolation rules (must-hold invariants)

1. `src/procedural/**` **imports only** `src/declarative/api.py` and `src/sensorimotor/api.py`. It does **not** import `networkx`, `sqlite3`, or anything under `mock_env`.
2. `src/declarative/**` and `src/sensorimotor/**` **do not import each other**. Cross-layer state travels only via the procedural layer's ReAct trace.
3. Both back-end layers expose a single `api.py` façade. Everything else in the layer is `_`-prefixed / internal.
4. All layer-to-layer data crosses as JSON-serializable dicts validated by Pydantic models (see F).
5. The LLM never sees Python objects. Only JSON strings.

Enforcement: a `tests/architecture/test_layer_isolation.py` grep-based check fails CI if any procedural file imports networkx/sqlite3, or if declarative/sensorimotor import each other.

### C.3 Two canonical data flows

**Flow 1 — belief lookup + sensor cross-check (Scenario A):**
```
query → ReAct(step 1: read_belief) → declarative.get_belief
      → ReAct(step 2: sense_lidar)  → sensorimotor.read_lidar
      → LLM detects conflict
      → ReAct(step 3: write_belief) → declarative.update_belief(confidence↓)
      → ReAct(step 4: log_provenance) → declarative.log_source
      → final answer
```

**Flow 2 — perspective triangulation (Scenario B):**
```
query → ReAct(step 1: get_user_belief)     → declarative.query_perspective(user)
      → ReAct(step 2: sense_camera)         → sensorimotor.read_camera
      → ReAct(step 3: query_provenance)     → declarative.query_provenance(item_id)
      → LLM composes 3-perspective answer (no merging)
```

---

## D. Final Repository Tree

```
csf407-i-agents/
├── README.md                          # section O of this plan
├── LICENSE                            # MIT [REC]
├── .gitignore                         # Python + venv + .env + *.db + __pycache__
├── .env.example                       # GROQ_API_KEY=..., LLM_MODEL=..., LOG_LEVEL=INFO
├── pyproject.toml                     # PEP 621 metadata, ruff/black/pytest config
├── requirements.txt                   # pinned runtime deps (see F.5)
├── requirements-dev.txt               # pytest, pytest-cov, ruff, black, mypy
├── CODEOWNERS                         # per-file ownership (section L)
│
├── .github/
│   ├── workflows/
│   │   └── ci.yml                     # ruff + pytest on push/PR
│   ├── pull_request_template.md       # checklist enforcing tests + logs
│   └── ISSUE_TEMPLATE/
│       ├── bug.md
│       └── feature.md
│
├── configs/
│   ├── default.yaml                   # top-level agent config (model, temps, paths)
│   ├── scenario_a.yaml                # initial world state for Scenario A
│   └── scenario_b.yaml                # initial world state for Scenario B
│
├── data/
│   ├── seed/
│   │   ├── initial_beliefs.json       # bootstrap graph nodes/edges
│   │   └── initial_provenance.sql     # bootstrap SQLite rows
│   └── runs/                          # gitignored; test/demo output logs land here
│       └── .gitkeep
│
├── docs/
│   ├── architecture.md                # long-form version of section C
│   ├── data_contracts.md              # JSON schemas + SQL DDL (section F)
│   ├── react_loop.md                  # section H spelled out
│   ├── scenarios.md                   # sections I + J
│   ├── verification_report.md         # the "desirable" deliverable
│   └── diagrams/                      # any exported PNG/SVGs
│
├── src/
│   └── iagent/                        # single top-level package
│       ├── __init__.py                # exports Agent, run_scenario
│       │
│       ├── declarative/               # LAYER 1 — Owner: Person 1
│       │   ├── __init__.py
│       │   ├── api.py                 # KnowledgeAPI façade (only public entry)
│       │   ├── belief_graph.py        # NetworkX wrapper
│       │   ├── provenance.py          # SQLite wrapper
│       │   ├── models.py              # Pydantic: Belief, Provenance, Perspective
│       │   ├── schema.sql             # CREATE TABLE statements
│       │   └── seeding.py             # load initial_beliefs.json + initial_provenance.sql
│       │
│       ├── sensorimotor/              # LAYER 3 — Owner: Person 2
│       │   ├── __init__.py
│       │   ├── api.py                 # SensorimotorAPI façade
│       │   ├── mock_env.py            # MockEnvironment class (world state)
│       │   ├── sensors.py             # read_lidar, read_camera, read_clock
│       │   ├── actuators.py           # move_forward, rotate, grasp
│       │   ├── models.py              # Pydantic: SensorPayload, ActionResult
│       │   └── scenarios.py           # world-state loaders from YAML
│       │
│       ├── procedural/                # LAYER 2 — Owner: Person 3
│       │   ├── __init__.py
│       │   ├── agent.py               # Agent.run() — top-level entry
│       │   ├── react_loop.py          # the loop: thought → action → observation
│       │   ├── llm_client.py          # Groq/OpenAI-compat wrapper + retries
│       │   ├── tools.py               # tool schemas + dispatch to layer APIs
│       │   ├── prompts.py             # system prompts, few-shot templates
│       │   ├── trace.py               # structured step logger → data/runs/*.json
│       │   └── models.py              # Pydantic: Thought, ToolCall, Observation, Trace
│       │
│       ├── cli.py                     # `python -m iagent` entrypoint — Owner: Person 4
│       └── config.py                  # YAML/env loader shared across layers
│
├── scripts/
│   ├── init_db.py                     # create SQLite file from schema.sql
│   ├── run_scenario_a.py              # one-shot demo of Scenario A
│   ├── run_scenario_b.py              # one-shot demo of Scenario B
│   ├── run_all_tests.py               # runs the 10-test suite + emits report
│   └── verify_isolation.py            # static check of layer imports
│
└── tests/                             # Owner: Person 4 (integration/test lead)
    ├── __init__.py
    ├── conftest.py                    # shared fixtures (tmp DB, seeded graph, mock env)
    │
    ├── unit/
    │   ├── test_belief_graph.py       # Person 1 writes
    │   ├── test_provenance.py         # Person 1 writes
    │   ├── test_mock_env.py           # Person 2 writes
    │   ├── test_sensors.py            # Person 2 writes
    │   ├── test_react_loop.py         # Person 3 writes (LLM mocked)
    │   └── test_tools.py              # Person 3 writes
    │
    ├── architecture/
    │   └── test_layer_isolation.py    # Person 4 writes — grep-based invariant check
    │
    └── epistemic/                     # THE 10 REQUIRED TEST LOGS — Person 4 owns
        ├── __init__.py
        ├── conftest.py
        ├── test_01_grounded_query.py
        ├── test_02_stale_map_conflict.py       # Scenario A
        ├── test_03_sensor_confidence_update.py
        ├── test_04_provenance_written.py
        ├── test_05_user_perspective.py         # Scenario B
        ├── test_06_egocentric_perspective.py
        ├── test_07_historical_perspective.py
        ├── test_08_conflicting_third_party.py
        ├── test_09_unknown_entity.py
        └── test_10_multi_step_reconciliation.py
```

**File counts:** ~55 source/test files at Day 1 completion of Phase 1. All are listed in section E with responsibility notes.

---

## E. File-by-File Responsibilities

Legend: **P1** = Declarative owner · **P2** = Sensorimotor owner · **P3** = Procedural owner · **P4** = Integration/Test lead.

### Root

| File | Owner | Purpose |
|---|---|---|
| `README.md` | P4 | User-facing intro. Content = section O of this plan. |
| `LICENSE` | P4 | MIT boilerplate. |
| `.gitignore` | P4 | Python + `.env` + `*.db` + `data/runs/` + IDE junk. |
| `.env.example` | P4 | Documents required env vars: `GROQ_API_KEY`, `LLM_MODEL`, `LLM_TEMPERATURE`, `IAGENT_DB_PATH`, `LOG_LEVEL`. |
| `pyproject.toml` | P4 | PEP 621 project metadata; ruff/black/pytest config. |
| `requirements.txt` | P4 | Pinned deps: `networkx`, `pydantic`, `pyyaml`, `python-dotenv`, `groq` (or `openai`), `rich` (CLI pretty-print). |
| `requirements-dev.txt` | P4 | `pytest`, `pytest-cov`, `ruff`, `black`, `mypy`. |
| `CODEOWNERS` | P4 | Enforces per-directory ownership (see L.4). |

### `.github/`

| File | Owner | Purpose |
|---|---|---|
| `workflows/ci.yml` | P4 | On push/PR: `ruff check`, `pytest -q`, isolation test. |
| `pull_request_template.md` | P4 | Checklist: unit tests added, layer isolation preserved, trace log attached if scenario changed. |
| `ISSUE_TEMPLATE/*.md` | P4 | Bug + feature templates. |

### `configs/`

| File | Owner | Purpose |
|---|---|---|
| `default.yaml` | P3 | Model name, temperature, max ReAct steps, DB path, log path. |
| `scenario_a.yaml` | P1+P2 | Initial belief graph + initial mock-env state for Scenario A. |
| `scenario_b.yaml` | P1+P2 | Same for Scenario B (user query, camera lighting, third-party bot log). |

### `data/`

| Path | Owner | Purpose |
|---|---|---|
| `seed/initial_beliefs.json` | P1 | Bootstrap nodes + edges for the graph. |
| `seed/initial_provenance.sql` | P1 | Seed rows for provenance table. |
| `runs/` | — | Gitignored. Each test/demo writes a JSON trace here. |

### `docs/`

| File | Owner | Purpose |
|---|---|---|
| `architecture.md` | P4 | Long-form section C — the layer diagram + isolation rules. |
| `data_contracts.md` | P1 | JSON schemas + SQL DDL — the source of truth for cross-layer types. |
| `react_loop.md` | P3 | The tool list, prompt template, decision rules. |
| `scenarios.md` | P4 | Scenario A + B walk-throughs (sections I, J). |
| `verification_report.md` | P4 | The "desirable" deliverable, filled in during Phase 9. |

### `src/iagent/declarative/` — **Owner: P1**

| File | Purpose | Key exports |
|---|---|---|
| `api.py` | Sole public façade. All procedural calls arrive here. | `KnowledgeAPI` class with `get_belief`, `update_belief`, `query_provenance`, `log_source`, `query_perspective`. |
| `belief_graph.py` | NetworkX MultiDiGraph wrapper. | `BeliefGraph` class. |
| `provenance.py` | SQLite CRUD. | `ProvenanceStore` class. |
| `models.py` | Pydantic types crossing the layer boundary. | `Belief`, `ProvenanceEntry`, `PerspectiveView`. |
| `schema.sql` | `CREATE TABLE` statements — canonical DDL. | — |
| `seeding.py` | Load `data/seed/*` into graph + DB. | `seed_from_config(config)`. |

### `src/iagent/sensorimotor/` — **Owner: P2**

| File | Purpose | Key exports |
|---|---|---|
| `api.py` | Sole façade. | `SensorimotorAPI` with `sense_lidar`, `sense_camera`, `sense_clock`, `act_move`, `act_rotate`, `act_grasp`. |
| `mock_env.py` | Holds mutable world state; sensor readings derive from it. | `MockEnvironment` class. |
| `sensors.py` | Pure functions turning world state → `SensorPayload`. | `read_lidar`, `read_camera`, `read_clock`. |
| `actuators.py` | Mutate world state; return `ActionResult`. | `move_forward`, `rotate`, `grasp`. |
| `models.py` | Pydantic. | `SensorPayload`, `ActionResult`, `WorldState`. |
| `scenarios.py` | Load YAML → `WorldState`. | `load_scenario(path)`. |

### `src/iagent/procedural/` — **Owner: P3**

| File | Purpose | Key exports |
|---|---|---|
| `agent.py` | Top-level `Agent`. Composes API objects + LLM client + loop. | `Agent`, `Agent.run(query) -> AgentResult`. |
| `react_loop.py` | The reason-act-observe loop. | `run_react(agent, query, max_steps) -> Trace`. |
| `llm_client.py` | Groq client + retry/backoff. | `LLMClient.chat_with_tools(messages, tools)`. |
| `tools.py` | JSON-Schema tool definitions + dispatch to API façades. | `TOOL_SPECS`, `dispatch(name, args, apis)`. |
| `prompts.py` | System prompt + per-scenario few-shot. | `SYSTEM_PROMPT`, `format_query(...)`. |
| `trace.py` | Every step written to `data/runs/<timestamp>.json`. | `Trace.append(step)`, `Trace.dump()`. |
| `models.py` | Pydantic. | `Thought`, `ToolCall`, `Observation`, `Step`, `Trace`. |

### `src/iagent/`

| File | Owner | Purpose |
|---|---|---|
| `cli.py` | P4 | `python -m iagent --scenario a` runs a scenario end-to-end and prints/logs the trace. |
| `config.py` | P4 | Load YAML + `.env`; validate; return `AppConfig` Pydantic model. |

### `scripts/`

| File | Owner | Purpose |
|---|---|---|
| `init_db.py` | P1 | `python scripts/init_db.py` — creates SQLite file from `schema.sql`. |
| `run_scenario_a.py` | P4 | Demo runner (calls into `cli.py`). |
| `run_scenario_b.py` | P4 | Same for B. |
| `run_all_tests.py` | P4 | Invokes pytest + aggregates JSON traces into one report. |
| `verify_isolation.py` | P4 | Static check of imports for layer isolation. Called by CI. |

### `tests/` — See section K for the 10 epistemic tests.

---

## F. Data Models & Schemas

### F.1 Belief Graph (NetworkX)

**Graph type:** `nx.MultiDiGraph` — multiple edges allowed between same pair (e.g. one `historical` and one `observed` version of the same relation).

**Node schema.** A node is any *cognitive entity*: a physical object, a location, a person, an abstract fact. Node attributes:

```python
{
    "id": "box_01",                    # str, primary key
    "type": "object" | "location" | "agent" | "fact",
    "labels": ["box", "container"],    # list[str]
    "created_at": "2026-09-06T10:00:00Z",
}
```

**Edge schema.** An edge is a *belief that a relation holds*. Attributes:

```python
{
    "relation": "located_at",          # str, controlled vocabulary
    "confidence": 0.85,                # float 0..1
    "source": "default_map",           # str — must match a Provenance.source row
    "perspective": "self",             # "self" | "user" | "third_party:<agent_id>"
    "status": "active",                # "active" | "downgraded" | "retracted"
    "observed_at": "2026-09-06T10:00:00Z",
    "provenance_id": 42,               # FK into provenance table
}
```

**Example — Scenario A initial state:**
```python
G.add_node("robot", type="agent", labels=["self"])
G.add_node("room_101", type="location", labels=["room"])
G.add_node("path_ahead", type="location", labels=["corridor"])

G.add_edge("robot", "room_101",
           relation="located_at", confidence=1.0,
           source="default_map", perspective="self",
           status="active", observed_at="2026-09-06T09:00:00Z",
           provenance_id=1)

G.add_edge("path_ahead", "path_ahead",   # self-loop expressing a property
           relation="is_clear", confidence=1.0,
           source="default_map", perspective="self",
           status="active", observed_at="2026-09-06T09:00:00Z",
           provenance_id=2)
```

After Scenario A resolves, the LLM (via a tool call) writes:
```python
# downgrade the stale belief
G["path_ahead"]["path_ahead"][0]["confidence"] = 0.2
G["path_ahead"]["path_ahead"][0]["status"] = "downgraded"

# add the fresh sensor-sourced belief
G.add_edge("path_ahead", "path_ahead",
           relation="is_blocked", confidence=0.95,
           source="lidar_front", perspective="self",
           status="active", observed_at="2026-09-06T10:15:00Z",
           provenance_id=17)
```

### F.2 Provenance Store (SQLite)

Canonical DDL — this is the file `src/iagent/declarative/schema.sql`:

```sql
-- Sources = origin registry (map file, sensor, external agent, user statement, LLM inference)
CREATE TABLE IF NOT EXISTS sources (
    source_id     TEXT PRIMARY KEY,        -- e.g. 'default_map', 'lidar_front', 'bot_02'
    kind          TEXT NOT NULL,           -- 'static_map'|'sensor'|'agent'|'user'|'llm'
    trust_prior   REAL NOT NULL DEFAULT 0.5,  -- baseline trust weight 0..1
    description   TEXT
);

-- Every belief write leaves a provenance row.
CREATE TABLE IF NOT EXISTS provenance (
    provenance_id INTEGER PRIMARY KEY AUTOINCREMENT,
    subject       TEXT NOT NULL,           -- graph node id
    predicate     TEXT NOT NULL,           -- relation, e.g. 'located_at'
    object        TEXT NOT NULL,           -- graph node id (or JSON literal for atoms)
    confidence    REAL NOT NULL,           -- 0..1
    source_id     TEXT NOT NULL REFERENCES sources(source_id),
    perspective   TEXT NOT NULL,           -- 'self'|'user'|'third_party:<agent_id>'
    observed_at   TEXT NOT NULL,           -- ISO-8601 UTC
    recorded_at   TEXT NOT NULL DEFAULT (datetime('now')),
    status        TEXT NOT NULL DEFAULT 'active',   -- active|downgraded|retracted
    payload       TEXT                     -- optional raw JSON blob (e.g. sensor reading)
);

CREATE INDEX IF NOT EXISTS idx_prov_subject   ON provenance(subject);
CREATE INDEX IF NOT EXISTS idx_prov_source    ON provenance(source_id);
CREATE INDEX IF NOT EXISTS idx_prov_perspect  ON provenance(perspective);

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
```

**Seed rows (`data/seed/initial_provenance.sql`):**
```sql
INSERT INTO sources VALUES ('default_map',  'static_map', 0.6, 'Baseline floor plan');
INSERT INTO sources VALUES ('lidar_front',  'sensor',     0.9, 'Front-facing LiDAR');
INSERT INTO sources VALUES ('camera_front', 'sensor',     0.7, 'Front-facing camera');
INSERT INTO sources VALUES ('bot_02',       'agent',      0.5, 'Maintenance bot #02');
INSERT INTO sources VALUES ('user',         'user',       0.4, 'Human operator statement');
INSERT INTO sources VALUES ('llm_infer',    'llm',        0.2, 'LLM inference (unverified)');
```

Trust priors give Scenario A its resolution rule: `lidar_front` (0.9) beats `default_map` (0.6), so the LLM prioritizes sensor.

### F.3 Sensor Payloads (JSON)

Every sensor returns a payload with the same envelope so the LLM's prompt only needs one interpretation rule:

```json
{
  "sensor": "lidar_front",
  "kind": "distance_cm",
  "value": 12,
  "status": "blocked",
  "observed_at": "2026-09-06T10:15:00.234Z",
  "confidence": 0.95,
  "raw": {"distance_cm": 12, "material_guess": "wall"}
}
```

Camera payload:

```json
{
  "sensor": "camera_front",
  "kind": "vision",
  "detections": [
    {"item_id": "box_01", "class": "box", "color_perceived": "brown",
     "bbox": [120, 88, 240, 210], "confidence": 0.82}
  ],
  "ambient": {"lighting": "yellow", "warning": "color perception unreliable"},
  "observed_at": "2026-09-06T10:15:00.234Z"
}
```

Action result:

```json
{
  "action": "move_forward",
  "params": {"distance_cm": 30},
  "success": false,
  "reason": "obstacle detected at 12cm",
  "world_delta": {},
  "observed_at": "2026-09-06T10:15:01.001Z"
}
```

### F.4 Cross-layer Pydantic models

All in `src/iagent/*/models.py`. Pydantic v2. Every JSON crossing a layer boundary is validated. Concrete definitions:

```python
# declarative/models.py
class Belief(BaseModel):
    subject: str
    predicate: str
    object: str
    confidence: float = Field(ge=0.0, le=1.0)
    source: str
    perspective: Literal["self", "user"] | str  # "third_party:<id>"
    status: Literal["active", "downgraded", "retracted"] = "active"
    observed_at: datetime
    provenance_id: int | None = None

class ProvenanceEntry(BaseModel):
    provenance_id: int | None = None
    subject: str; predicate: str; object: str
    confidence: float; source_id: str
    perspective: str; observed_at: datetime
    status: str = "active"
    payload: dict[str, Any] | None = None

class PerspectiveView(BaseModel):
    perspective: str                # "self"|"user"|"third_party:<id>"
    beliefs: list[Belief]
```

```python
# sensorimotor/models.py
class SensorPayload(BaseModel):
    sensor: str; kind: str
    value: Any; status: str
    observed_at: datetime
    confidence: float | None = None
    raw: dict[str, Any] | None = None
    detections: list[dict] | None = None
    ambient: dict[str, Any] | None = None

class ActionResult(BaseModel):
    action: str; params: dict
    success: bool
    reason: str | None = None
    world_delta: dict = {}
    observed_at: datetime
```

```python
# procedural/models.py
class ToolCall(BaseModel):
    name: str; arguments: dict
class Observation(BaseModel):
    tool: str; ok: bool; content: dict | str
class Step(BaseModel):
    idx: int
    thought: str | None = None
    tool_call: ToolCall | None = None
    observation: Observation | None = None
    final_answer: str | None = None
class Trace(BaseModel):
    query: str
    steps: list[Step]
    final_answer: str
    started_at: datetime
    ended_at: datetime
```

### F.5 Runtime dependencies (`requirements.txt`)

```
networkx==3.3
pydantic==2.9.2
pyyaml==6.0.2
python-dotenv==1.0.1
groq==0.11.0            # or: openai==1.51.0 with base_url override
rich==13.9.2            # pretty CLI output
```

Dev deps (`requirements-dev.txt`):
```
pytest==8.3.3
pytest-cov==5.0.0
ruff==0.6.9
black==24.10.0
mypy==1.11.2
```

---

## G. Layer APIs / Interfaces

The three façades below are the *only* public entry points. Every method signature is frozen at end of Phase 1 — changes require a PR review by all four owners.

### G.1 `KnowledgeAPI` (declarative façade)

```python
class KnowledgeAPI:
    def __init__(self, graph: BeliefGraph, store: ProvenanceStore): ...

    # ----- reads -----
    def get_belief(self, subject: str, predicate: str,
                   perspective: str = "self") -> Belief | None:
        """Return current best belief for (subject, predicate, perspective) or None."""

    def query_perspective(self, subject: str,
                          predicate: str | None = None) -> list[PerspectiveView]:
        """Return all perspectives held about `subject`, grouped."""

    def query_provenance(self, subject: str,
                         predicate: str | None = None,
                         limit: int = 20) -> list[ProvenanceEntry]:
        """Full history, newest first."""

    def list_sources(self) -> list[dict]:
        """Registered sources with trust priors."""

    # ----- writes -----
    def update_belief(self, belief: Belief, reason: str) -> ProvenanceEntry:
        """Insert a new provenance row AND update graph edge. Atomic."""

    def downgrade_belief(self, subject: str, predicate: str,
                         new_confidence: float, reason: str,
                         perspective: str = "self") -> ProvenanceEntry: ...

    def log_source(self, source_id: str, kind: str,
                   trust_prior: float, description: str) -> None:
        """Register a new source (idempotent)."""
```

**Error contract.** All methods raise `KnowledgeError` (subclass of `ValueError`) with a JSON-serializable message on invalid input. Never raises database exceptions upward.

### G.2 `SensorimotorAPI` (sensorimotor façade)

```python
class SensorimotorAPI:
    def __init__(self, env: MockEnvironment): ...

    def sense_lidar(self, direction: Literal["front","left","right","back"] = "front"
                    ) -> SensorPayload: ...
    def sense_camera(self, direction: Literal["front","left","right","back"] = "front"
                     ) -> SensorPayload: ...
    def sense_clock(self) -> SensorPayload: ...

    def act_move(self, distance_cm: int,
                 direction: Literal["forward","backward"] = "forward"
                 ) -> ActionResult: ...
    def act_rotate(self, degrees: int) -> ActionResult: ...
    def act_grasp(self, target_id: str) -> ActionResult: ...

    def describe_world(self) -> dict:
        """Debug-only. Not exposed to the LLM."""
```

**Error contract.** All sensor methods succeed and return a payload with `status="error"` on hardware-like failures. Actions return `ActionResult(success=False, reason=...)`. Never raises.

### G.3 Procedural side — `Agent`

```python
class Agent:
    def __init__(self, knowledge: KnowledgeAPI, sensors: SensorimotorAPI,
                 llm: LLMClient, config: AppConfig): ...

    def run(self, query: str, *, scenario: str | None = None,
            max_steps: int = 8) -> AgentResult:
        """Execute one ReAct episode. Returns final answer + full trace."""

class AgentResult(BaseModel):
    answer: str
    trace: Trace
    beliefs_written: list[Belief]
    tools_called: list[str]
```

### G.4 The tool list exposed to the LLM

Registered in `procedural/tools.py`, converted to the provider's JSON-Schema tool format, and dispatched to the appropriate façade:

| Tool name | Layer | Signature (JSON) | Purpose |
|---|---|---|---|
| `get_belief` | Declarative | `{subject: str, predicate: str, perspective?: str}` → `Belief \| null` | Read the agent's current belief. |
| `query_perspective` | Declarative | `{subject: str, predicate?: str}` → `PerspectiveView[]` | Fetch all perspectives on a subject. |
| `query_provenance` | Declarative | `{subject: str, predicate?: str, limit?: int}` → `ProvenanceEntry[]` | Inspect history. |
| `update_belief` | Declarative | `{subject, predicate, object, confidence, source, perspective, reason}` → `ProvenanceEntry` | Write a new belief + provenance row. |
| `downgrade_belief` | Declarative | `{subject, predicate, new_confidence, reason, perspective?}` → `ProvenanceEntry` | Lower confidence in a stale belief. |
| `sense_lidar` | Sensorimotor | `{direction?: str}` → `SensorPayload` | Live proximity read. |
| `sense_camera` | Sensorimotor | `{direction?: str}` → `SensorPayload` | Live vision read. |
| `sense_clock` | Sensorimotor | `{}` → `SensorPayload` | Current sim time. |
| `list_sources` | Declarative | `{}` → `{source_id, trust_prior, kind}[]` | Let the LLM reason about which source to trust. |
| `finalize_answer` | Procedural | `{answer: str, cited_beliefs?: string[], cited_sensors?: string[]}` → *terminates loop* | The one action that ends the episode. |

Actions like `act_move` are *not* exposed for the epistemic tests — they exist in `SensorimotorAPI` for future work but are behind a config flag (`enable_actuators: false` by default) to keep the epistemic tests deterministic.

---

## H. Procedural LLM + ReAct Workflow

### H.1 Framework choice

**Recommendation: hand-written ReAct loop over the Groq API's native tool-calling.** [REC]

Rationale:
1. The provider's tool-calling API already handles the LLM ↔ tool JSON round-trip. LangGraph adds a graph-of-nodes abstraction that this project doesn't need — we have exactly one loop.
2. Every step must be logged to a JSON trace (the deliverable is 10 test logs). A hand-written loop makes trace format an explicit choice, not something extracted from framework internals.
3. Zero framework upgrade risk over the 6-week timeline.
4. Students see every prompt token — pedagogically better and easier to debug.

Estimated LOC: ~200 lines for `react_loop.py` + `llm_client.py` combined.

### H.2 The loop, precisely

```
initialize:
    messages = [system_prompt, user_query]
    trace = Trace(query=user_query, steps=[])

for step_idx in 1..max_steps:
    response = llm.chat_with_tools(messages, tools=TOOL_SPECS)

    if response.tool_calls:
        for tc in response.tool_calls:
            observation = dispatch(tc.name, tc.arguments, apis)
            messages.append(assistant_tool_call(tc))
            messages.append(tool_result(tc.id, observation))
            trace.append(Step(idx=step_idx, tool_call=tc, observation=observation))

        if any(tc.name == "finalize_answer" for tc in response.tool_calls):
            final = extract_final(response.tool_calls)
            trace.final_answer = final.answer
            trace.ended_at = utcnow()
            return AgentResult(answer=final.answer, trace=trace, ...)

    else:  # LLM produced free-text without calling a tool — treat as thought
        trace.append(Step(idx=step_idx, thought=response.content))
        messages.append({"role": "user",
                         "content": "You must call a tool. Use finalize_answer to end."})

raise MaxStepsExceeded(trace)
```

### H.3 System prompt (source of truth: `procedural/prompts.py`)

```
You are "I, Agent" — a grounded reasoning agent. You have three
information sources:

1. Your declarative memory (belief graph + provenance log).
   Tools: get_belief, query_perspective, query_provenance, list_sources.
2. Your live sensors (LiDAR, camera, clock).
   Tools: sense_lidar, sense_camera, sense_clock.
3. Your writes: update_belief, downgrade_belief.

Rules you MUST follow:
- Never assert a fact you have not verified via a tool.
- When declarative memory and a sensor disagree, PRIORITIZE the source
  with the HIGHER trust_prior from list_sources. Log the conflict by
  calling downgrade_belief on the losing side and update_belief with
  the winning source.
- When asked about perspective, keep user / self (egocentric) /
  third-party views strictly separate. Never merge them.
- End every episode by calling finalize_answer exactly once. Cite the
  belief IDs and sensor readings you relied on.

Do not chain more than 8 tool calls. If unsure, ask for the specific
belief or sensor rather than guessing.
```

### H.4 Full happy-path walkthrough

Query: *"Is your route clear? Justify by inspecting your internal layers."*

| # | LLM action | Tool | Result |
|---|---|---|---|
| 1 | reason: check memory first | `get_belief(subject="path_ahead", predicate="is_clear")` | `Belief(confidence=1.0, source="default_map")` |
| 2 | verify with live sensor | `sense_lidar(direction="front")` | `SensorPayload(value=12, status="blocked", confidence=0.95)` |
| 3 | conflict — check trust priors | `list_sources` | `default_map=0.6, lidar_front=0.9` |
| 4 | downgrade stale belief | `downgrade_belief(subject="path_ahead", predicate="is_clear", new_confidence=0.2, reason="live LiDAR contradicts")` | `ProvenanceEntry(id=17)` |
| 5 | write fresh belief | `update_belief(subject="path_ahead", predicate="is_blocked", object="obstacle@12cm", confidence=0.95, source="lidar_front", perspective="self", reason="lidar reading")` | `ProvenanceEntry(id=18)` |
| 6 | end | `finalize_answer(answer="No — my static map claims the path is clear, but the front LiDAR reads 12 cm and status=blocked. I trust the sensor (prior 0.9) over the map (0.6), so I have downgraded the map belief to 0.2 and recorded a new blocked belief (provenance #18).", cited_beliefs=["path_ahead:is_clear","path_ahead:is_blocked"], cited_sensors=["lidar_front"])` | — |

### H.5 Guardrails

- **Structured errors:** every tool wraps its output in `{ok: bool, data: ...}`. On dispatch failure the LLM sees `{ok: false, error: "..."}` and can recover.
- **Loop cap:** hard cap at 8 steps (config). Exceeding → `MaxStepsExceeded`, still returns partial trace.
- **Token budget:** all messages truncated to N=8 most-recent (except system prompt) if context grows > 6k tokens.
- **Determinism for tests:** LLM `temperature=0.0`, and each epistemic test can mock the LLM entirely to a pre-recorded call sequence (see K).

---

## I. Scenario A Workflow — Groundedness

**Setup.**
- Belief graph seeded with `path_ahead is_clear (conf 1.0, source default_map)`.
- Mock env world state: obstacle at `distance_cm=12` in front of robot.
- User query: *"Is your route clear? Justify your response by inspecting your internal system layers."*

**Complete pseudocode.**

```python
def run_scenario_a(agent: Agent) -> AgentResult:
    query = "Is your route clear? Justify your response by inspecting your internal system layers."
    return agent.run(query, scenario="A", max_steps=8)

# Inside the loop the LLM will emit (temperature=0.0 → deterministic on Groq/Llama-3.1):

# STEP 1 — check declarative memory
tool: get_belief(subject="path_ahead", predicate="is_clear")
returns: {"subject":"path_ahead","predicate":"is_clear","object":"true",
          "confidence":1.0,"source":"default_map","perspective":"self",
          "status":"active","observed_at":"..."}

# STEP 2 — verify with sensor
tool: sense_lidar(direction="front")
returns: {"sensor":"lidar_front","kind":"distance_cm","value":12,
          "status":"blocked","observed_at":"...","confidence":0.95}

# STEP 3 — detect conflict, look up trust priors
tool: list_sources()
returns: [{"source_id":"default_map","trust_prior":0.6}, ...,
          {"source_id":"lidar_front","trust_prior":0.9}, ...]

# --- Conflict detection is done by the LLM. Reasoning in prompt:
#     get_belief says CLEAR, sense_lidar says BLOCKED → conflict.
#     trust(lidar_front)=0.9 > trust(default_map)=0.6 → sensor wins.

# STEP 4 — downgrade the losing belief
tool: downgrade_belief(subject="path_ahead", predicate="is_clear",
                       new_confidence=0.2,
                       reason="live LiDAR contradicts default_map at 12cm")
returns: {"provenance_id":17,"status":"downgraded",...}

# NetworkX side-effect inside KnowledgeAPI.downgrade_belief():
#   G["path_ahead"]["path_ahead"][edge_key]["confidence"] = 0.2
#   G["path_ahead"]["path_ahead"][edge_key]["status"]     = "downgraded"

# SQLite side-effect: a NEW row is inserted (we never mutate history), status='downgraded'.

# STEP 5 — record the fresh belief
tool: update_belief(subject="path_ahead", predicate="is_blocked",
                    object="obstacle@12cm", confidence=0.95,
                    source="lidar_front", perspective="self",
                    reason="live LiDAR reading")
returns: {"provenance_id":18,"status":"active",...}

# STEP 6 — end the episode
tool: finalize_answer(answer="No — my static map (default_map, confidence 1.0)
      claimed the path was clear, but a live front-LiDAR read of 12 cm with
      status=blocked contradicts it. Trust prior of lidar_front (0.9) exceeds
      default_map (0.6), so I have downgraded the stale belief to 0.2
      (provenance #17) and recorded a new is_blocked belief at 0.95
      (provenance #18). Route is NOT clear.",
      cited_beliefs=["path_ahead:is_clear","path_ahead:is_blocked"],
      cited_sensors=["lidar_front"])
```

**Post-conditions asserted by tests:**
1. `KnowledgeAPI.get_belief("path_ahead","is_clear").status == "downgraded"`
2. `KnowledgeAPI.get_belief("path_ahead","is_blocked").confidence >= 0.9`
3. Two new provenance rows exist with `source_id in ("lidar_front",)` and `source_id in ("llm_infer","default_map")` respectively.
4. Answer text contains the words `"downgraded"` and `"lidar"` (case-insensitive).
5. Tools called include `get_belief`, `sense_lidar`, `list_sources`, `downgrade_belief`, `update_belief`, `finalize_answer` — in that order or with `list_sources` optional.

---

## J. Scenario B Workflow — Perspective

**Setup.**
- User states: *"Find the red box."*
- Belief graph seeded with:
  - `box_01 has_color red · perspective="user" · source="user"`
  - `box_01 has_color blue · perspective="third_party:bot_02" · source="bot_02"` (from a maintenance log stating `action="painted_blue"`)
- Mock env: `box_01` is currently in view of `camera_front`; ambient lighting is `yellow`, so the camera reports `color_perceived="brown"`.
- Query: *"What color does the user think the object is, what color do you register it as, and what does your data history say its true state is?"*

**Data-model representation — the key isolation trick.**

Three separate belief edges exist on the same `(subject, predicate)` pair, distinguished by the `perspective` field:

```python
# user
G.add_edge("box_01","red_atom", relation="has_color",
           confidence=0.9, source="user",   perspective="user",
           status="active", observed_at="...", provenance_id=5)

# third-party historical
G.add_edge("box_01","blue_atom", relation="has_color",
           confidence=0.7, source="bot_02", perspective="third_party:bot_02",
           status="active", observed_at="...", provenance_id=6)

# egocentric will be written at query time from the camera
```

`KnowledgeAPI.query_perspective("box_01", predicate="has_color")` returns a `list[PerspectiveView]` — three separate views, never merged.

**Complete pseudocode.**

```python
def run_scenario_b(agent: Agent) -> AgentResult:
    query = ("What color does the user think the object is, "
             "what color do you register it as, "
             "and what does your data history say its true state is?")
    return agent.run(query, scenario="B", max_steps=8)

# STEP 1 — user perspective
tool: query_perspective(subject="box_01", predicate="has_color")
returns: [
  {"perspective":"user",
   "beliefs":[{"object":"red_atom","confidence":0.9,"source":"user"}]},
  {"perspective":"third_party:bot_02",
   "beliefs":[{"object":"blue_atom","confidence":0.7,"source":"bot_02"}]}
]

# STEP 2 — egocentric perception
tool: sense_camera(direction="front")
returns: {"sensor":"camera_front","kind":"vision",
          "detections":[{"item_id":"box_01","color_perceived":"brown",
                         "confidence":0.82}],
          "ambient":{"lighting":"yellow",
                     "warning":"color perception unreliable"},
          "observed_at":"..."}

# STEP 3 — record the fresh sensor observation as an egocentric belief
tool: update_belief(subject="box_01", predicate="has_color",
                    object="brown_atom", confidence=0.82,
                    source="camera_front", perspective="self",
                    reason="live camera reading; ambient=yellow")
returns: {"provenance_id":19,...}

# STEP 4 — full history (optional but expected)
tool: query_provenance(subject="box_01", predicate="has_color")
returns: [
  {"perspective":"self","object":"brown_atom","source_id":"camera_front",...},
  {"perspective":"third_party:bot_02","object":"blue_atom","source_id":"bot_02",...},
  {"perspective":"user","object":"red_atom","source_id":"user",...}
]

# STEP 5 — end
tool: finalize_answer(answer=
  "Three perspectives, kept separate:\n"
  "  • User perspective: red (from user statement, provenance #5).\n"
  "  • Egocentric (my current camera): brown, confidence 0.82 — but ambient "
  "    lighting is yellow so this reading is flagged unreliable (provenance #19).\n"
  "  • Historical / third-party (bot_02 maintenance log): blue "
  "    (provenance #6). This is the most recent verified true-state assertion.",
  cited_beliefs=["box_01:has_color@user","box_01:has_color@self",
                 "box_01:has_color@third_party:bot_02"],
  cited_sensors=["camera_front"])
```

**Guardrails that prevent mixing perspectives:**
1. `Belief.perspective` is a required Pydantic field. No default. LLM cannot forget to specify it because the tool schema marks it required.
2. `KnowledgeAPI.query_perspective` returns *grouped* views — the LLM never sees a flat list where perspectives are ambiguous.
3. System prompt has an explicit rule against merging.
4. Test #08 asserts the final answer string contains the exact tokens `user`, `egocentric`/`self`, and `historical`/`third-party` in distinct sentences.

---

## K. 10-Test Evaluation Suite

Every test lives under `tests/epistemic/`. Every test:
- uses fixtures from `tests/epistemic/conftest.py` that build a tmp SQLite DB + tmp NetworkX graph + a `MockEnvironment` from a scenario YAML.
- either calls the real LLM (Groq, temp=0) if `IAGENT_LIVE_LLM=1` is set, or a `RecordedLLMClient` that replays a canned sequence of tool calls (default, so CI stays deterministic).
- writes its trace to `data/runs/epistemic_<id>.json` — **this file is the required test log**.

### Test index

| # | ID | What it stresses |
|---|---|---|
| 01 | `grounded_query` | Baseline: agent answers a fact already in memory without hallucinating. |
| 02 | `stale_map_conflict` | **Scenario A.** Detect belief-vs-sensor conflict; prioritize sensor. |
| 03 | `sensor_confidence_update` | After A, the numerical confidence in the graph actually changes. |
| 04 | `provenance_written` | Every belief update leaves an inspectable SQLite row. |
| 05 | `user_perspective` | Agent isolates the user's belief on request. |
| 06 | `egocentric_perspective` | Agent reports its own live sensor perception. |
| 07 | `historical_perspective` | Agent surfaces a third-party log entry without merging it. |
| 08 | `perspective_no_merge` | **Scenario B full.** All 3 perspectives kept syntactically distinct. |
| 09 | `unknown_entity` | Agent replies "I don't know" instead of hallucinating. |
| 10 | `multi_source_reconciliation` | Two sensors disagree; agent picks by trust prior + logs both. |

### Test cards (contract per test)

#### Test 01 — `test_01_grounded_query.py`
- **Initial belief:** `robot located_at room_101 (conf 1.0, source default_map)`.
- **World:** empty.
- **Query:** *"Where are you?"*
- **Expected tool calls:** `get_belief("robot","located_at")` → `finalize_answer`.
- **Expected belief update:** none.
- **Expected provenance entry:** none new.
- **Expected answer contains:** `"room_101"`.
- **Failure mode tested:** LLM inventing a location instead of reading memory.

#### Test 02 — `test_02_stale_map_conflict.py`  ← **Scenario A**
Full spec in section I. Failure mode: LLM trusts stale map / doesn't downgrade.

#### Test 03 — `test_03_sensor_confidence_update.py`
- Runs Scenario A, then asserts: `get_belief("path_ahead","is_clear").confidence <= 0.3` and `.status == "downgraded"`.
- **Failure mode:** LLM speaks the words but doesn't call `downgrade_belief`.

#### Test 04 — `test_04_provenance_written.py`
- Runs Scenario A, then queries SQLite directly (using the fixture DB path) and asserts ≥ 2 new provenance rows with correct `source_id` values.
- **Failure mode:** in-memory graph updated but nothing persisted.

#### Test 05 — `test_05_user_perspective.py`
- **Initial belief:** `box_01 has_color red @ perspective=user`.
- **Query:** *"What color does the user believe box_01 is?"*
- **Expected calls:** `query_perspective("box_01","has_color")` → `finalize_answer`.
- **Expected answer contains:** `"red"` and `"user"`.
- **Failure mode:** LLM answers from third-party/sensor data.

#### Test 06 — `test_06_egocentric_perspective.py`
- Camera reports `brown`.
- **Query:** *"What do you currently see box_01 as?"*
- **Expected calls:** `sense_camera` → `finalize_answer` (may also `update_belief` with `perspective=self`).
- **Expected answer contains:** `"brown"` and one of `"currently"|"now"|"live"|"egocentric"`.
- **Failure mode:** LLM answers from historical data.

#### Test 07 — `test_07_historical_perspective.py`
- **Initial provenance:** third-party bot_02 row saying `painted_blue`.
- **Query:** *"What does the maintenance history say box_01's color is?"*
- **Expected calls:** `query_provenance("box_01","has_color")` (or `query_perspective` with third-party filter).
- **Expected answer contains:** `"blue"`, `"bot_02"` or `"maintenance"`.
- **Failure mode:** answers from user's belief or current sensor.

#### Test 08 — `test_08_conflicting_third_party.py` ← **Scenario B full**
- All three views seeded, camera reports brown, query = Scenario B.
- **Expected calls:** `query_perspective` + `sense_camera` + `update_belief` + `query_provenance` + `finalize_answer`.
- **Expected answer:** three distinct sentences/bullets, each naming exactly one perspective with its color.
- **Assertion helper:** `assert_perspectives_distinct(answer, {"user":"red","self":"brown","third_party":"blue"})`.
- **Failure mode:** merging perspectives (e.g. "the box is blue-brown to me and the user").

#### Test 09 — `test_09_unknown_entity.py`
- **Query:** *"Where is object `mystery_thing_42`?"*
- Nothing in graph, nothing in provenance.
- **Expected calls:** `get_belief` → `query_provenance` → `finalize_answer`.
- **Expected answer contains:** `"unknown"|"no information"|"cannot determine"`; **must not** contain a fabricated location.
- **Failure mode:** hallucination.

#### Test 10 — `test_10_multi_step_reconciliation.py`
- **Initial state:** `path_ahead is_clear (default_map, 1.0)` AND `path_ahead is_clear (camera_front, 0.7)`.
- **Live LiDAR:** blocked.
- **Query:** *"Can I proceed?"*
- **Expected:** agent calls both sensors, notices lidar disagrees, prefers lidar over camera (both are sensors but lidar has higher trust prior for distance), downgrades map + camera beliefs.
- **Post-condition:** exactly one active `is_blocked` belief with source `lidar_front`; two `downgraded` rows.
- **Failure mode:** conflict-avoidance (LLM ignores lidar because two other sources agree).

### Test log format

Each test writes `data/runs/epistemic_<id>.json`:

```json
{
  "test_id": "test_02_stale_map_conflict",
  "query": "Is your route clear? ...",
  "started_at": "...", "ended_at": "...",
  "steps": [
    {"idx":1,"tool_call":{"name":"get_belief","arguments":{...}},
             "observation":{"tool":"get_belief","ok":true,"content":{...}}},
    ...
  ],
  "final_answer": "...",
  "assertions": [
    {"name":"belief_downgraded","passed":true,
     "detail":"path_ahead:is_clear now confidence=0.2"},
    ...
  ],
  "verdict": "pass"
}
```

The 10 JSON files together = the deliverable.

---

## L. Four-Person Work Division

Ownership is by **file, not by feature**. Each person owns entire directories; nobody else edits inside another owner's directory without a PR review from that owner.

### L.1 Person 1 — **Declarative Layer Owner**

- **Role:** Knowledge engineer.
- **Owns (files):**
  - `src/iagent/declarative/**` (all files)
  - `data/seed/**`
  - `scripts/init_db.py`
  - `tests/unit/test_belief_graph.py`
  - `tests/unit/test_provenance.py`
  - `docs/data_contracts.md`
- **Implements:**
  - NetworkX belief graph with per-edge attributes.
  - SQLite schema + CRUD.
  - `KnowledgeAPI` façade — the single import point for other layers.
  - Seed loaders that materialize the initial graph + DB from `data/seed/`.
- **Writes tests:** graph invariants, provenance atomicity, `downgrade_belief` never mutates history (only inserts new rows), perspective grouping.
- **Depends on:** nobody after Phase 1 data contracts are agreed.
- **Must NOT modify:** anything under `src/iagent/procedural/` or `src/iagent/sensorimotor/`.
- **Definition of Done:** `KnowledgeAPI` covers every method in G.1; unit tests ≥ 90% coverage of `declarative/`; seed scripts run cleanly; `docs/data_contracts.md` finalized.

### L.2 Person 2 — **Sensorimotor Layer Owner**

- **Role:** Environment simulator.
- **Owns:**
  - `src/iagent/sensorimotor/**`
  - `configs/scenario_a.yaml`, `configs/scenario_b.yaml`
  - `tests/unit/test_mock_env.py`
  - `tests/unit/test_sensors.py`
- **Implements:**
  - `MockEnvironment` holding world state.
  - Sensor functions with deterministic responses given a world state.
  - Scenario YAML loader — the same file is consumed by tests.
  - Actuators (behind config flag, not needed for tests).
- **Writes tests:** given world X, `sense_lidar` returns Y; scenario loader produces expected world.
- **Depends on:** `SensorPayload` shape agreed in Phase 1 (with P1).
- **Must NOT modify:** anything under `declarative/` or `procedural/`.
- **DoD:** `SensorimotorAPI` complete per G.2; both scenario YAMLs load and produce the expected sensor responses; unit tests green.

### L.3 Person 3 — **Procedural Layer Owner**

- **Role:** LLM & ReAct engineer.
- **Owns:**
  - `src/iagent/procedural/**`
  - `configs/default.yaml`
  - `docs/react_loop.md`
  - `tests/unit/test_react_loop.py`
  - `tests/unit/test_tools.py`
- **Implements:**
  - `LLMClient` around Groq (OpenAI-compat), tool-calling flavor.
  - `run_react` loop.
  - Tool registry + dispatcher.
  - System prompt + trace logger.
  - `Agent` façade.
- **Writes tests:** with a `RecordedLLMClient` (fake), verify loop steps execute correctly; dispatcher routes to right façade; trace captures all steps.
- **Depends on:** frozen `KnowledgeAPI` (P1) + `SensorimotorAPI` (P2).
- **Must NOT modify:** `declarative/`, `sensorimotor/`, or the epistemic tests (only unit tests inside `procedural/`).
- **DoD:** `Agent.run` returns correct trace for a mocked 3-step query; all 10 tools registered; loop respects `max_steps`; unit tests green.

### L.4 Person 4 — **Integration & Test Lead**

- **Role:** Glue, CLI, CI, tests, documentation.
- **Owns:**
  - `src/iagent/cli.py`
  - `src/iagent/config.py`
  - `src/iagent/__init__.py`
  - `tests/architecture/**`
  - `tests/epistemic/**` (all 10 tests)
  - `tests/conftest.py`
  - `scripts/run_scenario_a.py`, `scripts/run_scenario_b.py`, `scripts/run_all_tests.py`, `scripts/verify_isolation.py`
  - `.github/**` (CI, PR template)
  - `README.md`, `LICENSE`, `.gitignore`, `.env.example`, `pyproject.toml`, `requirements*.txt`, `CODEOWNERS`
  - `docs/architecture.md`, `docs/scenarios.md`, `docs/verification_report.md`
- **Implements:** CLI, config loader, CI pipeline, layer-isolation static test, the 10 epistemic tests, all docs.
- **Depends on:** all three layer APIs; can begin the CLI + isolation test in parallel with layer work.
- **Must NOT modify:** internals of `declarative/`, `sensorimotor/`, `procedural/` (only their public façades via import).
- **DoD:** `python -m iagent --scenario a` runs end-to-end; all 10 epistemic tests pass with recorded LLM; CI green on `main`; README covers install → run → tests; verification report drafted.

### L.5 Workload balance sanity check

| Owner | Src LOC (est.) | Tests (files) | Docs (files) |
|---|---|---|---|
| P1 | ~450 | 2 | 1 |
| P2 | ~350 | 2 | 0 |
| P3 | ~500 | 2 | 1 |
| P4 | ~400 | 11 | 3 |

Balanced. P4 has fewer src LOC but writes the majority of tests and all cross-cutting docs.

### L.6 `CODEOWNERS` (put at repo root)

```
# Format: <pattern>  @<github-user>
/src/iagent/declarative/     @person1-gh
/data/seed/                  @person1-gh
/src/iagent/sensorimotor/    @person2-gh
/configs/scenario_*.yaml     @person2-gh
/src/iagent/procedural/      @person3-gh
/tests/epistemic/            @person4-gh
/tests/architecture/         @person4-gh
/.github/                    @person4-gh
/README.md                   @person4-gh
```

GitHub will auto-request the correct reviewer on every PR.

---

## M. GitHub / Git Workflow

### M.1 Branch model

`main` = always-green, protected. No direct pushes. Every change enters via PR.

**No `develop` branch.** For a 4-person 6-week project, a second long-lived branch adds coordination overhead without benefit. We integrate straight to `main` behind CI + review.

### M.2 Branch naming

```
<type>/<owner-initials>-<short-slug>
```

Types: `feat`, `fix`, `docs`, `test`, `chore`, `refactor`.

Examples:
```
feat/p1-belief-graph-api
feat/p2-mock-env-lidar
feat/p3-react-loop
test/p4-epistemic-scenario-a
docs/p4-architecture-overview
chore/p4-ci-workflow
```

### M.3 Commit convention

Conventional Commits. Subject ≤ 72 chars, imperative.

```
feat(declarative): add downgrade_belief with atomic provenance write
fix(procedural): cap ReAct loop at max_steps from config
test(epistemic): scenario A conflict detection

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>   # only if AI-assisted
```

### M.4 PR rules (in `.github/pull_request_template.md`)

Every PR must:
1. Have exactly one owner as author; be reviewed by ≥ 1 other person.
2. Pass CI (ruff + pytest + isolation test).
3. Not modify files outside the author's `CODEOWNERS` block without an approval from the affected owner.
4. Include added/updated tests when source under `src/` is changed.
5. If it changes cross-layer JSON, update `docs/data_contracts.md` in the same PR.

Squash-merge only. Never rebase-merge main. Never force-push shared branches.

### M.5 Concrete branch layout for the whole project

```
main
├── chore/p4-repo-scaffold                (Phase 1, day 1)
├── feat/p1-data-contracts                (Phase 1)
├── feat/p1-belief-graph                  (Phase 2)
├── feat/p1-provenance-store              (Phase 2)
├── feat/p1-knowledge-api                 (Phase 2)
├── feat/p2-mock-env                      (Phase 3)
├── feat/p2-sensors                       (Phase 3)
├── feat/p2-scenarios-yaml                (Phase 3)
├── feat/p3-llm-client                    (Phase 4)
├── feat/p3-tools                         (Phase 4)
├── feat/p3-react-loop                    (Phase 4)
├── feat/p3-agent                         (Phase 4)
├── feat/p4-cli                           (Phase 5)
├── test/p4-scenario-a                    (Phase 6)
├── test/p4-scenario-b                    (Phase 7)
├── test/p4-epistemic-01..10              (Phase 8, may be one PR per test)
└── docs/p4-verification-report           (Phase 9)
```

### M.6 Conflict avoidance

- Ownership by directory (see L) means two people almost never edit the same file.
- Cross-layer contracts (Pydantic models) live inside each layer's own `models.py`; contract *documentation* lives in `docs/data_contracts.md` and only P1 edits it (with input from others via PR review).
- Any change to a public façade signature (G.1, G.2) requires the *layer owner + P4* to approve.

### M.7 CI pipeline (`.github/workflows/ci.yml`, sketch)

```yaml
name: CI
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.11" }
      - run: pip install -r requirements.txt -r requirements-dev.txt
      - run: ruff check .
      - run: python scripts/verify_isolation.py
      - run: pytest -q --cov=src/iagent
```

Epistemic tests run with the `RecordedLLMClient` in CI (no API key needed). Live-LLM runs happen locally with `IAGENT_LIVE_LLM=1`.

---

## N. Implementation Timeline (6 weeks, 9 phases)

Convert the proposal's 4 milestones into 9 sequenced phases with explicit owners + acceptance criteria.

### Phase 1 — Scaffold + Data Contracts (Week 1, Days 1–3)
- **Tasks:**
  1. P4: create repo scaffold — every directory, all placeholder files, `.gitignore`, `.env.example`, `pyproject.toml`, `requirements*.txt`, `LICENSE`, `README.md` skeleton, `CI` workflow, `CODEOWNERS`, PR template.
  2. All 4: agree in one 60-min meeting on the Pydantic model shapes in F.4 and the tool list in G.4. Commit as `docs/data_contracts.md`.
  3. P1: write `schema.sql` + `src/iagent/declarative/models.py`.
  4. P2: write `src/iagent/sensorimotor/models.py` + one YAML scenario stub.
  5. P3: write `src/iagent/procedural/models.py` + tool-spec stubs.
- **Deliverable:** repo boots (`python -c "import iagent"` works), CI green on an empty test suite, contracts frozen.
- **Acceptance:** all 4 owners sign off on `docs/data_contracts.md`.

### Phase 2 — Declarative Layer (Week 1 Day 4 – Week 2 Day 3)
- **Owner:** P1.
- **Tasks:** implement `BeliefGraph`, `ProvenanceStore`, `KnowledgeAPI`, seeding, unit tests, `scripts/init_db.py`.
- **Dependencies:** Phase 1 contracts.
- **Deliverable:** `tests/unit/test_belief_graph.py` + `test_provenance.py` both green; `python scripts/init_db.py` creates a working SQLite file; `python -c "from iagent.declarative.api import KnowledgeAPI"` works.
- **Acceptance:** unit-test coverage ≥ 85%.

### Phase 3 — Sensorimotor Layer (Week 1 Day 4 – Week 2 Day 3, in parallel with Phase 2)
- **Owner:** P2.
- **Tasks:** implement `MockEnvironment`, `sense_*`, `act_*`, scenario YAML loader, unit tests.
- **Dependencies:** Phase 1 contracts.
- **Deliverable:** loading `scenario_a.yaml` + calling `sense_lidar` returns `{value:12, status:"blocked"}`; loading `scenario_b.yaml` + `sense_camera` returns `color_perceived:"brown"` with the ambient yellow warning.
- **Acceptance:** unit tests green.

### Phase 4 — Procedural Layer (Week 2 Day 4 – Week 3 Day 5)
- **Owner:** P3.
- **Dependencies:** frozen APIs from P1 + P2.
- **Tasks:** `LLMClient` (Groq), `tools.py`, `react_loop.py`, `Agent`, `RecordedLLMClient`, unit tests.
- **Deliverable:** with a recorded 3-step conversation, `Agent.run("test query")` returns the expected trace; the 10 tools all appear in `TOOL_SPECS`; loop respects `max_steps`.
- **Acceptance:** `test_react_loop.py` + `test_tools.py` green.

### Phase 5 — Integration (Week 3 Day 6 – Week 4 Day 1)
- **Owner:** P4 (with support from all).
- **Tasks:** `cli.py`, `config.py`, `scripts/run_scenario_a.py`, `verify_isolation.py`, wire everything through `Agent`.
- **Deliverable:** `python -m iagent --scenario a` runs end-to-end against the real Groq API and prints a coherent trace.
- **Acceptance:** manual demo works twice in a row with `temperature=0`.

### Phase 6 — Scenario A hardening (Week 4 Days 2–4)
- **Owners:** P4 primary; P1 supports (belief-write correctness); P2 supports (sensor tuning).
- **Tasks:** finalize `tests/epistemic/test_02..04`, iterate on system prompt if LLM fails to downgrade, add assertion helpers.
- **Deliverable:** tests 02, 03, 04 pass reliably with `RecordedLLMClient` and pass 5/5 runs with live Groq.

### Phase 7 — Scenario B hardening (Week 4 Days 5–7)
- **Owners:** P4 primary; P1 supports (perspective queries); P2 supports (camera output).
- **Tasks:** finalize `test_05..08`, `assert_perspectives_distinct` helper, refine prompt to prevent merging.
- **Deliverable:** tests 05–08 pass reliably.

### Phase 8 — Remaining epistemic tests + polish (Week 5 Days 1–4)
- **Owner:** P4.
- **Tasks:** implement tests 01, 09, 10; run full suite live 3× and record traces into `data/runs/`; commit the JSON logs (or attach them to a GitHub Release).
- **Deliverable:** all 10 tests green in CI; 10 JSON logs available for grading.

### Phase 9 — Docs, Verification Report, Demo (Week 5 Day 5 – Week 6)
- **Owner:** P4 (writer), all 4 (contributors).
- **Tasks:** finalize `README.md`, `docs/architecture.md`, `docs/scenarios.md`, and the "desirable" `docs/verification_report.md` (explains prompt design, information flow, and hallucination prevention). Prepare a 10-minute demo script running Scenarios A + B live.
- **Deliverable:** repo ready for grading; demo rehearsed.

### N.1 Milestone map back to proposal

| Proposal milestone | Our phases |
|---|---|
| Layer Initialization & Data Contracts | 1, 2, 3 |
| ReAct Tool Integration | 4, 5 |
| Groundedness Validation Suite | 6 (+ 8) |
| Perspective & Final Demo | 7 (+ 8, 9) |

---

## O. Complete `README.md`

*(This is the section-O deliverable: paste the block below into `README.md` on the `main` branch.)*

```markdown
# I, Agent — A Three-Layer Epistemic Architecture for Grounded Agents

[![CI](https://github.com/jewel-surabhi/csf407-i-agents/actions/workflows/ci.yml/badge.svg)](https://github.com/jewel-surabhi/csf407-i-agents/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

CSF407 course project. A grounded LLM agent that answers queries by reasoning
across three strictly-isolated cognitive layers, detects conflicts between
its stored beliefs and live sensor readings, and keeps user / self /
third-party perspectives cleanly separated.

## Table of Contents
1. [Overview](#overview)
2. [Problem statement](#problem-statement)
3. [Objectives](#objectives)
4. [Architecture](#architecture)
5. [Three layers](#three-layers)
6. [Repository structure](#repository-structure)
7. [Installation](#installation)
8. [Configuration](#configuration)
9. [Running the project](#running-the-project)
10. [Running tests](#running-tests)
11. [Example usage](#example-usage)
12. [Scenario A — Groundedness](#scenario-a--groundedness)
13. [Scenario B — Perspective](#scenario-b--perspective)
14. [Technology stack](#technology-stack)
15. [Team responsibilities](#team-responsibilities)
16. [Development workflow](#development-workflow)
17. [Project milestones](#project-milestones)
18. [Evaluation criteria](#evaluation-criteria)
19. [Future improvements](#future-improvements)

---

## Overview
Modern LLMs reason and plan well but lack **semantic grounding** (they cannot
verify that their text-based model matches the physical world) and **perspective
awareness** (they conflate historical facts, live sensor readings, and human
beliefs). This project builds an integrated cognitive system that isolates
these information sources into three API-bounded software layers and puts a
tool-calling LLM at the centre to reason across them.

## Problem statement
Given a knowledge graph that says *"the path is clear"* and a live sensor that
reports *"obstacle at 12 cm"*, a naive agent will echo whichever source it
saw first. Our agent must (a) detect the contradiction, (b) prioritize sources
by trust prior, (c) durably revise its beliefs, and (d) explain its reasoning
citing the exact sources it used.

## Objectives
- Architect a decoupled 3-layer cognitive system (Declarative / Procedural / Sensorimotor).
- Implement a tool-calling ReAct loop that reads from the belief graph and live sensors.
- Cross-reference LLM outputs against a provenance log so no fact is asserted
  without a source.
- Force the LLM to resolve conflicts between stored abstractions and raw
  telemetry, and log the resolution.
- Track three perspectives (user, egocentric, third-party historical) without
  merging them.

## Architecture

```
                      ┌──────────────────────────┐
                      │        USER (CLI)        │
                      └────────────┬─────────────┘
                                   │ NL query
                                   ▼
   ┌───────────────────────────────────────────────────────┐
   │        PROCEDURAL LAYER  (src/iagent/procedural)      │
   │  • ReAct loop         • Tool dispatch                 │
   │  • LLM (Groq/Llama-3) • Trace logger                  │
   └───────┬────────────────────────────────────┬──────────┘
           │ declarative tools                  │ sensorimotor tools
           ▼                                    ▼
 ┌────────────────────────┐        ┌────────────────────────┐
 │  DECLARATIVE LAYER     │        │  SENSORIMOTOR LAYER    │
 │  BeliefGraph (NetworkX)│        │  MockEnvironment       │
 │  ProvenanceStore(SQLite)│       │  Sensors + Actuators   │
 └────────────────────────┘        └────────────────────────┘
```

Cross-layer traffic is JSON only; layer isolation is enforced by
`tests/architecture/test_layer_isolation.py`.

## Three layers
- **Declarative** — long-term memory. NetworkX `MultiDiGraph` for beliefs +
  SQLite for provenance rows (subject, predicate, object, confidence, source,
  perspective, observed_at). Public interface: `KnowledgeAPI` in
  `src/iagent/declarative/api.py`.
- **Procedural** — the LLM orchestrator. A hand-written ReAct loop over the
  provider's native tool-calling. Every step is logged to `data/runs/*.json`.
  Public interface: `Agent` in `src/iagent/procedural/agent.py`.
- **Sensorimotor** — a `MockEnvironment` that returns JSON sensor payloads
  driven by a YAML scenario file. Public interface: `SensorimotorAPI` in
  `src/iagent/sensorimotor/api.py`.

## Repository structure
```
src/iagent/
  declarative/     BeliefGraph, ProvenanceStore, KnowledgeAPI
  sensorimotor/    MockEnvironment, sensors, actuators, SensorimotorAPI
  procedural/      Agent, ReAct loop, LLMClient, tools, trace logger
  cli.py           python -m iagent --scenario a
tests/
  unit/            per-layer unit tests
  architecture/    layer-isolation grep test
  epistemic/       the 10 required test logs
configs/           default.yaml + scenario_a.yaml + scenario_b.yaml
data/seed/         initial belief graph + provenance rows
scripts/           init_db, run_scenario_a, run_scenario_b, run_all_tests
docs/              architecture, data_contracts, scenarios, verification_report
```
Full tree: see `docs/architecture.md`.

## Installation
Requires Python **3.11+**.
```bash
git clone https://github.com/jewel-surabhi/csf407-i-agents.git
cd csf407-i-agents
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt -r requirements-dev.txt
python scripts/init_db.py            # creates iagent.db from schema.sql
```

## Configuration
1. Copy `.env.example` to `.env`.
2. Set `GROQ_API_KEY=...` (free at https://console.groq.com).
3. Optional: `LLM_MODEL=llama-3.1-8b-instant`, `LLM_TEMPERATURE=0.0`.

## Running the project
```bash
python -m iagent --scenario a        # runs Scenario A, prints trace
python -m iagent --scenario b        # runs Scenario B
python -m iagent --query "Where am I?"  # ad-hoc query
```
Traces land in `data/runs/<timestamp>.json`.

## Running tests
```bash
pytest -q                                        # unit + architecture (recorded LLM)
pytest tests/epistemic -q                        # the 10 required tests
IAGENT_LIVE_LLM=1 pytest tests/epistemic -q      # live Groq (needs API key)
```
CI runs the recorded variant on every push.

## Example usage
```python
from iagent import Agent, load_config
from iagent.declarative.api import KnowledgeAPI
from iagent.sensorimotor.api import SensorimotorAPI
from iagent.procedural.llm_client import LLMClient

cfg = load_config("configs/default.yaml")
knowledge = KnowledgeAPI.from_config(cfg)
sensors   = SensorimotorAPI.from_scenario("configs/scenario_a.yaml")
agent     = Agent(knowledge, sensors, LLMClient(cfg), cfg)

result = agent.run("Is your route clear?")
print(result.answer)
for step in result.trace.steps:
    print(step.model_dump_json(indent=2))
```

## Scenario A — Groundedness
The map says the path is clear; the LiDAR says it is blocked at 12 cm. The
agent must detect the conflict, prefer the sensor (`lidar_front` trust_prior
= 0.9 > `default_map` = 0.6), downgrade the stale belief, insert a fresh
belief, and cite provenance IDs. See `docs/scenarios.md#scenario-a`.

## Scenario B — Perspective
The user expects a red box; the camera sees brown (yellow ambient); a
maintenance bot logged the box as blue. The agent must return three distinct
perspectives without merging them. See `docs/scenarios.md#scenario-b`.

## Technology stack
| Layer | Tech |
|---|---|
| Declarative | NetworkX 3.3, SQLite (stdlib) |
| Procedural | Groq API (Llama-3.1-8B-Instruct, tool-calling), Pydantic v2 |
| Sensorimotor | Python + YAML scenarios |
| Tests | pytest, pytest-cov |
| Lint/format | ruff, black |
| CI | GitHub Actions |

## Team responsibilities
- **Person 1 — Declarative:** `src/iagent/declarative/`, seed data, `init_db.py`.
- **Person 2 — Sensorimotor:** `src/iagent/sensorimotor/`, scenario YAMLs.
- **Person 3 — Procedural:** `src/iagent/procedural/`, LLM + ReAct loop.
- **Person 4 — Integration/Tests:** CLI, config, CI, the 10 epistemic tests, docs.

`CODEOWNERS` enforces this on every PR.

## Development workflow
- Branch: `<type>/<owner-initials>-<slug>` (e.g. `feat/p1-belief-graph`).
- Commits: Conventional Commits.
- Every PR: passes CI (ruff + pytest + isolation) and gets ≥ 1 review.
- No direct pushes to `main`. Squash-merge only.

## Project milestones
1. Layer Initialization & Data Contracts — **Planned**
2. ReAct Tool Integration — **Planned**
3. Groundedness Validation Suite — **Planned**
4. Perspective & Final Demo — **Planned**

*(Update this list as phases complete: `Planned` → `In progress` → `Done`.)*

## Evaluation criteria
- Correctness on the 10 epistemic tests (JSON logs under `data/runs/`).
- Strict layer isolation (`test_layer_isolation.py` passes).
- Provenance completeness (every belief write leaves a SQLite row).
- Clarity of the Scenario A + B final answers (no merged perspectives).
- Verification report (`docs/verification_report.md`).

## Future improvements
- Swap `MockEnvironment` for a Gymnasium-based simulator (still behind the
  same `SensorimotorAPI` façade).
- Extend `KnowledgeAPI` with belief-decay over time.
- Multi-agent scenario (two `Agent` instances, each with its own belief graph,
  reconciling perspectives via a shared provenance log).
- Optional FastAPI + minimal HTML frontend for interactive querying.
```

---

## P. START HERE Checklist

Ordered by dependency. Anyone on the team can begin at any `[ ]` whose predecessors are done.

### Day 1 — Repo scaffold (P4 leads a 90-min pairing session with all 4)
```
[ ] P4: create branch chore/p4-repo-scaffold
[ ] P4: add .gitignore, LICENSE (MIT), pyproject.toml, requirements*.txt
[ ] P4: create the full directory tree from section D (empty __init__.py in every package)
[ ] P4: add .env.example with GROQ_API_KEY placeholder
[ ] P4: add .github/workflows/ci.yml, PR template, ISSUE_TEMPLATEs
[ ] P4: add CODEOWNERS with everyone's GitHub handle
[ ] P4: paste section O into README.md
[ ] P4: open PR → all 4 review → squash-merge to main
[ ] All: clone main, create your own local venv, run `pytest` (should collect 0 tests, exit 0)
```

### Day 2 — Data contracts meeting (all 4, 60 min)
```
[ ] All: walk through section F.1–F.4 together
[ ] All: agree on any Pydantic field renames — record decisions in docs/data_contracts.md
[ ] P1: land src/iagent/declarative/models.py + schema.sql
[ ] P2: land src/iagent/sensorimotor/models.py
[ ] P3: land src/iagent/procedural/models.py + TOOL_SPECS stub (json only, no dispatch yet)
[ ] P4: land docs/data_contracts.md as source of truth
[ ] All 4: sign off on the PRs — after this point, contract changes need collective approval
```

### Day 3–5 — Layer implementation in parallel
```
P1 (Declarative):
[ ] Implement BeliefGraph (belief_graph.py)
[ ] Implement ProvenanceStore (provenance.py)
[ ] Implement KnowledgeAPI (api.py) — all methods in G.1
[ ] Implement seeding.py + data/seed/initial_beliefs.json + initial_provenance.sql
[ ] scripts/init_db.py
[ ] tests/unit/test_belief_graph.py + test_provenance.py

P2 (Sensorimotor):
[ ] Implement MockEnvironment (mock_env.py) with world-state model
[ ] Implement sensors.py (read_lidar/camera/clock)
[ ] Implement actuators.py (behind a feature flag)
[ ] Implement scenarios.py (YAML loader)
[ ] Write configs/scenario_a.yaml + scenario_b.yaml
[ ] Implement SensorimotorAPI (api.py) — all methods in G.2
[ ] tests/unit/test_mock_env.py + test_sensors.py

P3 (Procedural):
[ ] Implement LLMClient (llm_client.py) — Groq wrapper with retries
[ ] Implement tools.py — full TOOL_SPECS + dispatch to façades
[ ] Implement prompts.py — SYSTEM_PROMPT from H.3
[ ] Implement react_loop.py — the loop in H.2
[ ] Implement trace.py — JSON logger writing to data/runs/
[ ] Implement Agent (agent.py) façade
[ ] Implement RecordedLLMClient for deterministic tests
[ ] tests/unit/test_react_loop.py + test_tools.py

P4 (Integration/tests scaffolding, in parallel):
[ ] Implement config.py (YAML + .env loader → AppConfig Pydantic model)
[ ] Implement cli.py (argparse: --scenario, --query, --max-steps)
[ ] Implement scripts/verify_isolation.py + tests/architecture/test_layer_isolation.py
[ ] Implement tests/conftest.py + tests/epistemic/conftest.py (fixtures)
[ ] docs/architecture.md — expand section C
```

### Week 2 — First end-to-end run
```
[ ] P4: scripts/run_scenario_a.py that wires everything through Agent
[ ] All 4: pair-debug the first live Groq call until Scenario A produces a coherent trace
[ ] P4: commit the resulting trace into data/runs/first_success.json as reference
```

### Week 3 — Scenario A hardening + Scenario B first run
```
[ ] P4: test_02_stale_map_conflict.py, test_03_sensor_confidence_update.py,
        test_04_provenance_written.py
[ ] P1 + P2: iterate on trust priors / sensor payloads if the LLM fails to downgrade
[ ] P3: refine SYSTEM_PROMPT if needed
[ ] P4: scripts/run_scenario_b.py + test_08_conflicting_third_party.py
```

### Week 4 — Remaining epistemic tests
```
[ ] P4: test_01 (grounded_query), test_05 (user), test_06 (egocentric),
        test_07 (historical), test_09 (unknown_entity), test_10 (multi-source)
[ ] P4: run the full suite live 5× — commit the 10 JSON logs
```

### Week 5 — Docs + demo
```
[ ] P4: fill in docs/verification_report.md (prompt design, info flow, hallucination guards)
[ ] P4: cut a demo script (~10 min): Scenario A live, Scenario B live, show 10 test logs
[ ] All 4: rehearse the demo twice
[ ] P4: tag v1.0 on main; attach the 10 JSON logs to the GitHub Release
```

### Week 6 — Buffer / polish
```
[ ] All: fix any flaky tests
[ ] P4: coverage report, ensure README badges are green
[ ] All: write a 1-page individual reflection (helps for grading)
```

---

**End of plan.** After Day 1 the four of you can open GitHub, pick your files, and start coding without another architecture meeting.
