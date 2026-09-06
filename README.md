# I, Agent — A Three-Layer Epistemic Architecture for Grounded Agents

[![CI](https://github.com/jewel-surabhi/csf407-i-agents/actions/workflows/ci.yml/badge.svg)](https://github.com/jewel-surabhi/csf407-i-agents/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

CSF407 course project. A grounded LLM agent that answers queries by reasoning
across three strictly-isolated cognitive layers, detects conflicts between
its stored beliefs and live sensor readings, and keeps user / self /
third-party perspectives cleanly separated.

> **Status:** scaffold. All modules are placeholders. See
> [`PROJECT_PLAN.md`](PROJECT_PLAN.md) for the full implementation plan and
> [`docs/`](docs/) for per-topic design notes.

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
Full tree: see [`docs/architecture.md`](docs/architecture.md).

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

*(These commands work once Phases 4–5 land; the scaffold ships placeholders.)*

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
belief, and cite provenance IDs. See [`docs/scenarios.md`](docs/scenarios.md).

## Scenario B — Perspective
The user expects a red box; the camera sees brown (yellow ambient); a
maintenance bot logged the box as blue. The agent must return three distinct
perspectives without merging them. See [`docs/scenarios.md`](docs/scenarios.md).

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
