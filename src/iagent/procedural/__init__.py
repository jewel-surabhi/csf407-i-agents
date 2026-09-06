"""Procedural layer — LLM + ReAct loop.

Public entrypoint: `iagent.procedural.agent.Agent`.

This layer MUST NOT import:
- networkx / sqlite3   (declarative internals)
- iagent.sensorimotor.mock_env or *.sensors  (sensorimotor internals)

It talks to the other layers ONLY via `KnowledgeAPI` and `SensorimotorAPI`.
`scripts/verify_isolation.py` enforces this.
"""
