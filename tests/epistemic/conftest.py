"""Fixtures shared across the 10 epistemic tests.

Owner: Person 4. Filled in during Phases 6–8.

Planned fixtures:
    seeded_knowledge(tmp_db, scenario_yaml)  -> KnowledgeAPI
    mock_sensors(scenario_yaml)              -> SensorimotorAPI
    recorded_agent(script)                   -> Agent (RecordedLLMClient)
    live_agent()                             -> Agent (real Groq), marked live_llm
"""

from __future__ import annotations

import os

import pytest

LIVE_LLM = os.getenv("IAGENT_LIVE_LLM") == "1"


def pytest_collection_modifyitems(config, items):
    """Auto-skip `live_llm`-marked tests unless IAGENT_LIVE_LLM=1."""
    if LIVE_LLM:
        return
    skip_live = pytest.mark.skip(reason="Set IAGENT_LIVE_LLM=1 to run live-LLM tests.")
    for item in items:
        if "live_llm" in item.keywords:
            item.add_marker(skip_live)
