"""Load a scenario YAML into a `WorldState` (plus the raw scenario dict).

Only the `world:` block is parsed into `WorldState`; the rest of the file
(`seed_beliefs`, `expected_answer_contains`, `sensors`, `query`, ...) is
returned untouched so other layers can read what they need without the
sensorimotor layer depending on them.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from iagent.sensorimotor.models import WorldState

__all__ = ["load_scenario"]


def load_scenario(path: str | Path) -> tuple[WorldState, dict[str, Any]]:
    """Parse a scenario YAML file.

    Returns
    -------
    (world_state, scenario_dict)
        `world_state` is built from the `world:` block. `scenario_dict` is the
        full parsed YAML — callers (e.g. the declarative layer's seed loader)
        pick out `seed_beliefs`, `expected_answer_contains`, etc.
    """
    p = Path(path)
    with p.open("r", encoding="utf-8") as fh:
        data: dict[str, Any] = yaml.safe_load(fh) or {}

    world_block: dict[str, Any] = data.get("world", {}) or {}

    world_state = WorldState(
        robot=world_block.get("robot", {}) or {},
        objects=list(world_block.get("objects", []) or []),
        obstacles=list(world_block.get("obstacles", []) or []),
        lighting=world_block.get("lighting", "normal") or "normal",
        extras={
            k: v
            for k, v in world_block.items()
            if k not in {"robot", "objects", "obstacles", "lighting"}
        },
    )

    return world_state, data
