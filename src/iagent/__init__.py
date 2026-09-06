"""I, Agent — Three-Layer Epistemic Architecture for Grounded Agents.

Top-level package. Import individual layers explicitly, e.g.:

    from iagent.declarative.api import KnowledgeAPI
    from iagent.sensorimotor.api import SensorimotorAPI
    from iagent.procedural.agent import Agent

`load_config` is re-exported here for convenience.
"""

from iagent.config import load_config

__all__ = ["load_config"]
__version__ = "0.1.0"
