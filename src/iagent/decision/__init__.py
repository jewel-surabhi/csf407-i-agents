"""Reads the procedural layer's decision: the Total map.

``total_map.decide`` gathers the declaration map (declarative layer) and the
sensor map (sensorimotor layer) through ``layer_readers``, hands them to the
procedural layer's grid task (the LLM), and returns what the LLM decided.
``total_map.write_back`` stores the LLM's sensor-backed winners in the
declarative layer.
"""
