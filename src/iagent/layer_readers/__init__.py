"""Readers that turn the declarative and sensorimotor layers into grid maps.

* ``declaration_map`` — what the declarative layer (``KnowledgeAPI``) believes
  about every path, decayed to a given hour.
* ``sensor_map`` — what the sensorimotor layer (``SensorimotorAPI``) reports
  from the robot's node: LiDAR state and distance, camera colour.
"""
