-- Registered sources with trust priors (0..1). Higher = more trusted by default.
-- Trust priors are the tie-breaker the LLM consults via list_sources when it detects a conflict.

INSERT OR IGNORE INTO sources VALUES ('default_map',  'static_map', 0.6, 'Baseline floor plan');
INSERT OR IGNORE INTO sources VALUES ('lidar_front',  'sensor',     0.9, 'Front-facing LiDAR');
INSERT OR IGNORE INTO sources VALUES ('camera_front', 'sensor',     0.7, 'Front-facing camera');
INSERT OR IGNORE INTO sources VALUES ('bot_02',       'agent',      0.5, 'Maintenance bot #02');
INSERT OR IGNORE INTO sources VALUES ('user',         'user',       0.4, 'Human operator statement');
INSERT OR IGNORE INTO sources VALUES ('llm_infer',    'llm',        0.2, 'LLM inference (unverified)');
