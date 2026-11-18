-- Registered sources with trust priors (0..1). Higher = more trusted by default.
-- Trust priors are the tie-breaker the LLM consults via list_sources when it detects a conflict.
--
-- decay_rate_per_hour models temporal confidence decay:
--   0.0 = never decays (static facts: maps, user statements)
--   0.1 = slow decay (log-based systems updated infrequently)
--   0.3+ = fast decay (sensor readings go stale quickly)
-- Effective confidence at read time = max(0, stored_confidence - decay_rate * age_hours).

INSERT OR IGNORE INTO sources VALUES ('default_map',  'static_map', 0.6, 0.00, 'Baseline floor plan');
INSERT OR IGNORE INTO sources VALUES ('lidar_front',  'sensor',     0.9, 0.30, 'Front-facing LiDAR');
INSERT OR IGNORE INTO sources VALUES ('camera_front', 'sensor',     0.7, 0.40, 'Front-facing camera');
INSERT OR IGNORE INTO sources VALUES ('bot_02',       'agent',      0.5, 0.10, 'Maintenance bot #02');
INSERT OR IGNORE INTO sources VALUES ('user',         'user',       0.4, 0.00, 'Human operator statement');
INSERT OR IGNORE INTO sources VALUES ('llm_infer',    'llm',        0.2, 0.50, 'LLM inference (unverified)');
INSERT OR IGNORE INTO sources VALUES ('self_reflection', 'agent',   0.3, 0.05, 'Agent-generated meta-belief (Reflexion-style)');
