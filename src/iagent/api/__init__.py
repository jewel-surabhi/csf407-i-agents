"""HTTP API over the Agent — composed interface for the React frontend.

This module sits at the SAME altitude as `cli.py` (both wrap the Agent for
different interfaces). It is NOT a fourth cognitive layer — layer isolation
rules do not apply here, since this module's job is to compose the three
layers behind an HTTP surface.
"""
