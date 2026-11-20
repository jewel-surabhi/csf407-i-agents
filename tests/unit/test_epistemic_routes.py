"""Tests for the read-only epistemic test results endpoint."""

from __future__ import annotations

from fastapi.testclient import TestClient

from iagent.api.server import app

client = TestClient(app)


def test_list_epistemic_tests_returns_both_modes():
    r = client.get("/api/epistemic/tests")
    assert r.status_code == 200
    body = r.json()
    assert len(body) >= 10  # at least the 10 recorded traces

    modes = {item["mode"] for item in body}
    assert "recorded" in modes

    numbers = {item["number"] for item in body}
    assert numbers == set(range(1, 11))


def test_each_test_has_a_verdict_and_steps():
    body = client.get("/api/epistemic/tests").json()
    for item in body:
        assert item["verdict"] in {"pass", "fail", "unknown"}
        assert item["step_count"] == len(item["steps"])
        assert item["test_id"]
        assert item["title"]
