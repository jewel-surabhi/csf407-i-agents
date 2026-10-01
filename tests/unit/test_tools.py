"""Unit tests for the tool registry + dispatcher. Owner: Person 3 (Phase 4)."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest

from iagent.declarative.api import Belief, KnowledgeError, PerspectiveView, ProvenanceEntry, Source
from iagent.procedural.tools import (
    FINALIZE_TOOL,
    TOOL_NAMES,
    TOOL_SPECS,
    DispatchError,
    dispatch,
)
from iagent.sensorimotor.api import SensorPayload

# --------------------------------------------------------------------------
# Fakes — layer isolation lets us swap the real façades for spies.
# --------------------------------------------------------------------------


class FakeKnowledge:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.next_error: Exception | None = None

    def _record(self, name: str, **kw: Any) -> None:
        if self.next_error is not None:
            err, self.next_error = self.next_error, None
            raise err
        self.calls.append((name, kw))

    def get_belief(self, subject, predicate, perspective="self"):
        self._record("get_belief", subject=subject, predicate=predicate, perspective=perspective)
        return Belief(
            subject=subject,
            predicate=predicate,
            object="clear",
            confidence=1.0,
            source="default_map",
            perspective=perspective,
            observed_at=datetime(2026, 1, 1, tzinfo=UTC),
            provenance_id=1,
        )

    def query_perspective(self, subject, predicate=None):
        self._record("query_perspective", subject=subject, predicate=predicate)
        return [PerspectiveView(perspective="self", beliefs=[])]

    def query_provenance(self, subject, predicate=None, limit=20):
        self._record("query_provenance", subject=subject, predicate=predicate, limit=limit)
        return []

    def list_sources(self):
        self._record("list_sources")
        return [Source(source_id="default_map", kind="static_map", trust_prior=0.6)]

    def update_belief(self, belief: Belief, reason: str):
        self._record("update_belief", belief=belief.model_dump(mode="json"), reason=reason)
        return ProvenanceEntry(
            provenance_id=42,
            subject=belief.subject,
            predicate=belief.predicate,
            object=belief.object,
            confidence=belief.confidence,
            source_id=belief.source,
            perspective=belief.perspective,
            observed_at=belief.observed_at,
        )

    def downgrade_belief(self, subject, predicate, new_confidence, reason, perspective="self"):
        self._record(
            "downgrade_belief",
            subject=subject,
            predicate=predicate,
            new_confidence=new_confidence,
            reason=reason,
            perspective=perspective,
        )
        return ProvenanceEntry(
            provenance_id=99,
            subject=subject,
            predicate=predicate,
            object="_",
            confidence=new_confidence,
            source_id="_",
            perspective=perspective,
            observed_at=datetime(2026, 1, 1, tzinfo=UTC),
            status="downgraded",
        )


class FakeSensors:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def sense_lidar(self, direction="front"):
        self.calls.append(("sense_lidar", {"direction": direction}))
        return SensorPayload(
            sensor=f"lidar_{direction}",
            kind="distance",
            value=12,
            status="blocked",
            observed_at=datetime(2026, 1, 1, tzinfo=UTC),
            confidence=0.95,
        )

    def sense_camera(self, direction="front"):
        self.calls.append(("sense_camera", {"direction": direction}))
        return SensorPayload(
            sensor=f"camera_{direction}",
            kind="frame",
            observed_at=datetime(2026, 1, 1, tzinfo=UTC),
        )

    def sense_clock(self):
        self.calls.append(("sense_clock", {}))
        return SensorPayload(
            sensor="clock",
            kind="time",
            value="2026-01-01T00:00:00+00:00",
            observed_at=datetime(2026, 1, 1, tzinfo=UTC),
        )


@pytest.fixture
def apis():
    return {"knowledge": FakeKnowledge(), "sensors": FakeSensors()}


# --------------------------------------------------------------------------
# TOOL_SPECS shape
# --------------------------------------------------------------------------


REQUIRED_TOOLS = frozenset(
    {
        "get_belief",
        "query_perspective",
        "query_provenance",
        "list_sources",
        "update_belief",
        "downgrade_belief",
        "sense_lidar",
        "sense_camera",
        "sense_clock",
        "finalize_answer",
    }
)


def test_all_required_tools_registered():
    assert TOOL_NAMES == REQUIRED_TOOLS


def test_finalize_tool_name_matches_constant():
    assert FINALIZE_TOOL == "finalize_answer"
    assert FINALIZE_TOOL in TOOL_NAMES


@pytest.mark.parametrize("spec", TOOL_SPECS, ids=lambda s: s["function"]["name"])
def test_tool_spec_shape(spec):
    assert spec["type"] == "function"
    fn = spec["function"]
    assert isinstance(fn["name"], str) and fn["name"]
    assert isinstance(fn["description"], str) and fn["description"]
    params = fn["parameters"]
    assert params["type"] == "object"
    assert isinstance(params.get("properties", {}), dict)


def test_tool_names_are_unique():
    names = [s["function"]["name"] for s in TOOL_SPECS]
    assert len(names) == len(set(names))


# --------------------------------------------------------------------------
# dispatch: happy paths
# --------------------------------------------------------------------------


def test_dispatch_get_belief_routes_to_knowledge(apis):
    result = dispatch(
        "get_belief",
        {"subject": "path_ahead", "predicate": "is_clear"},
        apis,
    )
    assert result["ok"] is True
    assert apis["knowledge"].calls == [
        (
            "get_belief",
            {"subject": "path_ahead", "predicate": "is_clear", "perspective": "self"},
        )
    ]
    assert result["data"]["subject"] == "path_ahead"


def test_dispatch_get_belief_passes_perspective(apis):
    dispatch(
        "get_belief",
        {"subject": "box_01", "predicate": "has_color", "perspective": "user"},
        apis,
    )
    assert apis["knowledge"].calls[0][1]["perspective"] == "user"


def test_dispatch_query_perspective(apis):
    result = dispatch("query_perspective", {"subject": "box_01"}, apis)
    assert result["ok"] is True
    assert isinstance(result["data"], list)


def test_dispatch_query_provenance_defaults_limit(apis):
    dispatch("query_provenance", {"subject": "path_ahead"}, apis)
    assert apis["knowledge"].calls[0][1]["limit"] == 20


def test_dispatch_list_sources(apis):
    result = dispatch("list_sources", {}, apis)
    assert result["ok"] is True
    assert result["data"][0]["source_id"] == "default_map"


def test_dispatch_update_belief_builds_belief_model(apis):
    result = dispatch(
        "update_belief",
        {
            "subject": "path_ahead",
            "predicate": "is_blocked",
            "object": "obstacle@12cm",
            "confidence": 0.95,
            "source": "lidar_front",
            "perspective": "self",
            "reason": "lidar reading",
        },
        apis,
    )
    assert result["ok"] is True
    assert result["data"]["provenance_id"] == 42
    assert apis["knowledge"].calls[0][0] == "update_belief"


def test_dispatch_downgrade_belief(apis):
    result = dispatch(
        "downgrade_belief",
        {
            "subject": "path_ahead",
            "predicate": "is_clear",
            "new_confidence": 0.2,
            "reason": "lidar contradicts",
        },
        apis,
    )
    assert result["ok"] is True
    assert result["data"]["status"] == "downgraded"


def test_dispatch_sense_lidar_default_direction(apis):
    result = dispatch("sense_lidar", {}, apis)
    assert result["ok"] is True
    assert result["data"]["status"] == "blocked"
    assert apis["sensors"].calls == [("sense_lidar", {"direction": "front"})]


def test_dispatch_sense_camera_and_clock(apis):
    dispatch("sense_camera", {"direction": "left"}, apis)
    dispatch("sense_clock", {}, apis)
    names = [c[0] for c in apis["sensors"].calls]
    assert names == ["sense_camera", "sense_clock"]


def test_dispatch_finalize_answer_passes_through(apis):
    result = dispatch(
        "finalize_answer",
        {
            "answer": "blocked",
            "cited_beliefs": ["path_ahead:is_blocked"],
            "cited_sensors": ["lidar_front"],
        },
        apis,
    )
    assert result["ok"] is True
    assert result["data"] == {
        "answer": "blocked",
        "cited_beliefs": ["path_ahead:is_blocked"],
        "cited_sensors": ["lidar_front"],
    }
    # finalize must NOT touch the façades
    assert apis["knowledge"].calls == []
    assert apis["sensors"].calls == []


def test_dispatch_finalize_answer_cited_lists_default_to_empty(apis):
    result = dispatch("finalize_answer", {"answer": "done"}, apis)
    assert result["data"]["cited_beliefs"] == []
    assert result["data"]["cited_sensors"] == []


# --------------------------------------------------------------------------
# dispatch: error handling
# --------------------------------------------------------------------------


def test_dispatch_unknown_tool_returns_error_envelope(apis):
    result = dispatch("teleport", {}, apis)
    assert result == {
        "ok": False,
        "error": {"kind": "unknown_tool", "message": "No tool named 'teleport'."},
    }


def test_dispatch_missing_required_argument(apis):
    result = dispatch("get_belief", {"subject": "x"}, apis)
    assert result["ok"] is False
    assert result["error"]["kind"] == "missing_argument"


def test_dispatch_invalid_confidence_reports_validation_error(apis):
    result = dispatch(
        "update_belief",
        {
            "subject": "x",
            "predicate": "y",
            "object": "z",
            "confidence": 1.5,  # out of [0,1]
            "source": "s",
            "reason": "r",
        },
        apis,
    )
    assert result["ok"] is False
    assert result["error"]["kind"] == "invalid_argument"


def test_dispatch_wraps_knowledge_error(apis):
    apis["knowledge"].next_error = KnowledgeError("no such subject")
    result = dispatch(
        "get_belief",
        {"subject": "ghost", "predicate": "exists"},
        apis,
    )
    assert result["ok"] is False
    assert result["error"]["kind"] == "knowledge_error"


def test_dispatch_wraps_unexpected_exception(apis):
    apis["knowledge"].next_error = RuntimeError("boom")
    result = dispatch(
        "get_belief",
        {"subject": "x", "predicate": "y"},
        apis,
    )
    assert result["ok"] is False
    assert result["error"]["kind"] == "tool_error"
    assert "boom" in result["error"]["message"]


def test_dispatch_missing_api_raises_dispatch_error(apis):
    with pytest.raises(DispatchError):
        dispatch("get_belief", {"subject": "x", "predicate": "y"}, {"sensors": apis["sensors"]})
