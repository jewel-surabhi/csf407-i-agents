"""Unit tests for temporal decay of belief confidence."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from iagent.declarative.decay import annotate_with_decay, effective_confidence
from iagent.declarative.models import Belief, Source

_NOW = datetime(2026, 11, 17, 12, 0, 0, tzinfo=UTC)


def _belief(src: str, conf: float, observed_at: datetime) -> Belief:
    return Belief(
        subject="path_ahead",
        predicate="is_clear",
        object="true",
        confidence=conf,
        source=src,
        observed_at=observed_at,
    )


def _src(sid: str, decay: float) -> Source:
    return Source(source_id=sid, kind="sensor", trust_prior=0.9, decay_rate_per_hour=decay)


def test_zero_decay_rate_passes_confidence_through_unchanged():
    eff, age = effective_confidence(0.9, _NOW - timedelta(hours=10), 0.0, now=_NOW)
    assert eff == 0.9
    assert age == 10.0


def test_linear_decay_subtracts_rate_times_age():
    # 0.95 base, 0.3/hour decay, 2 hours old -> 0.95 - 0.6 = 0.35
    eff, age = effective_confidence(0.95, _NOW - timedelta(hours=2), 0.3, now=_NOW)
    assert eff == 0.35
    assert age == 2.0


def test_decay_floors_at_zero_never_negative():
    # 0.5 base, 0.3/hour decay, 10 hours old -> would be -2.5, clamp to 0
    eff, _ = effective_confidence(0.5, _NOW - timedelta(hours=10), 0.3, now=_NOW)
    assert eff == 0.0


def test_future_observed_at_treated_as_zero_age():
    # Clock skew: observed_at later than now. Treat age as 0, confidence unchanged.
    eff, age = effective_confidence(0.8, _NOW + timedelta(hours=5), 0.3, now=_NOW)
    assert eff == 0.8
    assert age == 0.0


def test_annotate_populates_effective_confidence_and_age_on_belief():
    b = _belief("lidar_front", 0.95, _NOW - timedelta(hours=1))
    srcs = {"lidar_front": _src("lidar_front", 0.3)}
    out = annotate_with_decay(b, srcs, now=_NOW)
    assert abs(out.effective_confidence - 0.65) < 1e-9  # 0.95 - 0.3
    assert out.age_hours == 1.0
    # Original stored confidence must be untouched — ledger stays faithful.
    assert out.confidence == 0.95


def test_annotate_handles_unknown_source_as_no_decay():
    b = _belief("mystery_box", 0.5, _NOW - timedelta(hours=100))
    out = annotate_with_decay(b, {}, now=_NOW)
    assert out.effective_confidence == 0.5
    assert out.age_hours == 100.0
