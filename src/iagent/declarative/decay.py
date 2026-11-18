"""Temporal decay of belief confidence.

Beliefs lose confidence as they age. A LiDAR reading from 10 seconds ago is
more trustworthy than one from two hours ago. Each `Source` carries a
`decay_rate_per_hour`: the effective confidence of a belief at read time is
    max(0.0, stored_confidence - decay_rate * age_hours).

Design notes:
- Stored confidence stays frozen — only the READ-TIME effective value
  changes. This keeps the provenance ledger a faithful audit trail (what
  the source actually claimed at the time), while callers who care about
  present-day trustworthiness can read `Belief.effective_confidence`.
- A source with `decay_rate_per_hour = 0.0` (floor maps, user statements)
  never decays — its effective confidence equals its stored confidence.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import UTC, datetime

from iagent.declarative.models import Belief, Source


def effective_confidence(
    stored_confidence: float,
    observed_at: datetime,
    decay_rate_per_hour: float,
    *,
    now: datetime | None = None,
) -> tuple[float, float]:
    """Return (effective_confidence, age_hours) for a belief.

    `now` defaults to the current UTC time. `observed_at` is normalized to
    UTC if it carries no tzinfo (naive datetimes are treated as UTC).
    """
    reference = now or datetime.now(UTC)
    anchor = observed_at if observed_at.tzinfo else observed_at.replace(tzinfo=UTC)

    delta = reference - anchor
    age_hours = max(0.0, delta.total_seconds() / 3600.0)

    if decay_rate_per_hour <= 0.0:
        return stored_confidence, age_hours

    decayed = stored_confidence - decay_rate_per_hour * age_hours
    return max(0.0, decayed), age_hours


def annotate_with_decay(
    belief: Belief,
    sources_by_id: dict[str, Source],
    *,
    now: datetime | None = None,
) -> Belief:
    """Return a copy of `belief` with `effective_confidence` and `age_hours`
    populated from the matching source's decay rate.

    If the source is unknown (shouldn't happen in production — all writes go
    through `log_source`), defaults to no decay so the belief passes through
    unchanged.
    """
    source = sources_by_id.get(belief.source)
    decay = source.decay_rate_per_hour if source else 0.0
    eff, age = effective_confidence(
        belief.confidence, belief.observed_at, decay, now=now
    )
    return belief.model_copy(update={"effective_confidence": eff, "age_hours": age})


def annotate_many(
    beliefs: Iterable[Belief],
    sources: Iterable[Source],
    *,
    now: datetime | None = None,
) -> list[Belief]:
    """Batch variant of `annotate_with_decay` — computes the sources dict once."""
    sources_by_id = {s.source_id: s for s in sources}
    return [annotate_with_decay(b, sources_by_id, now=now) for b in beliefs]
