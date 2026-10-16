"""Pydantic models for the warehouse demo layer."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class Junction(BaseModel):
    id: str
    label: str
    x: float
    y: float


class SegmentSeed(BaseModel):
    a: str
    b: str
    state: Literal["clear", "blocked"]
    source: str
    confidence: float = Field(ge=0.0, le=1.0)


class StockSeed(BaseModel):
    source: str
    perspective: str
    predicate: str  # "count" | "shipped"
    value: int
    note: str | None = None


class Shelf(BaseModel):
    id: str
    near: str
    dx: float = 0
    dy: float = 0
    target: bool = False
    low: bool = False
    stock: list[StockSeed] = Field(default_factory=list)


class Topology(BaseModel):
    id: str
    start: str
    goal: str
    junctions: dict[str, Junction]
    segments: list[SegmentSeed]
    shelves: list[Shelf]


# ---- runtime result models (what the API returns) ----


class MoveOption(BaseModel):
    """One possible next move, scored by grounded confidence."""

    to: str
    to_label: str
    segment: str
    confidence: float = Field(ge=0.0, le=1.0)
    tier: Literal["high", "medium", "low"]
    source: str
    passable: bool
    reason: str


class SourceCount(BaseModel):
    source: str
    perspective: str
    predicate: str
    value: int
    note: str | None = None
    confidence: float


class InventoryView(BaseModel):
    sku: str
    reconciled: int
    confidence: float
    winning_source: str
    sources: list[SourceCount]
    explanation: str


class TransferResult(BaseModel):
    ok: bool
    sku: str
    quantity: int
    source_shelf: str
    dest_shelf: str
    source_after: int | None = None
    dest_after: int | None = None
    message: str
