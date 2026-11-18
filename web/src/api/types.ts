// Mirrors the backend Pydantic models (src/iagent/warehouse/models.py).

export interface Junction {
  id: string;
  label: string;
  x: number;
  y: number;
}

export interface SegmentSeed {
  a: string;
  b: string;
  state: "clear" | "blocked";
  source: string;
  confidence: number;
}

export interface StockSeed {
  source: string;
  perspective: string;
  predicate: string;
  value: number;
  note?: string | null;
}

export interface Shelf {
  id: string;
  near: string;
  dx: number;
  dy: number;
  target: boolean;
  low: boolean;
  stock: StockSeed[];
}

export type Tier = "high" | "medium" | "low";

export interface MoveOption {
  to: string;
  to_label: string;
  segment: string;
  confidence: number;
  tier: Tier;
  source: string;
  passable: boolean;
  reason: string;
}

export interface FiredEvent {
  event_id: string;
  action: string;
  message: string;
}

export interface WarehouseView {
  id: string;
  position: string;
  goal: string;
  junctions: Record<string, Junction>;
  segments: SegmentSeed[];
  shelves: Shelf[];
  options: MoveOption[];
  events_fired?: FiredEvent[];
  moves_made?: number;
}

export interface SourceCount {
  source: string;
  perspective: string;
  predicate: string;
  value: number;
  note?: string | null;
  confidence: number;
}

export interface InventoryView {
  sku: string;
  reconciled: number;
  confidence: number;
  winning_source: string;
  sources: SourceCount[];
  explanation: string;
}

export interface TransferResult {
  ok: boolean;
  sku: string;
  quantity: number;
  source_shelf: string;
  dest_shelf: string;
  source_after?: number | null;
  dest_after?: number | null;
  message: string;
}

export interface ShelfLot {
  sku: string;
  quantity: number;
  source: string;
  confidence: number;
  is_native: boolean;
}

export interface ShelfContents {
  shelf: string;
  lots: ShelfLot[];
  total_units: number;
}

// Mirrors declarative.Source — one row in the sources registry.
export interface Source {
  source_id: string;
  kind: "static_map" | "sensor" | "agent" | "user" | "llm";
  trust_prior: number;
  decay_rate_per_hour: number;
  description?: string | null;
}

// Mirrors declarative.Belief — includes decay annotations from the reader.
export interface Belief {
  subject: string;
  predicate: string;
  object: string;
  confidence: number;
  source: string;
  perspective: string;
  status: "active" | "downgraded" | "retracted";
  observed_at: string;
  provenance_id?: number | null;
  effective_confidence?: number | null;
  age_hours?: number | null;
}

// Mirrors declarative.ProvenanceEntry — one row in the SQLite history log.
export interface ProvenanceEntry {
  provenance_id?: number | null;
  subject: string;
  predicate: string;
  object: string;
  confidence: number;
  source_id: string;
  perspective: string;
  observed_at: string;
  recorded_at?: string | null;
  status: "active" | "downgraded" | "retracted";
  payload?: Record<string, unknown> | null;
}

// /api/query (the live agent)
export interface TraceStep {
  idx: number;
  thought?: string | null;
  tool_call?: { name: string; arguments: Record<string, unknown> } | null;
  observation?: { tool: string; ok: boolean; content: unknown } | null;
  final_answer?: string | null;
}

export interface AgentResult {
  answer: string;
  trace: { query: string; steps: TraceStep[]; final_answer: string };
  tools_called: string[];
}
