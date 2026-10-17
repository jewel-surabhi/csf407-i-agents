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

export interface WarehouseView {
  id: string;
  position: string;
  goal: string;
  junctions: Record<string, Junction>;
  segments: SegmentSeed[];
  shelves: Shelf[];
  options: MoveOption[];
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
