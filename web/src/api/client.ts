import type {
  AgentResult,
  Belief,
  InventoryView,
  ProvenanceEntry,
  ShelfContents,
  Source,
  TraceStep,
  TransferResult,
  WarehouseView,
} from "./types";

async function asJson<T>(r: Response): Promise<T> {
  if (!r.ok) {
    const text = await r.text().catch(() => r.statusText);
    throw new Error(`${r.status}: ${text}`);
  }
  return r.json() as Promise<T>;
}

const jsonHeaders = { "Content-Type": "application/json" };

export const api = {
  getWarehouse: () => fetch("/api/warehouse").then(asJson<WarehouseView>),

  move: (to: string) =>
    fetch("/api/warehouse/move", {
      method: "POST",
      headers: jsonHeaders,
      body: JSON.stringify({ to }),
    }).then(asJson<WarehouseView>),

  reset: () =>
    fetch("/api/warehouse/reset", { method: "POST" }).then(asJson<WarehouseView>),

  inventory: (sku: string) =>
    fetch(`/api/warehouse/inventory/${encodeURIComponent(sku)}`).then(
      asJson<InventoryView>,
    ),

  bin: (shelfId: string) =>
    fetch(`/api/warehouse/bin/${encodeURIComponent(shelfId)}`).then(
      asJson<ShelfContents>,
    ),

  provenance: (sku: string) =>
    fetch(`/api/warehouse/provenance/${encodeURIComponent(sku)}`).then(
      asJson<ProvenanceEntry[]>,
    ),

  transfer: (sku: string, dest: string, quantity: number) =>
    fetch("/api/warehouse/transfer", {
      method: "POST",
      headers: jsonHeaders,
      body: JSON.stringify({ sku, dest, quantity }),
    }).then(asJson<TransferResult>),

  // The live Groq agent — used by the "Explain" / ask flow.
  query: (query: string, scenario?: string) =>
    fetch("/api/query", {
      method: "POST",
      headers: jsonHeaders,
      body: JSON.stringify({ query, scenario: scenario ?? null }),
    }).then(asJson<AgentResult>),

  // Sources + their decay rates (drives the Trust panel badges).
  sources: () => fetch("/api/warehouse/sources").then(asJson<Source[]>),

  // Reflexion-style meta-beliefs. GET lists them; POST writes a new one.
  listReflections: () =>
    fetch("/api/warehouse/reflections").then(asJson<Belief[]>),

  addReflection: (lesson: string, topic = "general", confidence = 0.6) =>
    fetch("/api/warehouse/reflections", {
      method: "POST",
      headers: jsonHeaders,
      body: JSON.stringify({ lesson, topic, confidence }),
    }).then(asJson<Belief>),

  /**
   * Streaming version of `query` — emits steps one by one as the agent runs
   * them, then calls onDone with the final AgentResult (or onError).
   * Returns an abort function the caller can invoke to cancel the stream.
   */
  queryStream: (
    query: string,
    opts: {
      onStep?: (step: TraceStep) => void;
      onDone?: (result: AgentResult) => void;
      onError?: (message: string) => void;
      scenario?: string;
      maxSteps?: number;
    } = {},
  ): (() => void) => {
    const controller = new AbortController();
    (async () => {
      try {
        const res = await fetch("/api/query/stream", {
          method: "POST",
          headers: jsonHeaders,
          signal: controller.signal,
          body: JSON.stringify({
            query,
            scenario: opts.scenario ?? null,
            max_steps: opts.maxSteps ?? null,
          }),
        });
        if (!res.ok || !res.body) {
          opts.onError?.(`stream HTTP ${res.status}`);
          return;
        }
        const reader = res.body.getReader();
        const decoder = new TextDecoder();
        let buf = "";
        for (;;) {
          const { value, done } = await reader.read();
          if (done) break;
          buf += decoder.decode(value, { stream: true });
          // SSE frames are separated by a blank line.
          let idx = buf.indexOf("\n\n");
          while (idx !== -1) {
            const frame = buf.slice(0, idx);
            buf = buf.slice(idx + 2);
            const line = frame.split("\n").find((l) => l.startsWith("data: "));
            if (line) {
              try {
                const msg = JSON.parse(line.slice(6));
                if (msg.kind === "step") opts.onStep?.(msg.data as TraceStep);
                else if (msg.kind === "done") opts.onDone?.(msg.data as AgentResult);
                else if (msg.kind === "error") {
                  const err = msg.data || {};
                  opts.onError?.(err.message || "agent error");
                }
              } catch {
                /* malformed frame — skip */
              }
            }
            idx = buf.indexOf("\n\n");
          }
        }
      } catch (e) {
        if ((e as Error).name !== "AbortError") {
          opts.onError?.(String(e));
        }
      }
    })();
    return () => controller.abort();
  },
};
