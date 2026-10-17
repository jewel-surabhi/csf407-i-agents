import type {
  AgentResult,
  InventoryView,
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
};
