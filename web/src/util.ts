// Shared formatting helpers used by App and Inspector.

export function fmtTime(iso?: string | null): string {
  if (!iso) return "";
  try {
    const d = new Date(iso);
    return d.toLocaleTimeString(undefined, { hour: "2-digit", minute: "2-digit" });
  } catch {
    return iso;
  }
}

export function ageHours(iso?: string | null): number {
  if (!iso) return 0;
  const t = new Date(iso).getTime();
  if (Number.isNaN(t)) return 0;
  return Math.max(0, (Date.now() - t) / 3_600_000);
}

export function freshnessTier(hours: number): "fresh" | "aging" | "stale" {
  if (hours < 1) return "fresh";
  if (hours < 24) return "aging";
  return "stale";
}

export function fmtAge(hours: number): string {
  if (hours < 1 / 60) return "just now";
  if (hours < 1) return `${Math.round(hours * 60)}m`;
  if (hours < 24) return `${hours.toFixed(1)}h`;
  return `${Math.round(hours / 24)}d`;
}

// Human-readable label for a raw perspective slug like `third_party:erp`.
export function perspectiveLabel(p: string): string {
  if (p === "self") return "self";
  if (p === "user") return "user";
  if (p.startsWith("third_party:")) return p.slice("third_party:".length);
  return p;
}

// Class suffix driving perspective-chip color in CSS.
export function perspectiveKind(p: string): "self" | "user" | "third" {
  if (p === "self") return "self";
  if (p === "user") return "user";
  return "third";
}

// Arrow glyph from one point to another (map directions for move chips).
const GLYPH = { right: "→", left: "←", up: "↑", down: "↓" };
export function dirGlyph(from: { x: number; y: number }, to: { x: number; y: number }): string {
  if (to.x > from.x) return GLYPH.right;
  if (to.x < from.x) return GLYPH.left;
  if (to.y < from.y) return GLYPH.up;
  return GLYPH.down;
}
