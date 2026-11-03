import type { SegmentSeed, Tier } from "./types";

const SENSORS = new Set(["lidar_front", "warehouse_scanner", "camera_front"]);

export function tierOf(conf: number): Tier {
  return conf >= 0.8 ? "high" : conf >= 0.4 ? "medium" : "low";
}

export function tierColor(tier: Tier): string {
  return tier === "high"
    ? "var(--clear)"
    : tier === "medium"
      ? "var(--warn)"
      : "var(--blocked)";
}

export const SRC_LABEL: Record<string, string> = {
  lidar_front: "front LiDAR",
  warehouse_scanner: "floor scanner",
  camera_front: "dock camera",
  default_map: "floor map",
  erp_system: "ERP",
  sales_log: "sales log",
};

export function srcLabel(src: string): string {
  return SRC_LABEL[src] ?? src;
}

// Client-side mirror of the backend's score_segment, for the path-map overlay
// (which needs every segment scored, not just reachable ones).
export function segmentScore(seg: SegmentSeed): {
  confidence: number;
  tier: Tier;
  passable: boolean;
} {
  if (seg.state === "blocked") {
    const c = Math.round((1 - seg.confidence) * 100) / 100;
    return { confidence: c, tier: tierOf(c), passable: false };
  }
  const c = Math.round(seg.confidence * 100) / 100;
  return { confidence: c, tier: tierOf(c), passable: true };
}

export function isSensor(src: string): boolean {
  return SENSORS.has(src);
}
