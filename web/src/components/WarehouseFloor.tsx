import type { WarehouseView } from "../api/types";
import { segmentScore, tierColor } from "../api/util";

interface Props {
  view: WarehouseView;
  pathMap: boolean;
  onMove: (to: string) => void;
  onSelectShelf: (id: string) => void;
}

function segId(a: string, b: string): string {
  return "seg:" + [a, b].sort().join("|");
}

export default function WarehouseFloor({ view, pathMap, onMove, onSelectShelf }: Props) {
  const J = view.junctions;
  // segment canonical id -> reachable option (from current position)
  const reachable = new Map(view.options.map((o) => [o.segment, o]));

  return (
    <svg
      className="floor"
      viewBox="0 0 1040 660"
      preserveAspectRatio="xMidYMid meet"
      aria-label="Warehouse floor"
    >
      <rect x="8" y="8" width="1024" height="644" rx="18" fill="var(--floor)" stroke="var(--line)" />
      <g opacity="0.5">
        {Array.from({ length: 16 }, (_, i) => 70 + i * 58).map((x) => (
          <line key={"vx" + x} x1={x} y1="20" x2={x} y2="632" stroke="var(--floor-2)" strokeWidth="1" />
        ))}
        {Array.from({ length: 10 }, (_, i) => 70 + i * 58).map((y) => (
          <line key={"hz" + y} x1="20" y1={y} x2="1020" y2={y} stroke="var(--floor-2)" strokeWidth="1" />
        ))}
      </g>

      {/* zone strips */}
      <ZoneStrip label="LOADING BAY" x={24} y={90} w={116} h={470} hazardRight />
      <ZoneStrip label="DISPATCH" x={900} y={90} w={116} h={470} />

      {/* segments */}
      {view.segments.map((s) => {
        const id = segId(s.a, s.b);
        const A = J[s.a];
        const B = J[s.b];
        if (!A || !B) return null;
        const opt = reachable.get(id);
        const score = segmentScore(s);
        const show = pathMap;
        const base = show ? tierColor(score.tier) : "var(--rack-face)";
        return (
          <g
            key={id}
            className={"seg" + (opt ? " reachable" : "")}
            onClick={opt ? () => onMove(opt.to) : undefined}
          >
            {opt && (
              <line
                x1={A.x} y1={A.y} x2={B.x} y2={B.y}
                stroke={tierColor(opt.tier)} strokeWidth="17" strokeLinecap="round"
                opacity="0.22" className="pulse"
              />
            )}
            <line
              x1={A.x} y1={A.y} x2={B.x} y2={B.y}
              stroke={base} strokeWidth={opt ? 15 : 12} strokeLinecap="round"
              opacity={show ? 0.9 : 0.5}
            />
            {show && (
              <>
                <rect x={(A.x + B.x) / 2 - 20} y={(A.y + B.y) / 2 - 11} width="40" height="22" rx="7"
                  fill="var(--panel)" stroke="var(--line)" />
                <text x={(A.x + B.x) / 2} y={(A.y + B.y) / 2 + 5} textAnchor="middle" fontSize="12"
                  fontWeight="700" fontFamily="var(--f-mono)" fill={tierColor(score.tier)}>
                  {Math.round(score.confidence * 100)}
                </text>
              </>
            )}
          </g>
        );
      })}

      {/* junctions */}
      {Object.values(J).map((j) => (
        <circle key={j.id} cx={j.x} cy={j.y} r="7" fill="var(--panel)" stroke="var(--ink-3)" strokeWidth="2" />
      ))}

      {/* shelves */}
      {view.shelves.map((sh) => {
        const j = J[sh.near];
        if (!j) return null;
        const x = j.x + sh.dx - 34;
        const y = j.y + sh.dy - 20;
        const w = 68, h = 40;
        const disputed = sh.stock.length > 1;
        const stCol = disputed || sh.low ? "var(--warn)" : sh.target ? "var(--accent)" : "var(--clear)";
        return (
          <g key={sh.id} className="shelf" role="button" tabIndex={0}
            onClick={() => onSelectShelf(sh.id)}
            onKeyDown={(e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); onSelectShelf(sh.id); } }}>
            <rect x={x} y={y} width={w} height={h} rx="6" fill="var(--rack-face)" stroke="var(--rack)" />
            <rect x={x} y={y} width={w} height="11" rx="6" fill="var(--rack-top)" />
            <circle cx={x + w - 11} cy={y + 21} r="5" fill={stCol}
              className={disputed || sh.target ? "pulse" : undefined} />
            <text x={x + 7} y={y + 26} fontSize="11" fontFamily="var(--f-mono)" fill="var(--ink)">{sh.id}</text>
            {sh.target && (
              <text x={x + w / 2} y={y - 7} textAnchor="middle" fontSize="10" fontWeight="700"
                letterSpacing="1" fill="var(--accent)">★ GOAL</text>
            )}
          </g>
        );
      })}

      {/* robot */}
      {J[view.position] && (
        <g className="robot-g" style={{ transform: `translate(${J[view.position].x}px, ${J[view.position].y}px)` }}>
          <rect x="-17" y="-13" width="34" height="26" rx="7" fill="var(--robot)" />
          <rect x="-17" y="-13" width="34" height="9" rx="7" fill="#fff" opacity="0.18" />
          <circle cx="0" cy="0" r="4" fill="#fff" opacity="0.92" />
          <text x="0" y="27" textAnchor="middle" fontSize="10" fontWeight="700" letterSpacing="1" fill="var(--ink-3)">AGV-1</text>
        </g>
      )}
    </svg>
  );
}

function ZoneStrip({ label, x, y, w, h, hazardRight }: {
  label: string; x: number; y: number; w: number; h: number; hazardRight?: boolean;
}) {
  const sx = hazardRight ? x + w - 8 : x;
  return (
    <g>
      <rect x={x} y={y} width={w} height={h} rx="12" fill="var(--panel-2)" stroke="var(--line)" strokeWidth="1.5" />
      <rect x={sx} y={y + 10} width="8" height={h - 20} rx="3" fill="var(--lane)" opacity="0.85" />
      <text x={x + w / 2} y={y + h / 2} textAnchor="middle" fontSize="16" fontWeight="700" letterSpacing="2"
        fill="var(--ink-2)" transform={`rotate(-90 ${x + w / 2} ${y + h / 2})`}>{label}</text>
    </g>
  );
}
