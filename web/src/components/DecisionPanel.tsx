import type { WarehouseView } from "../api/types";

interface Props {
  view: WarehouseView;
  explain: boolean;
  onMove: (to: string) => void;
}

const GLYPH: Record<string, string> = { right: "→", left: "←", up: "↑", down: "↓" };

function dirGlyph(from: { x: number; y: number }, to: { x: number; y: number }): string {
  if (to.x > from.x) return GLYPH.right;
  if (to.x < from.x) return GLYPH.left;
  if (to.y < from.y) return GLYPH.up;
  return GLYPH.down;
}

export default function DecisionPanel({ view, explain, onMove }: Props) {
  const here = view.junctions[view.position];
  const goalShelf = view.shelves.find((s) => s.id === view.goal);
  const arrived = goalShelf && goalShelf.near === view.position;

  return (
    <div className="decide">
      <div className="decide-h">
        <span className="pin">
          <svg viewBox="0 0 24 24" fill="none" stroke="#fff" strokeWidth="2">
            <circle cx="12" cy="10" r="3" />
            <path d="M12 2a8 8 0 0 0-8 8c0 5 8 12 8 12s8-7 8-12a8 8 0 0 0-8-8z" />
          </svg>
        </span>
        <span className="loc">
          <small>AGV-1 at</small>
          {here?.label ?? view.position}
        </span>
      </div>

      {arrived ? (
        <div className="arrived">
          <div className="big">✓ Arrived at {view.goal}</div>
          <div className="sm">High-confidence route complete. Open the shelf to pick, or ask the agent.</div>
        </div>
      ) : (
        <>
          <div className="decide-note">
            Your next move — the agent scores each option on grounded confidence.
          </div>
          {view.options.map((o) => {
            const to = view.junctions[o.to];
            const pct = Math.round(o.confidence * 100);
            const goalHint = view.shelves.some((s) => s.id === view.goal && s.near === o.to);
            return (
              <button key={o.to} className="opt" onClick={() => onMove(o.to)}>
                <div className="opt-top">
                  <span className="opt-dir">{here && to ? dirGlyph(here, to) : "•"}</span>
                  <span className="opt-name">
                    {to?.label ?? o.to} {goalHint && <small>· goal shelf</small>}
                  </span>
                  <span className={"opt-conf tier-" + o.tier}>{pct}%</span>
                </div>
                <div className="opt-bar">
                  <i className={"bar-" + o.tier} style={{ width: pct + "%" }} />
                </div>
                <div className="opt-reason">
                  <span>{o.tier === "high" ? "✓" : o.tier === "medium" ? "⚠" : "✗"}</span>
                  <span>
                    {o.passable
                      ? o.tier === "high"
                        ? "Clear — sensor-confirmed"
                        : "Map only — unverified"
                      : "Sensor conflict — avoid"}
                  </span>
                </div>
                {explain && <div className="opt-why">{o.reason}</div>}
              </button>
            );
          })}
        </>
      )}
    </div>
  );
}
