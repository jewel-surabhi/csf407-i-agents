import { useMemo, useState } from "react";

// Hex swatches for the palette — kept separate from the design tokens
// because these ARE the subject of the demo, not chrome.
const COLORS: Record<string, { hex: string; label: string }> = {
  red: { hex: "#cf4330", label: "red" },
  brown: { hex: "#8b5a3c", label: "brown" },
  blue: { hex: "#3064b5", label: "blue" },
  green: { hex: "#2c9b6a", label: "green" },
  yellow: { hex: "#d9a830", label: "yellow" },
};

// Lighting tints that visibly skew the camera reading. Yellow light on a red
// box washes it out to brown; blue light shifts it cool. "dim" just drains
// saturation. These are rough — the point is visible, not colorimetric.
const LIGHTING: Record<string, { tint: string; mix: number; label: string }> = {
  white: { tint: "#ffffff", mix: 0, label: "neutral white" },
  yellow: { tint: "#d9a830", mix: 0.55, label: "warm yellow" },
  blue: { tint: "#3064b5", mix: 0.45, label: "cool blue" },
  dim: { tint: "#404040", mix: 0.55, label: "dim / shadow" },
};

const SOURCES = [
  "Maintenance Bot 02",
  "ERP record",
  "Sales log",
  "Night-shift operator",
];

interface ProvRow {
  id: number;
  source: string;
  color: string;
  confidence: number;
  recordedAtHour: number;
}

function cameraPerceives(truth: string, lighting: string): string {
  const base = COLORS[truth].hex;
  const l = LIGHTING[lighting];
  if (l.mix === 0) return base;
  return `color-mix(in srgb, ${base} ${Math.round((1 - l.mix) * 100)}%, ${l.tint} ${Math.round(l.mix * 100)}%)`;
}

// What NAME the camera would attach to that perceived color. It never maps
// back to the real name — that's the whole point of the scenario.
function cameraReportsName(truth: string, lighting: string): string {
  if (lighting === "white") return truth;
  if (lighting === "dim") return "dark / washed out";
  if (lighting === "yellow") {
    if (truth === "red") return "brown";
    if (truth === "blue") return "green";
    if (truth === "green") return "yellow-green";
    if (truth === "brown") return "brown";
    return "warm " + truth;
  }
  // blue light
  if (truth === "red") return "purple";
  if (truth === "yellow") return "green";
  if (truth === "brown") return "slate";
  return "cool " + truth;
}

export default function PerspectiveStudio() {
  const [truth, setTruth] = useState<string>("red");
  const [lighting, setLighting] = useState<string>("yellow");
  const [userExpects, setUserExpects] = useState<string>("red");
  const [hour, setHour] = useState(0);
  const [prov, setProv] = useState<ProvRow[]>([
    { id: 1, source: "Maintenance Bot 02", color: "blue", confidence: 0.80, recordedAtHour: 0 },
  ]);
  const [newSource, setNewSource] = useState(SOURCES[0]);
  const [newColor, setNewColor] = useState("blue");
  const [newConf, setNewConf] = useState(0.7);

  const cameraColor = useMemo(() => cameraPerceives(truth, lighting), [truth, lighting]);
  const cameraName = useMemo(() => cameraReportsName(truth, lighting), [truth, lighting]);

  // The agent's "third-party" view is the HIGHEST-CONFIDENCE live entry
  // after decay — 10%/hour, matching the demo convention in ghost.html.
  const decayedProv = prov
    .map((p) => ({ ...p, liveConf: Math.max(0, p.confidence - 0.1 * (hour - p.recordedAtHour)) }))
    .sort((a, b) => b.liveConf - a.liveConf);
  const thirdParty = decayedProv[0];
  const thirdPartyColor = thirdParty && thirdParty.liveConf > 0 ? thirdParty.color : null;

  const addRow = () => {
    setProv((rows) => [
      ...rows,
      {
        id: Date.now(),
        source: newSource,
        color: newColor,
        confidence: newConf,
        recordedAtHour: hour,
      },
    ]);
  };

  return (
    <section className="ps">
      <div className="ps-head">
        <div>
          <h3 className="sb-h" style={{ margin: 0 }}>Perspective Studio</h3>
          <p className="insp-note" style={{ margin: "4px 0 0" }}>
            Change the ground truth, the lighting, what the user was told, and what
            third-party logs claim. Watch the three perspectives stay separate —
            no source ever gets re-attributed to another.
          </p>
        </div>
        <div className="ps-hour">
          <span>Hour {hour}</span>
          <button className="icnbtn" onClick={() => setHour((h) => h + 1)}>Next hour ▸</button>
          <button className="icnbtn ghost" onClick={() => setHour(0)}>Reset</button>
        </div>
      </div>

      <div className="ps-grid">
        <div className="ps-reality">
          <div className="ps-cap">Reality (hidden from the agent)</div>
          <div
            className="ps-box"
            style={{
              background: COLORS[truth].hex,
              boxShadow: `inset 0 -40px 60px -20px ${LIGHTING[lighting].tint}${LIGHTING[lighting].mix === 0 ? "00" : "80"}`,
            }}
          >
            <span className="ps-box-tag">WIDGET-42</span>
          </div>
          <div className="ps-controls">
            <label>
              <span>Ground-truth color</span>
              <select value={truth} onChange={(e) => setTruth(e.target.value)}>
                {Object.entries(COLORS).map(([k, v]) => <option key={k} value={k}>{v.label}</option>)}
              </select>
            </label>
            <label>
              <span>Lighting</span>
              <select value={lighting} onChange={(e) => setLighting(e.target.value)}>
                {Object.entries(LIGHTING).map(([k, v]) => <option key={k} value={k}>{v.label}</option>)}
              </select>
            </label>
            <label>
              <span>User was told it is</span>
              <select value={userExpects} onChange={(e) => setUserExpects(e.target.value)}>
                {Object.keys(COLORS).map((k) => <option key={k} value={k}>{COLORS[k].label}</option>)}
              </select>
            </label>
          </div>
        </div>

        <div className="ps-perspectives">
          <div className="ps-persp psp-user-bg">
            <span className="psp psp-user">User</span>
            <div className="ps-swatch" style={{ background: COLORS[userExpects].hex }} />
            <p>Expects <b>{COLORS[userExpects].label}</b> — what they were told before ever seeing it.</p>
          </div>
          <div className="ps-persp psp-self-bg">
            <span className="psp psp-self">Self (camera)</span>
            <div className="ps-swatch" style={{ background: cameraColor }} />
            <p>Reports <b>{cameraName}</b> — camera feed through <i>{LIGHTING[lighting].label}</i> lighting.</p>
          </div>
          <div className="ps-persp psp-third-bg">
            <span className="psp psp-third">Third-party</span>
            <div className="ps-swatch" style={{ background: thirdPartyColor ? COLORS[thirdPartyColor].hex : "transparent", borderStyle: thirdPartyColor ? "solid" : "dashed" }} />
            {thirdParty && thirdParty.liveConf > 0 ? (
              <p>
                <b>{thirdParty.source}</b> says <b>{thirdParty.color}</b> ({(thirdParty.liveConf * 100).toFixed(0)}% now, after decay).
              </p>
            ) : (
              <p style={{ fontStyle: "italic" }}>No live third-party claim — all entries decayed to 0 or none recorded.</p>
            )}
          </div>
        </div>
      </div>

      <div className="ps-log">
        <div className="ps-log-h">
          <h4>Provenance log</h4>
          <div className="ps-log-add">
            <select value={newSource} onChange={(e) => setNewSource(e.target.value)}>
              {SOURCES.map((s) => <option key={s}>{s}</option>)}
            </select>
            <select value={newColor} onChange={(e) => setNewColor(e.target.value)}>
              {Object.keys(COLORS).map((k) => <option key={k} value={k}>{COLORS[k].label}</option>)}
            </select>
            <select value={newConf} onChange={(e) => setNewConf(Number(e.target.value))}>
              {[0.5, 0.6, 0.7, 0.8, 0.9, 0.95].map((c) => <option key={c} value={c}>{Math.round(c * 100)}%</option>)}
            </select>
            <button className="icnbtn" onClick={addRow}>Add report</button>
          </div>
        </div>
        <table className="ps-log-t">
          <thead>
            <tr><th>Source</th><th>Says</th><th>Stored conf.</th><th>Recorded</th><th>Live conf.</th></tr>
          </thead>
          <tbody>
            {decayedProv.map((r) => (
              <tr key={r.id} className={r.liveConf <= 0 ? "stale" : ""}>
                <td>{r.source}</td>
                <td><span className="ps-chip" style={{ background: COLORS[r.color].hex }} />{COLORS[r.color].label}</td>
                <td>{Math.round(r.confidence * 100)}%</td>
                <td>h{r.recordedAtHour}</td>
                <td>{r.liveConf > 0 ? `${(r.liveConf * 100).toFixed(0)}%` : "stale"}</td>
              </tr>
            ))}
            {decayedProv.length === 0 && (
              <tr><td colSpan={5} style={{ textAlign: "center", fontStyle: "italic", color: "var(--ink-3)" }}>No entries yet</td></tr>
            )}
          </tbody>
        </table>
      </div>
    </section>
  );
}
