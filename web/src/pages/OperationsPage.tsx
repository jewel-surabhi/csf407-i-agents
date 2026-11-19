import { useState } from "react";
import WarehouseFloor from "../components/WarehouseFloor";
import Inspector, { type InspectorTab } from "../components/Inspector";
import { useWarehouse } from "../context/WarehouseContext";

// Suggested commands grouped by category — the bottom command bar surfaces
// these instead of a chatbot prompt pile.
const SUGGESTIONS: { group: string; items: { label: string; query: string }[] }[] = [
  {
    group: "Routes",
    items: [
      {
        label: "Why is aisle B upper at 60%?",
        query: "Why is the segment to Aisle B Upper only at 60% confidence from where I am?",
      },
      {
        label: "Scenario A — path clear?",
        query:
          "Is the route from B-S to B-D clear? Justify by inspecting the belief graph — map claim vs live LiDAR — and explain which source wins.",
      },
    ],
  },
  {
    group: "Stock",
    items: [
      {
        label: "Three perspectives on WIDGET-42",
        query:
          "For WIDGET-42, report three perspectives separately: user, your own (self), and third-party ERP + sales. Do not merge them.",
      },
      {
        label: "Lowest-stock SKU",
        query: "Which SKU has the lowest reconciled count across all shelves?",
      },
    ],
  },
  {
    group: "Policy",
    items: [
      { label: "What are our hours?", query: "What are the facility's operating hours?" },
      { label: "Who runs the night shift?", query: "Who is the night-shift supervisor?" },
      {
        label: "How do you avoid hallucinating?",
        query:
          "Explain, citing your tool calls, how your answers are grounded in the belief graph rather than invented.",
      },
    ],
  },
];

export default function OperationsPage() {
  const { view, pathMap, setPathMap, doMove, selectShelf, busy, ask } = useWarehouse();
  const [tab, setTab] = useState<InspectorTab>("now");
  const [askText, setAskText] = useState("");

  if (!view) return null;

  const run = (text: string) => {
    setTab("now");
    ask(text);
  };

  const onSelectShelf = (id: string) => {
    selectShelf(id);
    setTab("selection");
  };

  return (
    <>
      <main className="stage">
        <section className="floor">
          <WarehouseFloor view={view} pathMap={pathMap} onMove={doMove} onSelectShelf={onSelectShelf} />

          {pathMap && (
            <div className="legend">
              <span className="l"><i style={{ background: "var(--clear)" }} />Sensor-confirmed</span>
              <span className="l"><i style={{ background: "var(--warn)" }} />Map only</span>
              <span className="l"><i style={{ background: "var(--blocked)" }} />Conflict / blocked</span>
            </div>
          )}

          <button
            className={"tgl floor-tgl" + (pathMap ? " on" : "")}
            onClick={() => setPathMap((v) => !v)}
            title="Overlay confidence scores on every segment"
          >
            <span className="sw" />Path map
          </button>
        </section>

        <Inspector tab={tab} setTab={setTab} />
      </main>

      <footer className="cmdbar">
        <div className="sugg">
          {SUGGESTIONS.map((g) => (
            <div className="sugg-g" key={g.group}>
              <span className="sugg-gn">{g.group}</span>
              {g.items.map((it) => (
                <button
                  className="sugg-i"
                  key={it.label}
                  onClick={() => { setAskText(it.query); run(it.query); }}
                  title={it.query}
                >
                  {it.label}
                </button>
              ))}
            </div>
          ))}
        </div>

        <form
          className="cmd"
          onSubmit={(e) => { e.preventDefault(); run(askText); }}
        >
          <span className="cmd-prefix">›</span>
          <input
            value={askText}
            onChange={(e) => setAskText(e.target.value)}
            placeholder="Ask the warehouse — routes, stock, policies, reasoning…"
            disabled={busy}
          />
          <button type="submit" className="btn-primary" disabled={busy || !askText.trim()}>
            {busy ? "Thinking…" : "Ask"}
          </button>
        </form>
      </footer>
    </>
  );
}
