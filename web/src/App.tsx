import { useCallback, useEffect, useRef, useState } from "react";
import { api } from "./api/client";
import type {
  AgentResult,
  Belief,
  InventoryView,
  ProvenanceEntry,
  Shelf,
  ShelfContents,
  Source,
  TraceStep,
  WarehouseView,
} from "./api/types";
import WarehouseFloor from "./components/WarehouseFloor";
import Inspector, { type InspectorTab } from "./components/Inspector";

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

interface Toast {
  id: number;
  dot: string;
  html: string;
}

export default function App() {
  const [view, setView] = useState<WarehouseView | null>(null);
  const [offline, setOffline] = useState(false);
  const [pathMap, setPathMap] = useState(false);
  const [tab, setTab] = useState<InspectorTab>("now");

  const [shelfSel, setShelfSel] = useState<Shelf | null>(null);
  const [inv, setInv] = useState<InventoryView | null>(null);
  const [bin, setBin] = useState<ShelfContents | null>(null);
  const [prov, setProv] = useState<ProvenanceEntry[] | null>(null);
  const [recentArrival, setRecentArrival] = useState<string | null>(null);
  const [xferDest, setXferDest] = useState<string>("");
  const [xferQty, setXferQty] = useState<number>(10);

  const [answer, setAnswer] = useState<AgentResult | null>(null);
  const [busy, setBusy] = useState(false);
  const [liveSteps, setLiveSteps] = useState<TraceStep[]>([]);
  const [askText, setAskText] = useState("");

  const [sources, setSources] = useState<Source[]>([]);
  const [reflections, setReflections] = useState<Belief[]>([]);
  const [newTopic, setNewTopic] = useState("");
  const [newLesson, setNewLesson] = useState("");

  const [toasts, setToasts] = useState<Toast[]>([]);
  const toastId = useRef(0);

  const toast = useCallback((dot: string, html: string) => {
    const id = ++toastId.current;
    setToasts((t) => [...t, { id, dot, html }]);
    setTimeout(() => setToasts((t) => t.filter((x) => x.id !== id)), 2600);
  }, []);

  const load = useCallback(async () => {
    try {
      const [v, srcs, refs] = await Promise.all([
        api.getWarehouse(),
        api.sources(),
        api.listReflections(),
      ]);
      setView(v);
      setSources(srcs);
      setReflections(refs);
      setOffline(false);
    } catch {
      setOffline(true);
    }
  }, []);

  useEffect(() => { load(); }, [load]);

  const refreshReflections = useCallback(async () => {
    try { setReflections(await api.listReflections()); } catch { /* ignore */ }
  }, []);

  const teachReflection = useCallback(async () => {
    const lesson = newLesson.trim();
    if (!lesson) return;
    try {
      const b = await api.addReflection(lesson, newTopic.trim() || "general");
      toast("var(--accent)", `Learned: <b>${b.predicate.split(":").pop()}</b>`);
      setNewLesson("");
      setNewTopic("");
      refreshReflections();
    } catch (e) {
      toast("var(--blocked)", String(e));
    }
  }, [newLesson, newTopic, toast, refreshReflections]);

  const doMove = useCallback(
    async (to: string) => {
      if (!view) return;
      const opt = view.options.find((o) => o.to === to);
      try {
        const next = await api.move(to);
        setView(next);
        if (opt && !opt.passable)
          toast("var(--blocked)", `Overrode agent · <b>${Math.round(opt.confidence * 100)}%</b>`);
        else if (opt)
          toast("var(--clear)", `Moved · <b>${Math.round(opt.confidence * 100)}%</b> confident`);
        for (const ev of next.events_fired ?? []) {
          toast(ev.action === "toast" ? "var(--accent)" : "var(--warn)",
            `⚡ <b>${ev.event_id}</b> · ${ev.message}`);
        }
        if ((next.events_fired ?? []).some((e) => e.action !== "toast") && shelfSel) {
          try {
            const [i, b, pr] = await Promise.all([
              api.inventory(shelfSel.id),
              api.bin(shelfSel.id),
              api.provenance(shelfSel.id),
            ]);
            setInv(i); setBin(b); setProv(pr);
          } catch { /* ignore */ }
        }
        const goal = next.shelves.find((s) => s.id === next.goal);
        if (goal && goal.near === next.position)
          toast("var(--accent)", `Reached <b>${next.goal}</b> ★`);
      } catch (e) {
        toast("var(--blocked)", String(e));
      }
    },
    [view, toast, shelfSel],
  );

  const selectShelf = useCallback(async (id: string) => {
    if (!view) return;
    const sh = view.shelves.find((s) => s.id === id) ?? null;
    setShelfSel(sh);
    setInv(null); setBin(null); setProv(null); setRecentArrival(null);
    const others = view.shelves.filter((s) => s.id !== id);
    setXferDest(others[0]?.id ?? "");
    setTab("selection"); // auto-focus the Inspector
    try {
      const [i, b, pr] = await Promise.all([
        api.inventory(id), api.bin(id), api.provenance(id),
      ]);
      setInv(i); setBin(b); setProv(pr);
    } catch { /* leave null */ }
  }, [view]);

  const closeSelection = useCallback(() => {
    setShelfSel(null);
    setInv(null); setBin(null); setProv(null); setRecentArrival(null);
    setTab("now");
  }, []);

  const doTransfer = useCallback(async () => {
    if (!shelfSel) return;
    try {
      const res = await api.transfer(shelfSel.id, xferDest, xferQty);
      if (!res.ok) { toast("var(--blocked)", res.message); return; }
      toast("var(--clear)", `Moved <b>${res.quantity} × ${res.sku}</b> → ${res.dest_shelf}`);
      const [v, srcInv, srcBin, destBin, srcProv] = await Promise.all([
        api.getWarehouse(), api.inventory(shelfSel.id),
        api.bin(shelfSel.id), api.bin(xferDest), api.provenance(shelfSel.id),
      ]);
      setView(v); setInv(srcInv); setBin(srcBin); setProv(srcProv);
      setRecentArrival(
        `+${res.quantity} ${res.sku} landed at ${res.dest_shelf} — ${res.dest_shelf} now holds ${destBin.total_units} total units (${destBin.lots.length} lot${destBin.lots.length === 1 ? "" : "s"}).`,
      );
      setTimeout(() => setRecentArrival(null), 6000);
    } catch (e) { toast("var(--blocked)", String(e)); }
  }, [shelfSel, xferDest, xferQty, toast]);

  const ask = useCallback((text: string) => {
    if (!text.trim()) return;
    setBusy(true);
    setAnswer(null);
    setLiveSteps([]);
    setTab("now"); // show the live reasoning inline
    api.queryStream(text, {
      onStep: (step) => setLiveSteps((prev) => [...prev, step]),
      onDone: (result) => { setAnswer(result); setBusy(false); },
      onError: (msg) => { toast("var(--blocked)", "Agent error: " + msg); setBusy(false); },
      maxSteps: 6,
    });
  }, [toast]);

  const reset = useCallback(async () => {
    try {
      setView(await api.reset());
      setShelfSel(null); setInv(null); setBin(null); setProv(null);
      setAnswer(null); setLiveSteps([]);
      toast("var(--accent)", "Warehouse reset to start");
    } catch (e) { toast("var(--blocked)", String(e)); }
  }, [toast]);

  const toggleTheme = () => {
    const cur = document.documentElement.getAttribute("data-theme");
    const sysDark = window.matchMedia("(prefers-color-scheme:dark)").matches;
    document.documentElement.setAttribute(
      "data-theme",
      cur ? (cur === "dark" ? "light" : "dark") : sysDark ? "light" : "dark",
    );
  };

  if (offline) {
    return (
      <div className="offline">
        <div className="card">
          <h2>Backend not reachable</h2>
          <p>Start the FastAPI server so the warehouse can load its grounded state:</p>
          <pre>$env:PYTHONPATH='src'; python -m uvicorn iagent.api.server:app --reload --port 8000 --app-dir src</pre>
          <p>Then reload this page. The dev server proxies <code>/api</code> to port 8000.</p>
          <button className="btn-primary" onClick={load}>Retry</button>
        </div>
      </div>
    );
  }

  if (!view) {
    return <div className="offline"><div className="card"><h2>Loading Warehouse 7…</h2></div></div>;
  }

  const here = view.junctions[view.position];

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <span className="brand-dot" />
          <h1>Warehouse&nbsp;7</h1>
          <span className="brand-sub">Ops&nbsp;AI</span>
        </div>

        <div className="status">
          <span className="status-l">AGV-1</span>
          <span className="status-v">{here?.label ?? view.position}</span>
          <span className="status-sep" />
          <span className="status-l">Goal</span>
          <span className="status-v">{view.goal}</span>
        </div>

        <div className="actions">
          <button
            className={"tgl" + (pathMap ? " on" : "")}
            onClick={() => setPathMap((v) => !v)}
            title="Overlay confidence scores on every segment"
          >
            <span className="sw" />Path map
          </button>
          <button className="icnbtn ghost" onClick={toggleTheme} aria-label="Toggle theme" title="Toggle theme">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <circle cx="12" cy="12" r="4" />
              <path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4" />
            </svg>
          </button>
          <button className="icnbtn" onClick={reset} title="Restore the warehouse to its seeded starting state">Reset</button>
        </div>
      </header>

      <main className="stage">
        <section className="floor">
          <WarehouseFloor view={view} pathMap={pathMap} onMove={doMove} onSelectShelf={selectShelf} />

          {pathMap && (
            <div className="legend">
              <span className="l"><i style={{ background: "var(--clear)" }} />Sensor-confirmed</span>
              <span className="l"><i style={{ background: "var(--warn)" }} />Map only</span>
              <span className="l"><i style={{ background: "var(--blocked)" }} />Conflict / blocked</span>
            </div>
          )}

          <div className="toasts">
            {toasts.map((t) => (
              <div className="toast" key={t.id}>
                <span className="d" style={{ background: t.dot }} />
                <span dangerouslySetInnerHTML={{ __html: t.html }} />
              </div>
            ))}
          </div>
        </section>

        <Inspector
          view={view}
          tab={tab}
          setTab={setTab}
          shelfSel={shelfSel}
          inv={inv}
          bin={bin}
          prov={prov}
          closeSelection={closeSelection}
          xferDest={xferDest}
          setXferDest={setXferDest}
          xferQty={xferQty}
          setXferQty={setXferQty}
          doTransfer={doTransfer}
          recentArrival={recentArrival}
          busy={busy}
          answer={answer}
          liveSteps={liveSteps}
          dismissAnswer={() => { setAnswer(null); setLiveSteps([]); }}
          sources={sources}
          reflections={reflections}
          newTopic={newTopic}
          setNewTopic={setNewTopic}
          newLesson={newLesson}
          setNewLesson={setNewLesson}
          teachReflection={teachReflection}
          onMove={doMove}
        />
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
                  onClick={() => { setAskText(it.query); ask(it.query); }}
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
          onSubmit={(e) => { e.preventDefault(); ask(askText); }}
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
    </div>
  );
}
